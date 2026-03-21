import os
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
sys.path.insert(0, "/home/jovyan/vasiliev/notebooks/Show-o")

import logging
import time
import math
from pathlib import Path
from typing import Union
from utils import get_optimizer
import gc
from omegaconf import OmegaConf
import mlflow
from mlflow.tracking import MlflowClient
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedType,
    DistributedDataParallelKwargs,
    # ProfileKwargs,
    set_seed
)


import mlflow
from mlflow.tracking import MlflowClient
from models.lr_schedulers import get_scheduler
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR, LambdaLR
from models.logger import set_verbosity_info, set_verbosity_error
from models import Showo, MAGVITv2, get_mask_chedule

from training.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_predict_next,
    create_attention_mask_for_mmu,
)
from training.patch_model import patch_model_with_moe
from training.moe_utils import LayerExpertStatsCollector
from training.eval_utils import (
    visualize_predictions,
    generate_images,
    log_grad_norm,
    collect_and_log_moe_activations,
    log_training_metrics,
    evaluate_mmu,
)
from benchmark.coco_dataset import COCODataset
from benchmark.fid_benchmark import ShowoBenchmark
from training.dataset_utils import create_dataloaders
from training.checkpoint_utils import save_checkpoint
from training.utils import (
    get_config,
    mask_or_random_replace_tokens,
    AverageMeter,
)
from training.moe_visualization import MoEVisualizer
from training.moe_mlflow_logger import MoEMLflowLogger
from training.profiling_context import get_profiling_context_torch
from training.sample_logger import SampleLogger


logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def prepare_inputs_and_labels(
    pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
    texts: Union[str, str],
    vq_model,
    uni_prompting,
    mask_id,
    config,
    mask_schedule,
    min_masking_rate: float = 0.0,
    is_train: bool = True,
):
    image_tokens = vq_model.get_code(pixel_values_or_image_ids)
    image_tokens = image_tokens + len(uni_prompting.text_tokenizer)

    # create MLM mask and labels
    input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
        image_tokens,
        mask_id,
        config,
        mask_schedule=mask_schedule,
        is_train=is_train,
    )
    # Correct T2I sequence: T2I → SOT → [Text tokens] → EOT → SOI → [Image tokens] → EOI
    input_ids, masks, labels = uni_prompting((texts, input_ids, labels), "t2i")
    return input_ids, labels, mask_prob, image_tokens


def set_moe_layers_global_step(model, accelerator, step_value):
    try:
        unwrapped_model = accelerator.unwrap_model(model)
    except Exception:
        unwrapped_model = model

    if not hasattr(unwrapped_model, "showo"):
        return

    model_layers = getattr(unwrapped_model.showo.model, "layers", [])
    for layer in model_layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "set_global_step"):
            layer.mlp.set_global_step(step_value)


def set_moe_layers_logging_preferences(model, accelerator, log_overall=None, log_domain=None):
    try:
        unwrapped_model = accelerator.unwrap_model(model)
    except Exception:
        unwrapped_model = model

    if not hasattr(unwrapped_model, "showo"):
        return

    model_layers = getattr(unwrapped_model.showo.model, "layers", [])
    for layer in model_layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "set_logging_preferences"):
            continue
        mlp.set_logging_preferences(log_overall=log_overall, log_domain=log_domain)


def train_step(
    batch,
    epoch,
    global_step,
    model,
    optimizer,
    lr_scheduler,
    balance_scheduler,
    temp_scheduler,
    accelerator,
    config,
    uni_prompting,
    vq_model,
    mask_dtype,
    mask_id,
    mask_schedule,
    batch_time_m,
    data_time_m,
    total_batch_size_per_gpu,
    mlflow_client,
    mlflow_run_id,
    pbar,
    sample_logger=None,
):
    batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
    # Проверяем, есть ли данные LM (может быть пустым, если batch_size_lm=0)
    batch_size_lm = config.training.get("batch_size_lm", 0)
    if batch_size_lm > 0 and "lm_flow" in batch and len(batch["lm_flow"]["input_ids"]) > 0:
        batch_size_lm = len(batch["lm_flow"]["input_ids"])
    else:
        batch_size_lm = 0
    domain_flows = [
        "vqav2_experiments_flow",
        "textvqa_experiments_flow",
        "clevr_flow",
        "vqav2_flow",
        "textvqa_flow",
        "docvqa_flow",
        "kvasir_flow",
    ]
    
    # Will be updated after collecting all domain data (если batch_size_mmu > 0)
    batch_size_mmu = config.training.get("batch_size_mmu", 0)
    domain_id = None
    domain_ids = []
    

    # Build T2I sequences
    pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
    pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
    input_ids_t2i, labels_t2i, mask_prob, image_tokens_ori = prepare_inputs_and_labels(
        pixel_values,
        texts,
        vq_model,
        uni_prompting,
        mask_id,
        config,
        mask_schedule,
        config.training.min_masking_rate,
    )
    
    # Debug: log T2I sequence shape BEFORE any padding
    if global_step == 0 and epoch == 0:
        logger.info(f"🔍 T2I SEQUENCE (raw from prompting):")
        logger.info(f"   input_ids_t2i shape: {input_ids_t2i.shape}")
        logger.info(f"   labels_t2i shape: {labels_t2i.shape}")
        logger.info(f"   Expected: max_text_len({uni_prompting.max_text_len}) + SOI(1) + image(1024) + EOI(1) = {uni_prompting.max_text_len + 1026}")
    
    attention_mask = create_attention_mask_predict_next(
        input_ids_t2i,
        pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
        soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        rm_pad_in_image=True,
        return_inverse_mask=True,
    )
    attention_mask = attention_mask.to(mask_dtype)

    # Build LM sequences (только если batch_size_lm > 0)
    if batch_size_lm > 0 and "lm_flow" in batch and len(batch["lm_flow"]["input_ids"]) > 0:
        texts_lm = batch["lm_flow"]["input_ids"]
        input_ids_lm, _, labels_lm = uni_prompting(
            (texts_lm, input_ids_t2i.shape[-1]), "lm"
        )
        attention_mask_lm = create_attention_mask_predict_next(
            input_ids_lm.to(input_ids_t2i.device),
            pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        )
        attention_mask_lm = attention_mask_lm.to(mask_dtype)
        attention_mask = torch.cat([attention_mask, attention_mask_lm], dim=0)
        input_ids = torch.cat((input_ids_t2i, input_ids_lm.to(input_ids_t2i.device)), dim=0)
        labels = torch.cat((labels_t2i, labels_lm.to(input_ids_t2i.device)), dim=0)
    else:
        # Нет LM данных - используем только T2I
        input_ids = input_ids_t2i
        labels = labels_t2i

    present_domain_flows = [flow_key for flow_key in domain_flows if flow_key in batch]
    
    # Проверяем, включен ли MMU в конфиге
    batch_size_mmu_config = config.training.get("batch_size_mmu", 0)
    
    # Collect MMU data from all present domain flows (только если batch_size_mmu > 0)
    all_mmu_input_ids = []
    all_mmu_labels = []
    all_mmu_attention_masks = []
    mmu_domain_assignments = []  # track which domain each MMU sample belongs to
    
    if batch_size_mmu_config > 0 and present_domain_flows:
        # Process each domain flow
        for flow_key in present_domain_flows:
            
            flow_domain = flow_key[:-5]
            pixel_values_domain = batch[flow_key]["images"].to(accelerator.device, non_blocking=True)
            input_ids_domain = batch[flow_key]["input_ids"].to(accelerator.device, non_blocking=True)
            labels_domain = batch[flow_key]["labels"].to(accelerator.device, non_blocking=True)
            
            image_tokens_domain = vq_model.get_code(pixel_values_domain)
            image_tokens_domain = image_tokens_domain + len(uni_prompting.text_tokenizer)

            input_ids_domain_proc = torch.cat([
                (torch.ones(input_ids_domain.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(accelerator.device),
                (torch.ones(input_ids_domain.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(accelerator.device),
                image_tokens_domain,
                (torch.ones(input_ids_domain.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(accelerator.device),
                input_ids_domain,
        ], dim=1).long()

            labels_domain_proc = torch.cat([
                (torch.ones(input_ids_domain.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                (torch.ones(input_ids_domain.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                torch.ones_like(image_tokens_domain) * uni_prompting.ignore_id,
                (torch.ones(input_ids_domain.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
                labels_domain.to(accelerator.device)
        ], dim=1).long()

            attention_mask_domain = create_attention_mask_for_mmu(
                input_ids_domain_proc,
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            ).to(mask_dtype)
            
            all_mmu_input_ids.append(input_ids_domain_proc)
            all_mmu_labels.append(labels_domain_proc)
            all_mmu_attention_masks.append(attention_mask_domain)
            mmu_domain_assignments.extend([flow_domain] * input_ids_domain_proc.shape[0])
        
        # Concatenate all domain data
        input_ids_mmu = torch.cat(all_mmu_input_ids, dim=0)
        labels_mmu = torch.cat(all_mmu_labels, dim=0)
        attention_mask_mmu = torch.cat(all_mmu_attention_masks, dim=0)
        batch_size_mmu = input_ids_mmu.shape[0]
        domain_id = present_domain_flows[0][:-5]  # use first domain as primary
        domain_ids = [flow[:-5] for flow in present_domain_flows]
    elif batch_size_mmu_config > 0 and "llava" in config.dataset.und_type and "mmu_flow" in batch:
        # Fallback to mmu_flow - data is already tokenized
        pixel_values_mmu = batch["mmu_flow"]["images"]
        input_ids_mmu = batch["mmu_flow"]["input_ids"]
        labels_mmu = batch["mmu_flow"]["labels"]
        
        pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
        input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)
        labels_mmu = labels_mmu.to(accelerator.device)
        
        # Get image tokens from VQ model
        image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
        image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
        
        # Create input sequence: <|mmu|> <|soi|> image_tokens <|eoi|> text_tokens
        input_ids_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(accelerator.device),
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(accelerator.device),
            image_tokens_mmu,
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(accelerator.device),
            input_ids_mmu,
        ], dim=1).long()
        
        # Create labels: ignore special tokens and image tokens
        labels_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),  # <|mmu|>
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),  # <|soi|>
            torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,  # image tokens
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),  # <|eoi|>
            labels_mmu.to(accelerator.device)  # text labels
        ], dim=1).long()
        
        attention_mask_mmu = create_attention_mask_for_mmu(
            input_ids_mmu,
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        ).to(mask_dtype)
        mmu_domain_assignments = [None] * input_ids_mmu.shape[0]
        batch_size_mmu = pixel_values_mmu.shape[0]
    elif batch_size_mmu_config > 0 and config.dataset.und_type == "captioning" and "mmu_flow" in batch:
        # Captioning mode - data has images and raw text captions (not tokenized)
        pixel_values_mmu = batch["mmu_flow"]["images"]
        captions_mmu = batch["mmu_flow"]["input_ids"]  # list of strings
        
        pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
        
        # Get image tokens from VQ model
        image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
        image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
        
        # Tokenize captions
        max_text_len = config.dataset.preprocessing.max_seq_length
        
        # Debug: log caption tokenization
        if global_step == 0 and epoch == 0:
            logger.info(f"🔍 MMU CAPTIONING:")
            logger.info(f"   max_text_len for tokenization: {max_text_len}")
            logger.info(f"   Number of captions: {len(captions_mmu)}")
            logger.info(f"   First caption: {captions_mmu[0][:100] if captions_mmu else 'N/A'}...")
        
        tokenized = uni_prompting.text_tokenizer(
            captions_mmu,
            padding="max_length",
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        text_input_ids = tokenized["input_ids"].to(accelerator.device)
        
        # Debug: log tokenized text shape
        if global_step == 0 and epoch == 0:
            logger.info(f"   text_input_ids shape after tokenization: {text_input_ids.shape}")
            logger.info(f"   image_tokens_mmu shape: {image_tokens_mmu.shape}")
        
        # Create input sequence: <|mmu|> <|soi|> image_tokens <|eoi|> text_tokens
        batch_size_caption = pixel_values_mmu.shape[0]
        input_ids_mmu = torch.cat([
            (torch.ones(batch_size_caption, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(accelerator.device),
            (torch.ones(batch_size_caption, 1) * uni_prompting.sptids_dict['<|soi|>']).to(accelerator.device),
            image_tokens_mmu,
            (torch.ones(batch_size_caption, 1) * uni_prompting.sptids_dict['<|eoi|>']).to(accelerator.device),
            text_input_ids,
        ], dim=1).long()
        
        # Debug: log final MMU sequence shape
        if global_step == 0 and epoch == 0:
            logger.info(f"   Final input_ids_mmu shape: {input_ids_mmu.shape}")
            logger.info(f"   Expected: MMU(1) + SOI(1) + image(1024) + EOI(1) + text({max_text_len}) = {1 + 1 + 1024 + 1 + max_text_len}")
        
        # Create labels: ignore special tokens and image tokens, predict text tokens
        labels_mmu = torch.cat([
            (torch.ones(batch_size_caption, 1) * uni_prompting.ignore_id).to(accelerator.device),  # <|mmu|>
            (torch.ones(batch_size_caption, 1) * uni_prompting.ignore_id).to(accelerator.device),  # <|soi|>
            torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,  # image tokens
            (torch.ones(batch_size_caption, 1) * uni_prompting.ignore_id).to(accelerator.device),  # <|eoi|>
            text_input_ids.clone(),  # text labels (predict caption)
        ], dim=1).long()
        
        # Mask padding tokens in labels
        labels_mmu[labels_mmu == uni_prompting.text_tokenizer.pad_token_id] = uni_prompting.ignore_id
        
        attention_mask_mmu = create_attention_mask_for_mmu(
            input_ids_mmu,
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        ).to(mask_dtype)
        mmu_domain_assignments = [None] * batch_size_caption
        batch_size_mmu = batch_size_caption
    else:
        # No MMU data
        input_ids_mmu = torch.empty((0, input_ids.shape[1]), dtype=input_ids.dtype, device=input_ids.device)
        labels_mmu = torch.empty((0, labels.shape[1]), dtype=labels.dtype, device=labels.device)
        attention_mask_mmu = torch.empty((0, 1, 0, 0), dtype=mask_dtype, device=attention_mask.device)
        mmu_domain_assignments = []
        batch_size_mmu = 0
    
    # Debug: log shapes on first step
    if global_step == 0 and epoch == 0:
        logger.info(f"🔍 BEFORE PADDING:")
        logger.info(f"   T2I input_ids shape: {input_ids.shape}")
        logger.info(f"   MMU input_ids_mmu shape: {input_ids_mmu.shape}")
        logger.info(f"   T2I attention_mask shape: {attention_mask.shape}")
        logger.info(f"   MMU attention_mask_mmu shape: {attention_mask_mmu.shape}")
        logger.info(f"   T2I labels shape: {labels.shape}")
        logger.info(f"   MMU labels_mmu shape: {labels_mmu.shape}")
        logger.info(f"   Config max_seq_length: {config.dataset.preprocessing.max_seq_length}")
        logger.info(f"   uni_prompting.max_text_len: {uni_prompting.max_text_len}")
    
    # Pad sequences to the same length if needed
    if batch_size_mmu > 0 and input_ids_mmu.shape[0] > 0:
        max_len = max(input_ids.shape[1], input_ids_mmu.shape[1])
        if global_step == 0 and epoch == 0:
            logger.info(f"   Padding to max_len={max_len} (T2I={input_ids.shape[1]}, MMU={input_ids_mmu.shape[1]})")
    else:
        max_len = input_ids.shape[1]
    
    if input_ids.shape[1] < max_len:
        pad_len = max_len - input_ids.shape[1]
        input_ids = torch.cat([input_ids, torch.full((input_ids.shape[0], pad_len), uni_prompting.pad_id, dtype=input_ids.dtype, device=input_ids.device)], dim=1)
        labels = torch.cat([labels, torch.full((labels.shape[0], pad_len), uni_prompting.ignore_id, dtype=labels.dtype, device=labels.device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.full((attention_mask.shape[0], 1, attention_mask.shape[2], pad_len), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=attention_mask.device)], dim=3)
        attention_mask = torch.cat([attention_mask, torch.full((attention_mask.shape[0], 1, pad_len, max_len), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=attention_mask.device)], dim=2)
    
    if batch_size_mmu > 0 and input_ids_mmu.shape[0] > 0:
        if input_ids_mmu.shape[1] < max_len:
            pad_len = max_len - input_ids_mmu.shape[1]
            input_ids_mmu = torch.cat([input_ids_mmu, torch.full((input_ids_mmu.shape[0], pad_len), uni_prompting.pad_id, dtype=input_ids_mmu.dtype, device=input_ids_mmu.device)], dim=1)
            labels_mmu = torch.cat([labels_mmu, torch.full((labels_mmu.shape[0], pad_len), uni_prompting.ignore_id, dtype=labels_mmu.dtype, device=labels_mmu.device)], dim=1)
            attention_mask_mmu = torch.cat([attention_mask_mmu, torch.full((attention_mask_mmu.shape[0], 1, attention_mask_mmu.shape[2], pad_len), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=attention_mask_mmu.device)], dim=3)
            attention_mask_mmu = torch.cat([attention_mask_mmu, torch.full((attention_mask_mmu.shape[0], 1, pad_len, max_len), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=attention_mask_mmu.device)], dim=2)
        
        attention_mask = torch.cat([attention_mask, attention_mask_mmu], dim=0)
        input_ids = torch.cat((input_ids, input_ids_mmu.to(input_ids.device)), dim=0)
        labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)
    # Если batch_size_mmu=0, то input_ids, labels, attention_mask уже содержат только T2I (+ LM если есть)
    
    # Create sample_domains: simple list where each element is the domain of that sample
    # [batch_size] where each element is domain name (None for T2I/LM, domain name for MMU)
    batch_size_total = input_ids.shape[0]
    sample_domains = (
        [None] * batch_size_t2i +           # T2I samples
        [None] * batch_size_lm +            # LM samples  
        mmu_domain_assignments              # MMU samples with their domains
    )
    
    if global_step <= 2:
        logger.info(f"Step {global_step}: sample_domains = {sample_domains}")
        logger.info(f"Step {global_step}: batch sizes: t2i={batch_size_t2i}, lm={batch_size_lm}, mmu={len(mmu_domain_assignments)}, total={batch_size_total}")

    if global_step == 0 and epoch == 0:
        logger.info(f"🔍 AFTER PADDING:")
        logger.info(f"   Final input_ids shape: {input_ids.shape}")
        logger.info(f"   Final labels shape: {labels.shape}")
        logger.info(f"   Final attention_mask shape: {attention_mask.shape}")
        
        logger.info(f"📊 First training step diagnostics:")
        logger.info(f"   input_ids shape: {input_ids.shape}")
        logger.info(f"   input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        logger.info(f"   labels range (non-ignore): [{labels[labels != uni_prompting.ignore_id].min().item()}, {labels[labels != uni_prompting.ignore_id].max().item()}]")
        logger.info(f"   mask_id used: {mask_id}")
        
        # Log sample token distribution
        for i in range(min(2, input_ids.shape[0])):
            sample = input_ids[i]
            pad_count = (sample == uni_prompting.pad_id).sum().item()
            mask_count = (sample == mask_id).sum().item()
            soi_count = (sample == int(uni_prompting.sptids_dict['<|soi|>'])).sum().item()
            eoi_count = (sample == int(uni_prompting.sptids_dict['<|eoi|>'])).sum().item()
            logger.info(f"   Sample {i}: pad={pad_count}, mask={mask_count}, soi={soi_count}, eoi={eoi_count}")
        logger.info(f"   text_tokenizer size: {len(uni_prompting.text_tokenizer)}")
        logger.info(f"   Expected image token range: [{len(uni_prompting.text_tokenizer)}, {len(uni_prompting.text_tokenizer) + 8192 - 1}]")

    current_temperature = temp_scheduler.get_last_lr()[0]
    should_log_layer_expert = global_step % config.experiment.log_every == 0
    if config.get("moe", {}).get("enabled", False):
        set_moe_layers_global_step(model, accelerator, global_step)


    logits, loss_t2i, loss_lm, loss_mmu = model(
        input_ids=input_ids,
        input_embeddings=None,
        attention_mask=attention_mask,
        labels=labels,
        label_smoothing=config.training.label_smoothing,
        batch_size_t2i=batch_size_t2i,
        batch_size_lm=batch_size_lm,
        batch_size_mmu=batch_size_mmu,
        max_seq_length=config.dataset.preprocessing.max_seq_length,
        moe_temperature=current_temperature,
        moe_domain_id=domain_id,
        moe_sample_domains=sample_domains,
    )
    
    # Диагностика t2i loss
    if global_step <= 5 or global_step % 100 == 0:
        if batch_size_t2i > 0:
            # Проверяем, что в labels есть не-ignore значения для t2i части
            t2i_labels = labels[:batch_size_t2i, config.dataset.preprocessing.max_seq_length + 1 :]
            non_ignore_count = (t2i_labels != uni_prompting.ignore_id).sum().item()
            total_t2i_tokens = t2i_labels.numel()
            logger.info(
                f"🔍 Step {global_step}: t2i diagnostics - "
                f"batch_size_t2i={batch_size_t2i}, "
                f"loss_t2i={loss_t2i.item():.4f}, "
                f"non_ignore_labels={non_ignore_count}/{total_t2i_tokens}, "
                f"mask_prob={mask_prob.mean().item():.4f}, "
                f"loss_t2i.requires_grad={loss_t2i.requires_grad}"
            )
            if non_ignore_count == 0:
                logger.warning(f"⚠️ Step {global_step}: Все t2i labels игнорируются! Loss может быть 0 или nan.")
            if torch.isnan(loss_t2i) or torch.isinf(loss_t2i):
                logger.error(f"❌ Step {global_step}: loss_t2i is nan/inf!")
        else:
            logger.warning(f"⚠️ Step {global_step}: batch_size_t2i = 0, t2i loss не вычисляется!")

    # Log samples for debugging
    if sample_logger is not None and sample_logger.should_log(global_step):
        # Log batch summary
        sample_logger.log_batch_summary(
            global_step=global_step,
            batch_size_t2i=batch_size_t2i,
            batch_size_lm=batch_size_lm,
            batch_size_mmu=batch_size_mmu,
            total_seq_length=input_ids.shape[1],
            mask_prob_mean=mask_prob.mean().item() if torch.is_tensor(mask_prob) else mask_prob,
        )
        
        # Log T2I samples
        for i in range(min(batch_size_t2i, sample_logger.max_samples_per_step)):
            sample_logger.log_t2i_sample(
                global_step=global_step,
                batch_idx=i,
                input_ids=input_ids[i],
                labels=labels[i],
                original_text=texts[i] if i < len(texts) else "",
                mask_prob=mask_prob[i].item() if torch.is_tensor(mask_prob) and mask_prob.dim() > 0 else mask_prob,
                image_tokens_ori=image_tokens_ori[i] if image_tokens_ori is not None else None,
            )
        
        # Log LM samples
        lm_start_idx = batch_size_t2i
        for i in range(min(batch_size_lm, sample_logger.max_samples_per_step)):
            idx = lm_start_idx + i
            lm_text = batch["lm_flow"]["input_ids"][i] if "lm_flow" in batch and i < len(batch["lm_flow"]["input_ids"]) else ""
            sample_logger.log_lm_sample(
                global_step=global_step,
                batch_idx=i,
                input_ids=input_ids[idx],
                labels=labels[idx],
                original_text=lm_text[:200] if isinstance(lm_text, str) else str(lm_text)[:200],
            )
        
        # Log MMU samples
        mmu_start_idx = batch_size_t2i + batch_size_lm
        for i in range(min(batch_size_mmu, sample_logger.max_samples_per_step)):
            idx = mmu_start_idx + i
            domain = mmu_domain_assignments[i] if i < len(mmu_domain_assignments) else None
            sample_logger.log_mmu_sample(
                global_step=global_step,
                batch_idx=i,
                input_ids=input_ids[idx],
                labels=labels[idx],
                domain=domain,
            )

    # Gather the losses across all processes for logging (if we use distributed training).
    # ВАЖНО: loss_t2i уже усреднён по батчу (из F.cross_entropy с reduction='mean')
    # Для distributed training мы повторяем его batch_size_t2i раз, собираем и усредняем
    # Для одного GPU это просто возвращает loss_t2i
    if batch_size_t2i > 0:
        avg_loss_t2i = accelerator.gather(
            loss_t2i.repeat(config.training.batch_size_t2i)
        ).mean()
    else:
        avg_loss_t2i = torch.tensor(0.0, device=loss_t2i.device)
    if batch_size_lm > 0:
        avg_loss_lm = accelerator.gather(
            loss_lm.repeat(batch_size_lm)
        ).mean()
    else:
        avg_loss_lm = torch.tensor(0.0, device=loss_lm.device)
    if batch_size_mmu > 0:
        avg_loss_mmu = accelerator.gather(
            loss_mmu.repeat(batch_size_mmu)
        ).mean()
    else:
        avg_loss_mmu = torch.tensor(0.0, device=loss_mmu.device)

    # MoE losses/coeff only if MoE enabled
    if config.get("moe", {}).get("enabled", False):
    balance_loss, orthogonal_loss, num_moe_layers = collect_moe_balance_losses(model)
    if num_moe_layers == 0:
        logger.warning(f"⚠️ MoE enabled but num_moe_layers = {num_moe_layers}")
            # Не логируем, если MoE фактически нет
            should_log_layer_expert = False
    balance_coeff = balance_scheduler.get_last_lr()[0]
    else:
        # MoE выключен — ставим нули и не логируем
        zero = loss_t2i.new_tensor(0.0)
        balance_loss, orthogonal_loss = zero, zero
        num_moe_layers = 0
        balance_coeff = 0.0
        should_log_layer_expert = False

    if (
        should_log_layer_expert
        and accelerator.is_main_process
        and mlflow_client is not None
        and mlflow_run_id is not None
    ):
        # Сбор статистики в отдельном блоке для освобождения памяти
        unwrapped_model = accelerator.unwrap_model(model)
        collector = LayerExpertStatsCollector(unwrapped_model)
        layer_expert_counts_by_modality, probability_map = collector.collect(global_step)
        
        if not layer_expert_counts_by_modality:
            raise Exception("No moe layers")
        visualizer = MoEVisualizer(num_experts=config.moe.num_experts)
        mlflow_logger = MoEMLflowLogger(mlflow_client=mlflow_client, mlflow_run_id=mlflow_run_id)
        
        for modality, layer_expert_counts in layer_expert_counts_by_modality.items():
            heatmap_bytes = visualizer.create_layer_expert_activation_heatmap(
                layer_expert_counts, global_step=global_step + 1
            )
            mlflow_logger.log_layer_expert_heatmap(global_step + 1, heatmap_bytes, suffix=modality)
            del heatmap_bytes

        for modality, layer_probabilities in probability_map.items():
            prob_heatmap = visualizer.create_layer_probability_heatmap(
                layer_probabilities, modality, global_step
            )
            mlflow_logger.log_layer_probability_heatmap(
                global_step + 1, prob_heatmap, suffix=modality
            )
            del prob_heatmap
        
        
        # Явно освобождаем объекты и GPU память
        del layer_expert_counts_by_modality, probability_map, collector, visualizer, mlflow_logger
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        if global_step <= 5:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"🔧 Step {global_step}: After MoE stats cleanup - GPU allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
    
    # Синхронизируем все процессы после логирования MoE статистики
    if should_log_layer_expert:
        accelerator.wait_for_everyone()

    # Проверка, что t2i loss валиден и участвует в общем loss
    if batch_size_t2i > 0:
        if torch.isnan(loss_t2i) or torch.isinf(loss_t2i):
            logger.error(f"❌ Step {global_step}: loss_t2i is nan/inf, заменяем на 0")
            loss_t2i = torch.tensor(0.0, device=loss_t2i.device, requires_grad=True)
        if loss_t2i.item() == 0.0:
            logger.warning(f"⚠️ Step {global_step}: loss_t2i = 0.0, возможно все labels игнорируются!")
    
    # Проверка MoE losses на NaN
    if torch.is_tensor(balance_loss) and (torch.isnan(balance_loss) or torch.isinf(balance_loss)):
        logger.error(f"❌ Step {global_step}: balance_loss is nan/inf ({balance_loss.item()}), заменяем на 0")
        balance_loss = torch.tensor(0.0, device=loss_t2i.device, requires_grad=True)
    if torch.is_tensor(orthogonal_loss) and (torch.isnan(orthogonal_loss) or torch.isinf(orthogonal_loss)):
        logger.error(f"❌ Step {global_step}: orthogonal_loss is nan/inf ({orthogonal_loss.item()}), заменяем на 0")
        orthogonal_loss = torch.tensor(0.0, device=loss_t2i.device, requires_grad=True)
    
    loss = (
        config.training.t2i_coeff * loss_t2i
        + config.training.lm_coeff * loss_lm
        + config.training.mmu_coeff * loss_mmu
        + balance_coeff * balance_loss
        + config.training.orthogonal_coeff * orthogonal_loss
    )
    
    # Диагностика общего loss
    if global_step <= 5 or global_step % 100 == 0:
        logger.info(
            f"📊 Step {global_step}: Total loss breakdown - "
            f"t2i_contrib={config.training.t2i_coeff * loss_t2i.item():.4f}, "
            f"mmu_contrib={config.training.mmu_coeff * loss_mmu.item():.4f}, "
            f"total={loss.item():.4f}"
        )

    # NOTE: Логирование loss перенесено в main loop для корректного усреднения по gradient accumulation steps
    # Метрики логируются как loss/t2i, loss/lm, loss/mmu, loss/balance, loss/orthogonal

    avg_masking_rate = accelerator.gather(
        mask_prob.repeat(config.training.batch_size_t2i)
    ).mean()

    # Диагностика GPU памяти перед backward
    if global_step <= 5:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"🔧 Step {global_step}: Before backward - GPU allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")

    # Backward pass
    accelerator.backward(loss)
    
    # Проверка градиентов для t2i loss (только для диагностики)
        # if (global_step <= 5 or global_step % 100 == 0) and batch_size_t2i > 0:
        #     try:
        #         unwrapped_model = accelerator.unwrap_model(model)
        #         # Проверяем градиенты в первом слое модели
        #         if hasattr(unwrapped_model, 'showo') and hasattr(unwrapped_model.showo, 'model'):
        #             first_param = next(unwrapped_model.showo.model.parameters())
        #             if first_param.grad is not None:
        #                 grad_norm = first_param.grad.norm().item()
        #                 logger.info(f"🔍 Step {global_step}: Gradient norm (first param) = {grad_norm:.6f}")
        #             else:
        #                 logger.warning(f"⚠️ Step {global_step}: Градиенты отсутствуют в первом параметре!")
        #     except Exception as e:
        #         logger.debug(f"Не удалось проверить градиенты: {e}")

    # Очистка CUDA кэша после каждого accumulation cycle
    if torch.cuda.is_available() and accelerator.sync_gradients:
        torch.cuda.empty_cache()

    if config.training.max_grad_norm is not None and accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

    optimizer.step()
    # Update ALL schedulers only on global steps (not micro-batches)
    if accelerator.sync_gradients:
        lr_scheduler.step()
        balance_scheduler.step()
        temp_scheduler.step()

    # log gradient norm before zeroing it
    if (
        accelerator.sync_gradients
        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
        and accelerator.is_main_process
    ):
        log_grad_norm(model, accelerator, global_step + 1, mlflow_client, mlflow_run_id)

    optimizer.zero_grad(set_to_none=True)

    # Log metrics
    if (
        accelerator.sync_gradients
        and (global_step + 1) % config.experiment.log_every == 0
        and accelerator.is_main_process
    ):
        samples_per_second_per_gpu = (
            config.training.gradient_accumulation_steps
            * total_batch_size_per_gpu
            / batch_time_m.val
        )
        log_training_metrics(
            avg_loss_t2i=avg_loss_t2i,
            avg_loss_lm=avg_loss_lm,
            avg_loss_mmu=avg_loss_mmu,
            balance_loss=balance_loss,
            balance_coeff=balance_coeff,
            orthogonal_loss=orthogonal_loss,
            avg_masking_rate=avg_masking_rate,
            lr_scheduler=lr_scheduler,
            batch_time_m=batch_time_m,
            data_time_m=data_time_m,
            samples_per_second_per_gpu=samples_per_second_per_gpu,
            global_step=global_step + 1,
            mlflow_client=mlflow_client,
            mlflow_run_id=mlflow_run_id,
            logger=logger,
        )

        if mlflow_client is not None and mlflow_run_id is not None and config.get("moe", {}).get("enabled", False):
            temperature = float(temp_scheduler.get_last_lr()[0])
            mlflow_client.log_metric(mlflow_run_id, "moe/temperature", temperature, step=global_step + 1)
            logger.info(f"[moe] temperature: {temperature:.4f}")
            
            mlflow_client.log_metric(mlflow_run_id, "moe/balance_coeff", float(balance_coeff), step=global_step + 1)
            logger.info(f"[moe] balance_coeff: {float(balance_coeff):.6f}")
            
            unwrapped_model = accelerator.unwrap_model(model)
            domain_bias_hardness = 0.0
            for layer in unwrapped_model.showo.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "get_domain_bias_hardness"):
                    domain_bias_hardness = layer.mlp.get_domain_bias_hardness()
                    break
            mlflow_client.log_metric(mlflow_run_id, "moe/domain_bias_hardness", domain_bias_hardness, step=global_step + 1)
            logger.info(f"[moe] domain_bias_hardness: {domain_bias_hardness:.4f}")

        # Reset time meters
        batch_time_m.reset()
        data_time_m.reset()

    if accelerator.is_main_process and pbar is not None:
        pbar.update(1)
        all_lrs = lr_scheduler.get_last_lr()
        if len(all_lrs) >= 4:
            lr_str = f"moe={all_lrs[0]:.2e}/base={all_lrs[2]:.2e}"
        else:
            lr_str = f"{all_lrs[0]:.2e}"
        pbar.set_postfix(
            {
                "loss_mmu": f"{avg_loss_mmu.item():.4f}",
                "loss_t2i": f"{avg_loss_t2i.item():.4f}",
                "lr": lr_str,
            }
        )

    return {
        "avg_loss_t2i": avg_loss_t2i,
        "avg_loss_lm": avg_loss_lm,
        "avg_loss_mmu": avg_loss_mmu,
        "balance_loss": balance_loss,
        "orthogonal_loss": orthogonal_loss,
        "avg_masking_rate": avg_masking_rate,
        "logits": logits,
        "input_ids": input_ids,
        "input_ids_t2i": input_ids_t2i,
        "labels": labels,
        "attention_mask": attention_mask,
        "image_tokens_ori": image_tokens_ori,
        "texts": texts,
        "batch_size_t2i": batch_size_t2i,
        "batch_size_lm": batch_size_lm,
        "batch_size_mmu": batch_size_mmu,
    }

def router_orth_loss(weight):
    W = weight # [experts_coune, hidden_size]
    W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    C = W_norm @ W_norm.t()  # shape: [experts_coune, experts_coune]
    I = torch.eye(C.size(0), device=C.device, dtype=C.dtype)
    loss = ((C - I)**2).sum()
    return loss

def collect_moe_balance_losses(model):
    total_balance_loss = 0.0
    num_moe_layers = 0
    if hasattr(model, "module"):
        unwrapped_model = model.module
    else:
        unwrapped_model = model
        
    gate_weights = []

    for layer_idx, layer in enumerate(unwrapped_model.showo.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            gate_loss = layer.mlp.get_balance_loss()
            total_balance_loss = total_balance_loss + gate_loss
            
            gate_weights.append(layer.mlp.gate.gate.weight)
            num_moe_layers += 1
            logger.debug(f"MoE layer {layer_idx}: balance_loss = {gate_loss.item():.6f}")
    
    # Нет MoE слоёв или gate_weights пуст — вернуть нули, но с grad
    if num_moe_layers == 0 or len(gate_weights) == 0:
        zero = next(unwrapped_model.parameters()).new_tensor(0.0, requires_grad=True)
        return zero, zero, num_moe_layers
            
    gate_weights = torch.cat(gate_weights, dim=0)
    orthogonal_loss = router_orth_loss(gate_weights)
            
    return total_balance_loss, orthogonal_loss, num_moe_layers


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    config = get_config()

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")

    #####################################
    # SET SEED FIRST - before anything else #
    # Это критично для детерминированности всей инициализации #
    # Используем обычный print, т.к. accelerate logger требует инициализации #
    #####################################
    if config.training.seed is not None:
        seed = config.training.seed
        print(f"🌱 Setting seed to {seed} (BEFORE Accelerator initialization)")
        set_seed(seed)
        # Дополнительная установка сидов для полной детерминированности
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"✅ Seed {seed} set for: torch, numpy, random, cuda")
    else:
        print("⚠️  No seed specified in config.training.seed - training will be non-deterministic!")

    # Enable TF32 on Ampere GPUs (после установки сида)
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("⚡ TF32 enabled (cudnn.deterministic=False for performance)")
    else:
        # Для полной детерминированности (может замедлить обучение)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("🔒 Full determinism enabled (cudnn.deterministic=True)")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # profile_kwargs = ProfileKwargs(
    #     activities=["cuda"],
    #     record_shapes=True
    # )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="mlflow",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
    )

    total_batch_size_per_gpu = (
        config.training.batch_size_t2i
        + config.training.get("batch_size_lm", 0)
        + config.training.get("batch_size_mmu", 0)
    )
    total_batch_size = (
        (
            config.training.batch_size_t2i
            + config.training.get("batch_size_lm", 0)
            + config.training.get("batch_size_mmu", 0)
        )
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = total_batch_size_per_gpu

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    mlflow_client = None
    mlflow_run_id = None
    if accelerator.is_main_process and config.get("mlflow", {}).get("enabled", False):
        mlflow_tracking_uri = config.mlflow.get("tracking_uri", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        experiment_name = config.mlflow.get(
            "experiment_name", config.experiment.project
        )
        try:
            experiment = mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow_client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except:
            experiment_id = mlflow_client.create_experiment(experiment_name)

        run = mlflow_client.create_run(
            experiment_id=experiment_id,
            run_name=config.experiment.name,
            tags=config.mlflow.get("tags", {}),
        )
        mlflow_run_id = run.info.run_id

        mlflow_client.log_param(
            mlflow_run_id, "batch_size_mmu", config.training.get("batch_size_mmu", 0)
        )
        mlflow_client.log_param(
            mlflow_run_id, "batch_size_t2i", config.training.batch_size_t2i
        )
        mlflow_client.log_param(
            mlflow_run_id, "batch_size_lm", config.training.get("batch_size_lm", 0)
        )
        mlflow_client.log_param(
            mlflow_run_id, "learning_rate", config.optimizer.params.learning_rate
        )
        mlflow_client.log_param(
            mlflow_run_id, "num_experts", config.moe.get("num_experts", 4)
        )
        mlflow_client.log_param(mlflow_run_id, "top_k", config.moe.get("top_k", 2))
        mlflow_client.log_param(
            mlflow_run_id, "max_train_steps", config.training.max_train_steps
        )
        mlflow_client.log_param(
            mlflow_run_id,
            "gradient_accumulation_steps",
            config.training.gradient_accumulation_steps,
        )

        logger.info(f"✅ MLflow run started: {mlflow_run_id}")

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        
        # Сохраняем конфиг в MLflow как артефакт
        if mlflow_client is not None and mlflow_run_id is not None:
            try:
                mlflow_client.log_artifact(mlflow_run_id, str(config_path))
                logger.info(f"📄 Config saved to MLflow as artifact: config.yaml")
            except Exception as e:
                logger.warning(f"Failed to save config to MLflow: {e}")

    # Seed уже установлен выше, перед инициализацией Accelerator
    # Это гарантирует детерминированность всей инициализации

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path, padding_side="left"
    )

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
    )

    # print("special tokens : \n", uni_prompting.sptids_dict)

    # Sample logger for debugging training data
    sample_logger = None
    if accelerator.is_main_process and config.get("sample_logging", {}).get("enabled", False):
        sample_log_dir = os.path.join(config.experiment.output_dir, "sample_logs")
        sample_logger = SampleLogger(
            log_dir=sample_log_dir,
            text_tokenizer=tokenizer,
            log_every_n_steps=config.get("sample_logging", {}).get("log_every_n_steps", 100),
            max_samples_per_step=config.get("sample_logging", {}).get("max_samples_per_step", 2),
            enabled=True,
        )
        sample_logger.set_special_tokens(uni_prompting.sptids_dict)
        logger.info(f"📝 Sample logging enabled: {sample_log_dir}")

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)["model"]
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(
            accelerator.device
        )
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Show-o model
    if config.model.showo.load_from_showo:
        logger.info(f"Loading model from {config.model.showo.pretrained_model_path}")
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(
            accelerator.device
        )
        logger.info(f"Original model vocab_size: {model.vocab_size}, mask_token_id: {model.mask_token_id}")
        logger.info(f"Original model codebook_size: {model.config.codebook_size}")
        logger.info(f"Config vocab_size: {config.model.showo.vocab_size}, codebook_size: {config.model.showo.codebook_size}")
        
        if config.model.showo.vocab_size != model.vocab_size:
            logger.warning(f"⚠️ Vocab size mismatch! Resizing embeddings from {model.vocab_size} to {config.model.showo.vocab_size}")
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = config.model.showo.vocab_size - 1
            model.mask_token_id = config.model.showo.vocab_size - 1
            logger.info(f"After resize - vocab_size: {model.vocab_size}, mask_token_id: {model.mask_token_id}")
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
        logger.info(f"Created new model - vocab_size: {model.vocab_size}, mask_token_id: {model.mask_token_id}")

    special_tokens = {
        "soi_id": uni_prompting.sptids_dict["<|soi|>"].item()
        if "<|soi|>" in uni_prompting.sptids_dict
        else None,
        "eoi_id": uni_prompting.sptids_dict["<|eoi|>"].item()
        if "<|eoi|>" in uni_prompting.sptids_dict
        else None,
        "sov_id": uni_prompting.sptids_dict["<|sov|>"].item()
        if "<|sov|>" in uni_prompting.sptids_dict
        else None,
        "eov_id": uni_prompting.sptids_dict["<|eov|>"].item()
        if "<|eov|>" in uni_prompting.sptids_dict
        else None,
    }

    if config.get("moe", {}).get("enabled", False):
        model = patch_model_with_moe(
            model,
            config.moe,
            mlflow_client=mlflow_client,
            mlflow_run_id=mlflow_run_id,
            special_tokens=special_tokens
        )
        logger.info("✅ MoE patching enabled")
    else:
        logger.info("⏭️  MoE disabled, skipping patching")
    
    mask_id = model.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer = get_optimizer(
        optimizer_config=config.optimizer,
        named_parameters=model.named_parameters(),
        logger=logger,
        moe_config=config.get("moe", None) if config.get("moe", {}).get("enabled", False) else None,
    )

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    evaluation_cfg = config.get("evaluation", None)
    coco_cfg = evaluation_cfg.get("coco", None)
    coco_eval_batch_size = int(
        coco_cfg.get("batch_size", config.training.batch_size_t2i)
    )
    metric_interval = evaluation_cfg.get("metric_interval", None)
    
    images_root = coco_cfg.get("images_root")
    ann_file = coco_cfg.get("ann_file")
    
    subset_size = coco_cfg.subset_size
    subset_seed = config.training.seed
    coco_eval_dataset = COCODataset(
        root=images_root,
        annFile=ann_file,
    )
    periodic_fid_seed = int(subset_seed) + 9999 if subset_seed is not None else 9999
    coco_eval_dataset.restrict_to_subset(
        subset_size=subset_size,
        seed=periodic_fid_seed,
    )
    logger.info(
        f"Initialized COCO periodic FID dataset with {len(coco_eval_dataset)} samples "
        f"(batch_size={coco_eval_dataset}, seed={periodic_fid_seed})"
    )


    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    balance_init_coeff = float(config.training["balance_coeff"])
    balance_warmup_steps = (
        config.moe["balance_warmup_steps"]
    )
    balance_gamma = float(config.moe["balance_gamma"])
    _dummy_param = nn.Parameter(torch.zeros((), device=accelerator.device))
    balance_optimizer = torch.optim.SGD([{"params": [_dummy_param], "lr": balance_init_coeff}])
    balance_warmup = LinearLR(balance_optimizer, start_factor=1e-8, total_iters=max(int(balance_warmup_steps), 1))
    balance_decay = ExponentialLR(balance_optimizer, gamma=float(balance_gamma))
    balance_scheduler = SequentialLR(
        balance_optimizer,
        schedulers=[balance_warmup, balance_decay],
        milestones=[max(int(balance_warmup_steps), 1)],
    )

    # Настройка температуры для MoE: если temp_fixed задан, используем фиксированную, иначе scheduler
    temp_fixed = config.moe.get("temp_fixed", None)
    if temp_fixed is not None:
        # Фиксированная температура
        temp_fixed_value = float(temp_fixed)
        _temp_dummy_param = nn.Parameter(torch.zeros((), device=accelerator.device))
        temp_optimizer = torch.optim.SGD([{"params": [_temp_dummy_param], "lr": temp_fixed_value}])
        def _constant_factor(step: int):
            return 1.0  # Всегда возвращаем 1.0, чтобы lr оставался temp_fixed_value
        temp_scheduler = LambdaLR(temp_optimizer, lr_lambda=_constant_factor)
    else:
        # Scheduler с изменением температуры
        temp_start = float(config.moe["temp_start"])
        temp_end = float(config.moe["temp_end"])
        temp_steps = int(config.moe["temp_steps"]) if "temp_steps" in config.moe else int(config.training.max_train_steps)
        _temp_dummy_param = nn.Parameter(torch.zeros((), device=accelerator.device))
        temp_optimizer = torch.optim.SGD([{"params": [_temp_dummy_param], "lr": temp_start}])
        def _cosine_factor(step: int):
            s = min(int(step), int(max(temp_steps, 1)))
            progress = s / max(temp_steps, 1)
            # Косинусное расписание: temp = temp_end + (temp_start - temp_end) * (1 + cos(π * progress)) / 2
            cosine_factor = (1 + math.cos(math.pi * progress)) / 2
            temp_current = temp_end + (temp_start - temp_end) * cosine_factor
            return temp_current / max(temp_start, 1e-8)

        temp_scheduler = LambdaLR(temp_optimizer, lr_lambda=_cosine_factor)

    ##################################
    #         DATALOADER             #
    #################################
    # Проверяем, нужна ли функция create_imagetext_dataloader для parquet режимов
    create_imagetext_dataloader_fn = None
    if (
        config.dataset.gen_type == "t2i_parquet"
        or config.dataset.und_type == "captioning_parquet"
    ):
        try:
            from datasets import (
                create_imagetext_dataloader as create_imagetext_dataloader_fn,
            )
        except ImportError:
            logger.warning(
                "create_imagetext_dataloader not available, but required for parquet mode"
            )

    combined_dataloader, num_update_steps_per_epoch, num_train_epochs = (
        create_dataloaders(
            config=config,
            accelerator=accelerator,
            tokenizer=tokenizer,
            create_imagetext_dataloader=create_imagetext_dataloader_fn,
        )
    )

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(
                f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin"
            )
            state_dict = torch.load(
                f"{path}/unwrapped_model/pytorch_model.bin", map_location="cpu"
            )
            model.load_state_dict(state_dict, strict=True)
            del state_dict

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    if hasattr(model, "module"):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = accelerator.unwrap_model(
            model
        ).showo.model.embed_tokens.weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    # Накопление loss для усреднения по gradient accumulation steps
    accumulated_loss_t2i = 0.0
    accumulated_loss_lm = 0.0
    accumulated_loss_mmu = 0.0
    accumulated_balance_loss = 0.0
    accumulated_orthogonal_loss = 0.0
    accumulated_masking_rate = 0.0
    accumulation_count = 0

    for epoch in range(first_epoch, num_train_epochs):
        model.train()

        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(
                total=config.training.max_train_steps - global_step,
                desc=f"Epoch {epoch}",
                initial=0,
                unit="step",
                colour="green",
            )

        try:
            for batch, batch_idx, dataloader_idx in combined_dataloader:
                data_time_m.update(time.time() - end)

                with accelerator.accumulate(model):
                    step_outputs = train_step(
                        batch=batch,
                        epoch=epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        balance_scheduler=balance_scheduler,
                        temp_scheduler=temp_scheduler,
                        accelerator=accelerator,
                        config=config,
                        uni_prompting=uni_prompting,
                        vq_model=vq_model,
                        mask_dtype=mask_dtype,
                        mask_id=mask_id,
                        mask_schedule=mask_schedule,
                        batch_time_m=batch_time_m,
                        data_time_m=data_time_m,
                        total_batch_size_per_gpu=total_batch_size_per_gpu,
                        mlflow_client=mlflow_client,
                        mlflow_run_id=mlflow_run_id,
                        pbar=pbar,
                        sample_logger=sample_logger,
                    )

                input_ids = step_outputs["input_ids"]
                input_ids_t2i = step_outputs["input_ids_t2i"]
                attention_mask = step_outputs["attention_mask"]
                labels = step_outputs["labels"]
                batch_size_t2i = step_outputs["batch_size_t2i"]
                batch_size_lm = step_outputs["batch_size_lm"]
                batch_size_mmu = step_outputs["batch_size_mmu"]
                image_tokens_ori = step_outputs["image_tokens_ori"]
                texts = step_outputs["texts"]
                logits = step_outputs["logits"]
                
                # Накапливаем loss для усреднения по gradient accumulation steps
                avg_loss_t2i = step_outputs["avg_loss_t2i"]
                avg_loss_lm = step_outputs["avg_loss_lm"]
                avg_loss_mmu = step_outputs["avg_loss_mmu"]
                avg_masking_rate = step_outputs["avg_masking_rate"]
                balance_loss = step_outputs["balance_loss"]
                orthogonal_loss = step_outputs["orthogonal_loss"]
                
                accumulated_loss_t2i += avg_loss_t2i.item() if torch.is_tensor(avg_loss_t2i) else avg_loss_t2i
                accumulated_loss_lm += avg_loss_lm.item() if torch.is_tensor(avg_loss_lm) else avg_loss_lm
                accumulated_loss_mmu += avg_loss_mmu.item() if torch.is_tensor(avg_loss_mmu) else avg_loss_mmu
                accumulated_balance_loss += balance_loss.item() if torch.is_tensor(balance_loss) else balance_loss
                accumulated_orthogonal_loss += orthogonal_loss.item() if torch.is_tensor(orthogonal_loss) else orthogonal_loss
                accumulated_masking_rate += avg_masking_rate.item() if torch.is_tensor(avg_masking_rate) else avg_masking_rate
                accumulation_count += 1

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # Усредняем накопленные loss по gradient accumulation steps
                    if accumulation_count > 0:
                        # ВАЖНО: Усредняем по количеству микробатчей (accumulation_count)
                        # Каждый микробатч уже имеет усреднённый loss (из F.cross_entropy с reduction='mean')
                        # Поэтому мы просто усредняем эти усреднённые значения по микробатчам
                        mean_loss_t2i = accumulated_loss_t2i / accumulation_count
                        mean_loss_lm = accumulated_loss_lm / accumulation_count
                        mean_loss_mmu = accumulated_loss_mmu / accumulation_count
                        mean_balance_loss = accumulated_balance_loss / accumulation_count
                        mean_orthogonal_loss = accumulated_orthogonal_loss / accumulation_count
                        mean_masking_rate = accumulated_masking_rate / accumulation_count
                        
                        # Диагностика: проверяем, что accumulation_count соответствует ожидаемому
                        expected_accumulation = config.training.gradient_accumulation_steps
                        if accumulation_count != expected_accumulation and (global_step + 1) % config.experiment.log_every == 0:
                            logger.warning(
                                f"⚠️ Step {global_step + 1}: accumulation_count={accumulation_count} != "
                                f"expected={expected_accumulation}. Это может указывать на проблему с gradient accumulation."
                            )
                        
                        # Логируем усреднённые значения в MLflow
                        if mlflow_client is not None and mlflow_run_id is not None:
                            try:
                                mlflow_client.log_metric(mlflow_run_id, "loss/t2i", mean_loss_t2i, step=global_step + 1)
                                mlflow_client.log_metric(mlflow_run_id, "loss/lm", mean_loss_lm, step=global_step + 1)
                                mlflow_client.log_metric(mlflow_run_id, "loss/mmu", mean_loss_mmu, step=global_step + 1)
                                mlflow_client.log_metric(mlflow_run_id, "loss/balance", mean_balance_loss, step=global_step + 1)
                                mlflow_client.log_metric(mlflow_run_id, "loss/orthogonal", mean_orthogonal_loss, step=global_step + 1)
                                mlflow_client.log_metric(mlflow_run_id, "masking_rate", mean_masking_rate, step=global_step + 1)
                            except Exception as e:
                                logger.warning(f"Failed to log accumulated losses: {e}")
                        
                        if (global_step + 1) % config.experiment.log_every == 0:
                            logger.info(
                                f"📊 Step {global_step + 1}: Avg over {accumulation_count} micro-batches - "
                                f"loss_t2i={mean_loss_t2i:.4f}, loss_mmu={mean_loss_mmu:.4f}, mask_rate={mean_masking_rate:.4f}"
                            )
                    
                    # Сбрасываем накопители
                    accumulated_loss_t2i = 0.0
                    accumulated_loss_lm = 0.0
                    accumulated_loss_mmu = 0.0
                    accumulated_balance_loss = 0.0
                    accumulated_orthogonal_loss = 0.0
                    accumulated_masking_rate = 0.0
                    accumulation_count = 0
                    
                    batch_time_m.update(time.time() - end)
                    end = time.time()
                    if (
                        (global_step + 1) % 100 == 0
                        and config.get("moe", {}).get("enabled", False)
                    ):
                        if accelerator.is_main_process:
                            collect_and_log_moe_activations(
                                model=model,
                                accelerator=accelerator,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                config=config,
                                batch_size_t2i=batch_size_t2i,
                                batch_size_lm=batch_size_lm,
                                batch_size_mmu=batch_size_mmu,
                                global_step=global_step + 1,
                                mlflow_client=mlflow_client,
                                mlflow_run_id=mlflow_run_id,
                            )
                        # Синхронизируем все процессы после сбора статистики
                        accelerator.wait_for_everyone()

                    step_plus_one = global_step + 1
                    should_generate = global_step % config.experiment.generate_every == 0
                    if should_generate and accelerator.is_main_process:
                        logger.info(
                            f"🎨 Step {step_plus_one}: should_generate={should_generate}, is_main_process={accelerator.is_main_process}"
                        )
                        generate_images(
                            model,
                            vq_model,
                            uni_prompting,
                            accelerator,
                            config,
                            global_step + 1,
                            mask_schedule=mask_schedule,
                            mlflow_client=mlflow_client,
                            mlflow_run_id=mlflow_run_id,
                        )

                        visualize_predictions(
                            model,
                            vq_model,
                            uni_prompting,
                            config,
                            global_step + 1,
                            input_ids_t2i,
                            image_tokens_ori,
                            batch["t2i_flow"]["images"],
                            texts,
                            logits,
                            mlflow_client=mlflow_client,
                            mlflow_run_id=mlflow_run_id,
                        )

                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()

                        # if not config.model.showo.get("w_clip_vit", False):
                        #     evaluate_mmu(
                        #         model,
                        #         vq_model,
                        #         uni_prompting,
                        #         accelerator,
                        #         config,
                        #         global_step + 1,
                        #         batch["mmu_flow"],
                        #         mlflow_client=mlflow_client,
                        #         mlflow_run_id=mlflow_run_id,
                        #     )

                    should_eval_metrics = (global_step % metric_interval == 0)
                    
                    if should_eval_metrics:
                        if accelerator.is_main_process:
                            print('Run distributed benchmark')
                        
                        device = str(accelerator.device)
                        rank = accelerator.process_index
                        world_size = accelerator.num_processes
                    
                        # Переводим модель в eval mode для бенчмарка
                        model.eval()
                        unwrapped_model = accelerator.unwrap_model(model)
                        mask_token_id = unwrapped_model.config.mask_token_id
                        
                        # Весь бенчмарк в no_grad для экономии памяти
                        with torch.no_grad():
                            # Бенчмарк только на rank 0 (torch_fidelity не поддерживает distributed)
                            if accelerator.is_main_process:
                        benchmark = ShowoBenchmark(
                            config=config,
                            coco_dataset=coco_eval_dataset,
                            model=unwrapped_model,
                            vq_model=vq_model,
                            mask_token_id=mask_token_id,
                            device=device,
                            save_comparisons=False,
                        )
                        
                        subset_size = config.evaluation.coco['subset_size']
                        seed = config.evaluation.coco['seed']
                        
                                print(f'Starting FID benchmark with {subset_size} samples...')
                                fid_score = benchmark.run_subset(subset_size=subset_size, seed=seed)

                                if fid_score is not None:
                                    print(f'Benchmark finished, FID: {fid_score:.4f}')
                            mlflow_client.log_metric(
                                mlflow_run_id,
                                "metrics/fid_coco",
                                        fid_score,
                                step=step_plus_one,
                            )
                                else:
                                    print('Benchmark returned None (skip_compute=True?)')
                                
                                # Очистка памяти
                                benchmark.model = None
                                benchmark.vq_model = None
                                benchmark.tokenizer = None
                                benchmark.uni_prompting = None
                        del benchmark
                            
                            accelerator.wait_for_everyone()
                        
                        # Возвращаем модель в train mode
                        model.train()
                        
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    
                    # Wait for all processes to finish benchmark
                    accelerator.wait_for_everyone()

                    global_step += 1

                    if global_step >= config.training.max_train_steps:
                        logger.info(f"Достигнут лимит шагов: {global_step} >= {config.training.max_train_steps}")
                        break
        except Exception as e:
            logger.error(f"Критическая ошибка в цикле обучения: {e}", exc_info=True)
            # Не останавливаем обучение, просто логируем ошибку
        finally:
            if accelerator.is_main_process:
                if pbar is not None:
                    pbar.close()

    accelerator.wait_for_everyone()
    # Save final checkpoint (отключено)
    # save_checkpoint(model, config, accelerator, global_step)

    # Завершаем MLflow run только если обучение действительно завершилось (достигнут max_train_steps)
    if mlflow_client is not None and mlflow_run_id is not None:
        if global_step >= config.training.max_train_steps:
            mlflow_client.set_terminated(mlflow_run_id, status="FINISHED")
            logger.info(f"MLflow run finished: достигнут лимит шагов {global_step} >= {config.training.max_train_steps}")
        else:
            logger.info(f"MLflow run продолжается: шаг {global_step} < {config.training.max_train_steps}")
            # НЕ завершаем run, если обучение не закончилось

    # Save final model (отключено)
    # if accelerator.is_main_process:
    #     model = accelerator.unwrap_model(model)
    #     model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    # Исправление для версий accelerate, где trackers может отсутствовать
    try:
        if hasattr(accelerator, 'trackers') and accelerator.trackers:
            accelerator.end_training()
        else:
            logger.info("Trackers not initialized, skipping end_training()")
    except AttributeError:
        logger.warning("accelerator.end_training() failed (trackers not available), skipping")


if __name__ == "__main__":
    main()
