import os
import sys
import random

sys.path.insert(0, "/home/jovyan/vasiliev/notebooks/Show-o")

import torch
import torch.nn.functional as F
from inference_t2i import get_model, get_vq_model_class
from benchmark.coco_dataset import COCODataset
from transformers import AutoTokenizer
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Sequence
import numpy as np
from omegaconf import OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance
from models import Showo, MAGVITv2, get_mask_chedule
from models import Showo
from training.utils import get_config
from PIL import Image, ImageDraw, ImageFont
from training.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_predict_next,
)

# from utils import create_comparison_image


def transpose_batch(arr):
    keys = arr[0].keys()
    return {key: [item[key] for item in arr] for key in keys}


class ShowoBenchmark:
    def __init__(
        self,
        config,
        coco_dataset,
        model,
        vq_model,
        mask_token_id,
        device="cuda:0",
        save_comparisons=False,
        output_dir="benchmark_output",
    ):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        assert config.mode == "t2i"

        self.config = config
        self.steps = 10
        self.batch_size = self.config.training.batch_size
        self.coco_dataset = coco_dataset
        self.save_comparisons = save_comparisons
        self.output_dir = output_dir

        if self.save_comparisons:
            os.makedirs(self.output_dir, exist_ok=True)
        # self.dataloader = DataLoader(self.coco_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     pin_memory=True,
        #     num_workers=4
        # )
        #
        # vq_model = get_vq_model_class(config.model.vq_model.type)
        # vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(
        #     device
        # )
        
        self.vq_model = vq_model
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()

        self.vq_model = vq_model
        # Отключаем distributed синхронизацию для FID, т.к. бенчмарк запускается только на rank 0
        self.fid_metric = FrechetInceptionDistance(feature=64, dist_sync_on_step=False, sync_on_compute=False).to(self.device)
        self.mask_token_id = mask_token_id

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.showo.llm_model_path, padding_side="left"
        )

        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
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

    def run(self):
        full_indices = list(range(len(self.coco_dataset)))
        return self._evaluate_indices(full_indices)

    def run_subset(self, subset_size=1000, seed=42):
        if subset_size <= 0:
            raise ValueError("subset_size must be positive")
        indices = list(range(len(self.coco_dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        subset = indices[: min(subset_size, len(indices))]
        return self._evaluate_indices(subset)

    def _evaluate_indices(self, indices: Sequence[int]):
        config = self.config
        uni_prompting = self.uni_prompting
        total = len(indices)
        for start in tqdm(range(0, total, self.batch_size)):
            current_indices = indices[start : start + self.batch_size]
            batch: List[Dict[str, Any]] = transpose_batch(
                [self.coco_dataset[i] for i in current_indices]
            )
            prompts = batch["caption"]
            gt_images = batch["image"]

            image_tokens = (
                torch.ones(
                    (len(prompts), self.config.model.showo.num_vq_tokens),
                    dtype=torch.long,
                    device=self.device,
                )
                * self.mask_token_id
            )

            input_ids, _ = uni_prompting((prompts, image_tokens), "t2i_gen")
            input_ids = input_ids.to(self.device)

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(
                    ([""] * len(prompts), image_tokens), "t2i_gen"
                )
                uncond_input_ids = uncond_input_ids.to(self.device)
                attention_mask = create_attention_mask_predict_next(
                    torch.cat([input_ids, uncond_input_ids], dim=0),
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )
            else:
                attention_mask = create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                    soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                    rm_pad_in_image=True,
                )
                uncond_input_ids = None

            attention_mask = attention_mask.to(self.device)

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(
                    config.training.get("mask_schedule", "cosine")
                )

            with torch.no_grad():
                gen_token_ids = self.model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(
                gen_token_ids, max=config.model.showo.codebook_size - 1, min=0
            )
            images = self.vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]

            transform_to_tensor = transforms.Compose(
                [
                    transforms.Resize(
                        (299, 299)
                    ),  # FID использует Inception, который ожидает 299x299
                    transforms.ToTensor(),
                ]
            )

            all_gt_tensors = []
            all_gen_tensors = []

            for img_idx, (generated_image, gt_image) in enumerate(
                zip(pil_images, gt_images)
            ):
                # Создаем сравнительную картинку если включено сохранение
                # if self.save_comparisons:
                #     caption = (
                #         prompts[img_idx] if img_idx < len(prompts) else "No caption"
                #     )
                #     comparison_path = create_comparison_image(
                #         generated_image,
                #         gt_image,
                #         caption,
                #         start,
                #         img_idx,
                #         self.output_dir,
                #     )
                #     print(f"Сохранено сравнение: {comparison_path}")

                gt_tensor = (
                    (transform_to_tensor(gt_image) * 255).unsqueeze(0).to(torch.uint8)
                )
                gen_tensor = (
                    (transform_to_tensor(generated_image) * 255)
                    .unsqueeze(0)
                    .to(torch.uint8)
                )

                all_gt_tensors.append(gt_tensor)
                all_gen_tensors.append(gen_tensor)

            if all_gt_tensors and all_gen_tensors:
                gt_batch = torch.cat(all_gt_tensors, dim=0)
                gen_batch = torch.cat(all_gen_tensors, dim=0)

                # FID expects uint8 in [0, 255]; just move tensors onto device
                gt_batch = gt_batch.to(self.device, non_blocking=True)
                gen_batch = gen_batch.to(self.device, non_blocking=True)

                self.fid_metric.update(gt_batch, real=True)
                self.fid_metric.update(gen_batch, real=False)
                processed = min(start + len(current_indices), total)
                print(
                    f"[Benchmark] Batch done: {processed}/{total} samples "
                    f"(batch_size={len(current_indices)})"
                )
                fid_score = self.fid_metric.compute()
                print(f"Calculated FID score: {fid_score.item()}")

        final_fid = self.fid_metric.compute()
        final_value = final_fid.item()
        print(f"Final FID: {final_value}")
        return final_value


if __name__ == "__main__":
    config_path = "../configs/showo_demo_w_clip_vit_512x512.yaml"
    config = OmegaConf.load(config_path)
    config.mode = "t2i"
    config.training.guidance_scale = 3.0
    config.training.generation_timesteps = 18
    config.training.mask_schedule = "cosine"
    config.training.noise_type = "mask"
    config.training.generation_temperature = 1.0

    model = get_model(config)

    coco_dataset = COCODataset(
        root="/home/jovyan/vasiliev/notebooks/Show-o/train2017",
        annFile="/home/jovyan/vasiliev/notebooks/Show-o/annotations/captions_train2017.json",
    )

    benchmark = ShowoBenchmark(
        config,
        coco_dataset,
        model,
        save_comparisons=False,  # Включаем сохранение сравнений
        # output_dir="benchmark_comparisons",
    )
    benchmark.run()
