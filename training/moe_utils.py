from typing import Dict
import torch
from collections import defaultdict

class LayerExpertStatsCollector:
    def __init__(self, unwrapped_model):
        self.model = unwrapped_model

    def collect(self, global_step) -> Dict[str, Dict[int, Dict[int, int]]]:
        layer_expert_counts: Dict[str, Dict[int, Dict[int, int]]] = defaultdict(dict)
        modality_layer_probs: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)

        for layer_idx, layer in enumerate(self.model.showo.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                # Получаем ссылки на историю без deepcopy (экономим память)
                distr_hist = layer.mlp._gate_distribution_history
                prob_history = layer.mlp._gate_probability_history

                for modality, history in prob_history.items():
                    probs = history.get(global_step)
                    if probs is not None:
                        modality_layer_probs[modality][layer_idx] = probs.detach().cpu()

                for modality, history in distr_hist.items():
                    if modality not in layer_expert_counts:
                        layer_expert_counts[modality] = {}

                    # Берем только данные текущего шага
                    step_counts = history.get(global_step)
                    if step_counts is not None and isinstance(step_counts, dict):
                        layer_expert_counts[modality][layer_idx] = step_counts.copy()
                    else:
                        layer_expert_counts[modality][layer_idx] = {}

        return layer_expert_counts, modality_layer_probs

    def collect_probabilities(self, global_step: int) -> Dict[str, Dict[int, torch.Tensor]]:
        modality_layer_probs: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)

        for layer_idx, layer in enumerate(self.model.showo.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "_gate_probability_history"):
                prob_history = layer.mlp._gate_probability_history
                for modality, history in prob_history.items():
                    probs = history.get(global_step)
                    if probs is not None:
                        modality_layer_probs[modality][layer_idx] = probs.detach().cpu()

        return modality_layer_probs


def compute_modality_bias_tensor(moe_layer, input_ids, num_tokens, dtype, device):
    if (
        not getattr(moe_layer, "use_modality_bias", False)
        or input_ids is None
        or moe_layer.modality_init_hardness <= 0
    ):
        return None

    if moe_layer._global_step < moe_layer.modality_init_steps:
        progress = moe_layer._global_step / max(moe_layer.modality_init_steps, 1)
        hardness = moe_layer.modality_init_hardness - (
            moe_layer.modality_init_hardness - moe_layer.modality_init_hardness_min
        ) * progress
    else:
        hardness = moe_layer.modality_init_hardness_min

    if hardness <= 0:
        return None

    modality = moe_layer._get_token_modality(input_ids.view(-1))
    if modality is None:
        return None

    text_mask = modality == 0
    image_mask = modality == 1

    modality_bias = torch.zeros(
        num_tokens,
        moe_layer.modality_bias_text.size(0),
        device=device,
        dtype=dtype,
    )
    if text_mask.any():
        modality_bias[text_mask] = moe_layer.modality_bias_text.unsqueeze(0) * hardness
    if image_mask.any():
        modality_bias[image_mask] = moe_layer.modality_bias_image.unsqueeze(0) * hardness

    return modality_bias


def compute_domain_bias_tensor(
    moe_layer,
    domain_id,
    num_tokens,
    dtype,
    device,
):
    if (
        not getattr(moe_layer, "use_domain_bias", False)
        or domain_id is None
    ):
        return None

    if moe_layer._global_step < moe_layer.domain_init_steps:
        progress = moe_layer._global_step / max(moe_layer.domain_init_steps, 1)
        hardness = moe_layer.domain_init_hardness - (
            moe_layer.domain_init_hardness - moe_layer.domain_init_hardness_min
        ) * progress
    else:
        hardness = moe_layer.domain_init_hardness_min

    if hardness <= 0:
        return None

    domain_key = str(domain_id)
    buffer_name = moe_layer._domain_bias_buffer_map.get(domain_key)
    if buffer_name is None:
        return None

    domain_bias_vector = getattr(moe_layer, buffer_name)
    return (
        domain_bias_vector.unsqueeze(0)
        .expand(num_tokens, -1)
        .to(device=device, dtype=dtype)
        * hardness
    )


def compute_domain_bias_from_sample_domains(
    moe_layer,
    sample_domains,
    batch_size,
    seq_len,
    num_tokens,
    dtype,
    device,
):
    """
    Вычисляет domain bias для каждого токена на основе sample_domains.
    
    Args:
        moe_layer: MoE layer
        sample_domains: [batch_size] - список доменов для каждого сэмпла (None для T2I/LM, имя домена для MMU)
        batch_size: размер батча
        seq_len: длина последовательности
        num_tokens: общее количество токенов (batch_size * seq_len)
        dtype: тип данных
        device: устройство
    
    Returns:
        domain_bias: [num_tokens, num_experts] - bias для каждого токена
    """
    if (
        not getattr(moe_layer, "use_domain_bias", False)
        or sample_domains is None
        or len(sample_domains) == 0
    ):
        return None

    if moe_layer._global_step < moe_layer.domain_init_steps:
        progress = moe_layer._global_step / max(moe_layer.domain_init_steps, 1)
        hardness = moe_layer.domain_init_hardness - (
            moe_layer.domain_init_hardness - moe_layer.domain_init_hardness_min
        ) * progress
    else:
        hardness = moe_layer.domain_init_hardness_min

    if hardness <= 0:
        return None

    # Создаем bias для каждого токена на основе его домена
    domain_bias = torch.zeros(num_tokens, moe_layer.num_experts, device=device, dtype=dtype)
    
    # Подсчитываем статистику по доменам для отладки
    domain_stats = {}
    
    # Для каждого токена определяем его домен
    for token_idx in range(num_tokens):
        batch_idx = token_idx // seq_len
        if batch_idx < len(sample_domains):
            domain_name = sample_domains[batch_idx]
            if domain_name is not None:
                domain_key = str(domain_name)
                buffer_name = moe_layer._domain_bias_buffer_map.get(domain_key)
                if buffer_name is not None:
                    domain_bias_vector = getattr(moe_layer, buffer_name)
                    domain_bias[token_idx] = domain_bias_vector.to(device=device, dtype=dtype) * hardness
                    
                    # Отладочная статистика
                    if moe_layer._global_step <= 2:
                        if domain_key not in domain_stats:
                            domain_stats[domain_key] = {
                                'tokens': 0,
                                'bias_vector': domain_bias_vector.cpu().clone(),
                                'hardness': hardness
                            }
                        domain_stats[domain_key]['tokens'] += 1
    
    # Отладочный вывод
    if moe_layer._global_step <= 2 and domain_stats:
        print(f"[Layer {moe_layer._layer_id}] Domain bias stats (step {moe_layer._global_step}):")
        for domain_key, stats in domain_stats.items():
            print(f"  {domain_key}: {stats['tokens']} tokens, hardness={stats['hardness']:.2f}, bias={stats['bias_vector'].tolist()}")
    
    return domain_bias


def save_moe_weights(model, path):
    moe_state = {
        k: v for k, v in model.state_dict().items() 
        if any(x in k for x in ['mlp.experts', 'mlp.gate', 'mlp.alpha'])
    }
    torch.save(moe_state, path)


def load_moe_weights(model, path):
    moe_weights = torch.load(path)
    model.load_state_dict(moe_weights, strict=False)
    return model

