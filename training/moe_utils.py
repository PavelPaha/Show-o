from typing import Dict
import torch
from copy import deepcopy
from collections import defaultdict

class LayerExpertStatsCollector:
    def __init__(self, unwrapped_model):
        self.model = unwrapped_model

    def collect(self, global_step) -> Dict[str, Dict[int, Dict[int, int]]]:
        layer_expert_counts: Dict[str, Dict[int, Dict[int, int]]] =  defaultdict(dict)
        modality_layer_probs: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)

        for layer_idx, layer in enumerate(self.model.showo.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                distr_hist = deepcopy(layer.mlp._gate_distribution_history)
                prob_history = deepcopy(layer.mlp._gate_probability_history)
                
                print(f'Modalities: {list(prob_history.keys())}')

                for modality, history in prob_history.items():
                    probs = history.get(global_step)
                    modality_layer_probs[modality][layer_idx] = probs.detach().cpu()
                print(f'{distr_hist=}')
                print(f'{prob_history=}')

                for modality, history in distr_hist.items():
                    if modality not in layer_expert_counts:
                        layer_expert_counts[modality] = {}

                    aggregated_counts: Dict[int, int] = {}
                    for step_counts in history.values():
                        if not isinstance(step_counts, dict):
                            continue
                        for expert_id, count in step_counts.items():
                            aggregated_counts[expert_id] = (
                                aggregated_counts.get(expert_id, 0) + int(count)
                            )

                    layer_expert_counts[modality][layer_idx] = aggregated_counts

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

