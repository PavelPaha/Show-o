import torch
import torch.nn as nn
import copy
import json
import os
import logging
from typing import Optional
from models.phi import PhiConfig
from models.moe_gates.gshard_gate import GShardGate
from training.moe_visualization import MoEVisualizer
from training.moe_utils import (
    compute_modality_bias_tensor,
    compute_domain_bias_tensor,
)

from collections import defaultdict


class SmallPhiMLP(nn.Module):
    def __init__(self, config: PhiConfig, scale_factor: int = 1):
        super().__init__()
        self.config = config
        intermediate_size = config.intermediate_size // scale_factor
        if config.hidden_act == "gelu_new":
            self.activation_fn = torch.nn.functional.gelu
        else:
            self.activation_fn = getattr(torch.nn.functional, config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states



class MoE(nn.Module):
    def __init__(
        self,
        config: PhiConfig,
        moe_config,
        template_mlp: Optional[nn.Module] = None,
        mlflow_logger = None,
        visualizer = None,
        layer_idx = None,
        special_tokens = None,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.num_experts = int(moe_config["num_experts"])
        self.top_k = int(moe_config["top_k"])
        noise_std = float(moe_config["noise_std"])
        use_modality_bias = bool(moe_config["use_modality_bias"])
        use_domain_bias = bool(moe_config["use_domain_bias"])
        modality_init_hardness = float(moe_config["modality_init_hardness"])
        modality_init_steps = int(moe_config["modality_init_steps"])
        modality_init_hardness_min = float(moe_config["modality_init_hardness_min"])
        domain_init_hardness = float(moe_config["domain_init_hardness"])
        domain_init_steps = int(moe_config["domain_init_steps"])
        domain_init_hardness_min = float(moe_config["domain_init_hardness_min"])
        gate_capacity = tuple(moe_config["gate_capacity"])
        assert len(gate_capacity) == 2
        random_routing = bool(moe_config["random_routing"])
        domain_to_expert_map = moe_config["domain_to_expert_map"]
        log_gates = bool(moe_config["log_gates"])
        log_activations = bool(moe_config["log_activations"])
        log_frequency = int(moe_config["log_frequency"])
        hidden_size = config.hidden_size
        num_experts = int(moe_config["num_experts"])
        top_k = int(moe_config["top_k"])
        
        self.gate = GShardGate(
            hidden_size, 
            num_experts, 
            world_size=4, 
            top_k=top_k, 
            capacity=gate_capacity,
            random_routing=random_routing,
            gate_bias=True
        )
        self.hidden_size = hidden_size
        num_text_experts = self.num_experts // 2
        tot_expert = self.num_experts * 4  # world_size=4
        self.register_buffer('modality_bias_text', torch.zeros(tot_expert))
        self.register_buffer('modality_bias_image', torch.zeros(tot_expert))
        init_bias_val = 0.0
        for rank in range(4):  # world_size=4
            start_text = rank * self.num_experts
            end_text = start_text + num_text_experts
            start_image = end_text
            end_image = start_image + num_text_experts
            self.modality_bias_text[start_text:end_text] = init_bias_val
            self.modality_bias_image[start_image:end_image] = init_bias_val
        self.use_modality_bias = bool(use_modality_bias)
        self.use_domain_bias = bool(use_domain_bias)
        self.modality_init_hardness = float(modality_init_hardness)
        self.modality_init_steps = int(modality_init_steps)
        self.modality_init_hardness_min = float(modality_init_hardness_min)
        self.domain_init_hardness = float(domain_init_hardness)
        self.domain_init_steps = int(domain_init_steps)
        self.domain_init_hardness_min = float(domain_init_hardness_min)
        self.domain_to_expert_map = domain_to_expert_map or {}
        self.world_size = 4
        self._domain_bias_buffer_map = {}
        for idx, (domain_name, expert_list) in enumerate(self.domain_to_expert_map.items()):
            if not expert_list:
                continue
            bias_vec = torch.zeros(tot_expert, dtype=torch.float32)
            for rank in range(self.world_size):
                base = rank * num_experts
                for expert_id in expert_list:
                    if 0 <= expert_id < num_experts:
                        bias_vec[base + expert_id] = 1.0
            buffer_name = f"_domain_bias_vec_{idx}"
            self.register_buffer(buffer_name, bias_vec)
            self._domain_bias_buffer_map[str(domain_name)] = buffer_name
        
        # Если use_modality_bias отключен, обнуляем bias buffers
        if not self.use_modality_bias:
            self.modality_bias_text.zero_()
            self.modality_bias_image.zero_()
        self.experts = nn.ModuleList()
        if template_mlp is not None:
            for _ in range(self.num_experts):
                expert = copy.deepcopy(template_mlp)
                with torch.no_grad():
                    for p in expert.parameters():
                        p.add_(torch.randn_like(p) * noise_std)
                self.experts.append(expert)
        else:
            self.experts = nn.ModuleList([SmallPhiMLP(config, scale_factor=1) for _ in range(self.num_experts)])
        # Постоянный множитель для регулировки выходов экспертов. Регистрируем как buffer,
        # чтобы избежать ошибок DDP, если некоторые эксперты не используются в итерации.
        self.register_buffer('alpha', torch.ones(self.num_experts))
        self._step_count = 0
        self._log_frequency = log_frequency
        self._global_step = 0
        self._layer_id = layer_idx
        self._soi_id = special_tokens.get("soi_id")
        self._eoi_id = special_tokens.get("eoi_id")
        self._sov_id = special_tokens.get("sov_id")
        self._eov_id = special_tokens.get("eov_id")
        self._gate_distribution_history = defaultdict(dict)
        self._mlflow_logger = mlflow_logger
        self._visualizer : MoEVisualizer = visualizer

        self._log_gates = log_gates
        self._log_activations = log_activations
        self._log_frequency = log_frequency

    def set_global_step(self, global_step):
        self._global_step = global_step


    def forward(
        self,
        hidden_states,
        input_ids=None,
        temperature: Optional[float] = None,
        domain_id: Optional[str] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        device = hidden_states.device
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [B*L, H]
        B = hidden_states_flat.shape[0]

        total_bias = None
        modality_bias = compute_modality_bias_tensor(
            self,
            input_ids,
            B,
            hidden_states.dtype,
            device,
        )
        if modality_bias is not None:
            total_bias = modality_bias

        domain_bias_tensor = compute_domain_bias_tensor(
            self,
            domain_id,
            B,
            hidden_states.dtype,
            device,
        )
        if domain_bias_tensor is not None:
            total_bias = domain_bias_tensor if total_bias is None else total_bias + domain_bias_tensor

        if bias is not None:
            bias = bias.to(device=device, dtype=hidden_states.dtype)
            if total_bias is None:
                total_bias = bias
            else:
                total_bias = total_bias + bias

        gate_idx, gate_score = self.gate(hidden_states_flat, temperature=temperature, bias=total_bias)
        overflowed_mask = (gate_idx[:, 0] == -1) & (gate_idx[:, 1] == -1)
        
        out_flat = torch.zeros(B, hidden_size, device=device, dtype=hidden_states.dtype)
        
        for k in range(self.top_k):
            expert_indices = gate_idx[:, k]
            weights = gate_score[:, k]
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id) & (~overflowed_mask)
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    weight = weights[mask] * self.alpha[expert_id]
                    out_flat[mask] += expert_output * weight.unsqueeze(-1)
        
        # When both experts exceed capacity, token passes via residual connection
        if overflowed_mask.any():
            out_flat[overflowed_mask] = hidden_states_flat[overflowed_mask]

        output = out_flat.view(batch_size, seq_len, hidden_size)
        self._step_count += 1
        should_log = self._global_step % self._log_frequency == 0
        if self._log_gates and should_log:
            self._log_gate_distribution(gate_idx, gate_score.detach(), input_ids, domain_id=domain_id)

        return output


    def _log_gate_distribution(self, gate_idx, gate_score, input_ids=None, domain_id=None):
        modality = None
        if isinstance(input_ids, torch.Tensor):
            modality = self._get_token_modality(input_ids.view(-1))
        
        expert_counts = {}
        for expert_id in range(self.num_experts):
            count = (gate_idx == expert_id).sum().item()
            expert_counts[expert_id] = count
        
        total_activations = sum(expert_counts.values())
        
        self._accumulate_history(self._gate_distribution_history['overall'], expert_counts)
        self._save_distribution_to_json(expert_counts, "overall")
        
        text_expert_counts = None
        image_expert_counts = None
        
        if modality is not None:
            text_mask = (modality == 0)
            image_mask = (modality == 1)
            video_mask = (modality == 2)
            
            modalities = [
                ("text", text_mask),
                ("image", image_mask),
                ("video", video_mask)
            ]
            
            for modality_name, mask in modalities:
                modality_gate_idx = gate_idx[mask]
                modality_gate_score = gate_score[mask]
                modality_expert_counts = {}
                for expert_id in range(self.num_experts):
                    count = (modality_gate_idx == expert_id).sum().item()
                    modality_expert_counts[expert_id] = count
                
                modality_total = sum(modality_expert_counts.values())
                
                self._accumulate_history(self._gate_distribution_history[modality_name], modality_expert_counts)
                self._save_distribution_to_json(modality_expert_counts, modality_name)
                
                if modality_name == "text":
                    text_expert_counts = modality_expert_counts
                elif modality_name == "image":
                    image_expert_counts = modality_expert_counts
                
                self._log_to_mlflow_modality_gates(
                    modality_expert_counts, modality_total, modality_gate_score, modality_name
                )
                    
                        
        
        if total_activations > 0:
            self._log_to_mlflow_gates(expert_counts, total_activations, gate_score)
        
        if domain_id is not None:
            self._ensure_domain_buffer(domain_id)
        if domain_id is not None:
            domain_expert_counts = {}
            for expert_id in range(self.num_experts):
                count = (gate_idx == expert_id).sum().item()
                domain_expert_counts[expert_id] = count
            
            self._accumulate_history(self._gate_distribution_history[domain_id], domain_expert_counts)
            self._save_distribution_to_json(domain_expert_counts, f"domain_{domain_id}")
        

        self._log_all_plots_to_mlflow(
            overall_expert_counts=expert_counts,
            overall_gate_score=gate_score,
            text_expert_counts=text_expert_counts,
            image_expert_counts=image_expert_counts,
            domain_id=domain_id
        )
    
    
    def _log_all_plots_to_mlflow(self, overall_expert_counts, 
                                    overall_gate_score, 
                                    text_expert_counts=None, image_expert_counts=None, domain_id=None):
        if self._mlflow_logger is None or self._visualizer is None:
            return
        
        overall_heatmap_bytes = self._visualizer.create_distribution_heatmap(
            self._gate_distribution_history, "overall", self._global_step
        )
        
        overall_histogram_bytes = self._visualizer.create_expert_activation_histogram(overall_expert_counts)
        
        text_history = self._gate_distribution_history["text"]
        image_history = self._gate_distribution_history["image"]
        combined_plot_bytes = self._visualizer.create_modality_combined_plot(
            text_history=text_history,
            image_history=image_history,
            text_expert_counts=text_expert_counts,
            image_expert_counts=image_expert_counts,
            global_step=self._global_step
        )
        
        domain_plot_bytes = None
        if domain_id is not None:
            domain_history = self._gate_distribution_history.get(domain_id, {})
            current_expert_counts = domain_history.get(self._global_step, {}) if domain_history else {}
            domain_plot_bytes = self._visualizer.create_domain_plot(
                domain_id, domain_history, self._global_step, current_expert_counts
            )
        
        all_domains_plot_bytes = self._visualizer.create_all_domains_combined_plot(
            self._gate_distribution_history, self._global_step
        )
        
        self._mlflow_logger.log_all_plots(
            layer_id=self._layer_id,
            global_step=self._global_step,
            overall_heatmap_bytes=overall_heatmap_bytes,
            overall_histogram_bytes=overall_histogram_bytes,
            combined_plot_bytes=combined_plot_bytes,
            domain_plot_bytes=domain_plot_bytes,
            all_domains_plot_bytes=all_domains_plot_bytes,
            domain_id=domain_id
        )
    
    def _accumulate_history(self, history_dict, counts):
        existing = history_dict.get(self._global_step)
        if existing is None:
            history_dict[self._global_step] = counts.copy()
        else:
            for expert_id, count in counts.items():
                existing[expert_id] = existing.get(expert_id, 0) + count


    def set_global_step(self, global_step):
        self._global_step = global_step
        
    def get_balance_loss(self, clear=True):
        gate_loss =  self.gate.get_loss(clear=clear)
        orthogonal_loss = torch.zeros_like(gate_loss)
        return gate_loss, orthogonal_loss
        

    def _ensure_domain_buffer(self, domain_id: str):
        domain_key = str(domain_id)
        buffer_name = self._domain_bias_buffer_map.get(domain_key)
        if buffer_name is None:
            buffer_name = f"_domain_bias_vec_extra_{len(self._domain_bias_buffer_map)}"
            bias_vec = torch.zeros(
                self.num_experts * self.world_size,
                dtype=torch.float32,
                device=self.modality_bias_text.device,
            )
            self.register_buffer(buffer_name, bias_vec)
            self._domain_bias_buffer_map[domain_key] = buffer_name
        return getattr(self, buffer_name)


    def _get_token_modality(self, input_ids_flat):
        if (
            input_ids_flat is None
            or not isinstance(input_ids_flat, torch.Tensor)
            or input_ids_flat.dtype not in (torch.int16, torch.int32, torch.int64)
            or self._soi_id is None
            or self._eoi_id is None
        ):
            return None
        modality = torch.zeros(input_ids_flat.shape[0], dtype=torch.long, device=input_ids_flat.device)
        soi_positions = (input_ids_flat == self._soi_id).nonzero(as_tuple=True)[0]
        eoi_positions = (input_ids_flat == self._eoi_id).nonzero(as_tuple=True)[0]
        for soi_pos, eoi_pos in zip(soi_positions, eoi_positions):
            if soi_pos < eoi_pos:
                modality[soi_pos:eoi_pos+1] = 1  # 1 = image tokens
        if self._sov_id is not None and self._eov_id is not None:
            sov_positions = (input_ids_flat == self._sov_id).nonzero(as_tuple=True)[0]
            eov_positions = (input_ids_flat == self._eov_id).nonzero(as_tuple=True)[0]
            
            for sov_pos, eov_pos in zip(sov_positions, eov_positions):
                if sov_pos < eov_pos:
                    modality[sov_pos:eov_pos+1] = 2  # 2 = video tokens
        
        return modality

    
    def _log_to_mlflow_gates(self, expert_counts, total_activations, gate_score):
        if self._mlflow_logger is None:
            return
        self._mlflow_logger.log_gate_metrics(
            layer_id=self._layer_id,
            global_step=self._global_step,
            expert_counts=expert_counts,
            total_activations=total_activations,
            gate_score_mean=gate_score.mean().item(),
            gate_score_std=gate_score.std().item()
        )
    

    def _log_to_mlflow_modality_gates(self, expert_counts, total_activations, gate_score, modality_name):
        if self._mlflow_logger is None:
            return
        
        self._mlflow_logger.log_modality_gate_metrics(
            layer_id=self._layer_id,
            global_step=self._global_step,
            expert_counts=expert_counts,
            total_activations=total_activations,
            gate_score_mean=gate_score.mean().item(),
            gate_score_std=gate_score.std().item(),
            modality_name=modality_name
        )
        
    
    def _save_distribution_to_json(self, expert_counts, modality_name="overall"):
        json_dir = "./gate_distributions"
        if self._layer_id is not None:
            json_dir = os.path.join(json_dir, f"layer_{self._layer_id}")
        os.makedirs(json_dir, exist_ok=True)
        filename = f"gate_distribution_{modality_name}_step_{self._global_step}.json"
        filepath = os.path.join(json_dir, filename)
        data = {
            "step": self._global_step,
            "layer_id": self._layer_id,
            "modality": modality_name,
            "expert_counts": expert_counts,
            "total_activations": sum(expert_counts.values())
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    

    def log_distribution_heatmap_to_mlflow(self, modality_name=None, alpha_value=None):
        if self._mlflow_logger is None or self._visualizer is None:
            return
        history = self._gate_distribution_history[modality_name]
        heatmap_bytes = self._visualizer.create_distribution_heatmap(
            history, modality_name, self._global_step, alpha_value
        )
        if heatmap_bytes is None:
            return
        
        self._mlflow_logger.log_distribution_heatmap(
            layer_id=self._layer_id,
            global_step=self._global_step,
            heatmap_bytes=heatmap_bytes,
            modality_name=modality_name
        )