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
    compute_domain_bias_from_sample_domains,
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
        # Use local experts only - bias size is num_experts (not tot_expert)
        self.register_buffer('modality_bias_text', torch.zeros(self.num_experts))
        self.register_buffer('modality_bias_image', torch.zeros(self.num_experts))
        init_bias_val = 0.0
        # Set bias for local experts: first half for text, second half for image
        self.modality_bias_text[:num_text_experts] = init_bias_val
        self.modality_bias_image[num_text_experts:] = init_bias_val
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
            # Use local experts only - bias size is num_experts
            bias_vec = torch.zeros(self.num_experts, dtype=torch.float32)
            for expert_id in expert_list:
                if 0 <= expert_id < self.num_experts:
                    bias_vec[expert_id] = 1.0
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
        self._gate_probability_history = defaultdict(dict)
        self._probability_sums = defaultdict(dict)
        self._probability_counts = defaultdict(dict)
        self._last_overall_logged_step = -1
        self._domain_last_logged_step = {}
        self._mlflow_logger = mlflow_logger
        self._visualizer : MoEVisualizer = visualizer

        self._log_gates = log_gates
        self._suppress_gate_logging = False
        self._log_activations = log_activations
        self._log_frequency = log_frequency
        self._log_overall_enabled = True
        self._log_domain_enabled = True

    def set_global_step(self, global_step):
        self._global_step = global_step

    def suppress_gate_logging(self, suppressed: bool):
        self._suppress_gate_logging = suppressed

    def set_logging_preferences(self, log_overall: Optional[bool] = None, log_domain: Optional[bool] = None):
        if log_overall is not None:
            self._log_overall_enabled = log_overall
        if log_domain is not None:
            self._log_domain_enabled = log_domain


    def forward(
        self,
        hidden_states,
        input_ids=None,
        temperature: Optional[float] = None,
        domain_id: Optional[str] = None,
        sample_domains=None,
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

        # Используем sample_domains для правильного применения domain bias к каждому токену
        domain_bias_tensor = compute_domain_bias_from_sample_domains(
            self,
            sample_domains,
            batch_size,
            seq_len,
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
        # gate_idx now contains local expert indices (0 to num_experts-1) directly
        overflowed_mask = (gate_idx[:, 0] == -1) & (gate_idx[:, 1] == -1)
        
        out_flat = torch.zeros(B, hidden_size, device=device, dtype=hidden_states.dtype)
        
        for k in range(self.top_k):
            expert_indices = gate_idx[:, k]  # Local indices (0 to num_experts-1)
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
        
        should_log = (
            self._log_gates 
            and not self._suppress_gate_logging
            and self._global_step % self._log_frequency == 0
        )
        if should_log:
            # gate_idx already contains local indices (0 to num_experts-1)
            self._log_gate_distribution(
                gate_idx, gate_score.detach(), input_ids, 
                batch_size=batch_size, seq_len=seq_len,
                domain_id=domain_id, sample_domains=sample_domains
            )

        return output


    def _log_gate_distribution(self, gate_idx, gate_score, input_ids=None, batch_size=None, seq_len=None, domain_id=None, sample_domains=None):
        modality = None
        if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
            modality = self._get_token_modality(input_ids.view(-1))  # [B*L]
        
        # Определяем домен каждого токена по его позиции в батче
        token_domains_flat = None
        if sample_domains is not None and batch_size is not None and seq_len is not None:
            # sample_domains[batch_idx] - домен сэмпла batch_idx
            # Для токена с индексом token_idx в flatten представлении:
            #   batch_idx = token_idx // seq_len
            #   domain = sample_domains[batch_idx]
            num_tokens = batch_size * seq_len
            token_domains_flat = [None] * num_tokens
            for token_idx in range(num_tokens):
                batch_idx = token_idx // seq_len
                if batch_idx < len(sample_domains):
                    token_domains_flat[token_idx] = sample_domains[batch_idx]
        
        expert_counts = {}
        for expert_id in range(self.num_experts):
            count = (gate_idx == expert_id).sum().item()
            expert_counts[expert_id] = count
        
        total_activations = sum(expert_counts.values())
        domain_key = str(domain_id)
        
        # if self._global_step <= 2:
        #     # print(f"[Layer {self._layer_id}] _log_gate_distribution: domain_id={domain_id}, sample_domains={sample_domains}")
        #     if token_domains_flat:
        #         unique_domains = set(d for d in token_domains_flat if d is not None)
        #         print(f"[Layer {self._layer_id}] Unique token domains: {unique_domains}")

        accumulate_overall = self._log_overall_enabled
        emit_overall = (
            accumulate_overall
            and self._last_overall_logged_step != self._global_step
        )
        log_domain = (
            self._log_domain_enabled
            and domain_key is not None
            and self._domain_last_logged_step.get(domain_key) != self._global_step
        )
        
        # print(f"[Layer {self._layer_id}] accumulate_overall={accumulate_overall}, emit_overall={emit_overall}, log_domain={log_domain}")

        # if not accumulate_overall and not log_domain:
        #     print(f"[Layer {self._layer_id}] Skipping logging (no accumulate/log flags)")
        #     return
        
        text_expert_counts = None
        image_expert_counts = None

        if accumulate_overall:
            self._accumulate_history(self._gate_distribution_history["overall"], expert_counts)
            overall_probs, overall_weight = self._compute_average_gate_probs(gate_idx, gate_score)
            self._store_probability(
                history_dict=self._gate_probability_history["overall"],
                history_key="overall",
                probs=overall_probs,
                weight=overall_weight,
                finalize=emit_overall,
            )
            if emit_overall:
                self._save_distribution_to_json(expert_counts, "overall")

            if modality is not None:
                text_mask = modality == 0
                image_mask = modality == 1
                video_mask = modality == 2

                modalities = [
                    ("text", text_mask),
                    ("image", image_mask),
                    ("video", video_mask),
                ]

                for modality_name, mask in modalities:
                    if not mask.any():
                        continue
                    modality_gate_idx = gate_idx[mask]
                    modality_gate_score = gate_score[mask]
                    modality_expert_counts = {}
                    for expert_id in range(self.num_experts):
                        count = (modality_gate_idx == expert_id).sum().item()
                        modality_expert_counts[expert_id] = count

                    modality_total = sum(modality_expert_counts.values())

                    self._accumulate_history(
                        self._gate_distribution_history[modality_name], modality_expert_counts
                    )
                    if emit_overall:
                        self._save_distribution_to_json(modality_expert_counts, modality_name)

                    if modality_name == "text":
                        text_expert_counts = modality_expert_counts
                    elif modality_name == "image":
                        image_expert_counts = modality_expert_counts

                    modality_probs, modality_weight = self._compute_average_gate_probs(
                        modality_gate_idx, modality_gate_score
                    )
                    self._store_probability(
                        history_dict=self._gate_probability_history[modality_name],
                        history_key=modality_name,
                        probs=modality_probs,
                        weight=modality_weight,
                        finalize=emit_overall,
                    )

                    if emit_overall:
                        self._log_to_mlflow_modality_gates(
                            modality_expert_counts, modality_total, modality_gate_score, modality_name
                        )

            if total_activations > 0 and emit_overall:
                self._log_to_mlflow_gates(expert_counts, total_activations, gate_score)

        # Process per-domain statistics from sample_domains
        # Простая логика: для каждого уникального домена в sample_domains собираем статистику
        # ВАЖНО: делаем это ДО вызова _log_all_plots_to_mlflow, чтобы домены были в history
        if token_domains_flat is not None and len(token_domains_flat) > 0:
            unique_domains = set(d for d in token_domains_flat if d is not None)
            # if self._global_step <= 2:
            #     print(f"[Layer {self._layer_id}] Found unique domains in sample_domains: {unique_domains}")
            
            for domain_name in unique_domains:
                # Создаем маску для токенов этого домена
                domain_mask = torch.tensor(
                    [d == domain_name for d in token_domains_flat],
                    dtype=torch.bool,
                    device=gate_idx.device
                )
                
                if not domain_mask.any():
                    continue
                
                # Фильтруем gate_idx и gate_score только для токенов этого домена
                domain_gate_idx = gate_idx[domain_mask]
                domain_gate_score = gate_score[domain_mask]
                
                # Подсчитываем активации экспертов для этого домена
                domain_expert_counts = {}
                for expert_id in range(self.num_experts):
                    count = (domain_gate_idx == expert_id).sum().item()
                    domain_expert_counts[expert_id] = count
                
                domain_key_str = str(domain_name)
                should_log_this_domain = self._domain_last_logged_step.get(domain_key_str) != self._global_step
                
                if should_log_this_domain:
                    if self._global_step <= 2:
                        num_tokens_in_domain = domain_mask.sum().item()
                        # print(f"[Layer {self._layer_id}] Logging domain: {domain_key_str}, tokens={num_tokens_in_domain}, counts={domain_expert_counts}")
                        # Проверяем, что все токены действительно из этого домена
                        domain_tokens_check = [token_domains_flat[i] for i in range(len(token_domains_flat)) if domain_mask[i]]
                        unique_in_domain = set(domain_tokens_check)
                        # if len(unique_in_domain) > 1:
                        #     print(f"[Layer {self._layer_id}] WARNING: Domain {domain_key_str} has mixed domains: {unique_in_domain}")
                    self._ensure_domain_buffer(domain_key_str)
                    self._accumulate_history(
                        self._gate_distribution_history[domain_key_str], domain_expert_counts
                    )
                    domain_probs, domain_weight = self._compute_average_gate_probs(domain_gate_idx, domain_gate_score)
                    self._store_probability(
                        history_dict=self._gate_probability_history[domain_key_str],
                        history_key=f"domain_{domain_key_str}",
                        probs=domain_probs,
                        weight=domain_weight,
                        finalize=True,
                    )
                    self._save_distribution_to_json(domain_expert_counts, domain_key_str)
                    self._log_to_mlflow_gates(domain_expert_counts, sum(domain_expert_counts.values()), domain_gate_score)
                    self._domain_last_logged_step[domain_key_str] = self._global_step

        # Теперь вызываем _log_all_plots_to_mlflow ПОСЛЕ обработки всех доменов
        if accumulate_overall and emit_overall:
            self._log_all_plots_to_mlflow(
                overall_expert_counts=expert_counts,
                overall_gate_score=gate_score,
                text_expert_counts=text_expert_counts,
                image_expert_counts=image_expert_counts,
                domain_id=None,  # Не используем domain_id для legacy логирования
            )
            self._last_overall_logged_step = self._global_step
    
    
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
        
        # print(f"[Layer {self._layer_id}] Creating all_domains plot, history keys: {list(self._gate_distribution_history.keys())}")
        all_domains_plot_bytes = self._visualizer.create_all_domains_combined_plot(
            self._gate_distribution_history, self._global_step
        )
        modality_prob_plot_bytes = self._visualizer.create_modality_probability_plot(
            self._gate_probability_history, self._global_step
        )
        
        self._mlflow_logger.log_all_plots(
            layer_id=self._layer_id,
            global_step=self._global_step,
            overall_heatmap_bytes=overall_heatmap_bytes,
            overall_histogram_bytes=overall_histogram_bytes,
            combined_plot_bytes=combined_plot_bytes,
            domain_plot_bytes=domain_plot_bytes,
            all_domains_plot_bytes=all_domains_plot_bytes,
            modality_probability_plot_bytes=modality_prob_plot_bytes,
            domain_id=domain_id
        )
    
    def _accumulate_history(self, history_dict, counts):
        existing = history_dict.get(self._global_step)
        if existing is None:
            history_dict[self._global_step] = counts.copy()
        else:
            for expert_id, count in counts.items():
                existing[expert_id] = existing.get(expert_id, 0) + count

    def _compute_average_gate_probs(self, gate_idx, gate_score, token_mask=None):
        num_tokens = gate_idx.shape[0]
        if num_tokens == 0:
            return None, 0.0

        if token_mask is None:
            mask = torch.ones(num_tokens, dtype=torch.bool, device=gate_idx.device)
        else:
            mask = token_mask.to(device=gate_idx.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.shape[0] != num_tokens:
                mask = mask.view(num_tokens)

        expert_valid = (gate_idx >= 0) & (gate_idx < self.num_experts)
        expanded_mask = mask.unsqueeze(1).expand_as(expert_valid)
        valid = expanded_mask & expert_valid

        if not valid.any():
            return None, 0.0

        cleaned_scores = torch.where(valid, gate_score, torch.zeros_like(gate_score))
        token_sums = cleaned_scores.sum(dim=1, keepdim=True)
        token_has_valid = (token_sums.squeeze(-1) > 0)
        if not token_has_valid.any():
            return None, 0.0

        token_sums = torch.where(token_sums > 0, token_sums, torch.ones_like(token_sums))
        normalized_scores = cleaned_scores / token_sums

        probs = torch.zeros(self.num_experts, device=gate_idx.device, dtype=gate_score.dtype)
        for k in range(self.top_k):
            expert_ids = gate_idx[:, k].long()  # Ensure long dtype for indexing
            weights = normalized_scores[:, k].to(dtype=probs.dtype)  # Ensure same dtype as probs
            valid_k = token_has_valid & expert_valid[:, k]
            if valid_k.any():
                probs.scatter_add_(0, expert_ids[valid_k], weights[valid_k])

        denom = token_has_valid.sum()
        if denom.item() == 0:
            return None, 0.0
        probs = probs / denom
        return probs, float(denom.item())

    def _store_probability(self, history_dict, history_key, probs, weight, finalize):
        if probs is None or weight <= 0.0:
            return
        probs_cpu = probs.detach().to("cpu")

        step = self._global_step
        sums = self._probability_sums[history_key]
        counts = self._probability_counts[history_key]
        if step in sums:
            sums[step] = sums[step] + probs_cpu * weight
        else:
            sums[step] = probs_cpu * weight
        counts[step] = counts.get(step, 0.0) + weight

        if not finalize:
            return

        total_weight = counts.pop(step, 0.0)
        sum_vec = sums.pop(step, None)
        if sum_vec is None or total_weight <= 0.0:
            return

        final_vec = sum_vec / total_weight
        norm = final_vec.sum().item()
        if norm <= 0:
            return
        final_vec = final_vec / norm
        history_dict[step] = final_vec

    def _update_prob_ema(self, key, new_probs):
        if new_probs is None:
            return
        new_probs_cpu = new_probs.detach().to("cpu")
        existing = self._gate_probability_ema.get(key)
        if existing is None:
            self._gate_probability_ema[key] = new_probs_cpu
        else:
            self._gate_probability_ema[key] = (
                self.prob_ema_decay * existing + (1.0 - self.prob_ema_decay) * new_probs_cpu
            )


    def set_global_step(self, global_step):
        self._global_step = global_step
        
    def get_balance_loss(self, clear=True):
        gate_loss =  self.gate.get_loss(clear=clear)
        return gate_loss
    
    def get_domain_bias_hardness(self):
        if not self.use_domain_bias:
            return 0.0
        
        if self._global_step < self.domain_init_steps:
            progress = self._global_step / max(self.domain_init_steps, 1)
            hardness = self.domain_init_hardness - (
                self.domain_init_hardness - self.domain_init_hardness_min
            ) * progress
        else:
            hardness = self.domain_init_hardness_min
        
        return float(hardness)
        

    def _ensure_domain_buffer(self, domain_id: str):
        domain_key = str(domain_id)
        buffer_name = self._domain_bias_buffer_map.get(domain_key)
        if buffer_name is None:
            buffer_name = f"_domain_bias_vec_extra_{len(self._domain_bias_buffer_map)}"
            # Use local experts only - bias size is num_experts
            bias_vec = torch.zeros(
                self.num_experts,
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