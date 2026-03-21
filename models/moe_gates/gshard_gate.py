import math
import torch
import torch.nn.functional as F
from .naive_gate import NaiveGate
from .utils import limit_by_capacity
from typing import Optional, Tuple


def prune_gate_by_capacity_vectorized(
    topk_idx: torch.Tensor,
    topk_score: torch.Tensor, 
    capacity_per_expert: int,
    num_expert: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Векторизованная версия capacity pruning.
    Возвращает pruned индексы И обновлённые веса (ренормализованные).
    """
    device = topk_idx.device
    S, top_k = topk_idx.shape
    
    # Копируем для модификации
    pruned_idx = topk_idx.clone()
    pruned_score = topk_score.clone()
    
    # Считаем сколько раз каждый эксперт выбран (по top-1)
    expert_counts = torch.zeros(num_expert, dtype=torch.long, device=device)
    
    # Простой подход: обрабатываем top-1, затем top-2
    for k in range(top_k):
        expert_indices = pruned_idx[:, k]
        
        for expert_id in range(num_expert):
            # Находим токены, выбравшие этого эксперта
            mask = (expert_indices == expert_id)
            if not mask.any():
                continue
            
            # Сколько уже назначено + сколько хотят
            current_count = expert_counts[expert_id].item()
            want_count = mask.sum().item()
            
            if current_count + want_count <= capacity_per_expert:
                # Все влезают
                expert_counts[expert_id] += want_count
            else:
                # Нужно отсечь лишних
                available = max(0, capacity_per_expert - current_count)
                if available > 0:
                    # Берём первые `available` токенов (можно сделать случайный выбор)
                    positions = torch.where(mask)[0]
                    keep_positions = positions[:available]
                    drop_positions = positions[available:]
                    
                    expert_counts[expert_id] += available
                    
                    # Отсекаем лишних
                    pruned_idx[drop_positions, k] = -1
                    pruned_score[drop_positions, k] = 0.0
                else:
                    # Отсекаем всех
                    pruned_idx[mask, k] = -1
                    pruned_score[mask, k] = 0.0
    
    # Ренормализуем веса для токенов с частично pruned экспертами
    valid_mask = pruned_idx >= 0
    weight_sum = pruned_score.sum(dim=1, keepdim=True)
    weight_sum = weight_sum.clamp(min=1e-8)
    pruned_score = pruned_score / weight_sum
    
    # Для полностью pruned токенов (оба эксперта = -1), веса будут 0
    # Это ОК, т.к. они используют residual connection
    
    return pruned_idx, pruned_score


class GShardGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size,
            top_k=2, capacity=(1.2, 2.4), random_routing=True, gate_bias=True, use_gumbel=False):
        assert top_k == 2, 'topk should be 2 in gshard'
        super().__init__(d_model, num_expert, world_size, top_k=2, gate_bias=gate_bias)
        self.capacity = capacity
        self.random_routing = random_routing
        self.use_gumbel = use_gumbel

    def forward(
        self, 
        x, 
        temperature: float | None = None,
        return_all_scores: bool = False,
        bias: Optional[torch.Tensor] = None,
    ):
        total_bias = bias.to(device=x.device) if bias is not None else None
        naive_outs = super().forward(x, return_all_scores=True, bias=total_bias)
        topk_idx, gate_score_topk, gate_logits = naive_outs
        
        # Apply Gumbel-Softmax noise for differentiable expert selection (training only)
        if self.use_gumbel and self.training:
            # Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(gate_logits) + 1e-10) + 1e-10)
            gate_logits_noisy = gate_logits + gumbel_noise
            gumbel_temp = temperature if temperature is not None and temperature > 0 else 1.0
            gate_logits_scaled = gate_logits_noisy / gumbel_temp
        elif temperature is not None and temperature > 0:
            gate_logits_scaled = gate_logits / temperature
        else:
            gate_logits_scaled = gate_logits
        
        # Get top-k experts based on (possibly noisy) logits
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate_logits_scaled, k=self.top_k, dim=-1, largest=True, sorted=False
        )
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        gate_score = F.softmax(gate_top_k_val, dim=-1)
        topk_idx = gate_top_k_idx.view(-1, self.top_k)
        
        # Use original logits for loss computation (not noisy)
        gate_logits_for_loss = gate_logits

        S = topk_idx.shape[0]
        
        # Balance loss (уменьшен множитель!)
        top1_idx = topk_idx[:, 0]
        c_e = torch.zeros(self.num_expert, device=top1_idx.device)
        for e in range(self.num_expert):
            c_e[e] = (top1_idx == e).float().sum() / S
        
        m_e = torch.mean(F.softmax(gate_logits_for_loss, dim=-1), dim=0)
        
        # Убрали множитель num_expert^2 - он слишком большой!
        gshard_loss = torch.sum(c_e * m_e) * self.num_expert
        target_load = 1.0 / self.num_expert
        variance_penalty = torch.sum((c_e - target_load) ** 2) * self.num_expert
        
        loss = gshard_loss + 0.1 * variance_penalty  # variance penalty меньше
        self.set_loss(loss)

        # Capacity pruning
        cap_rate = self.capacity[0 if self.training else 1]
        capacity_total = math.ceil(cap_rate * S)
        capacity_per_expert = max(1, capacity_total * self.top_k // self.num_expert)
        
        # Используем улучшенную функцию pruning
        topk_idx, gate_score = prune_gate_by_capacity_vectorized(
            topk_idx, gate_score, capacity_per_expert, self.num_expert
        )

        # Random routing (только в training)
        if self.random_routing and self.training:
            rand_routing_prob = torch.rand(S, device=x.device)
            # Отсекаем второй эксперт если его вес < случайного порога
            mask = (gate_score[:, 1] < rand_routing_prob * 0.5)  # Менее агрессивно
            topk_idx[:, 1] = torch.where(mask, torch.tensor(-1, device=x.device), topk_idx[:, 1])
            # Обнуляем вес и ренормализуем
            gate_score[:, 1] = torch.where(mask, torch.zeros_like(gate_score[:, 1]), gate_score[:, 1])
            weight_sum = gate_score.sum(dim=1, keepdim=True).clamp(min=1e-8)
            gate_score = gate_score / weight_sum

        return topk_idx, gate_score