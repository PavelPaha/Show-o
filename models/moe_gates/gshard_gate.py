import math
import torch
import torch.nn.functional as F
from .naive_gate import NaiveGate
from .utils import limit_by_capacity
from typing import Optional

import torch

def prune_gate_by_capacity_local(topk_idx: torch.Tensor,
                                  capacity: torch.Tensor) -> torch.Tensor:
    if topk_idx.dim() != 2:
        raise ValueError("topk_idx must be a 2D tensor of shape (S, top_k)")

    device = topk_idx.device
    num_expert = capacity.numel()
    pruned = topk_idx.clone()
    flat = pruned.view(-1)
    valid_mask = flat >= 0
    if valid_mask.sum() == 0:
        return pruned

    counts = torch.zeros(num_expert, dtype=torch.int32, device=device)
    cap = capacity.to(device=device, dtype=torch.int32)

    flat_np = flat
    idx_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    for pos in idx_positions:
        e = int(flat_np[pos].item())  # local expert id (0 to num_expert-1)
        if e < 0 or e >= num_expert:
            flat_np[pos] = -1
            continue
        if counts[e] < cap[e]:
            counts[e] += 1
        else:
            flat_np[pos] = -1

    pruned = flat_np.view_as(pruned)
    return pruned


def prune_gate_by_capacity(topk_idx: torch.Tensor,
                           capacity: torch.Tensor,
                           num_expert: int,
                           world_size: int) -> torch.Tensor:
    if topk_idx.dim() != 2:
        raise ValueError("topk_idx must be a 2D tensor of shape (S, top_k)")

    device = topk_idx.device
    top_k = topk_idx.size(1)
    tot_expert = capacity.numel()
    pruned = topk_idx.clone()
    flat = pruned.view(-1)
    valid_mask = flat >= 0
    if valid_mask.sum() == 0:
        return pruned

    counts = torch.zeros(tot_expert, dtype=torch.int32, device=device)
    cap = capacity.to(device=device, dtype=torch.int32)

    flat_np = flat
    idx_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    for pos in idx_positions:
        e = int(flat_np[pos].item())  # global expert id
        if e < 0 or e >= tot_expert:
            flat_np[pos] = -1
            continue
        if counts[e] < cap[e]:
            counts[e] += 1
        else:
            flat_np[pos] = -1

    pruned = flat_np.view_as(pruned)
    return pruned


class GShardGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size,
            top_k=2, capacity=(1.2, 2.4), random_routing=True, gate_bias=True, use_gumbel=False):
        assert top_k == 2, 'topk should be 2 in gshard'
        super().__init__(d_model, num_expert, world_size, top_k=2, gate_bias=gate_bias)
        self.capacity = capacity
        self.random_routing = random_routing

    def forward(
        self, 
        x, 
        temperature: float | None = None,
        return_all_scores: bool = False,
        bias: Optional[torch.Tensor] = None,
        ):

        total_bias = bias.to(device=x.device) if bias is not None else None
        naive_outs = super().forward(x, return_all_scores=True, bias=total_bias)
        topk_idx, gate_score_topk, gate_logits = naive_outs  # gate_logits are full logits, gate_score_topk is softmax for top-k
        
        # Apply temperature scaling to logits before recomputing top-k and softmax
        gate_logits_for_loss = gate_logits
        if temperature is not None and temperature > 0:
            gate_logits_scaled = gate_logits / temperature
            # Recompute top-k with temperature-scaled logits
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_logits_scaled, k=self.top_k, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
            gate_score = F.softmax(gate_top_k_val, dim=-1)
            topk_idx = gate_top_k_idx
            topk_val = gate_top_k_val
            gate_logits_for_loss = gate_logits_scaled
        else:
            # Use original values from NaiveGate
            # Extract top-k values from full logits using topk_idx
            topk_val = gate_logits.gather(1, topk_idx)
            gate_score = gate_score_topk  # Use softmax scores from NaiveGate

        S = topk_idx.shape[0]
        top1_idx = topk_idx.view((-1, self.top_k))[:, 0]
        # Use local experts only - indices are 0 to num_expert-1
        c_e = torch.scatter_add(
            torch.zeros(self.num_expert, device=top1_idx.device),
            0,
            top1_idx,
            torch.ones_like(top1_idx, dtype=torch.float),
        ) / S
        
        # For m_e, use full logits with temperature if applied
        m_e = torch.mean(F.softmax(gate_logits_for_loss, dim=-1), dim=0)
        gshard_loss = torch.mean(c_e * m_e) * (self.num_expert ** 2)
        target_load = 1.0 / self.num_expert
        variance_penalty = torch.mean((c_e - target_load) ** 2) * (self.num_expert ** 2)
        loss = gshard_loss + variance_penalty
        self.set_loss(loss)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        capacity = capacity * self.top_k // self.num_expert
        capacity = torch.ones(self.num_expert, dtype=torch.int32, device=topk_idx.device) * capacity
        topk_idx = prune_gate_by_capacity_local(topk_idx, capacity)

        if self.random_routing:
            rand_routing_prob = torch.rand(gate_score.size(0), device=x.device)
            mask = (2 * topk_val[:, 1] < rand_routing_prob)
            topk_idx[:, 1].masked_fill_(mask, -1)

        return topk_idx, topk_val