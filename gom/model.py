from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .graph import BASE_FEATURE_DIM, NUM_NODE_TYPES, NUM_RELATIONS, GraphBatch


@dataclass
class GOMConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.0
    n_solver_classes: int = 4
    n_action_classes: int = 5
    n_problem_types: int = 16
    value_dims: int = 3


class RelationAwareAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_relations: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rel_bias = nn.Embedding(n_relations, n_heads)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, relation: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.rel_bias(relation).permute(0, 3, 1, 2)
        scores = scores + bias
        scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        if self.dropout and self.training:
            attn = F.dropout(attn, p=self.dropout)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        y = self.out(y)
        return y.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class GOMBlock(nn.Module):
    def __init__(self, cfg: GOMConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = RelationAwareAttention(cfg.d_model, cfg.n_heads, NUM_RELATIONS, cfg.dropout)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, relation: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.norm1(x), relation, padding_mask)
        x = x + (F.dropout(h, p=self.dropout, training=self.training) if self.dropout else h)
        h = self.ff(self.norm2(x))
        x = x + (F.dropout(h, p=self.dropout, training=self.training) if self.dropout else h)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class GOMModel(nn.Module):
    def __init__(self, cfg: GOMConfig = GOMConfig()):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(BASE_FEATURE_DIM, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.node_type_embedding = nn.Embedding(NUM_NODE_TYPES, cfg.d_model)
        self.blocks = nn.ModuleList([GOMBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.solver_head = nn.Linear(cfg.d_model, cfg.n_solver_classes)
        self.action_head = nn.Linear(cfg.d_model, cfg.n_action_classes)
        self.value_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2), nn.SiLU(), nn.Linear(cfg.d_model // 2, cfg.value_dims))
        self.variable_score = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2), nn.SiLU(), nn.Linear(cfg.d_model // 2, 1))

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        x = self.input_proj(batch.x) + self.node_type_embedding(batch.node_type)
        x = x.masked_fill(batch.padding_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, batch.relation, batch.padding_mask)
        x = self.final_norm(x)
        global_state = x[:, 0]
        variable_logits = self.variable_score(x).squeeze(-1)
        variable_logits = variable_logits.masked_fill(~batch.variable_mask, -1e9)
        return {
            "solver_logits": self.solver_head(global_state),
            "action_logits": self.action_head(global_state),
            "variable_logits": variable_logits,
            "value": self.value_head(global_state),
            "node_embeddings": x,
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
