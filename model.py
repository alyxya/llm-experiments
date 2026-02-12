import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context_len: int
    n_layers: int
    n_heads: int
    embed_dim: int
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.embed_dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim, bias=False)
        self.out = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, T, C)
        return self.out(x)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim, bias=False)
        self.down = nn.Linear(4 * cfg.embed_dim, cfg.embed_dim, bias=False)
        # Dimension-change normalization for variance-preserving maps:
        # y = (x @ W^T) * sqrt(d_in / d_out)
        self._up_out_scale = math.sqrt(self.up.in_features / self.up.out_features)
        self._down_out_scale = math.sqrt(self.down.in_features / self.down.out_features)
        # With unit-norm vectors in high dimensions, GELU is near-linear around 0:
        # gelu(x) ~= 0.5x. Compensate to keep MLP branch scale near 1.
        self._gelu_out_scale = 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x) * self._up_out_scale
        h = F.gelu(h) * self._gelu_out_scale
        return self.down(h) * self._down_out_scale


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self._ln_out_scale = 1.0 / math.sqrt(cfg.embed_dim)
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x) * self._ln_out_scale)
        x = x + self.mlp(self.ln2(x) * self._ln_out_scale)
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.embed_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self._ln_out_scale = 1.0 / math.sqrt(cfg.embed_dim)
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            # Contraction dim for y = x @ W^T is W.shape[-1].
            d_contract = module.weight.shape[-1]
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(d_contract))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding rows are vectors in R^embed_dim.
            d_vector = module.weight.shape[-1]
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(d_vector))
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x) * self._ln_out_scale
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, eos_token: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_len:]
            logits = self(idx_cond)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token is not None and (next_id == eos_token).all():
                break
        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
