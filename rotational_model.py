"""
Rotational Gradient Descent model.

Instead of standard backprop (dL/dx), propagates rotational deltas:
    delta(a) = upstream * (b - (b·a) * a)   for dot product c = a·b

This is the Riemannian gradient on the unit sphere. The non-standard
projection formula (without dividing by ||a||²) creates a restoring
force toward unit norm — self-normalizing dynamics.

Key differences from standard GPT:
- No LayerNorm
- ReLU instead of GELU
- No softmax in attention
- Unit-norm initialization
- Manual forward/backward (no autograd)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    context_len: int
    n_layers: int
    n_heads: int
    embed_dim: int


def rotational_project(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Project grad onto tangent plane of unit sphere at x (per-vector, last dim).

    delta = grad - (grad · x) * x

    Uses x directly (not x/||x||²) so that when ||x|| > 1, a radial
    restoring force pushes toward unit norm.
    """
    return grad - (grad * x).sum(dim=-1, keepdim=True) * x


def init_unit_norm(shape: tuple[int, ...]) -> torch.Tensor:
    """Gaussian init with variance 1/d (d = last dim = contraction dimension).

    Each row has E[||row||²] = 1. In high dimensions, actual norms
    concentrate tightly around 1 (std of norm ≈ 1/√(2d)).
    """
    d = shape[-1]
    return torch.randn(shape) / math.sqrt(d)


class RotationalLinear(nn.Module):
    """Linear layer (no bias) with rotational forward/backward."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W = nn.Parameter(init_unit_norm((d_out, d_in)))
        self._x = self._y = self._delta_W = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._x = x
        self._y = x @ self.W.T
        return self._y

    def backward(self, delta_y: torch.Tensor) -> torch.Tensor:
        x, y = self._x, self._y

        # Activation delta (projected per-position)
        grad_x = delta_y @ self.W
        delta_x = rotational_project(grad_x, x)

        # Weight delta (averaged over batch positions, projected per-row)
        dy_flat = delta_y.reshape(-1, delta_y.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        grad_W = dy_flat.T @ x_flat / dy_flat.shape[0]
        self._delta_W = rotational_project(grad_W, self.W)

        return delta_x

    def apply_deltas(self, lr: float):
        if self._delta_W is not None:
            self.W.data.add_(self._delta_W, alpha=lr)
        self._x = self._y = self._delta_W = None


class RotationalAttention(nn.Module):
    """Causal self-attention without softmax."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.embed_dim // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = RotationalLinear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.out = RotationalLinear(cfg.embed_dim, cfg.embed_dim)
        self._saved = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv.forward(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Raw scaled dot-product scores, no softmax
        scores = (q @ k.transpose(-1, -2)) * self.scale  # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, 0.0)

        attn_out = scores @ v  # (B, H, T, d_k)

        self._saved = (q, k, v, scores, attn_out, mask, B, T, C)

        return self.out.forward(
            attn_out.transpose(1, 2).reshape(B, T, C)
        )

    def backward(self, delta_out: torch.Tensor) -> torch.Tensor:
        q, k, v, scores, attn_out, mask, B, T, C = self._saved

        # Through output projection
        delta_attn_flat = self.out.backward(delta_out)  # (B, T, C)
        delta_attn = delta_attn_flat.reshape(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)  # (B, H, T, d_k)

        # Through scores @ v
        grad_scores = delta_attn @ v.transpose(-1, -2)
        delta_scores = rotational_project(grad_scores, scores)
        delta_scores = delta_scores.masked_fill(~mask, 0.0)

        grad_v = scores.transpose(-1, -2) @ delta_attn
        delta_v = rotational_project(grad_v, v)

        # Through q @ k^T * scale
        delta_qk = delta_scores * self.scale

        grad_q = delta_qk @ k
        delta_q = rotational_project(grad_q, q)

        grad_k = delta_qk.transpose(-1, -2) @ q
        delta_k = rotational_project(grad_k, k)

        # Reassemble into (B, T, 3*C)
        delta_q = delta_q.transpose(1, 2).reshape(B, T, C)
        delta_k = delta_k.transpose(1, 2).reshape(B, T, C)
        delta_v = delta_v.transpose(1, 2).reshape(B, T, C)
        delta_qkv = torch.cat([delta_q, delta_k, delta_v], dim=-1)

        delta_x = self.qkv.backward(delta_qkv)
        self._saved = None
        return delta_x

    def apply_deltas(self, lr: float):
        self.qkv.apply_deltas(lr)
        self.out.apply_deltas(lr)


class RotationalMLP(nn.Module):
    """MLP: up → down (no activation)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = RotationalLinear(cfg.embed_dim, 4 * cfg.embed_dim)
        self.down = RotationalLinear(4 * cfg.embed_dim, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down.forward(self.up.forward(x))

    def backward(self, delta: torch.Tensor) -> torch.Tensor:
        delta = self.down.backward(delta)
        return self.up.backward(delta)

    def apply_deltas(self, lr: float):
        self.up.apply_deltas(lr)
        self.down.apply_deltas(lr)


class RotationalBlock(nn.Module):
    """Transformer block: attention + MLP with residual connections, no norm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = RotationalAttention(cfg)
        self.mlp = RotationalMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(x)
        x = x + self.mlp.forward(x)
        return x

    def backward(self, delta: torch.Tensor) -> torch.Tensor:
        # MLP residual: delta flows through both skip and sublayer
        delta_mlp_in = self.mlp.backward(delta)
        delta = delta + delta_mlp_in

        # Attention residual: same
        delta_attn_in = self.attn.backward(delta)
        delta = delta + delta_attn_in

        return delta

    def apply_deltas(self, lr: float):
        self.attn.apply_deltas(lr)
        self.mlp.apply_deltas(lr)


class RotationalGPT(nn.Module):
    """GPT with rotational gradient descent training."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Unit-norm embeddings
        self.tok_emb = nn.Parameter(init_unit_norm((cfg.vocab_size, cfg.embed_dim)))
        self.pos_emb = nn.Parameter(init_unit_norm((cfg.context_len, cfg.embed_dim)))

        self.blocks = nn.ModuleList(
            [RotationalBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # No output head — weight-tied with tok_emb

        # Backward state
        self._idx = None
        self._final_x = None
        self._delta_tok_emb = None
        self._delta_pos_emb = None

    @torch.no_grad()
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb[idx] + self.pos_emb[pos]

        for block in self.blocks:
            x = block.forward(x)

        self._idx = idx
        self._final_x = x

        # Output logits via weight-tied embedding
        return x @ self.tok_emb.T  # (B, T, vocab_size)

    @torch.no_grad()
    def backward(self, delta_logits: torch.Tensor):
        x = self._final_x  # (B, T, embed_dim)
        B, T, _ = x.shape

        # Through output matmul: logits = x @ tok_emb^T
        # Treat as RotationalLinear with W = tok_emb

        # Delta for final activations
        grad_x = delta_logits @ self.tok_emb  # (B, T, embed_dim)
        delta_x = rotational_project(grad_x, x)

        # Delta for tok_emb from output head (averaged over positions)
        dl_flat = delta_logits.reshape(-1, delta_logits.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        N = dl_flat.shape[0]
        grad_emb_out = dl_flat.T @ x_flat / N  # (vocab, embed_dim)
        delta_emb_out = rotational_project(grad_emb_out, self.tok_emb)

        # Through blocks (reverse order)
        for block in reversed(self.blocks):
            delta_x = block.backward(delta_x)

        # Through embedding addition: x = tok_emb[idx] + pos_emb[pos]
        # Accumulate tok_emb delta from input side
        delta_emb_in = torch.zeros_like(self.tok_emb)
        delta_emb_in.index_add_(
            0, self._idx.reshape(-1),
            delta_x.reshape(-1, delta_x.shape[-1]),
        )
        delta_emb_in = delta_emb_in / N
        delta_emb_in = rotational_project(delta_emb_in, self.tok_emb)

        # Accumulate pos_emb delta (sum over batch, average)
        delta_pos = delta_x.sum(dim=0)  # (T, embed_dim)
        delta_pos = delta_pos / B
        # Pad if T < context_len
        if T < self.cfg.context_len:
            full = torch.zeros_like(self.pos_emb)
            full[:T] = delta_pos
            delta_pos = full
        delta_pos = rotational_project(delta_pos, self.pos_emb)

        self._delta_tok_emb = delta_emb_out + delta_emb_in
        self._delta_pos_emb = delta_pos

    @torch.no_grad()
    def apply_deltas(self, lr: float):
        self.tok_emb.data.add_(self._delta_tok_emb, alpha=lr)
        self.pos_emb.data.add_(self._delta_pos_emb, alpha=lr)
        for block in self.blocks:
            block.apply_deltas(lr)
        self._idx = self._final_x = None
        self._delta_tok_emb = self._delta_pos_emb = None

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, eos_token: int | None = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_len :]
            logits = self.forward(idx_cond)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token is not None and (next_id == eos_token).all():
                break
        return idx

    def norm_stats(self) -> dict[str, float]:
        """Mean row norms for all weight matrices and embeddings."""
        stats = {}
        stats["tok_emb"] = self.tok_emb.data.norm(dim=-1).mean().item()
        stats["pos_emb"] = self.pos_emb.data.norm(dim=-1).mean().item()
        if self._final_x is not None:
            stats["final_x"] = self._final_x.norm(dim=-1).mean().item()
        for i, block in enumerate(self.blocks):
            stats[f"b{i}.qkv"] = block.attn.qkv.W.data.norm(dim=-1).mean().item()
            stats[f"b{i}.out"] = block.attn.out.W.data.norm(dim=-1).mean().item()
            stats[f"b{i}.up"] = block.mlp.up.W.data.norm(dim=-1).mean().item()
            stats[f"b{i}.down"] = block.mlp.down.W.data.norm(dim=-1).mean().item()
        return stats

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
