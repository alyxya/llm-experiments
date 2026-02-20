import torch
from torch.optim import Optimizer


class RotationalOptimizer(Optimizer):
    """Rotational optimizer using per-row norm-preserving updates.

    For each 2D weight matrix, rows are the dot-product vectors
    (y = x @ W^T dots each row of W with x). The update per row:
      1. Scale gradient row to match the weight row's norm
      2. w += lr * (-g_scaled - w)
      3. Re-normalize to preserve the original row norm

    1D parameters (e.g. LayerNorm) are updated with plain SGD.
    """

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim < 2:
                    # Plain SGD for 1D params (LayerNorm)
                    p.add_(p.grad, alpha=-lr)
                    continue

                # Row norms (contraction dim is always dim=-1)
                w_norm = p.data.norm(dim=-1, keepdim=True)
                g_norm = p.grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)

                # Scale each gradient row to match its weight row's norm
                g_scaled = p.grad * (w_norm / g_norm)

                # w += lr * (-g_scaled - w)
                p.data.add_(-g_scaled - p.data, alpha=lr)

                # Re-normalize each row to preserve original norm
                new_norm = p.data.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                p.data.mul_(w_norm / new_norm)

        return loss
