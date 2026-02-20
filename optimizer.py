import torch
from torch.optim import Optimizer


class RotationalOptimizer(Optimizer):
    """Rotational optimizer using Frobenius-norm-preserving updates.

    For each parameter:
      1. Scale gradient to match the Frobenius norm of the weights
      2. Update: W_new = W - lr * (g_scaled - W)
      3. Re-normalize W_new to preserve the original Frobenius norm

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

                w_norm = p.data.norm()
                g_norm = p.grad.norm().clamp(min=1e-12)

                # Scale gradient to match weight's Frobenius norm
                g_scaled = p.grad * (w_norm / g_norm)

                # Update: move away from gradient direction
                p.data.sub_(g_scaled - p.data, alpha=lr)

                # Re-normalize to preserve original Frobenius norm
                new_norm = p.data.norm().clamp(min=1e-12)
                p.data.mul_(w_norm / new_norm)

        return loss
