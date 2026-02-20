import math

import torch
from torch.optim import Optimizer


class RotationalOptimizer(Optimizer):
    """Rotational optimizer with multiple norm-preservation modes.

    Modes:
      - "row": Per-row norm preservation (default). Each row of W stays on
        its initialization-radius sphere.
      - "col": Per-column norm preservation. Each column of W stays on
        its initialization-radius sphere.
      - "frobenius": Whole-matrix Frobenius norm preservation.
      - "rowcol": Row+column compromise using sqrt(row_scale) * sqrt(col_scale)
        with fixed target norms derived from init_std.
      - "rowcol_drift": Same compromise but targets drift with the weights
        (tracks current norms rather than fixed targets).

    For all modes, 1D parameters are updated with plain SGD.
    """

    def __init__(self, params, lr=1e-3, mode="row", init_std=0.02):
        if mode not in ("row", "col", "frobenius", "rowcol", "rowcol_drift"):
            raise ValueError(f"Unsupported mode: {mode}")
        defaults = dict(lr=lr, mode=mode, init_std=init_std)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mode = group["mode"]
            init_std = group["init_std"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.ndim < 2:
                    p.add_(p.grad, alpha=-lr)
                    continue

                if mode == "row":
                    self._step_row(p, lr)
                elif mode == "col":
                    self._step_col(p, lr)
                elif mode == "frobenius":
                    self._step_frobenius(p, lr)
                elif mode == "rowcol":
                    self._step_rowcol_fixed(p, lr, init_std)
                elif mode == "rowcol_drift":
                    self._step_rowcol_drift(p, lr)

        return loss

    @staticmethod
    def _step_row(p, lr):
        w_norm = p.data.norm(dim=-1, keepdim=True)
        g_norm = p.grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        g_scaled = p.grad * (w_norm / g_norm)
        p.data.add_(-g_scaled - p.data, alpha=lr)
        new_norm = p.data.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        p.data.mul_(w_norm / new_norm)

    @staticmethod
    def _step_col(p, lr):
        w_norm = p.data.norm(dim=0, keepdim=True)
        g_norm = p.grad.norm(dim=0, keepdim=True).clamp(min=1e-12)
        g_scaled = p.grad * (w_norm / g_norm)
        p.data.add_(-g_scaled - p.data, alpha=lr)
        new_norm = p.data.norm(dim=0, keepdim=True).clamp(min=1e-12)
        p.data.mul_(w_norm / new_norm)

    @staticmethod
    def _step_frobenius(p, lr):
        w_norm = p.data.norm()
        g_norm = p.grad.norm().clamp(min=1e-12)
        g_scaled = p.grad * (w_norm / g_norm)
        p.data.add_(-g_scaled - p.data, alpha=lr)
        new_norm = p.data.norm().clamp(min=1e-12)
        p.data.mul_(w_norm / new_norm)

    @staticmethod
    def _step_rowcol_fixed(p, lr, init_std):
        m, n = p.shape
        target_row = init_std * math.sqrt(n)
        target_col = init_std * math.sqrt(m)

        g_row_norm = p.grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        g_col_norm = p.grad.norm(dim=0, keepdim=True).clamp(min=1e-12)
        g_scaled = p.grad * ((target_row / g_row_norm).sqrt() * (target_col / g_col_norm).sqrt())

        p.data.add_(-g_scaled - p.data, alpha=lr)

        row_norm_new = p.data.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        col_norm_new = p.data.norm(dim=0, keepdim=True).clamp(min=1e-12)
        p.data.mul_((target_row / row_norm_new).sqrt() * (target_col / col_norm_new).sqrt())

    @staticmethod
    def _step_rowcol_drift(p, lr):
        row_norm_orig = p.data.norm(dim=-1, keepdim=True)
        col_norm_orig = p.data.norm(dim=0, keepdim=True)

        g_row_norm = p.grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        g_col_norm = p.grad.norm(dim=0, keepdim=True).clamp(min=1e-12)
        g_scaled = p.grad * ((row_norm_orig / g_row_norm).sqrt() * (col_norm_orig / g_col_norm).sqrt())

        p.data.add_(-g_scaled - p.data, alpha=lr)

        row_norm_new = p.data.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        col_norm_new = p.data.norm(dim=0, keepdim=True).clamp(min=1e-12)
        p.data.mul_((row_norm_orig / row_norm_new).sqrt() * (col_norm_orig / col_norm_new).sqrt())
