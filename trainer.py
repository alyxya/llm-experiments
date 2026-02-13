import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    lr: float = 3e-4
    min_lr: float = 1e-5
    max_steps: int = 5000
    batch_size: int = 64
    eval_interval: int = 500
    log_interval: int = 100
    checkpoint_interval: int = 1000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    device: str = "auto"
    rotational_dot_products: bool = True
    optimizer: str = "adamw"  # "adamw" or "sgd"
    loss: str = "cross_entropy"  # "cross_entropy" or "sse_onehot_logits"


def get_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def _collect_dot_product_target_radii(model: nn.Module) -> dict[int, float]:
    """Initialization-based target radii for dot-product parameters.

    For Gaussian init with per-element std sigma and contraction dim d, the expected
    row norm is about sigma * sqrt(d). We estimate sigma from initialized weights.
    """
    target_radii: dict[int, float] = {}
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weight = getattr(module, "weight", None)
            if weight is None or not weight.requires_grad:
                continue
            pid = id(weight)
            if pid in target_radii:
                continue
            d = weight.shape[-1]
            sigma = weight.data.float().std(unbiased=False).item()
            if math.isfinite(sigma) and sigma > 0.0:
                target_radii[pid] = sigma * math.sqrt(d)
            else:
                # Fallback if std is degenerate for any reason.
                if weight.ndim > 1:
                    target_radii[pid] = max(
                        weight.data.float().norm(dim=-1).mean().item(), 1e-12
                    )
                else:
                    target_radii[pid] = max(weight.data.float().norm().item(), 1e-12)
    return target_radii


def _project_to_target_radius(
    grad: torch.Tensor,
    x: torch.Tensor,
    target_radius: float,
) -> torch.Tensor:
    """Project gradient to tangent space of sphere with radius target_radius.

    Equivalent to scaling by target_radius, projecting on the unit sphere, and
    scaling back. This matches initialization-dependent expected norms.
    """
    r2 = max(target_radius * target_radius, 1e-12)
    return grad - (grad * x).sum(dim=-1, keepdim=True) * (x / r2)


@torch.no_grad()
def _apply_rotational_dot_product_projection_(
    model: nn.Module,
    dot_target_radii: dict[int, float],
):
    for p in model.parameters():
        if p.grad is None:
            continue
        target_radius = dot_target_radii.get(id(p))
        if target_radius is not None:
            p.grad.copy_(_project_to_target_radius(p.grad, p.data, target_radius))


def _compute_train_loss(logits: torch.Tensor, targets: torch.Tensor, train_cfg: TrainConfig) -> torch.Tensor:
    if train_cfg.loss == "cross_entropy":
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0,
        )
    if train_cfg.loss == "sse_onehot_logits":
        one_hot = F.one_hot(targets, num_classes=logits.size(-1)).to(logits.dtype)
        per_token_sse = (logits - one_hot).pow(2).sum(dim=-1)
        mask = (targets != 0).to(logits.dtype)
        return (per_token_sse * mask).sum()
    raise ValueError(f"Unsupported loss: {train_cfg.loss}")


def train(
    model: nn.Module,
    get_batch: Callable[[int, str], tuple[torch.Tensor, torch.Tensor]],
    eval_fn: Callable[[nn.Module, str], dict],
    train_cfg: TrainConfig,
    experiment_config: Any = None,
) -> Path:
    device = get_device(train_cfg.device)
    print(f"Using device: {device}")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    run_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if experiment_config is not None:
        config_data = asdict(experiment_config) if hasattr(experiment_config, "__dataclass_fields__") else experiment_config
        (run_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    if train_cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    elif train_cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {train_cfg.optimizer}")
    dot_target_radii = (
        _collect_dot_product_target_radii(model)
        if train_cfg.rotational_dot_products
        else {}
    )
    loss_log = []

    for step in range(train_cfg.max_steps):
        model.train()
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_cfg.batch_size, device)
        logits = model(x)
        loss = _compute_train_loss(logits, y, train_cfg)

        optimizer.zero_grad()
        loss.backward()
        if train_cfg.rotational_dot_products:
            _apply_rotational_dot_product_projection_(model, dot_target_radii)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()

        loss_val = loss.item()
        loss_log.append({"step": step, "loss": loss_val, "lr": lr})

        if step % train_cfg.log_interval == 0:
            print(f"step {step:5d} | loss {loss_val:.4f} | lr {lr:.2e}")

        if step > 0 and step % train_cfg.eval_interval == 0:
            model.eval()
            metrics = eval_fn(model, device)
            print(f"step {step:5d} | eval: {metrics}")

        if step > 0 and step % train_cfg.checkpoint_interval == 0:
            torch.save(model.state_dict(), run_dir / f"ckpt_{step}.pt")

    # final eval
    model.eval()
    metrics = eval_fn(model, device)
    print(f"final eval: {metrics}")

    # save final checkpoint and loss log
    torch.save(model.state_dict(), run_dir / "ckpt_final.pt")
    (run_dir / "loss_log.json").write_text(json.dumps(loss_log))

    print(f"Run saved to {run_dir}")
    return run_dir
