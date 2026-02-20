import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizer import RotationalOptimizer


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
    optimizer: str = "adamw"  # "adamw" or "rotational"


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
    elif train_cfg.optimizer == "rotational":
        optimizer = RotationalOptimizer(model.parameters(), lr=train_cfg.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {train_cfg.optimizer}")

    loss_log = []

    for step in range(train_cfg.max_steps):
        model.train()
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_cfg.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=0,
        )

        optimizer.zero_grad()
        loss.backward()
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
