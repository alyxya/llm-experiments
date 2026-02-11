"""
Training loop for rotational gradient descent.

No optimizer, no loss function. The output delta is:
    target (one-hot) - logits
masked at PAD positions. Parameters are updated directly:
    W += lr * delta_W
Cross-entropy is logged for comparison with standard training.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    lr: float = 1.0
    max_steps: int = 5000
    batch_size: int = 64
    eval_interval: int = 500
    log_interval: int = 100
    checkpoint_interval: int = 1000
    device: str = "auto"


def get_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
    print(f"Model parameters: {model.count_parameters():,}")

    run_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if experiment_config is not None:
        config_data = (
            asdict(experiment_config)
            if hasattr(experiment_config, "__dataclass_fields__")
            else experiment_config
        )
        (run_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    loss_log = []

    for step in range(train_cfg.max_steps):
        x, y = get_batch(train_cfg.batch_size, device)

        with torch.no_grad():
            logits = model.forward(x)  # (B, T, vocab)

            # Log cross-entropy for comparison
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=0,
            ).item()

            # Snapshot norms before backward clears state
            norms = model.norm_stats() if step % train_cfg.log_interval == 0 else None

            # Output delta: target - logits
            target = torch.zeros_like(logits)
            target.scatter_(-1, y.unsqueeze(-1), 1.0)

            delta_logits = target - logits

            # Mask PAD positions (token 0)
            pad_mask = y == 0  # (B, T)
            delta_logits[pad_mask] = 0.0

            # Backward + update
            model.backward(delta_logits)
            model.apply_deltas(train_cfg.lr)

        loss_log.append({"step": step, "loss": ce_loss, "lr": train_cfg.lr})

        if norms is not None:
            norm_str = " ".join(f"{k}={v:.3f}" for k, v in norms.items())
            print(f"step {step:5d} | ce_loss {ce_loss:.4f} | {norm_str}")

        if step > 0 and step % train_cfg.eval_interval == 0:
            metrics = eval_fn(model, device)
            print(f"step {step:5d} | eval: {metrics}")

        if step > 0 and step % train_cfg.checkpoint_interval == 0:
            torch.save(model.state_dict(), run_dir / f"ckpt_{step}.pt")

    # Final eval
    metrics = eval_fn(model, device)
    print(f"final eval: {metrics}")

    torch.save(model.state_dict(), run_dir / "ckpt_final.pt")
    (run_dir / "loss_log.json").write_text(json.dumps(loss_log))

    print(f"Run saved to {run_dir}")
    return run_dir
