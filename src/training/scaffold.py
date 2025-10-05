from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    max_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    mixed_precision: bool = True
    early_stop_patience: int = 5


def default_bce_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float(), pos_weight=pos_weight)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0.0
    for batch in dataloader:
        images, labels = batch  # images: [B,3,H,W], labels: [B]
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast(True):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / max(1, len(dataloader.dataset))


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    metric_fn: Callable[[Iterable[float], Iterable[int]], Dict[str, float]],
    sigmoid: bool = True,
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    probs: list[float] = []
    labels_all: list[int] = []
    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        logits = model(images)
        if sigmoid:
            p = torch.sigmoid(logits).squeeze(-1)
        else:
            p = logits.squeeze(-1)
        probs.extend(p.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    return metric_fn(probs, labels_all)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = default_bce_loss,
    metric_fn: Optional[Callable[[Iterable[float], Iterable[int]], Dict[str, float]]] = None,
    cfg: TrainConfig = TrainConfig(),
) -> Dict[str, float]:
    scaler = torch.cuda.amp.GradScaler() if (cfg.mixed_precision and torch.cuda.is_available()) else None
    best_metric = -float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        _ = train_epoch(model, train_loader, optimizer, loss_fn, scaler)
        metrics = validate_epoch(model, val_loader, metric_fn or (lambda p, y: {}))
        # Expect "auroc" higher-is-better
        score = float(metrics.get("auroc", 0.0))
        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_auroc": best_metric}


