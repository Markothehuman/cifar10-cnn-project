from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.ndim > 1:
        targets = targets.argmax(dim=1)
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> EpochMetrics:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_examples = 0

    for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        metric_targets = targets

        optimizer.zero_grad()
        if batch_transform is not None:
            inputs, targets = batch_transform(inputs, targets)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_size = metric_targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == metric_targets).sum().item()
        running_examples += batch_size

        if max_batches is not None and batch_index >= max_batches:
            break

    return EpochMetrics(
        loss=running_loss / running_examples,
        accuracy=running_correct / running_examples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_examples = 0

    for batch_index, (inputs, targets) in enumerate(dataloader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        running_examples += batch_size

        if max_batches is not None and batch_index >= max_batches:
            break

    return EpochMetrics(
        loss=running_loss / running_examples,
        accuracy=running_correct / running_examples,
    )
