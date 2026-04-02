from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """A simple fully connected baseline for CIFAR-10."""

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Flatten()]
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
