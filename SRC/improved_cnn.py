from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


def make_downsample(
    channels: int,
    pooling_type: str,
) -> nn.Module:
    if pooling_type == "max":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    if pooling_type == "avg":
        return nn.AvgPool2d(kernel_size=2, stride=2)
    if pooling_type == "stride":
        return nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
    raise ValueError("pooling_type must be one of: 'max', 'avg', 'stride'.")


class ConvBNReLU(nn.Module):
    """A standard convolution block used in stronger CNN baselines."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeeperCIFAR10CNN(nn.Module):
    """
    A deeper CNN for CIFAR-10 with BatchNorm and configurable downsampling.

    Why this should improve on the first CNN:
    - more convolutional layers allow a richer hierarchy of features
    - BatchNorm usually stabilizes and speeds up optimization
    - we can compare downsampling choices instead of assuming max-pooling is best
    - a slightly larger classifier head gives the model more capacity
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        pooling_type: str = "max",
        dropout: float = 0.4,
        classifier_hidden: int = 256,
    ) -> None:
        super().__init__()

        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have the same length.")
        if classifier_hidden <= 0:
            raise ValueError("classifier_hidden must be positive.")

        stages: list[tuple[str, nn.Module]] = []
        in_channels = 3

        for stage_index, (stage_channels, num_blocks) in enumerate(
            zip(channels, blocks_per_stage),
            start=1,
        ):
            layers: list[nn.Module] = []

            for block_index in range(num_blocks):
                layers.append(
                    ConvBNReLU(
                        in_channels=in_channels,
                        out_channels=stage_channels,
                    )
                )
                in_channels = stage_channels

            if stage_index < len(channels):
                layers.append(make_downsample(stage_channels, pooling_type))

            stages.append((f"stage{stage_index}", nn.Sequential(*layers)))

        self.features = nn.Sequential(OrderedDict(stages))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=channels[-1], out_features=classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=classifier_hidden, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def describe_feature_shapes(
        self,
        input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    ) -> dict[str, tuple[int, ...]]:
        x = torch.zeros(input_shape)
        shapes = {"input": tuple(x.shape)}

        for name, module in self.features.named_children():
            x = module(x)
            shapes[name] = tuple(x.shape)

        x = self.global_pool(x)
        shapes["global_pool"] = tuple(x.shape)

        x = self.classifier(x)
        shapes["logits"] = tuple(x.shape)
        return shapes


def main() -> None:
    model = DeeperCIFAR10CNN()
    sample = torch.randn(4, 3, 32, 32)
    logits = model(sample)

    print(model)
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Feature map shapes: {model.describe_feature_shapes()}")


if __name__ == "__main__":
    main()
