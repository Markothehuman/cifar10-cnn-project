from __future__ import annotations

import torch
from torch import nn


class SimpleCIFAR10CNN(nn.Module):
    """
    A first CNN for CIFAR-10 with named layers so we can inspect what happens.

    Big idea:
    - Each convolution learns a bank of filters.
    - Each filter spans all input channels and produces one feature map.
    - Early filters often become edge or color detectors after training.
    - Deeper filters combine simpler patterns into more abstract ones.
    - Pooling reduces spatial size and makes the model less sensitive to
      small shifts in where a feature appears.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4) -> None:
        super().__init__()

        # Block 1: low-level features such as edges, corners, and color changes.
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: mid-level combinations of earlier features.
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: higher-level patterns built from earlier feature maps.
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU()

        # Global average pooling keeps strong class-relevant signals without
        # creating a very large fully connected classifier.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_activations(x, detach=False)
        return logits

    def forward_with_activations(
        self,
        x: torch.Tensor,
        detach: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Run a forward pass and also return intermediate activations.

        This is useful for understanding what each stage of the CNN is doing.
        """
        activations: dict[str, torch.Tensor] = {"input": x}

        x = self.conv1_1(x)
        activations["conv1_1"] = x
        x = self.relu1_1(x)
        activations["relu1_1"] = x

        x = self.conv1_2(x)
        activations["conv1_2"] = x
        x = self.relu1_2(x)
        activations["relu1_2"] = x

        x = self.pool1(x)
        activations["pool1"] = x

        x = self.conv2_1(x)
        activations["conv2_1"] = x
        x = self.relu2_1(x)
        activations["relu2_1"] = x

        x = self.conv2_2(x)
        activations["conv2_2"] = x
        x = self.relu2_2(x)
        activations["relu2_2"] = x

        x = self.pool2(x)
        activations["pool2"] = x

        x = self.conv3_1(x)
        activations["conv3_1"] = x
        x = self.relu3_1(x)
        activations["relu3_1"] = x

        x = self.conv3_2(x)
        activations["conv3_2"] = x
        x = self.relu3_2(x)
        activations["relu3_2"] = x

        x = self.global_pool(x)
        activations["global_pool"] = x

        logits = self.classifier(x)
        activations["logits"] = logits

        if detach:
            activations = {
                name: tensor.detach().cpu()
                for name, tensor in activations.items()
            }
            logits = logits.detach().cpu()

        return logits, activations

    @torch.no_grad()
    def describe_feature_shapes(
        self,
        input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    ) -> dict[str, tuple[int, ...]]:
        sample = torch.zeros(input_shape)
        _, activations = self.forward_with_activations(sample, detach=True)
        return {name: tuple(tensor.shape) for name, tensor in activations.items()}


def main() -> None:
    model = SimpleCIFAR10CNN()
    sample = torch.randn(4, 3, 32, 32)
    logits, activations = model.forward_with_activations(sample)

    print(model)
    print(f"Output shape: {tuple(logits.shape)}")
    print("Feature map shapes:")
    for name, tensor in activations.items():
        print(f"  {name}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
