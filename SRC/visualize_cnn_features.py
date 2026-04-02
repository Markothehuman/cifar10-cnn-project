from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .cifar10_data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, get_cifar10_dataloaders
    from .cnn_model import SimpleCIFAR10CNN
except ImportError:
    from cifar10_data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, get_cifar10_dataloaders
    from cnn_model import SimpleCIFAR10CNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize intermediate CNN feature maps.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-feature-maps", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    fallback = PROJECT_ROOT / "Data" / "cnn_feature_maps"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    image = image_tensor * std + mean
    return image.clamp(0.0, 1.0)


def save_input_image(image: torch.Tensor, label: int, output_dir: Path) -> Path:
    image = denormalize(image).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.set_title(f"Input image: {CIFAR10_CLASSES[label]}")
    ax.axis("off")
    fig.tight_layout()

    output_path = output_dir / "input_image.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_feature_grid(
    activation: torch.Tensor,
    layer_name: str,
    output_dir: Path,
    num_feature_maps: int,
) -> Path:
    channels = min(num_feature_maps, activation.shape[0])
    cols = 4
    rows = (channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes

    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for index, axis in enumerate(flat_axes):
        if index < channels:
            axis.imshow(activation[index].numpy(), cmap="viridis")
            axis.set_title(f"{layer_name} map {index}")
            axis.axis("off")
        else:
            axis.axis("off")

    fig.tight_layout()
    output_path = output_dir / f"{layer_name}_feature_maps.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    _, _, test_loader = get_cifar10_dataloaders(batch_size=1, augment=False)
    model = SimpleCIFAR10CNN()
    model.eval()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint provided. Visualizations will reflect random, untrained filters.")

    dataset = test_loader.dataset
    image, label = dataset[args.sample_index]
    batch = image.unsqueeze(0)

    logits, activations = model.forward_with_activations(batch, detach=True)
    predicted_label = int(logits.argmax(dim=1).item())

    print(f"Sample index: {args.sample_index}")
    print(f"True label: {CIFAR10_CLASSES[label]}")
    print(f"Predicted label: {CIFAR10_CLASSES[predicted_label]}")

    input_path = save_input_image(image, label, output_dir)
    print(f"Saved input image to: {input_path}")

    layers_to_plot = ["conv1_1", "pool1", "conv2_1", "pool2", "conv3_2"]
    for layer_name in layers_to_plot:
        feature_path = save_feature_grid(
            activation=activations[layer_name][0],
            layer_name=layer_name,
            output_dir=output_dir,
            num_feature_maps=args.num_feature_maps,
        )
        print(f"Saved {layer_name} feature maps to: {feature_path}")


if __name__ == "__main__":
    main()
