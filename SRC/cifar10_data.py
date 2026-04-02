from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Could not find CIFAR-10 data root: {root}")
    return root


def build_transforms(
    augment: bool = False,
    augmentation_policy: str = "basic",
    crop_padding: int = 4,
    random_erasing_prob: float = 0.0,
    color_jitter_brightness: float = 0.0,
    color_jitter_contrast: float = 0.0,
    color_jitter_saturation: float = 0.0,
    color_jitter_hue: float = 0.0,
    random_grayscale_prob: float = 0.0,
) -> tuple[transforms.Compose, transforms.Compose]:
    if augmentation_policy not in {"none", "basic", "autoaugment", "randaugment", "trivialaugmentwide"}:
        raise ValueError(
            "augmentation_policy must be one of: none, basic, autoaugment, randaugment, trivialaugmentwide."
        )

    train_steps: list = []
    if augment:
        train_steps.extend(
            [
                transforms.RandomCrop(32, padding=crop_padding, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
            ]
        )
        if any(
            value > 0.0
            for value in (
                color_jitter_brightness,
                color_jitter_contrast,
                color_jitter_saturation,
                color_jitter_hue,
            )
        ):
            train_steps.append(
                transforms.ColorJitter(
                    brightness=color_jitter_brightness,
                    contrast=color_jitter_contrast,
                    saturation=color_jitter_saturation,
                    hue=color_jitter_hue,
                )
            )
        if random_grayscale_prob > 0.0:
            train_steps.append(transforms.RandomGrayscale(p=random_grayscale_prob))
        if augmentation_policy == "autoaugment":
            train_steps.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
        elif augmentation_policy == "randaugment":
            train_steps.append(transforms.RandAugment())
        elif augmentation_policy == "trivialaugmentwide":
            train_steps.append(transforms.TrivialAugmentWide())

    normalization = [
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ]
    train_normalization = list(normalization)
    if random_erasing_prob > 0.0:
        train_normalization.append(
            transforms.RandomErasing(
                p=random_erasing_prob,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            )
        )

    train_transform = transforms.Compose([*train_steps, *train_normalization])
    eval_transform = transforms.Compose(normalization)
    return train_transform, eval_transform


def make_train_val_indices(
    dataset_size: int,
    validation_split: float,
    seed: int,
) -> tuple[Sequence[int], Sequence[int]]:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")

    validation_size = int(dataset_size * validation_split)
    if validation_size == 0:
        raise ValueError("validation_split is too small for the dataset size.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_indices = shuffled_indices[:validation_size]
    train_indices = shuffled_indices[validation_size:]
    return train_indices, val_indices


def get_cifar10_datasets(
    data_root: str | Path | None = None,
    validation_split: float = 0.1,
    augment: bool = False,
    augmentation_policy: str = "basic",
    crop_padding: int = 4,
    random_erasing_prob: float = 0.0,
    color_jitter_brightness: float = 0.0,
    color_jitter_contrast: float = 0.0,
    color_jitter_saturation: float = 0.0,
    color_jitter_hue: float = 0.0,
    random_grayscale_prob: float = 0.0,
    seed: int = 42,
    download: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    root = resolve_data_root(data_root)
    train_transform, eval_transform = build_transforms(
        augment=augment,
        augmentation_policy=augmentation_policy,
        crop_padding=crop_padding,
        random_erasing_prob=random_erasing_prob,
        color_jitter_brightness=color_jitter_brightness,
        color_jitter_contrast=color_jitter_contrast,
        color_jitter_saturation=color_jitter_saturation,
        color_jitter_hue=color_jitter_hue,
        random_grayscale_prob=random_grayscale_prob,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
        )
        train_dataset_full = datasets.CIFAR10(
            root=str(root),
            train=True,
            transform=train_transform,
            download=download,
        )
        validation_dataset_full = datasets.CIFAR10(
            root=str(root),
            train=True,
            transform=eval_transform,
            download=download,
        )
        test_dataset = datasets.CIFAR10(
            root=str(root),
            train=False,
            transform=eval_transform,
            download=download,
        )

    train_indices, val_indices = make_train_val_indices(
        dataset_size=len(train_dataset_full),
        validation_split=validation_split,
        seed=seed,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    validation_dataset = Subset(validation_dataset_full, val_indices)
    return train_dataset, validation_dataset, test_dataset


def get_cifar10_dataloaders(
    batch_size: int = 128,
    data_root: str | Path | None = None,
    validation_split: float = 0.1,
    augment: bool = False,
    augmentation_policy: str = "basic",
    crop_padding: int = 4,
    random_erasing_prob: float = 0.0,
    color_jitter_brightness: float = 0.0,
    color_jitter_contrast: float = 0.0,
    color_jitter_saturation: float = 0.0,
    color_jitter_hue: float = 0.0,
    random_grayscale_prob: float = 0.0,
    seed: int = 42,
    download: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    train_dataset, validation_dataset, test_dataset = get_cifar10_datasets(
        data_root=data_root,
        validation_split=validation_split,
        augment=augment,
        augmentation_policy=augmentation_policy,
        crop_padding=crop_padding,
        random_erasing_prob=random_erasing_prob,
        color_jitter_brightness=color_jitter_brightness,
        color_jitter_contrast=color_jitter_contrast,
        color_jitter_saturation=color_jitter_saturation,
        color_jitter_hue=color_jitter_hue,
        random_grayscale_prob=random_grayscale_prob,
        seed=seed,
        download=download,
    )

    use_pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, validation_loader, test_loader


def main() -> None:
    train_loader, validation_loader, test_loader = get_cifar10_dataloaders()
    train_images, train_labels = next(iter(train_loader))

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch image tensor shape: {tuple(train_images.shape)}")
    print(f"Batch label tensor shape: {tuple(train_labels.shape)}")
    print(f"Class names: {CIFAR10_CLASSES}")
    print(f"Image dtype: {train_images.dtype}")
    print(f"Label min/max: {int(train_labels.min())}/{int(train_labels.max())}")


if __name__ == "__main__":
    main()
