from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
import torch
import torch.hub
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .training import evaluate, train_one_epoch
except ImportError:
    from training import evaluate, train_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODEL_FACTORY = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "mobilenet_v2": models.mobilenet_v2,
}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning on CIFAR-10 using torchvision models.")
    parser.add_argument("--model", type=str, default="resnet18", choices=list(MODEL_FACTORY.keys()))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-epoch", type=int, default=4)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--open-plot", action="store_true")
    return parser.parse_args()


def resolve_output_dir() -> Path:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidates = [
        PROJECT_ROOT / "Outputs" / "transfer_learning" / run_name,
        PROJECT_ROOT / "Data" / "transfer_learning_outputs" / run_name,
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except PermissionError:
            continue

    raise PermissionError("Could not create an output directory for transfer learning artifacts.")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(augment: bool, resize: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[transforms.Transform] = []
    if augment:
        train_steps.extend(
            [
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ]
        )
    else:
        train_steps.append(transforms.Resize(resize))

    normalization = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    train_transform = transforms.Compose([*train_steps, *normalization])
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(
    batch_size: int,
    augment: bool,
    resize: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms(augment=augment, resize=resize)

    train_dataset = datasets.CIFAR10(
        root=str(PROJECT_ROOT / "Data"),
        train=True,
        transform=train_transform,
        download=False,
    )
    validation_dataset = datasets.CIFAR10(
        root=str(PROJECT_ROOT / "Data"),
        train=True,
        transform=eval_transform,
        download=False,
    )
    test_dataset = datasets.CIFAR10(
        root=str(PROJECT_ROOT / "Data"),
        train=False,
        transform=eval_transform,
        download=False,
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    val_size = int(0.1 * len(indices))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(validation_dataset, val_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def build_model(name: str) -> nn.Module:
    hub_dir = PROJECT_ROOT / "Data" / "torch_hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))

    weights = None
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT
    elif name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT

    model = MODEL_FACTORY[name](weights=weights)
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    return model


def set_trainable_layers(model: nn.Module, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad = trainable


def save_history(history: list[dict], output_dir: Path) -> None:
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_summary(summary: dict, output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def save_training_plot(history: list[dict], summary: dict, output_dir: Path) -> Path | None:
    if not history:
        return None

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_accuracy = [entry["train_accuracy"] for entry in history]
    val_accuracy = [entry["val_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation loss")
    axes[0].set_title("Transfer Learning Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_xticks(epochs)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, marker="o", label="Train accuracy")
    axes[1].plot(epochs, val_accuracy, marker="o", label="Validation accuracy")
    axes[1].axhline(
        summary["test_accuracy"],
        color="tab:green",
        linestyle="--",
        label=f"Test accuracy ({summary['test_accuracy']:.4f})",
    )
    axes[1].set_title("Transfer Learning Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(epochs)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def open_plot_file(plot_path: Path) -> None:
    if hasattr(Path, "open"):
        try:
            import os

            if hasattr(os, "startfile"):
                os.startfile(plot_path)  # type: ignore[attr-defined]
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = resolve_output_dir()

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        augment=args.augment,
        resize=args.resize,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = build_model(args.model).to(device)

    if args.freeze_backbone:
        set_trainable_layers(model, False)
        if args.model.startswith("resnet"):
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.classifier[1].parameters():
                param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict] = []
    best_val_accuracy = 0.0
    best_model_path = output_dir / "best_transfer_model.pt"

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Resize: {args.resize}")
    print(f"Augment: {args.augment}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Unfreeze epoch: {args.unfreeze_epoch}")

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            set_trainable_layers(model, True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate * 0.2,
                weight_decay=args.weight_decay,
            )

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.limit_train_batches,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.limit_val_batches,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_metrics.loss, 4),
            "train_accuracy": round(train_metrics.accuracy, 4),
            "val_loss": round(val_metrics.loss, 4),
            "val_accuracy": round(val_metrics.accuracy, 4),
        }
        history.append(epoch_record)

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            torch.save(model.state_dict(), best_model_path)

        partial_summary = {
            "model": args.model,
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "freeze_backbone": args.freeze_backbone,
            "unfreeze_epoch": args.unfreeze_epoch,
            "resize": args.resize,
            "augment": args.augment,
            "best_val_accuracy": round(best_val_accuracy, 4),
            "history": history,
        }
        save_history(history, output_dir)
        save_summary(partial_summary, output_dir)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics.loss:.4f} | train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | val_acc={val_metrics.accuracy:.4f}"
        )

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    summary = {
        "model": args.model,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_epoch": args.unfreeze_epoch,
        "resize": args.resize,
        "augment": args.augment,
        "best_val_accuracy": round(best_val_accuracy, 4),
        "test_loss": round(test_metrics.loss, 4),
        "test_accuracy": round(test_metrics.accuracy, 4),
        "history": history,
    }

    save_history(history, output_dir)
    summary_path = save_summary(summary, output_dir)
    plot_path = save_training_plot(history, summary, output_dir)

    if args.open_plot and plot_path is not None:
        open_plot_file(plot_path)

    print(f"Test loss: {test_metrics.loss:.4f}")
    print(f"Test accuracy: {test_metrics.accuracy:.4f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved training summary to: {summary_path}")
    if plot_path is not None:
        print(f"Saved training curves to: {plot_path}")


if __name__ == "__main__":
    main()
