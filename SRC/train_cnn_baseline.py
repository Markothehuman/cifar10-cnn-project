from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .cifar10_data import CIFAR10_CLASSES, get_cifar10_dataloaders
    from .cnn_model import SimpleCIFAR10CNN
    from .training import evaluate, train_one_epoch
except ImportError:
    from cifar10_data import CIFAR10_CLASSES, get_cifar10_dataloaders
    from cnn_model import SimpleCIFAR10CNN
    from training import evaluate, train_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple CNN baseline on CIFAR-10.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--open-plot", action="store_true")
    return parser.parse_args()


def resolve_output_dir() -> Path:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidates = [
        PROJECT_ROOT / "Outputs" / "cnn_baseline" / run_name,
        PROJECT_ROOT / "Data" / "cnn_baseline_outputs" / run_name,
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except PermissionError:
            continue

    raise PermissionError("Could not create an output directory for CNN training artifacts.")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    axes[0].set_title("CNN Baseline Loss")
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
    axes[1].set_title("CNN Baseline Accuracy")
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
    if hasattr(os, "startfile"):
        os.startfile(plot_path)  # type: ignore[attr-defined]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = resolve_output_dir()

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        augment=args.augment,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = SimpleCIFAR10CNN(num_classes=len(CIFAR10_CLASSES), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict] = []
    best_val_accuracy = 0.0
    best_model_path = output_dir / "best_cnn_baseline.pt"

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {CIFAR10_CLASSES}")
    print(f"Data augmentation enabled: {args.augment}")

    for epoch in range(1, args.epochs + 1):
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
            "model": "SimpleCIFAR10CNN",
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "validation_split": args.validation_split,
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
        "model": "SimpleCIFAR10CNN",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "validation_split": args.validation_split,
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
