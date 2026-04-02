from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re

import matplotlib
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .cifar10_data import CIFAR10_CLASSES, get_cifar10_dataloaders
    from .improved_cnn import DeeperCIFAR10CNN
    from .training import evaluate, train_one_epoch
except ImportError:
    from cifar10_data import CIFAR10_CLASSES, get_cifar10_dataloaders
    from improved_cnn import DeeperCIFAR10CNN
    from training import evaluate, train_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def slugify_label(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    slug = slug.strip("_")
    return slug or "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a deeper CNN on CIFAR-10.")
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--augmentation-policy",
        type=str,
        default="basic",
        choices=["none", "basic", "autoaugment", "randaugment", "trivialaugmentwide"],
    )
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--random-erasing-prob", type=float, default=0.0)
    parser.add_argument("--color-jitter-brightness", type=float, default=0.0)
    parser.add_argument("--color-jitter-contrast", type=float, default=0.0)
    parser.add_argument("--color-jitter-saturation", type=float, default=0.0)
    parser.add_argument("--color-jitter-hue", type=float, default=0.0)
    parser.add_argument("--random-grayscale-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--pooling-type", type=str, default="max", choices=["max", "avg", "stride"])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--channels", type=parse_int_tuple, default=(64, 128, 256, 512))
    parser.add_argument("--blocks-per-stage", type=parse_int_tuple, default=(2, 2, 2, 2))
    parser.add_argument("--classifier-hidden", type=int, default=256)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--open-plot", action="store_true")
    return parser.parse_args()


def resolve_output_dir(run_label: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{slugify_label(run_label)}_{timestamp}" if run_label else f"run_{timestamp}"
    candidates = [
        PROJECT_ROOT / "Outputs" / "improved_cnn" / run_name,
        PROJECT_ROOT / "Data" / "improved_cnn_outputs" / run_name,
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except PermissionError:
            continue

    raise PermissionError("Could not create an output directory for improved CNN training artifacts.")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )


def save_history(history: list[dict], output_dir: Path) -> None:
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_summary(summary: dict, output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def load_history(output_dir: Path) -> list[dict]:
    history_path = output_dir / "history.json"
    if not history_path.exists():
        return []
    return json.loads(history_path.read_text(encoding="utf-8"))


def load_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def save_resume_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    best_val_accuracy: float,
    history: list[dict],
) -> Path:
    checkpoint_path = output_dir / "latest_training_state.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "history": history,
        },
        checkpoint_path,
    )
    return checkpoint_path


class ClassificationLoss(nn.Module):
    def __init__(self, num_classes: int, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        smoothed_targets = targets
        if self.label_smoothing > 0.0:
            smoothed_targets = (
                smoothed_targets * (1.0 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )

        log_probs = F.log_softmax(logits, dim=1)
        return -(smoothed_targets * log_probs).sum(dim=1).mean()


def build_batch_transform(num_classes: int, args: argparse.Namespace) -> nn.Module | None:
    transforms_to_apply: list[nn.Module] = []
    if args.mixup_alpha > 0.0:
        transforms_to_apply.append(v2.MixUp(alpha=args.mixup_alpha, num_classes=num_classes))
    if args.cutmix_alpha > 0.0:
        transforms_to_apply.append(v2.CutMix(alpha=args.cutmix_alpha, num_classes=num_classes))

    if not transforms_to_apply:
        return None
    if len(transforms_to_apply) == 1:
        return transforms_to_apply[0]
    return v2.RandomChoice(transforms_to_apply)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if args.scheduler == "none":
        return None

    cosine_epochs = args.epochs - args.warmup_epochs
    if cosine_epochs <= 0:
        raise ValueError("--warmup-epochs must be smaller than --epochs when using cosine scheduling.")

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=args.min_learning_rate,
    )

    if args.warmup_epochs <= 0:
        return cosine_scheduler

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )


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
    axes[0].set_title("Improved CNN Loss")
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
    axes[1].set_title("Improved CNN Accuracy")
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

    if len(args.channels) != len(args.blocks_per_stage):
        raise ValueError("--channels and --blocks-per-stage must have the same number of values.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.resume_dir).resolve() if args.resume_dir else resolve_output_dir(args.run_label)

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        augment=args.augment,
        augmentation_policy=args.augmentation_policy,
        crop_padding=args.crop_padding,
        random_erasing_prob=args.random_erasing_prob,
        color_jitter_brightness=args.color_jitter_brightness,
        color_jitter_contrast=args.color_jitter_contrast,
        color_jitter_saturation=args.color_jitter_saturation,
        color_jitter_hue=args.color_jitter_hue,
        random_grayscale_prob=args.random_grayscale_prob,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = DeeperCIFAR10CNN(
        num_classes=len(CIFAR10_CLASSES),
        channels=args.channels,
        blocks_per_stage=args.blocks_per_stage,
        pooling_type=args.pooling_type,
        dropout=args.dropout,
        classifier_hidden=args.classifier_hidden,
    ).to(device)

    num_classes = len(CIFAR10_CLASSES)
    criterion = ClassificationLoss(num_classes=num_classes, label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(
        optimizer_name=args.optimizer,
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer=optimizer, args=args)
    batch_transform = build_batch_transform(num_classes=num_classes, args=args)

    history: list[dict] = []
    best_val_accuracy = 0.0
    start_epoch = 1
    best_model_path = output_dir / "best_improved_cnn.pt"
    resume_checkpoint_path = output_dir / "latest_training_state.pt"

    if args.resume_dir:
        checkpoint_loaded = False
        if resume_checkpoint_path.exists():
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            history = checkpoint.get("history", [])
            best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
            start_epoch = checkpoint.get("epoch", 0) + 1
            checkpoint_loaded = True
        else:
            if best_model_path.exists():
                model.load_state_dict(torch.load(best_model_path, map_location=device))
            history = load_history(output_dir)
            summary = load_summary(output_dir)
            best_val_accuracy = float(summary.get("best_val_accuracy", 0.0))
            start_epoch = len(history) + 1
            if scheduler is not None:
                for _ in history:
                    scheduler.step()

        print(f"Resuming from: {output_dir}")
        print(f"Resume checkpoint loaded: {checkpoint_loaded}")
        print(f"Completed epochs found: {len(history)}")

    print(f"Device: {device}")
    print(f"Run label: {args.run_label or 'none'}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {CIFAR10_CLASSES}")
    print(f"Architecture channels: {args.channels}")
    print(f"Blocks per stage: {args.blocks_per_stage}")
    print(f"Pooling type: {args.pooling_type}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Data augmentation enabled: {args.augment}")
    print(f"Augmentation policy: {args.augmentation_policy}")
    print(f"Crop padding: {args.crop_padding}")
    print(f"Random erasing probability: {args.random_erasing_prob}")
    print(
        "Color jitter: "
        f"brightness={args.color_jitter_brightness}, "
        f"contrast={args.color_jitter_contrast}, "
        f"saturation={args.color_jitter_saturation}, "
        f"hue={args.color_jitter_hue}"
    )
    print(f"Random grayscale probability: {args.random_grayscale_prob}")
    print(f"MixUp alpha: {args.mixup_alpha}")
    print(f"CutMix alpha: {args.cutmix_alpha}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Feature map shapes: {model.describe_feature_shapes()}")

    if start_epoch > args.epochs:
        print("Requested epoch count already reached in the resume directory. Running final evaluation only.")

    for epoch in range(start_epoch, args.epochs + 1):
        current_learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            batch_transform=batch_transform,
            grad_clip_norm=args.grad_clip_norm,
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
            "learning_rate": round(current_learning_rate, 8),
        }
        history.append(epoch_record)

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            torch.save(model.state_dict(), best_model_path)

        if scheduler is not None:
            scheduler.step()

        partial_summary = {
            "model": "DeeperCIFAR10CNN",
            "run_label": args.run_label,
            "resume_dir": args.resume_dir,
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "validation_split": args.validation_split,
            "augment": args.augment,
            "augmentation_policy": args.augmentation_policy,
            "crop_padding": args.crop_padding,
            "random_erasing_prob": args.random_erasing_prob,
            "color_jitter_brightness": args.color_jitter_brightness,
            "color_jitter_contrast": args.color_jitter_contrast,
            "color_jitter_saturation": args.color_jitter_saturation,
            "color_jitter_hue": args.color_jitter_hue,
            "random_grayscale_prob": args.random_grayscale_prob,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip_norm": args.grad_clip_norm,
            "pooling_type": args.pooling_type,
            "optimizer": args.optimizer,
            "channels": args.channels,
            "blocks_per_stage": args.blocks_per_stage,
            "classifier_hidden": args.classifier_hidden,
            "best_val_accuracy": round(best_val_accuracy, 4),
            "history": history,
        }
        save_history(history, output_dir)
        save_summary(partial_summary, output_dir)
        save_resume_checkpoint(
            output_dir=output_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_accuracy=best_val_accuracy,
            history=history,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"lr={current_learning_rate:.6f} | "
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
        "model": "DeeperCIFAR10CNN",
        "run_label": args.run_label,
        "resume_dir": args.resume_dir,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "validation_split": args.validation_split,
        "augment": args.augment,
        "augmentation_policy": args.augmentation_policy,
        "crop_padding": args.crop_padding,
        "random_erasing_prob": args.random_erasing_prob,
        "color_jitter_brightness": args.color_jitter_brightness,
        "color_jitter_contrast": args.color_jitter_contrast,
        "color_jitter_saturation": args.color_jitter_saturation,
        "color_jitter_hue": args.color_jitter_hue,
        "random_grayscale_prob": args.random_grayscale_prob,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip_norm": args.grad_clip_norm,
        "pooling_type": args.pooling_type,
        "optimizer": args.optimizer,
        "channels": args.channels,
        "blocks_per_stage": args.blocks_per_stage,
        "classifier_hidden": args.classifier_hidden,
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
