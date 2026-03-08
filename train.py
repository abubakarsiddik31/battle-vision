#!/usr/bin/env python
"""
Main training script for KIIT-MiTA multi-label classification.

Usage:
    # Train with transfer learning (ResNet-18)
    python train.py --model resnet18

    # Train custom CNN baseline
    python train.py --model custom --epochs 30

    # Train with wandb disabled
    python train.py --model resnet18 --no-wandb
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kiit_mita.dataloaders import create_dataloaders
from kiit_mita.trainer import Trainer, create_trainer
from kiit_mita.models import CustomCNN
from kiit_mita import NUM_CLASSES, CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Train KIIT-MiTA Multi-Label Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet", "vit", "custom"],
        help="Model architecture to train"
    )

    # Training options
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=10,
        help="Epochs for head training (transfer learning only)"
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=20,
        help="Epochs for fine-tuning (transfer learning only)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=30,
        help="Total epochs (custom CNN only)"
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=1e-3,
        dest="lr",
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size"
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )

    # WandB options
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="kiit-mita-classification",
        help="WandB project name"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("KIIT-MiTA Multi-Label Classification Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Classes ({NUM_CLASSES}): {', '.join(CLASS_NAMES)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"WandB: {not args.no_wandb}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Create trainer and model
    if args.model == "custom":
        # Custom CNN - train from scratch
        import torch
        import torch.nn as nn

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = CustomCNN(num_classes=NUM_CLASSES)
        trainer = Trainer(
            model=model,
            model_name="custom_cnn",
            device=device,
            learning_rate=args.lr,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
        )
        trainer.train_from_scratch(
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )
    else:
        # Transfer learning
        trainer = create_trainer(
            model_type=args.model,
            learning_rate=args.lr,
            use_wandb=not args.no_wandb,
            device=args.device,
        )
        trainer.train(
            train_loader,
            val_loader,
            num_epochs_head=args.epochs_head,
            num_epochs_finetune=args.epochs_finetune,
            learning_rate=args.lr,
        )

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    test_loss, test_metrics = trainer.evaluate(test_loader, trainer.val_metrics)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.4f}")
    print(f"Micro F1: {test_metrics['micro']['f1']:.4f}")
    print(f"Macro F1: {test_metrics['macro']['f1']:.4f}")

    # Save results
    import json
    results_path = Path("results") / f"{args.model}_test_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
