#!/usr/bin/env python
"""
Main training script for KIIT-MiTA multi-label classification.

Usage:
    # Train with transfer learning (ResNet-18)
    python train.py --model resnet18

    # Train with experiment notes
    python train.py --model resnet18 --notes "Testing higher learning rate"

    # Train with tags for categorization
    python train.py --model resnet18 --tags baseline resnet

    # Train custom CNN baseline
    python train.py --model custom --epochs 30

    # Train with trackio disabled
    python train.py --model resnet18 --no-trackio

    # View experiments dashboard
    trackio show --project kiit-mita-classification

    # List all experiments
    python -c "from src.kiit_mita.experiment_logger import ExperimentLogger; import json; print(json.dumps(ExperimentLogger.list_experiments(), indent=2))"
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

    # Experiment options
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Experiment notes (describe what you're testing)"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Experiment tags (e.g., baseline hyperparam-tuning)"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Custom experiment name"
    )

    # Trackio options
    parser.add_argument(
        "--no-trackio",
        action="store_true",
        help="Disable trackio logging"
    )
    parser.add_argument(
        "--trackio-project",
        type=str,
        default="kiit-mita-classification",
        help="Trackio project name"
    )

    args = parser.parse_args()

    # Generate default experiment name if not provided
    if args.exp_name is None:
        from datetime import datetime
        args.exp_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build notes
    notes = args.notes or ""
    if notes:
        print(f"Experiment Notes: {notes}")

    print("=" * 60)
    print("KIIT-MiTA Multi-Label Classification Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Experiment Name: {args.exp_name}")
    print(f"Classes ({NUM_CLASSES}): {', '.join(CLASS_NAMES)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Trackio: {not args.no_trackio}")
    if args.tags:
        print(f"Tags: {', '.join(args.tags)}")

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
            use_trackio=not args.no_trackio,
            experiment_name=args.exp_name,
            experiment_notes=notes,
            experiment_tags=args.tags,
        )

        # Add additional config note
        trainer.add_note(f"Training from scratch for {args.epochs} epochs")

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
            use_trackio=not args.no_trackio,
            device=args.device,
            experiment_name=args.exp_name,
            experiment_notes=notes,
            experiment_tags=args.tags,
        )

        # Add training strategy note
        trainer.add_note(
            f"Transfer learning: {args.epochs_head} epochs head, "
            f"{args.epochs_finetune} epochs finetune"
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

    # Add test results as note
    trainer.add_note(
        f"Test Results: Loss={test_loss:.4f}, "
        f"Acc={test_metrics['exact_match_accuracy']:.4f}, "
        f"F1={test_metrics['micro']['f1']:.4f}"
    )

    # Save results
    import json
    results_path = Path("results") / f"{args.model}_test_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Finish and save experiment
    summary, exp_file = trainer.finish_experiment(test_metrics)
    print(f"\nExperiment complete! View with: trackio show --project {args.trackio_project}")


if __name__ == "__main__":
    main()
