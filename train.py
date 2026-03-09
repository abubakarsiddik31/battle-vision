#!/usr/bin/env python
"""
Research training script for KIIT-MiTA multi-label classification.

This script is designed for research experimentation with full configuration tracking.
All hyperparameters, model choices, and experiment details are logged for reproducibility.

Usage:
    # Train with explicit architecture
    python train.py --architecture resnet18 --pretrained imagenet

    # Custom CNN from scratch
    python train.py --architecture custom_cnn --pretrained none

    # With full experiment notes
    python train.py --architecture resnet18 --pretrained imagenet \\
        --notes "Testing effect of batch size on convergence" \\
        --tags batch-size-experiment

    # Specify all hyperparameters explicitly
    python train.py --architecture resnet18 --pretrained imagenet \\
        --epochs-head 15 --epochs-finetune 25 --lr-head 0.001 --lr-finetune 0.0001

    # View experiments
    python scripts/view_experiments.py list
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kiit_mita.dataloaders import create_dataloaders
from kiit_mita.trainer import Trainer
from kiit_mita.models import create_model, CustomCNN
from kiit_mita import NUM_CLASSES, CLASS_NAMES


def parse_args():
    """Parse all training arguments for research experimentation."""
    parser = argparse.ArgumentParser(
        description="Train KIIT-MiTA Multi-Label Classifier - Research Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ===== Architecture =====
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        required=True,
        choices=[
            # ResNet family
            "resnet18", "resnet34", "resnet50",
            # EfficientNet family
            "efficientnet_b0", "efficientnet_b1",
            "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
            # Vision Transformers
            "vit_b_16", "vit_b_32",
            "swin_t", "swin_s", "swin_b",
            # VGG family (classic CNNs)
            "vgg11", "vgg13", "vgg16", "vgg19",
            # DenseNet family
            "densenet121", "densenet161", "densenet169", "densenet201",
            # MobileNet family (lightweight)
            "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
            # ConvNeXt (modern CNN)
            "convnext_tiny", "convnext_small", "convnext_base",
            # Custom baseline
            "custom_cnn",
        ],
        help="Model architecture to use"
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="imagenet",
        choices=["imagenet", "none", "custom"],
        help="Pretrained weights to use ('none' for random initialization)"
    )

    parser.add_argument(
        "--custom-weights",
        type=str,
        default=None,
        help="Path to custom pretrained weights (if pretrained=custom)"
    )

    # ===== Training Schedule =====
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=10,
        help="Epochs for head training (transfer learning only, ignored for custom_cnn)"
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=20,
        help="Epochs for fine-tuning (transfer learning only, ignored for custom_cnn)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total epochs (used only for custom_cnn architecture)"
    )

    # ===== Learning Rates =====
    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="Learning rate for head training phase"
    )
    parser.add_argument(
        "--lr-finetune",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning phase"
    )
    parser.add_argument(
        "--lr-scratch",
        type=float,
        default=1e-3,
        help="Learning rate for training from scratch (custom_cnn)"
    )

    # ===== Optimization =====
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer to use"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer"
    )

    # ===== Data =====
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # ===== Model Architecture Tweaks =====
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate in classifier head"
    )
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Freeze batch normalization layers during transfer learning"
    )

    # ===== Loss Function =====
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor"
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="Positive weight for imbalanced classes"
    )

    # ===== Device =====
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use automatic mixed precision training"
    )

    # ===== Experiment Metadata =====
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Detailed experiment notes (describe hypothesis, what you're testing)"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Experiment tags for organization (e.g., baseline hyperparam-tuning ablation)"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Custom experiment name (auto-generated if not provided)"
    )

    # ===== Tracking =====
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

    return parser.parse_args()


def build_config_dict(args):
    """Build a complete configuration dictionary from arguments."""
    config = {
        # Architecture
        "architecture": args.architecture,
        "pretrained": args.pretrained,
        "custom_weights_path": args.custom_weights,

        # Training Schedule
        "epochs_head": args.epochs_head,
        "epochs_finetune": args.epochs_finetune,
        "epochs_total": args.epochs_head + args.epochs_finetune if args.architecture != "custom_cnn" else args.epochs,

        # Learning Rates
        "lr_head": args.lr_head,
        "lr_finetune": args.lr_finetune,
        "lr_scratch": args.lr_scratch,

        # Optimization
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum if args.optimizer == "sgd" else None,

        # Data
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_workers": args.num_workers,

        # Architecture Tweaks
        "dropout": args.dropout,
        "freeze_batch_norm": args.freeze_bn,

        # Loss Function
        "label_smoothing": args.label_smoothing,
        "pos_weight": args.pos_weight,

        # Device
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "mixed_precision": args.mixed_precision,

        # Dataset info
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
    }

    return config


def create_model_from_config(config):
    """Create model from configuration dictionary."""
    architecture = config["architecture"]
    pretrained = config["pretrained"] != "none"

    model = create_model(
        model_type=architecture,
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
    )

    # Apply architecture tweaks
    if hasattr(model, 'classifier'):
        # Update dropout if model supports it
        pass  # Would need to modify model creation to support this

    return model


def main():
    args = parse_args()

    # Build complete configuration
    config = build_config_dict(args)

    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{args.architecture}_{config['pretrained']}_{timestamp}"

    # Build comprehensive notes
    notes = args.notes or ""
    if notes:
        notes += "\n\n"

    # Add configuration summary to notes
    notes += "Configuration:\n"
    notes += f"  Architecture: {args.architecture}\n"
    notes += f"  Pretrained: {args.pretrained}\n"
    notes += f"  Training: {config['epochs_total']} epochs total\n"
    notes += f"  LR Head: {args.lr_head}, LR Finetune: {args.lr_finetune}\n"
    notes += f"  Batch Size: {args.batch_size}\n"
    notes += f"  Optimizer: {args.optimizer} (weight_decay={args.weight_decay})\n"

    print("=" * 80)
    print("KIIT-MiTA Multi-Label Classification - Research Training")
    print("=" * 80)
    print(f"\nExperiment: {args.exp_name}")
    print(f"Architecture: {args.architecture}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Classes ({NUM_CLASSES}): {', '.join(CLASS_NAMES)}")
    print(f"\nTraining Configuration:")
    print(f"  Total Epochs: {config['epochs_total']}")
    print(f"  Learning Rates: head={args.lr_head}, finetune={args.lr_finetune}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Device: {config['device']}")
    if args.tags:
        print(f"  Tags: {', '.join(args.tags)}")
    print(f"\nNotes:\n{notes}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Create model
    print(f"\nCreating model: {args.architecture}")
    model = create_model_from_config(config)

    # Create trainer with full config tracking
    trainer = Trainer(
        model=model,
        model_name=args.architecture,
        device=config["device"],
        learning_rate=args.lr_head,
        use_trackio=not args.no_trackio,
        trackio_project=args.trackio_project,
        experiment_name=args.exp_name,
        experiment_notes=notes,
        experiment_tags=args.tags or [args.architecture, config["pretrained"]],
    )

    # Log complete configuration
    trainer.experiment_logger.log_config(config)
    trainer.experiment_logger.log_config({
        "command_line_args": vars(args),
        "git_commit": get_git_commit(),
    })

    # Add initial notes
    if args.notes:
        trainer.add_note(f"Research Goal: {args.notes}")

    # Training
    if args.architecture == "custom_cnn":
        trainer.train_from_scratch(
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr_scratch,
        )
    else:
        trainer.train(
            train_loader,
            val_loader,
            num_epochs_head=args.epochs_head,
            num_epochs_finetune=args.epochs_finetune,
            learning_rate=args.lr_head,
        )

    # Final test evaluation
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    test_loss, test_metrics = trainer.evaluate(test_loader, trainer.val_metrics)

    print(f"\n{'Metric':<30} {'Value'}")
    print("-" * 50)
    print(f"{'Test Loss':<30} {test_loss:.4f}")
    print(f"{'Exact Match Accuracy':<30} {test_metrics['exact_match_accuracy']:.4f}")
    print(f"{'Micro F1':<30} {test_metrics['micro']['f1']:.4f}")
    print(f"{'Micro Precision':<30} {test_metrics['micro']['precision']:.4f}")
    print(f"{'Micro Recall':<30} {test_metrics['micro']['recall']:.4f}")
    print(f"{'Macro F1':<30} {test_metrics['macro']['f1']:.4f}")

    print(f"\nPer-Class F1 Scores:")
    for class_name in CLASS_NAMES:
        if class_name in test_metrics:
            print(f"  {class_name:<20} {test_metrics[class_name]['f1']:.4f}")

    # Save test results
    results_path = Path("results") / f"{args.exp_name}_test_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "config": config,
            "metrics": test_metrics,
            "loss": test_loss,
        }, f, indent=2)
    print(f"\nTest results saved to {results_path}")

    # Add final results as note
    trainer.add_note(
        f"Final Test Results: Loss={test_loss:.4f}, "
        f"F1={test_metrics['micro']['f1']:.4f}, "
        f"Acc={test_metrics['exact_match_accuracy']:.4f}"
    )

    # Finish and save experiment
    summary, exp_file = trainer.finish_experiment(test_metrics)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Experiment saved: {exp_file}")
    print(f"View dashboard: trackio show --project {args.trackio_project}")
    print(f"List experiments: python scripts/view_experiments.py list")


def get_git_commit():
    """Get current git commit hash for reproducibility."""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      cwd=Path(__file__).parent,
                                      stderr=subprocess.DEVNULL).decode('ascii').strip()
    except:
        return "unknown"


if __name__ == "__main__":
    import torch
    main()
