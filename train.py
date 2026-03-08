"""
Training and evaluation script for KIIT-MiTA multi-label classification.

Includes:
- Training loop with transfer learning strategy (freeze -> train head -> fine-tune)
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix generation
- Model checkpointing
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import create_dataloaders
from dataset import KiitMitaDataset, load_metadata
from models import create_model


# Paths
CHECKPOINT_DIR = Path("/home/abubakar/Desktop/Research/DL-assignment/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("/home/abubakar/Desktop/Research/DL-assignment/results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load metadata
METADATA = load_metadata("/home/abubakar/Desktop/Research/DL-assignment/data/metadata.json")
CLASS_NAMES = METADATA["class_names"]
NUM_CLASSES = METADATA["num_classes"]


class MultiLabelMetrics:
    """Compute metrics for multi-label classification."""

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(self, probs: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with a batch.

        Args:
            probs: Predicted probabilities (B, C)
            labels: Ground truth labels (B, C)
        """
        preds = (probs > 0.5).float()
        self.all_preds.append(preds.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())
        self.all_probs.append(probs.detach().cpu().numpy())

    def compute(self) -> Dict:
        """Compute all metrics."""
        # Concatenate all batches
        all_preds = np.concatenate(self.all_preds, axis=0)
        all_labels = np.concatenate(self.all_labels, axis=0)
        all_probs = np.concatenate(self.all_probs, axis=0)

        # Per-class metrics
        metrics = {}
        for i, class_name in enumerate(self.class_names):
            tp = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 1))
            fp = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 0))
            fn = np.sum((all_preds[:, i] == 0) & (all_labels[:, i] == 1))
            tn = np.sum((all_preds[:, i] == 0) & (all_labels[:, i] == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

            metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            }

        # Micro-averaged metrics
        total_tp = np.sum((all_preds == 1) & (all_labels == 1))
        total_fp = np.sum((all_preds == 1) & (all_labels == 0))
        total_fn = np.sum((all_preds == 0) & (all_labels == 1))
        total_tn = np.sum((all_preds == 0) & (all_labels == 0))

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        micro_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)

        # Macro-averaged metrics
        macro_precision = np.mean([metrics[c]["precision"] for c in self.class_names])
        macro_recall = np.mean([metrics[c]["recall"] for c in self.class_names])
        macro_f1 = np.mean([metrics[c]["f1"] for c in self.class_names])

        # Exact match accuracy (all labels must match)
        exact_match = np.all(all_preds == all_labels, axis=1).mean()

        metrics["micro"] = {
            "precision": float(micro_precision),
            "recall": float(micro_recall),
            "f1": float(micro_f1),
            "accuracy": float(micro_accuracy),
        }
        metrics["macro"] = {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        }
        metrics["exact_match_accuracy"] = float(exact_match)

        return metrics

    def get_confusion_matrix(self, class_idx: int) -> np.ndarray:
        """Get confusion matrix for a specific class."""
        all_preds = np.concatenate(self.all_preds, axis=0)
        all_labels = np.concatenate(self.all_labels, axis=0)

        tp = np.sum((all_preds[:, class_idx] == 1) & (all_labels[:, class_idx] == 1))
        fp = np.sum((all_preds[:, class_idx] == 1) & (all_labels[:, class_idx] == 0))
        fn = np.sum((all_preds[:, class_idx] == 0) & (all_labels[:, class_idx] == 1))
        tn = np.sum((all_preds[:, class_idx] == 0) & (all_labels[:, class_idx] == 0))

        return np.array([[tp, fp], [fn, tn]])


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: MultiLabelMetrics,
) -> tuple[float, Dict]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    metrics.reset()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * images.size(0)
            metrics.update(probs, labels)

    avg_loss = total_loss / len(dataloader.dataset)
    results = metrics.compute()

    return avg_loss, results


def print_metrics(metrics: Dict, prefix: str = ""):
    """Print metrics in a formatted way."""
    print(f"\n{prefix}Results:")
    print(f"  Exact Match Accuracy: {metrics.get('exact_match_accuracy', 0):.4f}")
    print(f"  Micro F1: {metrics['micro']['f1']:.4f}")
    print(f"  Macro F1: {metrics['macro']['f1']:.4f}")
    print(f"\n  Per-class metrics:")
    for class_name, class_metrics in metrics.items():
        if class_name in ["micro", "macro"] or class_name == "exact_match_accuracy":
            continue
        print(f"    {class_name}:")
        print(f"      Precision: {class_metrics['precision']:.4f}")
        print(f"      Recall: {class_metrics['recall']:.4f}")
        print(f"      F1: {class_metrics['f1']:.4f}")


def save_checkpoint(model, optimizer, epoch, loss, metrics, path):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def train_model(
    model_type: str = "resnet18",
    num_epochs_head: int = 10,
    num_epochs_finetune: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    """
    Train a model with transfer learning strategy.

    Phase 1: Freeze backbone, train only the classification head
    Phase 2: Unfreeze and fine-tune the entire network
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=batch_size)

    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type, num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Loss function (BCEWithLogitsLoss for multi-label)
    criterion = nn.BCEWithLogitsLoss()

    # Metrics
    val_metrics = MultiLabelMetrics(NUM_CLASSES, CLASS_NAMES)

    # Phase 1: Train only the head (freeze backbone)
    print("\n" + "=" * 60)
    print("PHASE 1: Training classification head (backbone frozen)")
    print("=" * 60)

    # Freeze backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    best_val_f1 = 0.0

    for epoch in range(num_epochs_head):
        print(f"\nEpoch {epoch + 1}/{num_epochs_head}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_results = evaluate(model, val_loader, criterion, device, val_metrics)
        print_metrics(val_results, prefix="Validation ")

        # Save best model
        if val_results["micro"]["f1"] > best_val_f1:
            best_val_f1 = val_results["micro"]["f1"]
            save_checkpoint(model, optimizer, epoch, val_loss, val_results,
                          CHECKPOINT_DIR / f"{model_type}_best_head.pth")

    # Phase 2: Fine-tune entire network
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning entire network")
    print("=" * 60)

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=learning_rate / 10)  # Lower LR for fine-tuning
    best_val_f1 = 0.0

    for epoch in range(num_epochs_finetune):
        print(f"\nEpoch {epoch + 1}/{num_epochs_finetune}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_results = evaluate(model, val_loader, criterion, device, val_metrics)
        print_metrics(val_results, prefix="Validation ")

        if val_results["micro"]["f1"] > best_val_f1:
            best_val_f1 = val_results["micro"]["f1"]
            save_checkpoint(model, optimizer, epoch, val_loss, val_results,
                          CHECKPOINT_DIR / f"{model_type}_best_finetuned.pth")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    test_metrics = MultiLabelMetrics(NUM_CLASSES, CLASS_NAMES)
    test_loss, test_results = evaluate(model, test_loader, criterion, device, test_metrics)
    print_metrics(test_results, prefix="Test ")

    # Save final results
    results_path = RESULTS_DIR / f"{model_type}_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return model, test_results


def train_custom_cnn(
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    """
    Train the custom CNN from scratch (baseline).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=batch_size)

    # Create custom CNN
    print("\nCreating custom CNN...")
    from models import CustomCNN
    model = CustomCNN(num_classes=NUM_CLASSES)
    model = model.to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics
    val_metrics = MultiLabelMetrics(NUM_CLASSES, CLASS_NAMES)
    best_val_f1 = 0.0

    print("\n" + "=" * 60)
    print("TRAINING CUSTOM CNN (BASELINE)")
    print("=" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_results = evaluate(model, val_loader, criterion, device, val_metrics)
        print_metrics(val_results, prefix="Validation ")

        if val_results["micro"]["f1"] > best_val_f1:
            best_val_f1 = val_results["micro"]["f1"]
            save_checkpoint(model, optimizer, epoch, val_loss, val_results,
                          CHECKPOINT_DIR / "custom_cnn_best.pth")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    test_metrics = MultiLabelMetrics(NUM_CLASSES, CLASS_NAMES)
    test_loss, test_results = evaluate(model, test_loader, criterion, device, test_metrics)
    print_metrics(test_results, prefix="Test ")

    # Save results
    results_path = RESULTS_DIR / "custom_cnn_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return model, test_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train KIIT-MiTA classifier")
    parser.add_argument("--model", type=str, default="resnet18",
                       choices=["resnet18", "efficientnet", "vit", "custom"],
                       help="Model type to train")
    parser.add_argument("--epochs-head", type=int, default=10,
                       help="Epochs for head training (transfer learning)")
    parser.add_argument("--epochs-finetune", type=int, default=20,
                       help="Epochs for fine-tuning (transfer learning)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Total epochs (custom CNN)")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    if args.model == "custom":
        train_custom_cnn(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        train_model(
            model_type=args.model,
            num_epochs_head=args.epochs_head,
            num_epochs_finetune=args.epochs_finetune,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )
