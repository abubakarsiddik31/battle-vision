"""
Training and evaluation module with trackio integration for experiment tracking.
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

try:
    import trackio

    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False

from .dataloaders import create_dataloaders
from .dataset import KiitMitaDataset, load_metadata
from .models import create_model
from .experiment_logger import ExperimentLogger, format_results_for_logging


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α_t(1 - p_t)^γ log(p_t)

    Where:
    - p_t is the model's estimated probability for the correct class
    - γ (gamma) is the focusing parameter (default: 2.0)
    - α (alpha) is the weighting factor (not implemented here, using pos_weight instead)
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size, num_classes)

        Returns:
            Focal loss value
        """
        # Compute binary cross entropy with logits (without reduction)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=self.pos_weight
        )

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t: probability of the true class
        p_t = probs * labels + (1 - probs) * (1 - labels)

        # Compute focal loss: (1 - p_t)^gamma * BCE
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce

        return focal_loss.mean()


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# Load metadata
METADATA = load_metadata(str(PROJECT_ROOT / "data" / "metadata.json"))
CLASS_NAMES = METADATA["class_names"]
NUM_CLASSES = METADATA["num_classes"]


class TrackioLogger:
    """Wrapper for trackio logging with graceful fallback."""

    def __init__(self, project: str, config: Optional[Dict] = None):
        self.enabled = TRACKIO_AVAILABLE
        if self.enabled:
            trackio.init(project=project, config=config)
            print(f"Trackio logging enabled for project: {project}")
            print(f"View dashboard with: trackio show --project {project}")
        else:
            print("trackio not available. Using console logging only.")

    def log(self, metrics: Dict, step: Optional[int] = None):
        if self.enabled:
            trackio.log(metrics)
        else:
            metrics_str = " | ".join(
                [
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in metrics.items()
                ]
            )
            print(f"  [Step {step}] {metrics_str}" if step else f"  {metrics_str}")

    def finish(self):
        if self.enabled:
            trackio.finish()


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
        """Update metrics with a batch."""
        preds = (probs > 0.5).float()
        self.all_preds.append(preds.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())
        self.all_probs.append(probs.detach().cpu().numpy())

    def compute(self) -> Dict:
        """Compute all metrics."""
        all_preds = np.concatenate(self.all_preds, axis=0)
        all_labels = np.concatenate(self.all_labels, axis=0)

        metrics = {}
        for i, class_name in enumerate(self.class_names):
            tp = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 1))
            fp = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 0))
            fn = np.sum((all_preds[:, i] == 0) & (all_labels[:, i] == 1))
            tn = np.sum((all_preds[:, i] == 0) & (all_labels[:, i] == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = (
                (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            )

            metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }

        # Micro & macro averages
        total_tp = np.sum((all_preds == 1) & (all_labels == 1))
        total_fp = np.sum((all_preds == 1) & (all_labels == 0))
        total_fn = np.sum((all_preds == 0) & (all_labels == 1))
        total_tn = np.sum((all_preds == 0) & (all_labels == 0))

        metrics["micro"] = {
            "precision": float(total_tp / (total_tp + total_fp))
            if (total_tp + total_fp) > 0
            else 0.0,
            "recall": float(total_tp / (total_tp + total_fn))
            if (total_tp + total_fn) > 0
            else 0.0,
            "f1": float(2 * total_tp / (2 * total_tp + total_fp + total_fn))
            if (2 * total_tp + total_fp + total_fn) > 0
            else 0.0,
            "accuracy": float(
                (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
            ),
        }
        metrics["macro"] = {
            "precision": float(
                np.mean([metrics[c]["precision"] for c in self.class_names])
            ),
            "recall": float(np.mean([metrics[c]["recall"] for c in self.class_names])),
            "f1": float(np.mean([metrics[c]["f1"] for c in self.class_names])),
        }
        metrics["exact_match_accuracy"] = float(
            np.all(all_preds == all_labels, axis=1).mean()
        )

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


class Trainer:
    """
    Trainer class with trackio and experiment logging for multi-label classification.

    Supports:
    - Transfer learning with freeze/fine-tune phases
    - Automatic checkpointing
    - trackio experiment tracking (local-first, free)
    - Comprehensive experiment logging with notes
    - Iteration tracking for tweaks
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        use_trackio: bool = True,
        trackio_project: str = "kiit-mita-classification",
        experiment_name: Optional[str] = None,
        experiment_notes: Optional[str] = None,
        experiment_tags: Optional[List[str]] = None,
        loss_type: str = "bce",
        pos_weight: float = 1.0,
        class_weights: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model_name = model_name

        # Create loss function
        self.criterion = self._create_criterion(
            loss_type=loss_type,
            pos_weight=pos_weight,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )

        # Metrics
        self.train_metrics = MultiLabelMetrics(NUM_CLASSES, CLASS_NAMES)
        self.val_metrics = MultiLabelMetrics(NUM_CLASSES, CLASS_NAMES)

        # Experiment Logger
        self.experiment_logger = ExperimentLogger(
            project=trackio_project,
            experiment_name=experiment_name
            or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            notes=experiment_notes,
            tags=experiment_tags or [model_name],
        )

        # Log initial config
        self.experiment_logger.log_config(
            {
                "model": model_name,
                "learning_rate": learning_rate,
                "num_classes": NUM_CLASSES,
                "classes": CLASS_NAMES,
                "device": str(self.device),
            }
        )

        # Trackio
        self.use_trackio = use_trackio and TRACKIO_AVAILABLE
        self.trackio_logger = None
        if self.use_trackio:
            self.trackio_logger = TrackioLogger(
                project=trackio_project,
                config={
                    "model": model_name,
                    "learning_rate": learning_rate,
                    "num_classes": NUM_CLASSES,
                    "classes": CLASS_NAMES,
                    "experiment_id": self.experiment_logger.experiment_id,
                },
            )

        print(f"Trainer initialized on {self.device}")
        print(f"Experiment ID: {self.experiment_logger.experiment_id}")

    def _create_criterion(
        self,
        loss_type: str,
        pos_weight: float,
        class_weights: Optional[List[float]],
        focal_gamma: float,
    ) -> nn.Module:
        """
        Create the loss function based on specified parameters.

        Args:
            loss_type: Type of loss ('bce' or 'focal')
            pos_weight: Positive weight for BCE loss (upweights positive examples)
            class_weights: Per-class weights (for auto class weighting)
            focal_gamma: Gamma parameter for focal loss

        Returns:
            Loss function module
        """
        # Handle pos_weight (convert to tensor)
        pos_weight_tensor = None
        if pos_weight != 1.0:
            pos_weight_tensor = torch.tensor(
                [pos_weight] * NUM_CLASSES, device=self.device
            )
            print(f"Using pos_weight={pos_weight} for all classes")

        # Handle class_weights (auto or manual)
        if class_weights == "auto":
            # Compute inverse frequency weights
            class_weights_tensor = self._compute_class_weights()
            print(f"Using auto class weights: {class_weights_tensor.tolist()}")
            pos_weight_tensor = class_weights_tensor
        elif class_weights is not None:
            pos_weight_tensor = torch.tensor(class_weights, device=self.device)
            print(f"Using manual class weights: {class_weights}")

        # Create loss function
        if loss_type == "focal":
            print(f"Using Focal Loss with gamma={focal_gamma}")
            criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight_tensor)
        else:  # Default: BCEWithLogitsLoss
            if pos_weight_tensor is not None:
                print(f"Using BCEWithLogitsLoss with pos_weight")
            else:
                print(f"Using BCEWithLogitsLoss (standard)")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        return criterion

    def _compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights based on inverse frequency.

        Weight for class i = total_samples / (num_classes * samples_in_class_i)

        Returns:
            Tensor of shape (num_classes,) with weights for each class
        """
        # Load training data to compute class frequencies
        train_dataset = KiitMitaDataset(
            annotations_path=f"{PROJECT_ROOT}/data/train_annotations.json",
            dataset_root=str(PROJECT_ROOT / "KIIT-MiTA"),
            transform=None,  # Don't need transforms for counting
        )

        # Count positive examples for each class
        class_counts = torch.zeros(NUM_CLASSES)
        for _, labels in train_dataset:
            class_counts += labels

        # Compute inverse frequency weights
        total_samples = len(train_dataset)
        weights = total_samples / (
            NUM_CLASSES * (class_counts + 1e-6)
        )  # Add epsilon to avoid div by zero

        # Normalize weights to have mean of 1.0 (helps with stability)
        weights = weights / weights.mean()

        return weights.to(self.device)

    def add_note(self, note: str):
        """Add a note to the experiment log."""
        self.experiment_logger.add_note(note)
        if self.trackio_logger:
            # Log note to trackio as well
            self.trackio_logger.log({"note": note})

    def add_tag(self, tag: str):
        """Add a tag to the experiment."""
        self.experiment_logger.add_tag(tag)

    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str,
        total_epochs: int,
    ):
        """
        Create a learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule
            scheduler_type: Type of scheduler ('step', 'cosine', 'onecycle', 'plateau')
            total_epochs: Total number of epochs for scheduling

        Returns:
            Scheduler instance or None
        """
        if scheduler_type == "step":
            # Step LR: reduce by 0.1 every 10 epochs
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, total_epochs // 3),
                gamma=0.1,
            )
        elif scheduler_type == "cosine":
            # Cosine annealing
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=1e-6,
            )
        elif scheduler_type == "onecycle":
            # OneCycle policy (simulated with CyclicLR)
            return optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=optimizer.param_groups[0]["lr"] / 10,
                max_lr=optimizer.param_groups[0]["lr"] * 10,
                step_size_up=total_epochs // 2,
                mode="triangular",
            )
        elif scheduler_type == "plateau":
            # Reduce on plateau
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )
        else:
            return None

    def train_one_epoch(
        self, train_loader: DataLoader, optimizer: optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            probs = torch.sigmoid(logits)
            self.train_metrics.update(probs, labels)
            total_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(
        self, dataloader: DataLoader, metrics: MultiLabelMetrics
    ) -> tuple[float, Dict]:
        """Evaluate model on a dataset."""
        self.model.eval()
        metrics.reset()
        total_loss = 0.0

        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * images.size(0)
            metrics.update(probs, labels)

        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss, metrics.compute()

    def log_metrics(
        self, epoch: int, phase: str, loss: float, metrics: Dict, step: int
    ):
        """Log metrics to console, trackio, and experiment logger."""
        prefix = f"[{phase.upper()}]" if phase else ""
        print(f"\n{prefix} Epoch {epoch}: Loss={loss:.4f}")
        print(f"  Exact Match Acc: {metrics['exact_match_accuracy']:.4f}")
        print(
            f"  Micro F1: {metrics['micro']['f1']:.4f} | Macro F1: {metrics['macro']['f1']:.4f}"
        )

        # Log to trackio
        if self.trackio_logger:
            log_dict = {
                f"{phase}/loss": loss,
                f"{phase}/exact_match_accuracy": metrics["exact_match_accuracy"],
                f"{phase}/micro_f1": metrics["micro"]["f1"],
                f"{phase}/macro_f1": metrics["macro"]["f1"],
                f"{phase}/micro_precision": metrics["micro"]["precision"],
                f"{phase}/micro_recall": metrics["micro"]["recall"],
                "epoch": epoch,
            }
            # Log per-class metrics
            for class_name in CLASS_NAMES:
                if class_name in metrics:
                    log_dict[f"{phase}/{class_name}_f1"] = metrics[class_name]["f1"]
            self.trackio_logger.log(log_dict)

        # Log to experiment logger
        epoch_metrics = {
            "epoch": epoch,
            "phase": phase,
            "loss": loss,
            **format_results_for_logging(metrics),
        }
        self.experiment_logger.log_metrics(epoch_metrics, step=step)

    def save_checkpoint(self, epoch: int, loss: float, metrics: Dict, phase: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss": loss,
            "metrics": metrics,
            "model_name": self.model_name,
            "experiment_id": self.experiment_logger.experiment_id,
        }
        path = CHECKPOINT_DIR / f"{self.model_name}_{phase}_best.pth"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs_head: int = 10,
        num_epochs_finetune: int = 20,
        learning_rate: float = 1e-3,
        scheduler_type: Optional[str] = None,
    ):
        """Train with transfer learning strategy."""
        global_step = 0

        # Log training config
        self.experiment_logger.log_config(
            {
                "num_epochs_head": num_epochs_head,
                "num_epochs_finetune": num_epochs_finetune,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "scheduler": scheduler_type or "none",
            }
        )

        # Phase 1: Train head only
        print("\n" + "=" * 60)
        print("PHASE 1: Training classification head (backbone frozen)")
        print("=" * 60)

        self.experiment_logger.start_iteration(
            "head_training",
            {
                "phase": "head",
                "epochs": num_epochs_head,
                "lr": learning_rate,
                "scheduler": scheduler_type or "none",
            },
        )

        for param in self.model.backbone.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        scheduler = (
            self._create_scheduler(optimizer, scheduler_type, num_epochs_head)
            if scheduler_type
            else None
        )
        if scheduler:
            print(f"Using scheduler: {scheduler_type}")

        best_val_f1 = 0.0

        for epoch in range(num_epochs_head):
            train_loss = self.train_one_epoch(train_loader, optimizer)
            train_metrics = self.train_metrics.compute()

            val_loss, val_metrics_dict = self.evaluate(val_loader, self.val_metrics)

            self.log_metrics(epoch + 1, "head", val_loss, val_metrics_dict, global_step)
            self.experiment_logger.log_iteration_metrics(val_metrics_dict)

            if val_metrics_dict["micro"]["f1"] > best_val_f1:
                best_val_f1 = val_metrics_dict["micro"]["f1"]
                self.save_checkpoint(epoch, val_loss, val_metrics_dict, "head")

            if scheduler:
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                if epoch % 5 == 0 or epoch == num_epochs_head - 1:
                    print(f"  LR after epoch {epoch + 1}: {current_lr:.6f}")

            global_step += 1

        self.experiment_logger.finish_iteration({"best_val_f1": best_val_f1})

        # Phase 2: Fine-tune entire network
        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning entire network")
        print("=" * 60)

        self.experiment_logger.start_iteration(
            "finetuning",
            {
                "phase": "finetune",
                "epochs": num_epochs_finetune,
                "lr": learning_rate / 10,
                "scheduler": scheduler_type or "none",
            },
        )

        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate / 10)

        scheduler = (
            self._create_scheduler(optimizer, scheduler_type, num_epochs_finetune)
            if scheduler_type
            else None
        )
        if scheduler:
            print(f"Using scheduler: {scheduler_type}")
        best_val_f1 = 0.0

        for epoch in range(num_epochs_finetune):
            train_loss = self.train_one_epoch(train_loader, optimizer)
            train_metrics = self.train_metrics.compute()

            val_loss, val_metrics_dict = self.evaluate(val_loader, self.val_metrics)

            self.log_metrics(
                epoch + 1, "finetune", val_loss, val_metrics_dict, global_step
            )
            self.experiment_logger.log_iteration_metrics(val_metrics_dict)

            if val_metrics_dict["micro"]["f1"] > best_val_f1:
                best_val_f1 = val_metrics_dict["micro"]["f1"]
                self.save_checkpoint(epoch, val_loss, val_metrics_dict, "finetune")

            if scheduler:
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                if epoch % 5 == 0 or epoch == num_epochs_finetune - 1:
                    print(f"  LR after epoch {epoch + 1}: {current_lr:.6f}")

            global_step += 1

        self.experiment_logger.finish_iteration({"best_val_f1": best_val_f1})

        if self.trackio_logger:
            self.trackio_logger.finish()

        return self.model

    def train_from_scratch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
    ):
        """Train model from scratch (for custom CNN)."""
        print("\n" + "=" * 60)
        print(f"TRAINING {self.model_name.upper()} (FROM SCRATCH)")
        print("=" * 60)

        # Log training config
        self.experiment_logger.log_config(
            {
                "num_epochs": num_epochs,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
            }
        )

        self.experiment_logger.start_iteration(
            "from_scratch", {"epochs": num_epochs, "lr": learning_rate}
        )

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_f1 = 0.0
        global_step = 0

        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(train_loader, optimizer)
            train_metrics = self.train_metrics.compute()

            val_loss, val_metrics_dict = self.evaluate(val_loader, self.val_metrics)

            self.log_metrics(epoch + 1, "", val_loss, val_metrics_dict, global_step)
            self.experiment_logger.log_iteration_metrics(val_metrics_dict)

            if val_metrics_dict["micro"]["f1"] > best_val_f1:
                best_val_f1 = val_metrics_dict["micro"]["f1"]
                self.save_checkpoint(epoch, val_loss, val_metrics_dict, "best")

            global_step += 1

        self.experiment_logger.finish_iteration({"best_val_f1": best_val_f1})

        if self.trackio_logger:
            self.trackio_logger.finish()

        return self.model

    def finish_experiment(self, test_results: Optional[Dict] = None):
        """Finish the experiment and save all results."""
        print("\n" + "=" * 60)
        print("FINISHING EXPERIMENT")
        print("=" * 60)

        summary, exp_file = self.experiment_logger.finish(test_results)
        print(f"Experiment saved to: {exp_file}")

        # Print summary
        self.experiment_logger.print_summary()

        # Update markdown summary for git tracking
        try:
            import subprocess

            script_path = PROJECT_ROOT / "scripts" / "update_experiments_summary.py"
            if script_path.exists():
                subprocess.run(
                    ["python", str(script_path)], check=False, cwd=PROJECT_ROOT
                )
                summary_path = PROJECT_ROOT / "results" / "experiments" / "SUMMARY.md"
                print(f"\nMarkdown summary updated: {summary_path}")
                print(
                    f"Git command: git add results/experiments/ && git commit -m 'Update experiments'"
                )
        except Exception as e:
            print(f"Note: Could not update markdown summary: {e}")

        return summary, exp_file


def create_trainer(
    model_type: str = "resnet18",
    learning_rate: float = 1e-3,
    use_trackio: bool = True,
    device: Optional[str] = None,
    experiment_name: Optional[str] = None,
    experiment_notes: Optional[str] = None,
    experiment_tags: Optional[List[str]] = None,
) -> Trainer:
    """Factory function to create a trainer with a model."""
    model = create_model(model_type, num_classes=NUM_CLASSES, pretrained=True)
    return Trainer(
        model,
        model_type,
        device,
        learning_rate,
        use_trackio,
        experiment_name=experiment_name,
        experiment_notes=experiment_notes,
        experiment_tags=experiment_tags,
    )


# Import datetime for experiment naming
from datetime import datetime
