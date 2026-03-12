"""
KIIT-MiTA Multi-Label Image Classification Package

A deep learning project for classifying military objects in miniature art images.
Supports multi-label classification with 7 classes.
"""

__version__ = "0.1.0"

from .dataset import KiitMitaDataset, load_metadata

# Load metadata
_metadata = load_metadata("data/metadata.json")
METADATA = _metadata
CLASS_NAMES = _metadata["class_names"]
NUM_CLASSES = _metadata["num_classes"]
from .dataloaders import create_dataloaders, TrainTransforms, EvalTransforms
from .models import (
    create_model,
    CustomCNN,
    MultiLabelClassifier,
    BattleNet,
    create_battlenet,
    create_resnet18_classifier,
    create_efficientnet_classifier,
    create_vit_classifier,
)
from .trainer import (
    Trainer,
    MultiLabelMetrics,
    TrackioLogger,
    create_trainer,
)
from .experiment_logger import (
    ExperimentLogger,
    format_results_for_logging,
)

__all__ = [
    # Dataset
    "KiitMitaDataset",
    "load_metadata",
    "METADATA",
    "CLASS_NAMES",
    "NUM_CLASSES",
    # Dataloaders
    "create_dataloaders",
    "TrainTransforms",
    "EvalTransforms",
    # Models
    "create_model",
    "CustomCNN",
    "MultiLabelClassifier",
    "BattleNet",
    "create_battlenet",
    "create_resnet18_classifier",
    "create_efficientnet_classifier",
    "create_vit_classifier",
    # Training
    "Trainer",
    "MultiLabelMetrics",
    "TrackioLogger",
    "create_trainer",
    # Experiment Logging
    "ExperimentLogger",
    "format_results_for_logging",
]
