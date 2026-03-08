"""
PyTorch Dataset for KIIT-MiTA multi-label classification.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class KiitMitaDataset(Dataset):
    """
    PyTorch Dataset for KIIT-MiTA multi-label image classification.

    Args:
        annotations_path: Path to JSON file containing image annotations
        dataset_root: Root directory of the KIIT-MiTA dataset
        transform: Optional transform to apply to images
    """

    def __init__(
        self,
        annotations_path: str,
        dataset_root: str,
        transform: Optional[Callable] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, torch.Tensor]:
        """
        Get an item from the dataset.

        Returns:
            image: PIL Image
            label: Tensor of shape (num_classes,) with multi-hot encoded labels
        """
        annotation = self.annotations[idx]

        # Load image
        image_path = self.dataset_root / annotation["image_path"]
        image = Image.open(image_path).convert("RGB")

        # Get label (multi-hot encoded)
        label = torch.tensor(annotation["label"], dtype=torch.float32)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def get_class_names() -> List[str]:
        """Return the list of class names."""
        return [
            "Artilary",
            "Missile",
            "Radar",
            "M. Rocket Launcher",
            "Soldier",
            "Tank",
            "Vehicle"
        ]

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.get_class_names())


def load_metadata(metadata_path: str) -> Dict:
    """Load metadata file containing dataset statistics."""
    with open(metadata_path, 'r') as f:
        return json.load(f)
