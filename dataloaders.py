"""
Data transforms and dataloaders for KIIT-MiTA multi-label classification.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from dataset import KiitMitaDataset


# Dataset paths
DATASET_ROOT = "/home/abubakar/Desktop/Research/DL-assignment/KIIT-MiTA"
DATA_DIR = "/home/abubakar/Desktop/Research/DL-assignment/data"

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainTransforms:
    """Training transforms with augmentation."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Random resized crop
        img = self._random_resized_crop(img, scale=(0.8, 1.0))

        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            img = img.rotate(angle, expand=False, fillcolor=(124, 116, 104))

        # Convert to tensor and normalize
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0

        # Normalize (H, W, C) -> (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        for i in range(3):
            img_array[i] = (img_array[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

        return torch.from_numpy(img_array)

    def _random_resized_crop(self, img: Image.Image, scale=(0.8, 1.0)):
        """Simulate random resized crop."""
        w, h = img.size
        target_ratio = scale[0] + np.random.random() * (scale[1] - scale[0])
        new_w = int(w * target_ratio)
        new_h = int(h * target_ratio)

        # Random crop position
        left = np.random.randint(0, max(1, w - new_w))
        top = np.random.randint(0, max(1, h - new_h))

        return img.crop((left, top, left + new_w, top + new_h))


class EvalTransforms:
    """Evaluation transforms (no augmentation)."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Resize and convert to tensor
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0

        # Normalize (H, W, C) -> (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        for i in range(3):
            img_array[i] = (img_array[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

        return torch.from_numpy(img_array)


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Target image size
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = KiitMitaDataset(
        annotations_path=f"{DATA_DIR}/train_annotations.json",
        dataset_root=DATASET_ROOT,
        transform=TrainTransforms(image_size),
    )

    val_dataset = KiitMitaDataset(
        annotations_path=f"{DATA_DIR}/test_annotations.json",  # Using test as val
        dataset_root=DATASET_ROOT,
        transform=EvalTransforms(image_size),
    )

    test_dataset = KiitMitaDataset(
        annotations_path=f"{DATA_DIR}/valid_annotations.json",
        dataset_root=DATASET_ROOT,
        transform=EvalTransforms(image_size),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Num classes: {train_dataset.num_classes}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=8)

    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label example: {labels[0]}")
