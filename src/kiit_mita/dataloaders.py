"""
Data transforms and dataloaders for KIIT-MiTA multi-label classification.

Supports multiple augmentation strategies for experimentation.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageEnhance, ImageFilter

from .dataset import KiitMitaDataset


# Dataset paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = str(PROJECT_ROOT / "KIIT-MiTA")
DATA_DIR = str(PROJECT_ROOT / "data")

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation strategy types
AugmentationStrategy = Literal[
    "none",
    "baseline",
    "strong",
    "light",
    "color",
    "geometric",
    "aggressive",
]

# Augmentation strategy descriptions
AUGMENTATION_DESCRIPTIONS = {
    "none": "No augmentation - only resize and normalize",
    "baseline": "Standard: random crop (0.8-1.0), h-flip (50%), rotation (-15 to +15)",
    "strong": "Stronger: random crop (0.6-1.0), h-flip (50%), rotation (-30 to +30), v-flip (20%)",
    "light": "Light: random crop (0.9-1.0), h-flip (30%), rotation (-5 to +5)",
    "color": "Color-focused: baseline + color jitter (brightness, contrast, saturation)",
    "geometric": "Geometric-focused: baseline + stronger rotation (-45 to +45) + v-flip",
    "aggressive": "Aggressive: strong + color jitter + gaussian blur",
}


class TrainTransforms:
    """Training transforms with configurable augmentation strategy."""

    def __init__(
        self,
        image_size: int = 224,
        augmentation: AugmentationStrategy = "baseline",
    ):
        self.image_size = image_size
        self.augmentation = augmentation
        self._setup_augmentation()

    def _setup_augmentation(self):
        """Configure augmentation parameters based on strategy."""
        # Default (baseline) parameters
        self.crop_scale = (0.8, 1.0)
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.0
        self.rotation_range = 15
        self.color_jitter = False
        self.gaussian_blur = False

        if self.augmentation == "none":
            self.crop_scale = (1.0, 1.0)
            self.h_flip_prob = 0.0
            self.rotation_range = 0

        elif self.augmentation == "strong":
            self.crop_scale = (0.6, 1.0)
            self.h_flip_prob = 0.5
            self.v_flip_prob = 0.2
            self.rotation_range = 30

        elif self.augmentation == "light":
            self.crop_scale = (0.9, 1.0)
            self.h_flip_prob = 0.3
            self.rotation_range = 5

        elif self.augmentation == "color":
            self.color_jitter = True

        elif self.augmentation == "geometric":
            self.rotation_range = 45
            self.v_flip_prob = 0.3

        elif self.augmentation == "aggressive":
            self.crop_scale = (0.6, 1.0)
            self.h_flip_prob = 0.5
            self.v_flip_prob = 0.2
            self.rotation_range = 30
            self.color_jitter = True
            self.gaussian_blur = True

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Apply augmentation
        img = self._apply_augmentation(img)

        # Convert to tensor and normalize
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0

        # Normalize (H, W, C) -> (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        for i in range(3):
            img_array[i] = (img_array[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

        return torch.from_numpy(img_array)

    def _apply_augmentation(self, img: Image.Image) -> Image.Image:
        """Apply augmentation strategy to image."""
        # Random resized crop
        img = self._random_resized_crop(img, scale=self.crop_scale)

        # Random horizontal flip
        if np.random.random() < self.h_flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if np.random.random() < self.v_flip_prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Random rotation
        if self.rotation_range > 0 and np.random.random() > 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            img = img.rotate(angle, expand=False, fillcolor=(124, 116, 104))

        # Color jitter
        if self.color_jitter and np.random.random() > 0.5:
            # Brightness
            factor = np.random.uniform(0.7, 1.3)
            img = ImageEnhance.Brightness(img).enhance(factor)
            # Contrast
            factor = np.random.uniform(0.7, 1.3)
            img = ImageEnhance.Contrast(img).enhance(factor)
            # Saturation
            factor = np.random.uniform(0.7, 1.3)
            img = ImageEnhance.Color(img).enhance(factor)

        # Gaussian blur
        if self.gaussian_blur and np.random.random() > 0.7:
            img = img.filter(
                ImageFilter.GaussianBlur(radius=np.random.uniform(0.1, 2.0))
            )

        return img

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

    def get_description(self) -> str:
        """Get human-readable description of the augmentation strategy."""
        return AUGMENTATION_DESCRIPTIONS.get(self.augmentation, "Unknown")


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
    augmentation: AugmentationStrategy = "baseline",
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train, validation, and test dataloaders.

    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Target image size
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation: Augmentation strategy for training data

    Returns:
        train_loader, val_loader, test_loader, augmentation_info
    """
    # Create transforms
    train_transform = TrainTransforms(image_size, augmentation=augmentation)
    eval_transform = EvalTransforms(image_size)

    # Create datasets
    train_dataset = KiitMitaDataset(
        annotations_path=f"{DATA_DIR}/train_annotations.json",
        dataset_root=DATASET_ROOT,
        transform=train_transform,
    )

    val_dataset = KiitMitaDataset(
        annotations_path=f"{DATA_DIR}/test_annotations.json",  # Using test as val
        dataset_root=DATASET_ROOT,
        transform=eval_transform,
    )

    test_dataset = KiitMitaDataset(
        annotations_path=f"{DATA_DIR}/valid_annotations.json",
        dataset_root=DATASET_ROOT,
        transform=eval_transform,
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

    # Augmentation info for logging
    augmentation_info = {
        "strategy": augmentation,
        "description": train_transform.get_description(),
        "crop_scale": train_transform.crop_scale,
        "h_flip_prob": train_transform.h_flip_prob,
        "v_flip_prob": train_transform.v_flip_prob,
        "rotation_range": train_transform.rotation_range,
        "color_jitter": train_transform.color_jitter,
        "gaussian_blur": train_transform.gaussian_blur,
    }

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Num classes: {train_dataset.num_classes}")
    print(f"Augmentation: {augmentation}")
    print(f"  → {train_transform.get_description()}")

    return train_loader, val_loader, test_loader, augmentation_info


if __name__ == "__main__":
    # Test dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=8)

    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label example: {labels[0]}")
