"""
CNN models for KIIT-MiTA multi-label classification.

Includes:
1. Transfer learning models (ResNet, EfficientNet, ViT)
2. Custom CNN baseline
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelClassifier(nn.Module):
    """
    Multi-label classification head on top of a backbone.

    Args:
        backbone: Feature extractor (e.g., ResNet, EfficientNet)
        num_classes: Number of output classes (7 for KIIT-MiTA)
        dropout_rate: Dropout probability for the classifier head
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 7, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = backbone

        # Get the number of features from the backbone
        if hasattr(backbone, 'fc'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove original classification head
        elif hasattr(backbone, 'classifier'):
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported backbone architecture")

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for multi-label classification baseline.

    Architecture:
    - 4 convolutional blocks with batch normalization and pooling
    - Global average pooling
    - 2 fully connected layers with dropout
    """

    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.3):
        super().__init__()

        # Conv Block 1: 3 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Conv Block 2: 64 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Conv Block 3: 128 -> 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Conv Block 4: 256 -> 512
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def create_resnet18_classifier(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create a ResNet-18 based classifier with transfer learning."""
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)
    except ImportError:
        # Fallback: create a simple ResNet-like backbone
        backbone = SimpleResNetBackbone()
    return MultiLabelClassifier(backbone, num_classes)


def create_efficientnet_classifier(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create an EfficientNet-B0 based classifier with transfer learning."""
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        if pretrained:
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b0(weights=None)
    except ImportError:
        # Fallback to custom CNN
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


def create_vit_classifier(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create a Vision Transformer based classifier with transfer learning."""
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        if pretrained:
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            backbone = vit_b_16(weights=None)
    except ImportError:
        # Fallback to custom CNN
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


class SimpleResNetBackbone(nn.Module):
    """Simple ResNet-like backbone for when torchvision is unavailable."""

    def __init__(self):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Residual blocks (simplified)
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Placeholder fc (will be replaced by MultiLabelClassifier)
        self.fc = nn.Linear(512, 7)

    def _make_layer(self, in_channels, out_channels, blocks):
        """Create a residual layer."""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def create_model(model_type: str = "resnet18", num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('resnet18', 'efficientnet', 'vit', 'custom')
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights (for transfer learning models)

    Returns:
        Model instance
    """
    models_map = {
        "resnet18": create_resnet18_classifier,
        "efficientnet": create_efficientnet_classifier,
        "vit": create_vit_classifier,
        "custom": lambda **kwargs: CustomCNN(num_classes),
    }

    if model_type not in models_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models_map.keys())}")

    return models_map[model_type](num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    # Test custom CNN
    custom_cnn = CustomCNN(num_classes=7)
    x = torch.randn(2, 3, 224, 224)
    out = custom_cnn(x)
    print(f"Custom CNN output shape: {out.shape}")

    # Test ResNet (with fallback)
    resnet_model = create_model("resnet18", num_classes=7)
    out = resnet_model(x)
    print(f"ResNet model output shape: {out.shape}")

    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nCustom CNN parameters: {count_params(custom_cnn):,}")
    print(f"ResNet model parameters: {count_params(resnet_model):,}")
