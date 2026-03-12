"""
CNN models for KIIT-MiTA multi-label classification.

Includes:
1. Transfer learning models (ResNet, EfficientNet, ViT)
2. Custom CNN baseline
3. BattleNet - Novel dual-backbone hybrid fusion architecture
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
        self.use_avgpool = False

        # Get the number of features from the backbone
        if hasattr(backbone, 'fc'):
            # ResNet style
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove original classification head
        elif hasattr(backbone, 'head'):
            # ViT/Swin style - has head attribute
            in_features = backbone.head.in_features
            backbone.head = nn.Identity()
        elif hasattr(backbone, 'heads'):
            # Some models use heads (plural)
            if hasattr(backbone.heads, '__getitem__'):
                in_features = backbone.heads[0].in_features
            else:
                in_features = backbone.heads.in_features
            backbone.heads = nn.Identity()
        elif hasattr(backbone, 'classifier'):
            # Check if it's VGG (classifier is Sequential with first layer taking 25088)
            if isinstance(backbone.classifier, nn.Sequential) and len(backbone.classifier) > 0:
                first_layer = backbone.classifier[0]
                if isinstance(first_layer, nn.Linear) and first_layer.in_features == 25088:
                    # VGG style - use features part only
                    in_features = 25088  # VGG16/19 feature size
                    # Keep avgpool, remove classifier
                    backbone.classifier = nn.Identity()
                    self.use_avgpool = True
                else:
                    # EfficientNet/MobileNet style
                    in_features = backbone.classifier[-1].in_features
                    backbone.classifier = nn.Identity()
            else:
                in_features = backbone.classifier.in_features
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
        # VGG returns features without flattening, need to flatten
        if features.dim() > 2 and not self.use_avgpool:
            features = torch.flatten(features, 1)
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


def create_resnet_variant(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create ResNet variants (18, 34, 50)."""
    try:
        if variant == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")
    except ImportError:
        backbone = SimpleResNetBackbone()
    return MultiLabelClassifier(backbone, num_classes)


def create_efficientnet_variant(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create EfficientNet variants (b0, b1)."""
    try:
        if variant == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "efficientnet_b1":
            from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
            backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")
    except ImportError:
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


def create_vit_variant(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create Vision Transformer variants (b_16, b_32)."""
    try:
        if variant == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "vit_b_32":
            from torchvision.models import vit_b_32, ViT_B_32_Weights
            backbone = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown ViT variant: {variant}")
    except ImportError:
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


def create_vgg_variant(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create VGG variants (vgg11, vgg13, vgg16, vgg19)."""
    try:
        if variant == "vgg11":
            from torchvision.models import vgg11, VGG11_Weights
            backbone = vgg11(weights=VGG11_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "vgg13":
            from torchvision.models import vgg13, VGG13_Weights
            backbone = vgg13(weights=VGG13_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "vgg16":
            from torchvision.models import vgg16, VGG16_Weights
            backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "vgg19":
            from torchvision.models import vgg19, VGG19_Weights
            backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown VGG variant: {variant}")
    except ImportError:
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


def create_densenet_variant(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create DenseNet variants (densenet121, densenet161, densenet169, densenet201)."""
    try:
        if variant == "densenet121":
            from torchvision.models import densenet121, DenseNet121_Weights
            backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "densenet161":
            from torchvision.models import densenet161, DenseNet161_Weights
            backbone = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "densenet169":
            from torchvision.models import densenet169, DenseNet169_Weights
            backbone = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "densenet201":
            from torchvision.models import densenet201, DenseNet201_Weights
            backbone = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown DenseNet variant: {variant}")
    except ImportError:
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


def create_mobilenet_variant(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create MobileNet variants (mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large)."""
    try:
        if variant == "mobilenet_v2":
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "mobilenet_v3_small":
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "mobilenet_v3_large":
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown MobileNet variant: {variant}")
    except ImportError:
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


def create_convnext_variant(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create ConvNeXt variants (convnext_tiny, convnext_small, convnext_base)."""
    try:
        if variant == "convnext_tiny":
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "convnext_small":
            from torchvision.models import convnext_small, ConvNeXt_Small_Weights
            backbone = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "convnext_base":
            from torchvision.models import convnext_base, ConvNeXt_Base_Weights
            backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")
    except ImportError:
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


def create_swin_variant(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create Swin Transformer variants (swin_t, swin_s, swin_b)."""
    try:
        if variant == "swin_t":
            from torchvision.models import swin_t, Swin_T_Weights
            backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "swin_s":
            from torchvision.models import swin_s, Swin_S_Weights
            backbone = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "swin_b":
            from torchvision.models import swin_b, Swin_B_Weights
            backbone = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown Swin variant: {variant}")
    except ImportError:
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


class SqueezeExcitation(nn.Module):
    """Channel-wise SE recalibration: learns to amplify useful feature channels."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class LabelCoOccurrence(nn.Module):
    """
    Learnable label co-occurrence module.

    After the direct head produces per-class logits, this module refines them
    by allowing each class prediction to be informed by all other class predictions
    via a learned interaction matrix. This is applied to logits (not features),
    so scale is controlled and there's no explosion risk.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # Init as identity + small perturbation — starts as pass-through
        self.W = nn.Parameter(torch.eye(num_classes) + torch.randn(num_classes, num_classes) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, C) — apply learned label interaction
        return logits + torch.matmul(logits, self.W.t()) * 0.1 + self.bias


class BattleNetBackbone(nn.Module):
    """
    Dual-backbone module: EfficientNet-V2-S (CNN) + Swin-T (Transformer).

    Treated as the frozen 'backbone' during Phase 1 training, then
    unfrozen for end-to-end fine-tuning in Phase 2.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # CNN Branch: EfficientNet-V2-S — strong local texture/shape features
        try:
            from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
            self.cnn = efficientnet_v2_s(
                weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.cnn_dim = self.cnn.classifier[-1].in_features  # 1280
            self.cnn.classifier = nn.Identity()
        except ImportError:
            raise RuntimeError("torchvision required for BattleNet")

        # Transformer Branch: Swin-T — strong global context features
        try:
            from torchvision.models import swin_t, Swin_T_Weights
            self.transformer = swin_t(
                weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.trans_dim = self.transformer.head.in_features  # 768
            self.transformer.head = nn.Identity()
        except ImportError:
            raise RuntimeError("torchvision required for BattleNet")

    def forward(self, x: torch.Tensor):
        feat_cnn = self.cnn(x)        # (B, 1280)
        feat_trans = self.transformer(x)  # (B, 768)
        return feat_cnn, feat_trans


class BattleNetClassifier(nn.Module):
    """
    BattleNet fusion head (v2):
    - Projects both features to shared dim
    - Bilinear interaction (Hadamard product of projections) captures cross-modal correlations
    - SE channel recalibration on fused features
    - Label co-occurrence refinement on logits
    - Ensemble: direct head + co-occurrence head

    All stable by design: no raw dot-product attention, proper normalization throughout.
    """

    def __init__(self, cnn_dim: int, trans_dim: int, num_classes: int = 7,
                 fusion_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        # Project CNN features to fusion_dim
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Project Transformer features to fusion_dim
        self.trans_proj = nn.Sequential(
            nn.Linear(trans_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Bilinear fusion: combine additive, multiplicative, and concatenated signals
        # [f_cnn + f_trans, f_cnn * f_trans] -> fusion_dim via MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),  # [cnn, trans, cnn*trans]
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # SE channel recalibration — learns which fused channels are most discriminative
        self.se = SqueezeExcitation(fusion_dim, reduction=16)

        # Direct multi-label head
        self.direct_head = nn.Linear(fusion_dim, num_classes)

        # Label co-occurrence refinement (applied on logits — no scale explosion)
        self.label_cooccur = LabelCoOccurrence(num_classes)

    def forward(self, backbone_output):
        feat_cnn, feat_trans = backbone_output

        # Project to shared space
        c = self.cnn_proj(feat_cnn)    # (B, D)
        t = self.trans_proj(feat_trans)  # (B, D)

        # Bilinear fusion: concat [additive, multiplicative, CNN-only] signals
        # The element-wise product c*t captures feature co-activation patterns
        fused_input = torch.cat([c + t, c * t, c - t], dim=-1)  # (B, 3D)
        fused = self.fusion_mlp(fused_input)   # (B, D)

        # SE recalibration
        fused = self.se(fused)

        # Classification
        logits = self.direct_head(fused)

        # Label co-occurrence refinement (small residual correction)
        logits = self.label_cooccur(logits)

        return logits


class BattleNet(nn.Module):
    """
    BattleNet: Novel dual-backbone hybrid fusion architecture for
    multi-label military object classification in miniature art.

    Key design principles:
    1. Dual pretrained backbones — EfficientNet-V2-S for local CNN features
       (texture, shape, camouflage detail) + Swin-T for global transformer
       features (spatial relationships, scene context)
    2. Bilinear interaction fusion — captures pairwise feature correlations
       between CNN and Transformer branches via element-wise products
    3. SE channel recalibration — amplifies the most task-relevant feature channels
    4. Label co-occurrence refinement — learnable interaction matrix corrects
       predictions by modeling which objects co-appear (tanks with soldiers, etc.)

    Compatible with Trainer's 2-phase strategy:
    - Phase 1: backbone frozen → train fusion head only
    - Phase 2: end-to-end fine-tuning at lower LR
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True,
                 fusion_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        self.backbone = BattleNetBackbone(pretrained=pretrained)

        self.classifier = BattleNetClassifier(
            cnn_dim=self.backbone.cnn_dim,
            trans_dim=self.backbone.trans_dim,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_out = self.backbone(x)
        logits = self.classifier(backbone_out)
        return logits


def create_battlenet(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create a BattleNet dual-backbone hybrid fusion classifier."""
    return BattleNet(num_classes=num_classes, pretrained=pretrained)


def create_efficientnet_variant_v2(variant: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-V2 variants (efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l)."""
    try:
        if variant == "efficientnet_v2_s":
            from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
            backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "efficientnet_v2_m":
            from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
            backbone = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None)
        elif variant == "efficientnet_v2_l":
            from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
            backbone = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown EfficientNet-V2 variant: {variant}")
    except ImportError:
        return CustomCNN(num_classes)
    return MultiLabelClassifier(backbone, num_classes)


def create_model(model_type: str = "resnet18", num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1',
                      'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
                      'vit_b_16', 'vit_b_32', 'swin_t', 'swin_s', 'swin_b',
                      'vgg11', 'vgg13', 'vgg16', 'vgg19',
                      'densenet121', 'densenet161', 'densenet169', 'densenet201',
                      'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                      'convnext_tiny', 'convnext_small', 'convnext_base',
                      'battlenet', 'custom_cnn')
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights (for transfer learning models)

    Returns:
        Model instance
    """
    # Map model types to their creators
    if model_type.startswith("resnet"):
        return create_resnet_variant(model_type, num_classes, pretrained)
    elif model_type.startswith("efficientnet_v2"):
        return create_efficientnet_variant_v2(model_type, num_classes, pretrained)
    elif model_type.startswith("efficientnet"):
        return create_efficientnet_variant(model_type, num_classes, pretrained)
    elif model_type.startswith("vit"):
        return create_vit_variant(model_type, num_classes, pretrained)
    elif model_type.startswith("swin"):
        return create_swin_variant(model_type, num_classes, pretrained)
    elif model_type.startswith("vgg"):
        return create_vgg_variant(model_type, num_classes, pretrained)
    elif model_type.startswith("densenet"):
        return create_densenet_variant(model_type, num_classes, pretrained)
    elif model_type.startswith("mobilenet"):
        return create_mobilenet_variant(model_type, num_classes, pretrained)
    elif model_type.startswith("convnext"):
        return create_convnext_variant(model_type, num_classes, pretrained)
    elif model_type == "battlenet":
        return create_battlenet(num_classes, pretrained)
    elif model_type == "custom_cnn":
        return CustomCNN(num_classes)
    else:
        # Legacy support for old model names
        models_map = {
            "resnet18": lambda: create_resnet_variant("resnet18", num_classes, pretrained),
            "efficientnet": lambda: create_efficientnet_variant("efficientnet_b0", num_classes, pretrained),
            "vit": lambda: create_vit_variant("vit_b_16", num_classes, pretrained),
            "custom": lambda: CustomCNN(num_classes),
        }
        if model_type in models_map:
            return models_map[model_type]()

    raise ValueError(f"Unknown model type: {model_type}")


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
