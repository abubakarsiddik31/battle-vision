# Experiment Plan: KIIT-MiTA Multi-Label Classification

**Last Updated**: 2026-03-09

**Best Model**: Swin-Tiny + WD 1e-3 (Micro F1: 0.8326, Acc: 0.6706) ⭐

---

## Progress Summary

| Phase | Status | Experiments | Key Findings |
|-------|--------|-------------|--------------|
| **Phase 0** | ✅ Complete | 7/8 | Swin-Tiny wins (F1: 0.8083) - Custom CNN skipped |
| **Phase 1** | ✅ Complete | 5/5 | EffNet-B0 wins (F1: 0.7653) |
| **Phase 2** | ✅ Complete | 10/10 | Both models improved with WD 1e-3! |
| **Phase 3** | 🟡 In Progress | 0/7 | Testing different augmentation strategies |
| **Phase 4** | ⏸️ Skipped | 0/4 | Training Strategies (deferred) |
| **Phase 5** | ⏸️ Skipped | 0/4 | Learning Rate Schedules (deferred) |
| **Phase 6** | ✅ Complete | 6/6 | Class imbalance techniques HURT overall performance |

**Total**: 28/41 experiments completed (68%)

---

## Phase 0: Architecture Survey (Reduced) 🟡 IN PROGRESS
*[Goal: Train representative architectures from each family for report]*

| ID | Architecture | Command | Status | Micro F1 | Accuracy |
|----|--------------|---------|--------|----------|----------|
| **Exp-V1** | VGG16 | `uv run python train.py -a vgg16 --pretrained imagenet --notes "VGG16 - classic deep CNN"` | ✅ Done | 0.7021 | 0.5118 |
| **Exp-D1** | DenseNet121 | `uv run python train.py -a densenet121 --pretrained imagenet --notes "DenseNet121 - dense connectivity"` | ✅ Done | **0.7830** | **0.6412** ⭐ |
| **Exp-M1** | MobileNetV2 | `uv run python train.py -a mobilenet_v2 --pretrained imagenet --notes "MobileNetV2 - lightweight mobile"` | ✅ Done | 0.7286 | 0.5588 |
| **Exp-C1** | ConvNeXt-Tiny | `uv run python train.py -a convnext_tiny --pretrained imagenet --notes "ConvNeXt-Tiny - modern CNN"` | ✅ Done | **0.7854** | **0.6235** ⭐ |
| **Exp-T1** | ViT-B/16 | `uv run python train.py -a vit_b_16 --pretrained imagenet --notes "ViT-B/16 - vision transformer"` | ✅ Done | 0.7563 | 0.5882 |
| **Exp-S1** | Swin-Tiny | `uv run python train.py -a swin_t --pretrained imagenet --notes "Swin-Tiny - hierarchical transformer"` | ✅ Done | **0.8083** | **0.6706** ⭐ |
| **Exp-E1** | EffNetV2-S | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --notes "EfficientNetV2-S - improved V1"` | ✅ Done | 0.7926 | 0.6412 |
| **Exp-U1** | Custom CNN | `uv run python train.py -a custom_cnn --pretrained none --epochs 30 --notes "Custom CNN from scratch"` | ⏸️ Skipped | - | - |

### Phase 0 Results Summary

| Rank | Architecture | Micro F1 | Accuracy | Notes |
|------|--------------|----------|----------|-------|
| 🥇 1st | **Swin-Tiny** | **0.8083** | **0.6706** | Hierarchical transformer - BEST |
| 🥈 2nd | EfficientNetV2-S | 0.7926 | 0.6412 | Improved V1 |
| 🥉 3rd | ConvNeXt-Tiny | 0.7854 | 0.6235 | Modern CNN |
| 4th | DenseNet121 | 0.7830 | 0.6412 | Dense connectivity |
| 5th | ViT-B/16 | 0.7563 | 0.5882 | Vision transformer |
| 6th | MobileNetV2 | 0.7286 | 0.5588 | Lightweight |
| 7th | VGG16 | 0.7021 | 0.5118 | Classic CNN |

---

## Phase 1: Architecture Exploration ✅ COMPLETE
*[Goal: Find the best backbone architecture]*

- [x] **Exp 1**: ResNet18 + ImageNet → F1: 0.7189 (too light)
- [x] **Exp 2**: ResNet50 + ImageNet → F1: 0.7598 (baseline)
- [x] **Exp 3a**: ResNet34 + ImageNet → F1: 0.7209 (run 1 - high variance)
- [x] **Exp 3b**: ResNet34 + ImageNet → F1: 0.7583 (run 2 - better)
- [x] **Exp 4**: EfficientNet-B0 + ImageNet → F1: **0.7653** ⭐ **WINNER**
- [ ] **Exp 5**: EfficientNet-B1 + ImageNet (optional - EffNet-B0 already best)

---

## Phase 2: Regularization & Optimization 🟡 IN PROGRESS
*[Goal: Tune best models from each family - Swin-Tiny (Transformer) & EffNetV2-S (CNN)]*

### Swin-Tiny Experiments (Best Transformer)

| ID | Config | Command | Status | Micro F1 |
|----|--------|---------|--------|----------|
| **Exp-SW1** | Weight Decay 1e-4 | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-4 --notes "Swin-T + weight_decay 1e-4"` | ✅ Done | 0.8000 |
| **Exp-SW2** | Weight Decay 1e-3 | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --notes "Swin-T + weight_decay 1e-3"` | ✅ Done | **0.8326** ⭐ |
| **Exp-SW3** | Label Smoothing 0.1 | `uv run python train.py -a swin_t --pretrained imagenet --label-smoothing 0.1 --notes "Swin-T + label_smoothing 0.1"` | ✅ Done | 0.7954 |
| **Exp-SW4** | AdamW Optimizer | `uv run python train.py -a swin_t --pretrained imagenet --optimizer adamw --weight-decay 1e-4 --notes "Swin-T + AdamW optimizer"` | ✅ Done | 0.8037 |
| **Exp-SW5** | Lower Finetune LR | `uv run python train.py -a swin_t --pretrained imagenet --lr-finetune 5e-5 --notes "Swin-T + lower finetune LR"` | ✅ Done | 0.8046 |
### EfficientNetV2-S Experiments (Best CNN)

| ID | Config | Command | Status | Micro F1 |
|----|--------|---------|--------|----------|
| **Exp-EW1** | Weight Decay 1e-4 | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --weight-decay 1e-4 --notes "EffNetV2-S + weight_decay 1e-4"` | ✅ Done | 0.7925 |
| **Exp-EW2** | Weight Decay 1e-3 | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --weight-decay 1e-3 --notes "EffNetV2-S + weight_decay 1e-3"` | ✅ Done | **0.8084** ⭐ |
| **Exp-EW3** | Label Smoothing 0.1 | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --label-smoothing 0.1 --notes "EffNetV2-S + label_smoothing 0.1"` | ✅ Done | 0.7825 |
| **Exp-EW4** | AdamW Optimizer | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --optimizer adamw --weight-decay 1e-4 --notes "EffNetV2-S + AdamW optimizer"` | ✅ Done | 0.7896 |
| **Exp-EW5** | Lower Finetune LR | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --lr-finetune 5e-5 --notes "EffNetV2-S + lower finetune LR"` | ✅ Done | 0.7720 |

### Phase 2 Results Summary

| Model | Config | Micro F1 | vs Baseline |
|-------|--------|----------|-------------|
| **Swin-T** | Baseline | 0.8083 | - |
| **Swin-T** | + WD 1e-3 | **0.8326** | ✅ **+0.0243 BEST** |
| Swin-T | + Lower Finetune LR | 0.8046 | -0.0037 |
| Swin-T | + AdamW | 0.8037 | -0.0046 |
| Swin-T | + WD 1e-4 | 0.8000 | -0.0083 |
| Swin-T | + Label Smoothing 0.1 | 0.7954 | -0.0129 |
| **EffNetV2-S** | Baseline | 0.7926 | - |
| **EffNetV2-S** | + WD 1e-3 | **0.8084** | ✅ **+0.0158** |
| EffNetV2-S | + WD 1e-4 | 0.7925 | -0.0001 |
| EffNetV2-S | + AdamW | 0.7896 | -0.0030 |
| EffNetV2-S | + Label Smoothing 0.1 | 0.7825 | -0.0101 |
| EffNetV2-S | + Lower Finetune LR | 0.7720 | -0.0206 |

---
|----|--------|---------|--------|----------|
| **Exp-EW1** | Weight Decay 1e-4 | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --weight-decay 1e-4 --notes "EffNetV2-S + weight_decay 1e-4"` | ⬜ Todo | - |
| **Exp-EW2** | Weight Decay 1e-3 | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --weight-decay 1e-3 --notes "EffNetV2-S + weight_decay 1e-3"` | ⬜ Todo | - |
| **Exp-EW3** | Label Smoothing 0.1 | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --label-smoothing 0.1 --notes "EffNetV2-S + label_smoothing 0.1"` | ⬜ Todo | - |
| **Exp-EW4** | AdamW Optimizer | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --optimizer adamw --weight-decay 1e-4 --notes "EffNetV2-S + AdamW optimizer"` | ⬜ Todo | - |
| **Exp-EW5** | Lower Finetune LR | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --lr-finetune 5e-5 --notes "EffNetV2-S + lower finetune LR"` | ⬜ Todo | - |

---

## Phase 2 Complete! ✅

### Final Best Models from Each Family:

| Family | Best Config | Micro F1 | Accuracy |
|--------|-------------|----------|----------|
| 🥇 **Transformer** | **Swin-T + WD 1e-3** | **0.8326** | **0.6706** |
| 🥈 **CNN** | **EffNetV2-S + WD 1e-3** | **0.8084** | **0.6706** |

### Key Finding: **Weight Decay 1e-3 is the winning hyperparameter!**

- Improved Swin-T by +2.43%
- Improved EffNetV2-S by +1.58%
- AdamW, label smoothing, and lower LR all hurt performance

---

## Phase 6: Class Imbalance Handling ✅ COMPLETE
*[Goal: Improve performance on minority classes - Vehicle (0.67) & Soldier (0.71)]*

### Swin-Tiny + WD 1e-3 (Best Overall) Experiments

| ID | Config | Command | Status | Micro F1 | Vehicle F1 |
|----|--------|---------|--------|----------|------------|
| **Exp-SI1** | Pos Weight 2.0 | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --pos-weight 2.0` | ✅ Done | 0.7974 (↓0.035) | 0.75 (↑0.08) |
| **Exp-SI2** | Pos Weight 3.0 | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --pos-weight 3.0` | ✅ Done | 0.7919 (↓0.041) | 0.67 (flat) |
| **Exp-SI3** | Focal Loss γ=2.0 | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --loss focal --gamma 2.0` | ✅ Done | 0.7981 (↓0.035) | 0.74 (↑0.07) |
| **Exp-SI4** | Class Weights Auto | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --class-weights auto` | ✅ Done | 0.7907 (↓0.042) | 0.68 (↑0.01) |

### EffNetV2-S + WD 1e-3 (Best CNN) Experiments

| ID | Config | Command | Status | Micro F1 | Vehicle F1 |
|----|--------|---------|--------|----------|------------|
| **Exp-EI1** | Pos Weight 2.0 | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --weight-decay 1e-3 --pos-weight 2.0` | ✅ Done | 0.7883 (↓0.020) | 0.70 (↑0.03) |
| **Exp-EI2** | Pos Weight 3.0 | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --weight-decay 1e-3 --pos-weight 3.0` | ✅ Done | 0.7919 (↓0.017) | **0.74** (↑0.07) |

### Phase 6 Results Summary

**Key Finding**: Class imbalance techniques HURT overall performance while helping Vehicle

| Model | Config | Micro F1 | vs Baseline | Vehicle F1 |
|-------|--------|----------|-------------|-----------|
| **Swin-T + WD 1e-3** | Baseline | **0.8326** | - | 0.67 |
| Swin-T | + Pos Weight 2.0 | 0.7974 | ↓ 0.035 | 0.75 |
| Swin-T | + Focal Loss | 0.7981 | ↓ 0.035 | 0.74 |
| Swin-T | + Pos Weight 3.0 | 0.7919 | ↓ 0.041 | 0.67 |
| Swin-T | + Auto Weights | 0.7907 | ↓ 0.042 | 0.68 |
| **EffNetV2-S + WD 1e-3** | Baseline | **0.8084** | - | 0.67 |
| EffNetV2-S | + Pos Weight 2.0 | 0.7883 | ↓ 0.020 | 0.70 |
| EffNetV2-S | + Pos Weight 3.0 | 0.7919 | ↓ 0.017 | 0.74 |

**Conclusion**: The baseline with standard BCE loss and WD 1e-3 remains the best approach. Class imbalance techniques improve Vehicle F1 but hurt overall performance.

---

## Phase 3: Data Augmentation 🟡 IN PROGRESS
*[Goal: Find optimal augmentation strategy for best models]*

### Swin-Tiny + WD 1e-3 (Best Overall) Experiments

| ID | Augmentation | Description | Command | Status | Micro F1 |
|----|--------------|-------------|---------|--------|----------|
| **Exp-SA1** | None | No augmentation, only resize & normalize | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --augmentation none --notes "Swin-T + WD 1e-3 + no augmentation"` | ✅ Done | 0.8121 (↓0.021) |
| **Exp-SA2** | Light | Minimal: crop (0.9-1.0), h-flip (30%), rotation (-5 to +5) | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --augmentation light --notes "Swin-T + WD 1e-3 + light augmentation"` | ✅ Done | 0.7739 (↓0.059) |
| **Exp-SA3** | Strong | Stronger: crop (0.6-1.0), h-flip (50%), rotation (-30 to +30), v-flip (20%) | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --augmentation strong --notes "Swin-T + WD 1e-3 + strong augmentation"` | ✅ Done | 0.7981 (↓0.035) |
| **Exp-SA2** | Light | Minimal: crop (0.9-1.0), h-flip (30%), rotation (-5 to +5) | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --augmentation light --notes "Swin-T + WD 1e-3 + light augmentation"` | ⬜ Todo | - |
| **Exp-SA3** | Strong | Stronger: crop (0.6-1.0), h-flip (50%), rotation (-30 to +30), v-flip (20%) | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --augmentation strong --notes "Swin-T + WD 1e-3 + strong augmentation"` | ⬜ Todo | - |
| **Exp-SA4** | Color | Baseline + color jitter (brightness, contrast, saturation) | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --augmentation color --notes "Swin-T + WD 1e-3 + color augmentation"` | ⬜ Todo | - |
| **Exp-SA5** | Aggressive | Strong + color jitter + gaussian blur | `uv run python train.py -a swin_t --pretrained imagenet --weight-decay 1e-3 --augmentation aggressive --notes "Swin-T + WD 1e-3 + aggressive augmentation"` | ⬜ Todo | - |

### EffNetV2-S + WD 1e-3 (Best CNN) Experiments

| ID | Augmentation | Description | Command | Status | Micro F1 |
|----|--------------|-------------|---------|--------|----------|
| **Exp-EA1** | Strong | Stronger augmentation for CNN | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --weight-decay 1e-3 --augmentation strong --notes "EffNetV2-S + WD 1e-3 + strong augmentation"` | ⬜ Todo | - |
| **Exp-EA2** | Color | Color-based augmentation for CNN | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --weight-decay 1e-3 --augmentation color --notes "EffNetV2-S + WD 1e-3 + color augmentation"` | ⬜ Todo | - |

---

## Phase 4-5: Skipped for Now

---

## Experiment Results Summary

### All Completed Experiments

| ID | Architecture | Config | Micro F1 | Accuracy | Duration | Status |
|----|--------------|--------|----------|----------|----------|--------|
| **321d3507** | **EfficientNet-B0** | Baseline | **0.7653** | 0.5941 | 179s | ⭐ **BEST** |
| 8aa0313a | EffNet-B0 | dropout=0.5 | 0.7614 | 0.5824 | 183s | Dropout hurt |
| b720ead5 | ResNet50 | Baseline | 0.7598 | 0.6118 | 309s | Previous best |
| 7b9e888b | ResNet34 | Baseline | 0.7583 | 0.5824 | 202s | Good but slower |
| ed7424bf | ResNet34 | Baseline | 0.7209 | 0.5588 | 197s | High variance |
| e5fc6708 | ResNet18 | Baseline | 0.7189 | 0.5529 | 136s | Too light |
| a03df067 | ResNet50 | Extended epochs | 0.7443 | 0.5471 | 303s | Overfitting |
| 4bc915d1 | ResNet50 | Early test | 0.3470 | 0.1529 | 92s | Ignored |

### Key Observations

1. **EfficientNet-B0 is best**: F1=0.7653, fastest (179s)
2. **Training variance exists**: ResNet34 ranged from 0.72 to 0.76
3. **Higher dropout hurt**: 0.5 dropout performed worse than 0.3
4. **Longer training hurt**: 50 epochs degraded ResNet50 performance
5. **ResNet50 has higher accuracy**: 0.6118 vs 0.5941, but lower F1

---

## Per-Class Performance (Best Model: EffNet-B0)

| Class | F1 Score | Status | Trend |
|-------|----------|--------|-------|
| Radar | 0.93 | ✅ Strong | Excellent |
| Artilary | 0.87 | ✅ Strong | Excellent |
| M. Rocket Launcher | 0.79 | ✅ Good | Good |
| Missile | 0.76 | ✅ Good | Good |
| Tank | 0.74 | ✅ Good | Good |
| Soldier | 0.71 | ⚠️ Moderate | Needs work |
| **Vehicle** | **0.67** | ⚠️ Weak | **Priority** |

---

## Quick Reference Commands

### Run Experiments
```bash
# Phase 0 - All Architectures (copy from table above)
# Example:
uv run python train.py -a vgg16 --pretrained imagenet --notes "VGG16 baseline"

# Phase 2 - Regularization
uv run python train.py -a efficientnet_b0 --pretrained imagenet --dropout 0.2 --notes "Lower dropout test"
uv run python train.py -a efficientnet_b0 --pretrained imagenet --weight-decay 1e-4 --notes "Weight decay 1e-4"
uv run python train.py -a efficientnet_b0 --pretrained imagenet --label-smoothing 0.1 --notes "Label smoothing 0.1"

# Phase 3 - Optimizer
uv run python train.py -a efficientnet_b0 --pretrained imagenet --optimizer adamw --notes "AdamW optimizer"
uv run python train.py -a efficientnet_b0 --pretrained imagenet --optimizer sgd --momentum 0.9 --notes "SGD+Momentum"

# Phase 6 - Class Imbalance
uv run python train.py -a efficientnet_b0 --pretrained imagenet --pos-weight 2.0 --notes "Pos weight 2.0"
```

### View Results
```bash
# View all experiments
uv run python scripts/view_experiments.py list

# Show experiment details
uv run python scripts/view_experiments.py show <exp_id>

# Update summary after new experiments
uv run python scripts/update_experiments_summary.py
```

---

## Architecture Comparison (For Report)

### Architecture Families

| Family | Models | Key Characteristics |
|--------|--------|---------------------|
| **VGG** | vgg11, vgg13, vgg16, vgg19 | Classic deep CNN, simple stack of conv layers |
| **ResNet** | resnet18, resnet34, resnet50 | Residual connections, skip connections |
| **DenseNet** | densenet121, densenet161, densenet169, densenet201 | Dense connectivity, feature reuse |
| **MobileNet** | mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large | Depthwise separable conv, mobile-optimized |
| **EfficientNet** | efficientnet_b0, b1, v2_s, v2_m, v2_l | Compound scaling, efficient |
| **ConvNeXt** | convnext_tiny, small, base | Modern CNN with transformer-style design |
| **ViT** | vit_b_16, vit_b_32 | Pure vision transformer, patch-based |
| **Swin** | swin_t, swin_s, swin_b | Hierarchical vision transformer |
| **Custom** | custom_cnn | Baseline CNN from scratch |

### Expected Performance (Hypothesis)

| Architecture | Expected F1 | Expected Speed | Notes |
|--------------|-------------|----------------|-------|
| VGG11/13 | 0.70-0.73 | Fast | Lightweight classic |
| VGG16/19 | 0.72-0.75 | Slow | Heavy, may overfit |
| DenseNet121 | 0.74-0.76 | Medium | Efficient dense connections |
| DenseNet161+ | 0.75-0.77 | Slow | Very deep, diminishing returns |
| MobileNetV2 | 0.70-0.73 | Very Fast | Lightweight, good for edge |
| MobileNetV3 | 0.71-0.74 | Very Fast | Improved V2 |
| ConvNeXt-T | 0.75-0.77 | Medium | Modern competitive |
| ViT-B/16 | 0.73-0.76 | Slow | Needs more data |
| Swin-T | 0.74-0.77 | Medium | Hierarchical helps |
| EffNetV2-S | 0.76-0.78 | Fast | Improved V1 |
