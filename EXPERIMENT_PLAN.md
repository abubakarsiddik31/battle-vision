# Experiment Plan: KIIT-MiTA Multi-Label Classification

**Last Updated**: 2026-03-09

**Best Model**: ConvNeXt-Tiny (Micro F1: 0.7854, Acc: 0.6235) ⭐ NEW!

---

## Progress Summary

| Phase | Status | Experiments | Key Findings |
|-------|--------|-------------|--------------|
| **Phase 0** | 🟡 In Progress | 1/8 | VGG16: F1=0.7021 (slower than EffNet) |
| **Phase 1** | ✅ Complete | 5/5 | EffNet-B0 wins (F1: 0.7653) |
| **Phase 2** | 🟡 In Progress | 1/6 | Dropout 0.5 hurt performance |
| **Phase 3** | ⬜ Pending | 0/5 | - |
| **Phase 4** | ⬜ Pending | 0/4 | - |
| **Phase 5** | ⬜ Pending | 0/4 | - |
| **Phase 6** | ⬜ Pending | 0/4 | - |

**Total**: 6/36 experiments completed (17%)

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
| **Exp-S1** | Swin-Tiny | `uv run python train.py -a swin_t --pretrained imagenet --notes "Swin-Tiny - hierarchical transformer"` | ⬜ Todo | - | - |
| **Exp-E1** | EffNetV2-S | `uv run python train.py -a efficientnet_v2_s --pretrained imagenet --notes "EfficientNetV2-S - improved V1"` | ⬜ Todo | - | - |
| **Exp-U1** | Custom CNN | `uv run python train.py -a custom_cnn --pretrained none --epochs 30 --notes "Custom CNN from scratch"` | ⬜ Todo | - | - |

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

## Phase 2: Overfitting Reduction - Dropout & Regularization 🟡 IN PROGRESS
*[Goal: Improve generalization using regularization techniques]*

- [x] **Exp 6**: `dropout=0.5` → F1: 0.7614 (worse than baseline 0.7653)
- [ ] **Exp 7**: `dropout=0.2` (decrease from 0.3)
- [ ] **Exp 8**: `weight_decay=1e-4` (L2 regularization with Adam)
- [ ] **Exp 9**: `weight_decay=1e-3` (stronger L2 regularization)
- [ ] **Exp 10**: `label_smoothing=0.1` (soften labels)
- [ ] **Exp 11**: `label_smoothing=0.2` (stronger smoothing)

---

## Phase 3: Optimizer & Learning Rate Strategies
*[Goal: Find optimal optimization strategy]*

- [ ] **Exp 12**: AdamW optimizer (decoupled weight decay)
- [ ] **Exp 13**: SGD + Momentum (classic optimizer)
- [ ] **Exp 14**: Lower finetune LR (0.00005 instead of 0.0001)
- [ ] **Exp 15**: Higher head LR (0.005 instead of 0.001)
- [ ] **Exp 16**: Learning rate scheduler (ReduceLROnPlateau)

---

## Phase 4: Data Augmentation
*[Goal: Improve generalization through augmented training data]*

- [ ] **Exp 17**: Strong augmentation (more aggressive transforms)
- [ ] **Exp 18**: Mixup augmentation (blend images/labels)
- [ ] **Exp 19**: CutMix augmentation (patch-based mixing)
- [ ] **Exp 20**: Random erasing (simulate occlusion)

---

## Phase 5: Training Strategies
*[Goal: Optimize training process]*

- [ ] **Exp 21**: Early stopping (patience=5)
- [ ] **Exp 22**: Gradient clipping (clip_grad_norm=1.0)
- [ ] **Exp 23**: Longer training with cosine annealing
- [ ] **Exp 24**: Two-phase with different epochs (15 head + 30 finetune)

---

## Phase 6: Class Imbalance Handling
*[Goal: Improve performance on minority classes]*

- [ ] **Exp 25**: `pos_weight=2.0` (upweight positive class)
- [ ] **Exp 26**: `pos_weight=3.0` (stronger upweighting)
- [ ] **Exp 27**: Focal Loss (focus on hard examples)
- [ ] **Exp 28**: Class-weighted loss (manual class weights)

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
