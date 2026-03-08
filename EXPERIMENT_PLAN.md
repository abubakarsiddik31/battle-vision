# Experiment Plan: KIIT-MiTA Multi-Label Classification

**Baseline**: ResNet50 (Micro F1: 0.7598, Acc: 0.6118)

---

## Phase 1: Architecture Exploration
*[Goal: Find the best backbone architecture]*

- [ ] **Exp 1**: ResNet18 + ImageNet (lighter, faster)
- [x] **Exp 2**: ResNet50 + ImageNet (baseline - done: F1=0.7598)
- [ ] **Exp 3**: ResNet34 + ImageNet (middle ground)
- [ ] **Exp 4**: EfficientNet-B0 + ImageNet (efficient scaling)
- [ ] **Exp 5**: EfficientNet-B1 + ImageNet (slightly larger)

---

## Phase 2: Overfitting Reduction - Dropout & Regularization
*[Goal: Improve generalization using regularization techniques]*

- [ ] **Exp 6**: `dropout=0.5` (increase from 0.3)
- [ ] **Exp 7**: `dropout=0.2` (decrease from 0.3)
- [ ] **Exp 8**: `weight_decay=1e-4` (L2 regularization with Adam)
- [ ] **Exp 9**: `weight_decay=1e-3` (stronger L2 regularization)
- [ ] **Exp 10**: `label_smoothing=0.1` ( soften labels)
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
*[Goal: Improve performance on minority classes (Vehicle: 0.61 F1)*

- [ ] **Exp 25**: `pos_weight=2.0` (upweight positive class)
- [ ] **Exp 26**: `pos_weight=3.0` (stronger upweighting)
- [ ] **Exp 27**: Focal Loss (focus on hard examples)
- [ ] **Exp 28**: Class-weighted loss (manual class weights)

---

## Current Best Model

| Experiment | Architecture | Micro F1 | Accuracy | Notes |
|------------|--------------|----------|----------|-------|
| b720ead5 | ResNet50 | 0.7598 | 0.6118 | Baseline |
| a03df067 | ResNet50 (extended) | 0.7443 | 0.5471 | Overfitting observed |

---

## Per-Class Performance (Latest)

| Class | F1 Score | Status |
|-------|----------|--------|
| Radar | 0.92 | ✅ Strong |
| Artilary | 0.85 | ✅ Strong |
| M. Rocket Launcher | 0.83 | ✅ Good |
| Missile | 0.77 | ✅ Good |
| Tank | 0.75 | ✅ Good |
| Soldier | 0.74 | ⚠️ Moderate |
| **Vehicle** | **0.61** | ❌ Weak (focus area) |

---

## Usage

To run experiments one by one:

```bash
# Example: Run ResNet18
uv run train.py --architecture resnet18 --pretrained imagenet \
  --epochs-head 10 --epochs-finetune 20 \
  --notes "ResNet18 - lighter architecture" \
  --tags arch-explore,resnet18

# View all experiments
uv run python scripts/view_experiments.py list

# Update this plan after each run
```

**After each experiment**: Update the checkbox ✅ and note results in "Current Best Model" section.
