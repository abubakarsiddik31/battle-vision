# Experiment Plan: KIIT-MiTA Multi-Label Classification

**Baseline**: ResNet50 (Micro F1: 0.7598, Acc: 0.6118)

---

## Phase 1: Architecture Exploration ✅ COMPLETE
*[Goal: Find the best backbone architecture]*

- [x] **Exp 1**: ResNet18 + ImageNet → F1: 0.7189 (too light)
- [x] **Exp 2**: ResNet50 + ImageNet → F1: 0.7598 (baseline)
- [x] **Exp 3**: ResNet34 + ImageNet → F1: 0.7583 (good but slower than EffNet)
- [x] **Exp 4**: EfficientNet-B0 + ImageNet → F1: **0.7653** ⭐ **WINNER**
- [ ] **Exp 5**: EfficientNet-B1 + ImageNet (slightly larger)

---

## Phase 2: Overfitting Reduction - Dropout & Regularization
*[Goal: Improve generalization using regularization techniques]*

- [x] **Exp 6**: `dropout=0.5` → F1: 0.7614 (no improvement)
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
*[Goal: Improve performance on minority classes (Vehicle: 0.67 F1)*

- [ ] **Exp 25**: `pos_weight=2.0` (upweight positive class)
- [ ] **Exp 26**: `pos_weight=3.0` (stronger upweighting)
- [ ] **Exp 27**: Focal Loss (focus on hard examples)
- [ ] **Exp 28**: Class-weighted loss (manual class weights)

---

## Current Best Model

| Experiment | Architecture | Micro F1 | Accuracy | Duration | Notes |
|------------|--------------|----------|----------|----------|-------|
| **321d3507** | **EfficientNet-B0** | **0.7653** | 0.5941 | 179s | ⭐ **BEST** |
| 8aa0313a | EffNet-B0 + dropout 0.5 | 0.7614 | 0.5824 | 183s | Slight drop |
| b720ead5 | ResNet50 | 0.7598 | 0.6118 | 309s | Baseline |
| 7b9e888b | ResNet34 | 0.7583 | 0.5824 | 202s | Good but slower |
| e5fc6708 | ResNet18 | 0.7189 | 0.5529 | 136s | Too light |

---

## Per-Class Performance (Best Model: EffNet-B0)

| Class | F1 Score | Status |
|-------|----------|--------|
| Radar | 0.93 | ✅ Strong |
| Artilary | 0.87 | ✅ Strong |
| M. Rocket Launcher | 0.79 | ✅ Good |
| Missile | 0.76 | ✅ Good |
| Tank | 0.74 | ✅ Good |
| Soldier | 0.71 | ⚠️ Moderate |
| **Vehicle** | **0.67** | ⚠️ Needs improvement |

---

## Phase 1 Summary: Architecture Winner

**EfficientNet-B0 is the clear winner:**
- ✅ Best F1 score (0.7653)
- ✅ Fastest training (179s vs 309s for ResNet50)
- ✅ Best Vehicle class performance (0.67 F1)
- ✅ 40% faster than ResNet50

**Dropout 0.5 didn't help** - model actually performed worse (0.7614 vs 0.7653)

---

## Usage

To run experiments one by one:

```bash
# View all experiments
uv run python scripts/view_experiments.py list

# Show experiment details
uv run python scripts/view_experiments.py show <exp_id>
```

**After each experiment**: Update the checkbox ✅ and note results in "Current Best Model" section.
