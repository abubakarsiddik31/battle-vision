# Experiment Plan: KIIT-MiTA Multi-Label Classification

**Last Updated**: 2026-03-09

**Best Model**: EfficientNet-B0 (Micro F1: 0.7653, Acc: 0.5941)

---

## Progress Summary

| Phase | Status | Experiments | Key Findings |
|-------|--------|-------------|--------------|
| **Phase 1** | ✅ Complete | 5/5 | EffNet-B0 wins (F1: 0.7653) |
| **Phase 2** | 🟡 In Progress | 1/6 | Dropout 0.5 hurt performance |
| **Phase 3** | ⬜ Pending | 0/5 | - |
| **Phase 4** | ⬜ Pending | 0/4 | - |
| **Phase 5** | ⬜ Pending | 0/4 | - |
| **Phase 6** | ⬜ Pending | 0/4 | - |

**Total**: 6/28 experiments completed (21%)

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

### All Experiments (8 runs, 6 unique configs)

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

## Recommendations

### Next Experiments (Priority Order)

1. **Exp 8**: Weight decay (more effective than dropout for transfer learning)
2. **Exp 12**: AdamW optimizer (decoupled weight decay often helps)
3. **Exp 14**: Lower finetune LR (prevent catastrophic forgetting)
4. **Exp 25**: Class weighting (improve Vehicle class)

### Skip These

- ❌ Exp 5 (EfficientNet-B1) - B0 is already fast and accurate
- ❌ Exp 7 (Lower dropout) - dropout didn't help in Exp 6
- ❌ Exp 24 (Longer training) - already caused overfitting

---

## Usage

```bash
# View all experiments
uv run python scripts/view_experiments.py list

# Show experiment details
uv run python scripts/view_experiments.py show <exp_id>

# Update summary after new experiments
uv run python scripts/update_experiments_summary.py
```
