# Experiments Summary

Generated: 2026-03-09 04:43:17

Total Experiments: 6

## Quick Reference

| Date | Experiment | Architecture | Pretrained | Micro F1 | Acc | Tags |
|------|------------|--------------|------------|----------|-----|------|
| 2026-03-09 | resnet34_imagenet_20260309_043933 | resnet34 | imagenet | 0.7209 | 0.5588 | arch-explore,resnet3 |
| 2026-03-09 | resnet18_imagenet_20260309_043357 | resnet18 | imagenet | 0.7189 | 0.5529 | arch-explore,resnet1 |
| 2026-03-09 | efficientnet_b0_imagenet_20260309_042229 | efficientnet_b0 | imagenet | 0.7653 | 0.5941 | baseline,efficientne |
| 2026-03-09 | resnet50_imagenet_20260309_040438 | resnet50 | imagenet | 0.7443 | 0.5471 | baseline,resnet50,lo |
| 2026-03-09 | resnet50_imagenet_20260309_034115 | resnet50 | imagenet | 0.7598 | 0.6118 | baseline,resnet50,in |
| 2026-03-09 | resnet50_imagenet_20260309_030219 | resnet50 | imagenet | 0.3470 | 0.1529 | baseline,resnet50,in |

## Detailed Results

### resnet34_imagenet_20260309_043933

**ID:** `ed7424bf`

**Date:** 2026-03-09T04:39:58.238862

**Duration:** 196.7s

**Tags:** `arch-explore,resnet34`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `resnet34`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.5588235294117647
- Micro F1: 0.7209302325581395
- Micro Precision: 0.7311320754716981
- Micro Recall: 0.7110091743119266
- Macro F1: 0.7312068026864453

**Per-Class F1:**
- Artilary: 0.7692307692307692
- Missile: 0.72
- Radar: 0.8888888888888888
- M. Rocket Launcher: 0.7169811320754718
- Soldier: 0.7105263157894737
- Tank: 0.6461538461538462
- Vehicle: 0.6666666666666667

---

### resnet18_imagenet_20260309_043357

**ID:** `e5fc6708`

**Date:** 2026-03-09T04:34:11.880623

**Duration:** 136.3s

**Tags:** `arch-explore,resnet18`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `resnet18`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.5529411764705883
- Micro F1: 0.7188940092165899
- Micro Precision: 0.7222222222222222
- Micro Recall: 0.7155963302752294
- Macro F1: 0.7398369098587801

**Per-Class F1:**
- Artilary: 0.7777777777777778
- Missile: 0.76
- Radar: 0.8260869565217391
- M. Rocket Launcher: 0.7692307692307692
- Soldier: 0.7605633802816902
- Tank: 0.6756756756756757
- Vehicle: 0.6095238095238096

---

### efficientnet_b0_imagenet_20260309_042229

**ID:** `321d3507`

**Date:** 2026-03-09T04:22:37.008801

**Duration:** 178.9s

**Tags:** `baseline,efficientnet,arch-compare`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `efficientnet_b0`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.5941176470588235
- Micro F1: 0.7652582159624414
- Micro Precision: 0.7836538461538461
- Micro Recall: 0.7477064220183486
- Macro F1: 0.7821183052748177

**Per-Class F1:**
- Artilary: 0.8717948717948718
- Missile: 0.7555555555555556
- Radar: 0.9259259259259259
- M. Rocket Launcher: 0.7924528301886793
- Soldier: 0.7123287671232877
- Tank: 0.7428571428571429
- Vehicle: 0.6739130434782609

---

### resnet50_imagenet_20260309_040438

**ID:** `a03df067`

**Date:** 2026-03-09T04:04:39.494347

**Duration:** 303.3s

**Tags:** `baseline,resnet50,longer-training`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `resnet50`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.5470588235294118
- Micro F1: 0.7442922374429224
- Micro Precision: 0.740909090909091
- Micro Recall: 0.7477064220183486
- Macro F1: 0.7641156836285737

**Per-Class F1:**
- Artilary: 0.8292682926829269
- Missile: 0.7692307692307692
- Radar: 0.8260869565217391
- M. Rocket Launcher: 0.830188679245283
- Soldier: 0.736842105263158
- Tank: 0.7466666666666667
- Vehicle: 0.6105263157894737

---

### resnet50_imagenet_20260309_034115

**ID:** `b720ead5`

**Date:** 2026-03-09T03:42:56.316555

**Duration:** 308.6s

**Tags:** `baseline`, `resnet50`, `initial-run`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `resnet50`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.611764705882353
- Micro F1: 0.7598039215686274
- Micro Precision: 0.8157894736842105
- Micro Recall: 0.7110091743119266
- Macro F1: 0.7725425152470571

**Per-Class F1:**
- Artilary: 0.8500000000000001
- Missile: 0.7916666666666667
- Radar: 0.923076923076923
- M. Rocket Launcher: 0.7083333333333333
- Soldier: 0.7000000000000001
- Tank: 0.7605633802816901
- Vehicle: 0.6741573033707866

---

### resnet50_imagenet_20260309_030219

**ID:** `4bc915d1`

**Date:** 2026-03-09T03:02:19.183261

**Duration:** 91.6s

**Tags:** `baseline`, `resnet50`, `initial-run`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `resnet50`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.15294117647058825
- Micro F1: 0.3470031545741325
- Micro Precision: 0.5555555555555556
- Micro Recall: 0.25229357798165136
- Macro F1: 0.24926826806958172

**Per-Class F1:**
- Artilary: 0.0
- Missile: 0.06896551724137932
- Radar: 0.6785714285714286
- M. Rocket Launcher: 0.06666666666666667
- Soldier: 0.0
- Tank: 0.46153846153846156
- Vehicle: 0.4691358024691358

---

