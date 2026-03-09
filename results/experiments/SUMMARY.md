# Experiments Summary

Generated: 2026-03-09 13:39:43

Total Experiments: 13

## Quick Reference

| Date | Experiment | Architecture | Pretrained | Micro F1 | Acc | Tags |
|------|------------|--------------|------------|----------|-----|------|
| 2026-03-09 | vit_b_16_imagenet_20260309_132111 | vit_b_16 | imagenet | 0.7563 | 0.5882 | vit_b_16,imagenet |
| 2026-03-09 | convnext_tiny_imagenet_20260309_130623 | convnext_tiny | imagenet | 0.7854 | 0.6235 | convnext_tiny,imagen |
| 2026-03-09 | mobilenet_v2_imagenet_20260309_130004 | mobilenet_v2 | imagenet | 0.7286 | 0.5588 | mobilenet_v2,imagene |
| 2026-03-09 | densenet121_imagenet_20260309_124854 | densenet121 | imagenet | 0.7830 | 0.6412 | densenet121,imagenet |
| 2026-03-09 | vgg16_imagenet_20260309_122812 | vgg16 | imagenet | 0.7021 | 0.5118 | vgg16,imagenet |
| 2026-03-09 | efficientnet_b0_imagenet_20260309_050538 | efficientnet_b0 | imagenet | 0.7614 | 0.5824 | regularization,dropo |
| 2026-03-09 | resnet34_imagenet_20260309_050206 | resnet34 | imagenet | 0.7583 | 0.5824 | arch-explore,resnet3 |
| 2026-03-09 | resnet34_imagenet_20260309_043933 | resnet34 | imagenet | 0.7209 | 0.5588 | arch-explore,resnet3 |
| 2026-03-09 | resnet18_imagenet_20260309_043357 | resnet18 | imagenet | 0.7189 | 0.5529 | arch-explore,resnet1 |
| 2026-03-09 | efficientnet_b0_imagenet_20260309_042229 | efficientnet_b0 | imagenet | 0.7653 | 0.5941 | baseline,efficientne |
| 2026-03-09 | resnet50_imagenet_20260309_040438 | resnet50 | imagenet | 0.7443 | 0.5471 | baseline,resnet50,lo |
| 2026-03-09 | resnet50_imagenet_20260309_034115 | resnet50 | imagenet | 0.7598 | 0.6118 | baseline,resnet50,in |
| 2026-03-09 | resnet50_imagenet_20260309_030219 | resnet50 | imagenet | 0.3470 | 0.1529 | baseline,resnet50,in |

## Detailed Results

### vit_b_16_imagenet_20260309_132111

**ID:** `1262227c`

**Date:** 2026-03-09T13:21:13.617992

**Duration:** 1107.0s

**Tags:** `vit_b_16`, `imagenet`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `vit_b_16`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.5882352941176471
- Micro F1: 0.7562642369020501
- Micro Precision: 0.751131221719457
- Micro Recall: 0.7614678899082569
- Macro F1: 0.7747412496755884

**Per-Class F1:**
- Artilary: 0.8571428571428572
- Missile: 0.7692307692307692
- Radar: 0.9019607843137256
- M. Rocket Launcher: 0.8070175438596492
- Soldier: 0.676056338028169
- Tank: 0.7228915662650603
- Vehicle: 0.6888888888888888

---

### convnext_tiny_imagenet_20260309_130623

**ID:** `3385c192`

**Date:** 2026-03-09T13:06:56.814110

**Duration:** 434.0s

**Tags:** `convnext_tiny`, `imagenet`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `convnext_tiny`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.6235294117647059
- Micro F1: 0.7853881278538812
- Micro Precision: 0.7818181818181819
- Micro Recall: 0.7889908256880734
- Macro F1: 0.7954696976030922

**Per-Class F1:**
- Artilary: 0.8181818181818181
- Missile: 0.7234042553191491
- Radar: 0.9433962264150944
- M. Rocket Launcher: 0.8076923076923077
- Soldier: 0.7999999999999999
- Tank: 0.8311688311688312
- Vehicle: 0.6444444444444444

---

### mobilenet_v2_imagenet_20260309_130004

**ID:** `0b9775fb`

**Date:** 2026-03-09T13:00:09.101303

**Duration:** 147.0s

**Tags:** `mobilenet_v2`, `imagenet`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `mobilenet_v2`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.5588235294117647
- Micro F1: 0.7285714285714285
- Micro Precision: 0.7574257425742574
- Micro Recall: 0.7018348623853211
- Macro F1: 0.738456934878114

**Per-Class F1:**
- Artilary: 0.8
- Missile: 0.5957446808510639
- Radar: 0.9811320754716981
- M. Rocket Launcher: 0.7307692307692307
- Soldier: 0.6956521739130436
- Tank: 0.7222222222222223
- Vehicle: 0.6436781609195402

---

### densenet121_imagenet_20260309_124854

**ID:** `f51c7866`

**Date:** 2026-03-09T12:49:06.163340

**Duration:** 327.8s

**Tags:** `densenet121`, `imagenet`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `densenet121`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.6411764705882353
- Micro F1: 0.7830188679245284
- Micro Precision: 0.8058252427184466
- Micro Recall: 0.7614678899082569
- Macro F1: 0.7953561789400433

**Per-Class F1:**
- Artilary: 0.8780487804878049
- Missile: 0.6666666666666665
- Radar: 0.9615384615384615
- M. Rocket Launcher: 0.830188679245283
- Soldier: 0.782608695652174
- Tank: 0.7671232876712328
- Vehicle: 0.6813186813186813

---

### vgg16_imagenet_20260309_122812

**ID:** `d5c872cb`

**Date:** 2026-03-09T12:28:14.391716

**Duration:** 540.1s

**Tags:** `vgg16`, `imagenet`

**Notes:** See full experiment file for details

**Configuration:**
- Architecture: `vgg16`
- Pretrained: `imagenet`
- Epochs: 30
- LR Head: 0.001
- LR Finetune: 0.0001
- Batch Size: 32
- Optimizer: `adam`

**Results:**
- Exact Match Accuracy: 0.5117647058823529
- Micro F1: 0.7020785219399538
- Micro Precision: 0.7069767441860465
- Micro Recall: 0.6972477064220184
- Macro F1: 0.7142708740686744

**Per-Class F1:**
- Artilary: 0.7906976744186046
- Missile: 0.6046511627906976
- Radar: 0.9259259259259259
- M. Rocket Launcher: 0.7083333333333333
- Soldier: 0.6428571428571429
- Tank: 0.7297297297297297
- Vehicle: 0.5977011494252874

---

### efficientnet_b0_imagenet_20260309_050538

**ID:** `8aa0313a`

**Date:** 2026-03-09T05:05:39.667674

**Duration:** 183.0s

**Tags:** `regularization,dropout,efficientnet_b0`

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
- Exact Match Accuracy: 0.5823529411764706
- Micro F1: 0.7614457831325301
- Micro Precision: 0.8020304568527918
- Micro Recall: 0.7247706422018348
- Macro F1: 0.7756159841778996

**Per-Class F1:**
- Artilary: 0.8571428571428572
- Missile: 0.7111111111111111
- Radar: 0.9056603773584906
- M. Rocket Launcher: 0.8235294117647057
- Soldier: 0.6857142857142857
- Tank: 0.7428571428571429
- Vehicle: 0.7032967032967034

---

### resnet34_imagenet_20260309_050206

**ID:** `7b9e888b`

**Date:** 2026-03-09T05:02:07.477614

**Duration:** 201.9s

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
- Exact Match Accuracy: 0.5823529411764706
- Micro F1: 0.7582938388625592
- Micro Precision: 0.7843137254901961
- Micro Recall: 0.7339449541284404
- Macro F1: 0.7782114705107465

**Per-Class F1:**
- Artilary: 0.8205128205128205
- Missile: 0.8085106382978724
- Radar: 0.9259259259259259
- M. Rocket Launcher: 0.8235294117647057
- Soldier: 0.7164179104477612
- Tank: 0.7142857142857142
- Vehicle: 0.6382978723404256

---

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

