# Experiments Summary

This directory tracks all training experiments with full reproducibility.

## Experiment Files

Each experiment is saved as `results/experiments/<exp_name>_<id>.json` containing:
- Complete configuration
- Hyperparameters
- Training notes
- Per-epoch metrics
- Final results

## Index

All experiments are indexed in `index.json` for easy querying.

## Viewing Experiments

```bash
# List all experiments
python scripts/view_experiments.py list

# Show experiment details
python scripts/view_experiments.py show <exp_id>

# Search by tag
python scripts/view_experiments.py search --tag baseline

# Compare experiments
python scripts/view_experiments.py compare <exp_id1> <exp_id2>
```

## Git Tracking

- ✅ Experiment JSON files are tracked in git
- ❌ Model checkpoints (.pth) are excluded (use git-lfs if needed)
- ✅ Index file is tracked for quick reference

## Recent Experiments

<!-- Auto-generated summary will be added below -->
| Date | Experiment | Architecture | F1 | Accuracy | Notes |
|------|------------|--------------|-----|----------|-------|
| 2026-03-09 | efficientnet_v2_s_imagenet_202 | efficientnet_v2_s | 0.7896 | 0.6118 | efficientnet_v2_s,imagenet |
| 2026-03-09 | efficientnet_v2_s_imagenet_202 | efficientnet_v2_s | 0.7925 | 0.6294 | efficientnet_v2_s,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_17384 | swin_t | 0.7954 | 0.6235 | swin_t,imagenet |
| 2026-03-09 | efficientnet_v2_s_imagenet_202 | efficientnet_v2_s | 0.8084 | 0.6706 | efficientnet_v2_s,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_15410 | swin_t | 0.8037 | 0.6529 | swin_t,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_15243 | swin_t | 0.8326 | 0.6706 | swin_t,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_14095 | swin_t | 0.8000 | 0.6412 | swin_t,imagenet |
| 2026-03-09 | efficientnet_v2_s_imagenet_202 | efficientnet_v2_s | 0.7926 | 0.6412 | efficientnet_v2_s,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_13421 | swin_t | 0.8083 | 0.6706 | swin_t,imagenet |
| 2026-03-09 | vit_b_16_imagenet_20260309_132 | vit_b_16 | 0.7563 | 0.5882 | vit_b_16,imagenet |
| 2026-03-09 | convnext_tiny_imagenet_2026030 | convnext_tiny | 0.7854 | 0.6235 | convnext_tiny,imagenet |
| 2026-03-09 | mobilenet_v2_imagenet_20260309 | mobilenet_v2 | 0.7286 | 0.5588 | mobilenet_v2,imagenet |
| 2026-03-09 | densenet121_imagenet_20260309_ | densenet121 | 0.7830 | 0.6412 | densenet121,imagenet |
| 2026-03-09 | vgg16_imagenet_20260309_122812 | vgg16 | 0.7021 | 0.5118 | vgg16,imagenet |
| 2026-03-09 | efficientnet_b0_imagenet_20260 | efficientnet_b0 | 0.7614 | 0.5824 | regularization,dropout,efficie |
| 2026-03-09 | resnet34_imagenet_20260309_050 | resnet34 | 0.7583 | 0.5824 | arch-explore,resnet34 |
| 2026-03-09 | resnet34_imagenet_20260309_043 | resnet34 | 0.7209 | 0.5588 | arch-explore,resnet34 |
| 2026-03-09 | resnet18_imagenet_20260309_043 | resnet18 | 0.7189 | 0.5529 | arch-explore,resnet18 |
| 2026-03-09 | efficientnet_b0_imagenet_20260 | efficientnet_b0 | 0.7653 | 0.5941 | baseline,efficientnet,arch-com |
| 2026-03-09 | resnet50_imagenet_20260309_040 | resnet50 | 0.7443 | 0.5471 | baseline,resnet50,longer-train |
