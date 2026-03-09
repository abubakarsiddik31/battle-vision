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
| 2026-03-09 | efficientnet_v2_s_imagenet_202 | efficientnet_v2_s | 0.7919 | 0.6059 | efficientnet_v2_s,imagenet |
| 2026-03-09 | efficientnet_v2_s_imagenet_202 | efficientnet_v2_s | 0.7883 | 0.6059 | efficientnet_v2_s,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_21011 | swin_t | 0.7907 | 0.6353 | swin_t,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_20532 | swin_t | 0.7981 | 0.6294 | swin_t,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_20441 | swin_t | 0.7919 | 0.6059 | swin_t,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_19415 | swin_t | 0.7974 | 0.6118 | swin_t,imagenet |
| 2026-03-09 | efficientnet_v2_s_imagenet_202 | efficientnet_v2_s | 0.7720 | 0.6000 | efficientnet_v2_s,imagenet |
| 2026-03-09 | swin_t_imagenet_20260309_19041 | swin_t | 0.8046 | 0.6529 | swin_t,imagenet |
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
