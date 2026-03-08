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
| 2026-03-09 | resnet34_imagenet_20260309_043 | resnet34 | 0.7209 | 0.5588 | arch-explore,resnet34 |
| 2026-03-09 | resnet18_imagenet_20260309_043 | resnet18 | 0.7189 | 0.5529 | arch-explore,resnet18 |
| 2026-03-09 | efficientnet_b0_imagenet_20260 | efficientnet_b0 | 0.7653 | 0.5941 | baseline,efficientnet,arch-com |
| 2026-03-09 | resnet50_imagenet_20260309_040 | resnet50 | 0.7443 | 0.5471 | baseline,resnet50,longer-train |
| 2026-03-09 | resnet50_imagenet_20260309_034 | resnet50 | 0.7598 | 0.6118 | baseline,resnet50,initial-run |
| 2026-03-09 | resnet50_imagenet_20260309_030 | resnet50 | 0.3470 | 0.1529 | baseline,resnet50,initial-run |
