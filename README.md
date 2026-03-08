# KIIT-MiTA Multi-Label Classification

A deep learning project for multi-label classification of military objects in miniature art images using PyTorch.

## Overview

The KIIT-MiTA (Miniature Art Training Archive) dataset contains images of 7 different military object classes. This project implements CNN-based classifiers with transfer learning support to detect multiple objects present in each image.

## Classes

| Class | Index | Description |
|-------|-------|-------------|
| Artilary | 0 | Artillery pieces |
| Missile | 1 | Missile systems |
| Radar | 2 | Radar installations |
| M. Rocket Launcher | 3 | Multiple rocket launchers |
| Soldier | 4 | Military personnel |
| Tank | 5 | Armored tanks |
| Vehicle | 6 | Military vehicles |

## Dataset Statistics

| Split | Images | Avg Objects/Image |
|-------|--------|-------------------|
| Train | 1,360 | 1.27 |
| Validation | 170 | 1.24 |
| Test | 170 | 1.28 |

## Project Structure

```
KIIT-MiTA-Classification/
├── data/                      # Prepared dataset annotations
│   ├── train_annotations.json
│   ├── test_annotations.json
│   ├── valid_annotations.json
│   └── metadata.json
├── src/kiit_mita/             # Source package
│   ├── __init__.py
│   ├── dataset.py            # Dataset class
│   ├── dataloaders.py        # Data transforms & loaders
│   ├── models.py             # Model architectures
│   └── trainer.py            # Training with trackio integration
├── scripts/                   # Utility scripts
│   └── dataset_prep.py       # Parse YOLO labels
├── KIIT-MiTA/                # Original dataset
├── checkpoints/              # Model checkpoints
├── results/                  # Evaluation results
├── configs/                  # Configuration files
├── train.py                  # Main training script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- trackio (optional, for experiment tracking)

### Setup

```bash
# Clone or navigate to the project
cd /path/to/KIIT-MiTA-Classification

# Install dependencies
pip install -r requirements.txt

# Prepare dataset annotations
python scripts/dataset_prep.py
```

## Usage

### Quick Start

```bash
# Train with transfer learning (ResNet-18) - Recommended
python train.py --model resnet18

# Train custom CNN baseline
python train.py --model custom --epochs 30

# Train with EfficientNet
python train.py --model efficientnet

# Train with Vision Transformer
python train.py --model vit

# View experiments dashboard
trackio show --project kiit-mita-classification
```

### Training Options

```bash
python train.py --help

Options:
  --model, -m            Model architecture [resnet18|efficientnet|vit|custom]
  --epochs-head          Epochs for head training (default: 10)
  --epochs-finetune      Epochs for fine-tuning (default: 20)
  --epochs, -e           Total epochs for custom CNN (default: 30)
  --lr, --learning-rate  Learning rate (default: 1e-3)
  --batch-size, -b       Batch size (default: 32)
  --device               Device [cuda|cpu|mps]
  --no-trackio           Disable trackio logging
  --trackio-project      Trackio project name
```

### Training Strategy

For transfer learning models (ResNet, EfficientNet, ViT), training occurs in two phases:

1. **Phase 1 - Head Training**: Backbone frozen, only classification head is trained
2. **Phase 2 - Fine-tuning**: Entire network is trained with lower learning rate

## Evaluation Metrics

The following metrics are computed for multi-label classification:

- **Exact Match Accuracy**: All labels must match exactly
- **Micro F1/Precision/Recall**: Computed across all instances
- **Macro F1/Precision/Recall**: Averaged across classes
- **Per-class metrics**: Individual F1, precision, recall for each of 7 classes

## Experiment Tracking

The project integrates with [Trackio](https://github.com/huggingface/trackio) - a lightweight, local-first experiment tracking library from Hugging Face:

- Automatic logging of loss, accuracy, F1 scores
- Per-class metric tracking
- Model checkpointing
- Comparison across runs
- Local dashboard with `trackio show`
- Optional cloud sync to Hugging Face Spaces

### Viewing Experiments

```bash
# Launch the dashboard
trackio show

# View specific project
trackio show --project kiit-mita-classification

# Query experiments via CLI
trackio get runs --project kiit-mita-classification --json
```

To disable trackio:
```bash
python train.py --model resnet18 --no-trackio
```

## Model Architectures

### Transfer Learning Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| ResNet-18 | ~11M | Lightweight residual network |
| EfficientNet-B0 | ~5M | Efficient architecture |
| ViT-B/16 | ~86M | Vision Transformer |

### Custom CNN

A 4-block CNN baseline architecture:
- 4 convolutional blocks with BatchNorm
- Global average pooling
- 2 fully connected layers with dropout
- ~4.8M parameters

## Results

Results are saved to `results/<model>_test_results.json` containing per-class and aggregated metrics.

Example:
```json
{
  "micro": {"f1": 0.85, "precision": 0.87, "recall": 0.83},
  "macro": {"f1": 0.82, "precision": 0.84, "recall": 0.81},
  "exact_match_accuracy": 0.72,
  "Artilary": {"f1": 0.75, "precision": 0.80, "recall": 0.71},
  ...
}
```

## Development

### Project Layout

```
src/kiit_mita/
├── __init__.py       # Package exports
├── dataset.py        # KiitMitaDataset class
├── dataloaders.py    # TrainTransforms, EvalTransforms, create_dataloaders()
├── models.py         # Model factories and architectures
└── trainer.py        # Trainer class with trackio integration
```

### Adding a New Model

```python
# In src/kiit_mita/models.py
def create_your_model(num_classes: int = 7, pretrained: bool = True):
    backbone = your_backbone(pretrained=pretrained)
    return MultiLabelClassifier(backbone, num_classes)
```

## Citation

If you use this code or dataset, please cite:

```bibtex
@misc{kiit_mita,
  title={KIIT-MiTA: Miniature Art Training Archive},
  author={Your Name},
  year={2025},
  description={Multi-label military object classification dataset}
}
```

## License

This project is provided for educational and research purposes.

## Acknowledgments

- KIIT University for providing the dataset
- PyTorch team for the deep learning framework
- Hugging Face for the Trackio experiment tracking library
