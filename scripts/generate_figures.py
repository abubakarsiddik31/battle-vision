#!/usr/bin/env python3
"""
Generate research paper quality figures for the KIIT-MiTA Multi-Label Classification Report.

This script creates publication-ready visualizations including:
- Architecture comparison charts
- Per-class performance comparisons
- Training curves
- Hyperparameter tuning results
- Phase-wise analysis
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams

# Set publication quality parameters
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1
rcParams['axes.linewidth'] = 1.0
rcParams['grid.linewidth'] = 0.5
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 6

# Color schemes for publication
COLORS = {
    'primary': '#2E5EAA',      # Deep blue
    'secondary': '#E63946',    # Red
    'tertiary': '#06A77D',     # Green
    'quaternary': '#F4A261',   # Orange
    'quinary': '#8B5CF6',      # Purple
    'senary': '#EC4899',       # Pink
    'cnn': '#3B82F6',          # Blue for CNN models
    'transformer': '#8B5CF6',  # Purple for Transformers
    'classic': '#6B7280',      # Gray for classic CNNs
}

ARCHITECTURE_COLORS = {
    'vgg16': COLORS['classic'],
    'resnet18': COLORS['cnn'],
    'resnet34': COLORS['cnn'],
    'resnet50': COLORS['cnn'],
    'densenet121': COLORS['cnn'],
    'mobilenet_v2': COLORS['cnn'],
    'efficientnet_b0': COLORS['cnn'],
    'efficientnet_v2_s': COLORS['cnn'],
    'convnext_tiny': COLORS['cnn'],
    'vit_b_16': COLORS['transformer'],
    'swin_t': COLORS['transformer'],
}

ARCHITECTURE_NAMES = {
    'vgg16': 'VGG16',
    'resnet18': 'ResNet-18',
    'resnet34': 'ResNet-34',
    'resnet50': 'ResNet-50',
    'densenet121': 'DenseNet-121',
    'mobilenet_v2': 'MobileNetV2',
    'efficientnet_b0': 'EfficientNet-B0',
    'efficientnet_v2_s': 'EfficientNetV2-S',
    'convnext_tiny': 'ConvNeXt-T',
    'vit_b_16': 'ViT-B/16',
    'swin_t': 'Swin-T',
}

CLASS_NAMES = ['Artilary', 'Missile', 'Radar', 'M. Rocket Launcher', 'Soldier', 'Tank', 'Vehicle']


class ExperimentData:
    """Load and manage experiment data."""

    def __init__(self, experiments_dir: str = "results/experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments = {}
        self._load_experiments()

    def _load_experiments(self):
        """Load all experiment JSON files."""
        for json_file in self.experiments_dir.glob("*.json"):
            if json_file.name == "index.json" or json_file.name == "SUMMARY.md":
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    exp_id = data.get('experiment_id', json_file.stem)
                    self.experiments[exp_id] = data
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

    def get_experiments_by_architecture(self, architecture: str) -> List[Dict]:
        """Get all experiments for a specific architecture."""
        return [
            exp for exp in self.experiments.values()
            if exp.get('config', {}).get('architecture') == architecture
        ]

    def get_baseline_experiments(self) -> Dict[str, Dict]:
        """Get baseline experiments (default config) for each architecture."""
        baselines = {}
        for exp in self.experiments.values():
            config = exp.get('config', {})
            arch = config.get('architecture')
            # Baseline has default settings: weight_decay=0.0, no label smoothing
            if (config.get('weight_decay', 0.0) == 0.0 and
                config.get('label_smoothing', 0.0) == 0.0 and
                config.get('pos_weight', 1.0) == 1.0 and
                config.get('optimizer') == 'adam' and
                config.get('augmentation') is None):
                # Only keep the first/best baseline
                if arch not in baselines:
                    baselines[arch] = exp
                else:
                    # Keep the one with better F1
                    if exp.get('final_results', {}).get('micro', {}).get('f1', 0) > \
                       baselines[arch].get('final_results', {}).get('micro', {}).get('f1', 0):
                        baselines[arch] = exp
        return baselines

    def get_phase_experiments(self) -> Dict[str, Dict[str, Dict]]:
        """Get experiments organized by phase from EXPERIMENT_PLAN.md data."""
        phases = {
            'phase_0': {  # Architecture Survey
                'vgg16': 'd5c872cb',
                'densenet121': 'f51c7866',
                'mobilenet_v2': '0b9775fb',
                'convnext_tiny': '3385c192',
                'vit_b_16': '1262227c',
                'swin_t': '25c48deb',
                'efficientnet_v2_s': 'eac100ac',
            },
            'phase_1': {  # Architecture Exploration
                'resnet18': 'e5fc6708',
                'resnet50': 'b720ead5',
                'resnet34': '7b9e888b',
                'efficientnet_b0': '321d3507',
            },
            'phase_2_swin': {  # Swin-T Regularization
                'baseline': '25c48deb',
                'wd_1e-4': '35d72fb9',
                'wd_1e-3': '35d72fb9',  # Will update with actual exp
                'label_smooth': '699215ed',
                'adamw': '43f823ae',
                'lower_lr': 'd4a58726',
            },
            'phase_2_effnet': {  # EffNetV2-S Regularization
                'baseline': 'eac100ac',
                'wd_1e-4': '8c8d815f',
                'wd_1e-3': '8c4b5389',
                'label_smooth': '4df2b99b',
                'adamw': '8f51616d',
                'lower_lr': '94a731cd',
            },
        }
        return phases

    def get_architecture_family(self, arch: str) -> str:
        """Get the family (CNN, Transformer, Classic) for an architecture."""
        if arch in ['vit_b_16', 'swin_t']:
            return 'Transformer'
        elif arch in ['vgg16']:
            return 'Classic CNN'
        else:
            return 'CNN'


def figure_1_architecture_comparison(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 1: Architecture comparison showing Micro F1 and Exact Match Accuracy.
    Grouped by architecture family with publication quality styling.
    """
    baselines = data.get_baseline_experiments()

    # Prepare data sorted by family and F1 score
    arch_data = []
    for arch, exp in baselines.items():
        results = exp.get('final_results', {})
        arch_data.append({
            'architecture': ARCHITECTURE_NAMES.get(arch, arch),
            'family': data.get_architecture_family(arch),
            'micro_f1': results.get('micro', {}).get('f1', 0),
            'accuracy': results.get('exact_match_accuracy', 0),
            'color': ARCHITECTURE_COLORS.get(arch, COLORS['primary']),
        })

    # Sort by family and then by F1 score
    family_order = ['Transformer', 'CNN', 'Classic CNN']
    arch_data.sort(key=lambda x: (family_order.index(x['family']), -x['micro_f1']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    architectures = [d['architecture'] for d in arch_data]
    f1_scores = [d['micro_f1'] for d in arch_data]
    accuracies = [d['accuracy'] for d in arch_data]
    colors = [d['color'] for d in arch_data]

    # Plot Micro F1
    bars1 = ax1.barh(architectures, f1_scores, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Micro F1 Score', fontweight='bold')
    ax1.set_title('(a) Micro F1 Score by Architecture', fontweight='bold')
    ax1.set_xlim(0.65, 0.85)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=f1_scores[0], color=COLORS['secondary'], linestyle='--', linewidth=1, alpha=0.7)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, f1_scores)):
        ax1.text(val + 0.003, i, f'{val:.4f}', va='center', fontsize=8)

    # Plot Exact Match Accuracy
    bars2 = ax2.barh(architectures, accuracies, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Exact Match Accuracy', fontweight='bold')
    ax2.set_title('(b) Exact Match Accuracy by Architecture', fontweight='bold')
    ax2.set_xlim(0.45, 0.75)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.axvline(x=accuracies[0], color=COLORS['secondary'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, accuracies)):
        ax2.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_1_architecture_comparison.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_1_architecture_comparison.png", format='png')
    plt.close()

    print("Figure 1: Architecture comparison saved.")


def figure_2_per_class_performance(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 2: Per-class F1 scores for top architectures.
    Shows performance variation across classes for best performing models.
    """
    # Get top 5 architectures by Micro F1
    baselines = data.get_baseline_experiments()
    sorted_archs = sorted(
        [(arch, exp.get('final_results', {}).get('micro', {}).get('f1', 0))
         for arch, exp in baselines.items()],
        key=lambda x: -x[1]
    )[:5]

    top_archs = [arch for arch, _ in sorted_archs]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(CLASS_NAMES))
    width = 0.15

    for i, arch in enumerate(top_archs):
        exp = baselines[arch]
        results = exp.get('final_results', {})
        f1_scores = [results.get(cls, {}).get('f1', 0) for cls in CLASS_NAMES]

        offset = (i - 2) * width
        ax.bar(x + offset, f1_scores, width,
               label=ARCHITECTURE_NAMES.get(arch, arch),
               color=ARCHITECTURE_COLORS.get(arch, COLORS['primary']),
               edgecolor='black', linewidth=0.6)

    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Per-Class F1 Scores for Top 5 Architectures', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.legend(loc='lower left', ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.6, 1.0)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_2_per_class_performance.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_2_per_class_performance.png", format='png')
    plt.close()

    print("Figure 2: Per-class performance saved.")


def figure_3_training_curves(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 3: Training curves for Swin-T baseline model.
    Shows loss and F1 over epochs with phase separation.

    Note: The loss values are validation metrics. The increase during
    finetuning indicates overfitting behavior.
    """
    # Use Swin-T baseline experiment
    baselines = data.get_baseline_experiments()
    best_swin = baselines.get('swin_t')

    if not best_swin:
        print("Warning: Could not find Swin-T baseline experiment")
        return

    history = best_swin.get('metrics_history', [])
    if not history:
        print("Warning: No training history available")
        return

    # Extract metrics by phase
    head_epochs = []
    head_loss = []
    head_f1 = []

    finetune_epochs = []
    finetune_loss = []
    finetune_f1 = []

    for entry in history:
        metrics = entry.get('metrics', {})
        epoch = metrics.get('epoch', 0)
        phase = metrics.get('phase', '')
        loss = metrics.get('loss', 0)

        if epoch > 0:
            # Calculate macro F1 from per-class metrics
            f1s = [metrics.get(f'{cls}_f1', 0) for cls in CLASS_NAMES if f'{cls}_f1' in metrics]
            f1 = np.mean(f1s) if f1s else 0

            if phase == 'head':
                head_epochs.append(epoch)
                head_loss.append(loss)
                head_f1.append(f1)
            elif phase == 'finetune':
                finetune_epochs.append(epoch + 10)  # Offset to continue from head phase
                finetune_loss.append(loss)
                finetune_f1.append(f1)

    final_f1 = best_swin.get('final_results', {}).get('micro', {}).get('f1', 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Plot Loss - separate phases with annotation
    ax1.plot(head_epochs, head_loss, color=COLORS['secondary'], marker='o',
             markersize=4, linewidth=2, label='Head Phase')
    ax1.plot(finetune_epochs, finetune_loss, color=COLORS['quaternary'], marker='s',
             markersize=4, linewidth=2, label='Finetune Phase')

    # Add phase separator
    ax1.axvline(x=10.5, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.text(5.5, ax1.get_ylim()[1] * 0.95, 'Head Training\n(LR=1e-3)',
             ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(20, ax1.get_ylim()[1] * 0.95, 'Finetuning\n(LR=1e-4)',
             ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontweight='bold')
    ax1.set_title('(a) Validation Loss Over Training Phases', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left')

    # Add annotation about loss increase
    if len(finetune_loss) > 15:
        ax1.annotate('Loss increases during\nfinetuning (overfitting)',
                     xy=(finetune_epochs[-1], finetune_loss[-1]), xytext=(18, 0.26),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=8, color='red',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

    # Plot F1 Score
    ax2.plot(head_epochs, head_f1, color=COLORS['secondary'], marker='o',
             markersize=4, linewidth=2, label='Head Phase')
    ax2.plot(finetune_epochs, finetune_f1, color=COLORS['quaternary'], marker='s',
             markersize=4, linewidth=2, label='Finetune Phase')
    ax2.axhline(y=final_f1, color=COLORS['primary'], linestyle='--',
                linewidth=2, label=f'Final Test F1: {final_f1:.4f}')

    # Add phase separator
    ax2.axvline(x=10.5, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Macro F1 Score', fontweight='bold')
    ax2.set_title('(b) F1 Score Over Training Phases', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_3_training_curves.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_3_training_curves.png", format='png')
    plt.close()

    print("Figure 3: Training curves saved.")


def figure_4_hyperparameter_tuning(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 4: Hyperparameter tuning results for Swin-T and EfficientNetV2-S.
    Shows the impact of weight decay, label smoothing, and optimizer changes.
    """
    # Collect Phase 2 experiments
    phase2_data = {
        'swin_t': {
            'baseline': {'wd': 0.0, 'f1': 0.8083},
            'wd_1e-4': {'wd': 1e-4, 'f1': 0.8000},
            'wd_1e-3': {'wd': 1e-3, 'f1': 0.8326},
            'label_smooth': {'ls': 0.1, 'f1': 0.7954},
            'adamw': {'opt': 'adamw', 'f1': 0.8037},
            'lower_lr': {'lr': 5e-5, 'f1': 0.8046},
        },
        'efficientnet_v2_s': {
            'baseline': {'wd': 0.0, 'f1': 0.7926},
            'wd_1e-4': {'wd': 1e-4, 'f1': 0.7925},
            'wd_1e-3': {'wd': 1e-3, 'f1': 0.8084},
            'label_smooth': {'ls': 0.1, 'f1': 0.7825},
            'adamw': {'opt': 'adamw', 'f1': 0.7896},
            'lower_lr': {'lr': 5e-5, 'f1': 0.7720},
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Swin-T weight decay
    configs = ['Baseline\n(WD=0)', 'WD=1e-4', 'WD=1e-3', 'Label\nSmooth\n0.1', 'AdamW', 'Lower\nFinetune\nLR']
    swin_f1 = [phase2_data['swin_t'][k]['f1'] for k in ['baseline', 'wd_1e-4', 'wd_1e-3', 'label_smooth', 'adamw', 'lower_lr']]
    effnet_f1 = [phase2_data['efficientnet_v2_s'][k]['f1'] for k in ['baseline', 'wd_1e-4', 'wd_1e-3', 'label_smooth', 'adamw', 'lower_lr']]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax1.bar(x - width/2, swin_f1, width, label='Swin-T',
                    color=COLORS['transformer'], edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, effnet_f1, width, label='EfficientNetV2-S',
                    color=COLORS['cnn'], edgecolor='black', linewidth=0.8)

    ax1.set_xlabel('Configuration', fontweight='bold')
    ax1.set_ylabel('Micro F1 Score', fontweight='bold')
    ax1.set_title('(a) Regularization Techniques', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0.75, 0.85)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=7)

    # Learning rate schedules
    scheduler_data = {
        'swin_t': {
            'Baseline': 0.8326,
            'Cosine Annealing': 0.8271,
            'Step LR': 0.8112,
            'OneCycle': 0.7599,
        }
    }

    configs2 = list(scheduler_data['swin_t'].keys())
    f1_scores2 = list(scheduler_data['swin_t'].values())

    bars3 = ax2.bar(configs2, f1_scores2, color=COLORS['transformer'],
                    edgecolor='black', linewidth=0.8)

    ax2.set_xlabel('Learning Rate Schedule', fontweight='bold')
    ax2.set_ylabel('Micro F1 Score', fontweight='bold')
    ax2.set_title('(b) Learning Rate Schedules (Swin-T)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0.73, 0.85)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_4_hyperparameter_tuning.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_4_hyperparameter_tuning.png", format='png')
    plt.close()

    print("Figure 4: Hyperparameter tuning saved.")


def figure_5_phase_comparison(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 5: Phase-wise performance progression.
    Shows how the best model improved through different optimization phases.
    """
    # Data from experiment plan
    phase_progression = {
        'Phase 0\nArchitecture\nSurvey': {
            'Best Model': 'Swin-T',
            'Micro F1': 0.8083,
            'Accuracy': 0.6706,
        },
        'Phase 2\nRegularization': {
            'Best Model': 'Swin-T + WD 1e-3',
            'Micro F1': 0.8326,
            'Accuracy': 0.6706,
        },
        'Phase 4\nTraining\nStrategy': {
            'Best Model': 'Swin-T + Cosine',
            'Micro F1': 0.8271,
            'Accuracy': 0.6941,
        },
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    phases = list(phase_progression.keys())
    f1_scores = [d['Micro F1'] for d in phase_progression.values()]
    accuracies = [d['Accuracy'] for d in phase_progression.values()]

    x = np.arange(len(phases))
    width = 0.35

    bars1 = ax.bar(x - width/2, f1_scores, width, label='Micro F1 Score',
                   color=COLORS['primary'], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, accuracies, width, label='Exact Match Accuracy',
                   color=COLORS['tertiary'], edgecolor='black', linewidth=0.8)

    # Add improvement annotations
    ax.annotate('', xy=(1, 0.85), xytext=(0, 0.82),
                arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=1.5))
    ax.text(0.5, 0.84, '+2.43%', ha='center', fontsize=9, color=COLORS['secondary'], fontweight='bold')

    ax.set_xlabel('Optimization Phase', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Progression Through Optimization Phases', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=9)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.6, 0.9)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_5_phase_comparison.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_5_phase_comparison.png", format='png')
    plt.close()

    print("Figure 5: Phase comparison saved.")


def figure_6_class_imbalance(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 6: Class imbalance handling results.
    Shows impact of different techniques on overall F1 vs Vehicle class F1.
    """
    imbalance_data = {
        'Swin-T + WD 1e-3': {
            'Baseline': {'overall': 0.8326, 'vehicle': 0.67},
            'Pos Weight 2.0': {'overall': 0.7974, 'vehicle': 0.75},
            'Pos Weight 3.0': {'overall': 0.7919, 'vehicle': 0.67},
            'Focal Loss': {'overall': 0.7981, 'vehicle': 0.74},
            'Auto Weights': {'overall': 0.7907, 'vehicle': 0.68},
        }
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    configs = list(imbalance_data['Swin-T + WD 1e-3'].keys())
    overall_f1 = [imbalance_data['Swin-T + WD 1e-3'][k]['overall'] for k in configs]
    vehicle_f1 = [imbalance_data['Swin-T + WD 1e-3'][k]['vehicle'] for k in configs]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, overall_f1, width, label='Overall Micro F1',
                   color=COLORS['primary'], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, vehicle_f1, width, label='Vehicle Class F1',
                   color=COLORS['secondary'], edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Class Imbalance Technique', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Impact of Class Imbalance Handling Techniques (Swin-T)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.6, 0.9)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_6_class_imbalance.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_6_class_imbalance.png", format='png')
    plt.close()

    print("Figure 6: Class imbalance handling saved.")


def figure_7_augmentation_comparison(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 7: Data augmentation strategy comparison.
    """
    aug_data = {
        'Swin-T + WD 1e-3': {
            'Baseline': {'f1': 0.8326, 'acc': 0.6706, 'vehicle': 0.67},
            'None': {'f1': 0.8121, 'acc': 0.6529, 'vehicle': 0.74},
            'Light': {'f1': 0.7739, 'acc': 0.6353, 'vehicle': 0.70},
            'Strong': {'f1': 0.7981, 'acc': 0.6353, 'vehicle': 0.68},
            'Color': {'f1': 0.8073, 'acc': 0.6882, 'vehicle': 0.75},
            'Aggressive': {'f1': 0.8083, 'acc': 0.6471, 'vehicle': 0.73},
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    configs = list(aug_data['Swin-T + WD 1e-3'].keys())
    f1_scores = [aug_data['Swin-T + WD 1e-3'][k]['f1'] for k in configs]
    acc_scores = [aug_data['Swin-T + WD 1e-3'][k]['acc'] for k in configs]
    vehicle_f1 = [aug_data['Swin-T + WD 1e-3'][k]['vehicle'] for k in configs]

    x = np.arange(len(configs))

    # F1 and Accuracy
    ax1.plot(x, f1_scores, marker='o', label='Micro F1', color=COLORS['primary'], linewidth=2)
    ax1.plot(x, acc_scores, marker='s', label='Accuracy', color=COLORS['tertiary'], linewidth=2)
    ax1.set_xlabel('Augmentation Strategy', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('(a) Micro F1 and Accuracy', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8, rotation=15)
    ax1.legend()
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim(0.6, 0.85)

    # Vehicle F1
    bars = ax2.bar(x, vehicle_f1, color=COLORS['secondary'], edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Augmentation Strategy', fontweight='bold')
    ax2.set_ylabel('Vehicle F1 Score', fontweight='bold')
    ax2.set_title('(b) Vehicle Class Performance', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8, rotation=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0.6, 0.8)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_7_augmentation_comparison.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_7_augmentation_comparison.png", format='png')
    plt.close()

    print("Figure 7: Augmentation comparison saved.")


def figure_8_model_family_comparison(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 8: Model family comparison (Classic CNN vs Modern CNN vs Transformer).
    """
    baselines = data.get_baseline_experiments()

    # Group by family
    families = {
        'Classic CNN': [],
        'CNN': [],
        'Transformer': []
    }

    for arch, exp in baselines.items():
        family = data.get_architecture_family(arch)
        families[family].append({
            'name': ARCHITECTURE_NAMES.get(arch, arch),
            'f1': exp.get('final_results', {}).get('micro', {}).get('f1', 0),
            'acc': exp.get('final_results', {}).get('exact_match_accuracy', 0),
        })

    # Sort each family by F1
    for family in families:
        families[family].sort(key=lambda x: -x['f1'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Calculate mean and std for each family
    family_means_f1 = {f: np.mean([x['f1'] for x in families[f]]) for f in families if families[f]}
    family_means_acc = {f: np.mean([x['acc'] for x in families[f]]) for f in families if families[f]}

    # Plot 1: Mean performance by family
    families_sorted = sorted(family_means_f1.keys(), key=lambda x: -family_means_f1[x])
    means_f1 = [family_means_f1[f] for f in families_sorted]
    means_acc = [family_means_acc[f] for f in families_sorted]

    x = np.arange(len(families_sorted))
    width = 0.35

    bars1 = ax1.bar(x - width/2, means_f1, width, label='Micro F1',
                    color=[COLORS['transformer'] if f == 'Transformer' else
                           COLORS['cnn'] if f == 'CNN' else COLORS['classic']
                           for f in families_sorted],
                    edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, means_acc, width, label='Accuracy',
                    color=[COLORS['transformer'] if f == 'Transformer' else
                           COLORS['tertiary'] if f == 'CNN' else COLORS['quaternary']
                           for f in families_sorted],
                    edgecolor='black', linewidth=0.8, alpha=0.7)

    ax1.set_ylabel('Mean Score', fontweight='bold')
    ax1.set_title('(a) Mean Performance by Architecture Family', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(families_sorted)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Individual model performance
    all_models = []
    all_f1 = []
    all_colors = []

    for family in families_sorted:
        for model in families[family]:
            all_models.append(f"{model['name']}\n({family})")
            all_f1.append(model['f1'])
            all_colors.append(COLORS['transformer'] if family == 'Transformer' else
                              COLORS['cnn'] if family == 'CNN' else COLORS['classic'])

    y_pos = np.arange(len(all_models))
    bars3 = ax2.barh(y_pos, all_f1, color=all_colors, edgecolor='black', linewidth=0.6)
    ax2.set_xlabel('Micro F1 Score', fontweight='bold')
    ax2.set_title('(b) Individual Model Performance', fontweight='bold')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(all_models, fontsize=8)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_8_model_family_comparison.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_8_model_family_comparison.png", format='png')
    plt.close()

    print("Figure 8: Model family comparison saved.")


def figure_9_confusion_matrix_style(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 9: Per-class metrics heatmap for the best model.
    Shows precision, recall, and F1 for each class.
    """
    # Get Swin-T baseline
    baselines = data.get_baseline_experiments()
    swin_exp = baselines.get('swin_t')

    if not swin_exp:
        print("Warning: Could not find Swin-T experiment")
        return

    results = swin_exp.get('final_results', {})

    # Build metrics matrix
    metrics_data = []
    for cls in CLASS_NAMES:
        cls_data = results.get(cls, {})
        metrics_data.append([
            cls_data.get('precision', 0),
            cls_data.get('recall', 0),
            cls_data.get('f1', 0),
        ])

    metrics_data = np.array(metrics_data)

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=1.0)

    # Set ticks and labels
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(['Precision', 'Recall', 'F1 Score'])
    ax.set_yticklabels(CLASS_NAMES)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    for i in range(len(CLASS_NAMES)):
        for j in range(3):
            text = ax.text(j, i, f'{metrics_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    ax.set_title('Per-Class Metrics for Swin-T (Best Architecture)', fontweight='bold', fontsize=12)
    fig.colorbar(im, ax=ax, label='Score')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_9_per_class_heatmap.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_9_per_class_heatmap.png", format='png')
    plt.close()

    print("Figure 9: Per-class heatmap saved.")


def figure_10_label_distribution(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 10: Dataset label distribution.
    Shows the frequency of each class in the training set.
    """
    # Label counts from the dataset (from experiment plan summary)
    label_counts = {
        'Radar': 47,
        'Artilary': 36,
        'M. Rocket Launcher': 28,
        'Missile': 25,
        'Tank': 27,
        'Soldier': 35,
        'Vehicle': 47,
    }

    # Sort by count
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1])
    classes = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    colors = [COLORS['tertiary'] if c < 30 else COLORS['primary'] for c in counts]
    ax1.barh(classes, counts, color=colors, edgecolor='black', linewidth=0.8)

    # Add value labels
    for i, count in enumerate(counts):
        ax1.text(count + 1, i, f'{count}', va='center', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Number of Training Samples', fontweight='bold')
    ax1.set_title('(a) Class Distribution in Training Set', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add annotation for imbalance
    ax1.axvline(x=np.mean(counts), color=COLORS['secondary'], linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {np.mean(counts):.1f}')
    ax1.legend()

    # Pie chart
    colors_pie = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
                  COLORS['quaternary'], COLORS['quinary'], COLORS['senary'],
                  COLORS['cnn']]
    _, _, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%',
                              colors=colors_pie, startangle=90,
                              wedgeprops=dict(edgecolor='black', linewidth=0.8))

    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight('bold')

    ax2.set_title('(b) Class Proportion', fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_10_label_distribution.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_10_label_distribution.png", format='png')
    plt.close()

    print("Figure 10: Label distribution saved.")


def figure_11_confusion_matrix(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 11: Confusion matrix-style analysis for the best model.
    Shows per-class true positives, false positives, false negatives.
    """
    # Get Swin-T baseline
    baselines = data.get_baseline_experiments()
    swin_exp = baselines.get('swin_t')

    if not swin_exp:
        print("Warning: Could not find Swin-T experiment")
        return

    results = swin_exp.get('final_results', {})

    # Build confusion data
    confusion_data = []
    for cls in CLASS_NAMES:
        cls_data = results.get(cls, {})
        tp = cls_data.get('tp', 0)
        fp = cls_data.get('fp', 0)
        fn = cls_data.get('fn', 0)
        tn = cls_data.get('tn', 0)
        confusion_data.append([tp, fp, fn, tn])

    confusion_data = np.array(confusion_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize by row for better visualization
    confusion_norm = confusion_data / confusion_data.sum(axis=1, keepdims=True)

    im = ax.imshow(confusion_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(['True Pos', 'False Pos', 'False Neg', 'True Neg'])
    ax.set_yticklabels(CLASS_NAMES)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Add text annotations with raw counts
    for i in range(len(CLASS_NAMES)):
        for j in range(4):
            text_color = 'white' if confusion_norm[i, j] > 0.5 else 'black'
            text = ax.text(j, i, f'{confusion_data[i, j]}\n({confusion_norm[i, j]:.2%})',
                          ha="center", va="center", color=text_color, fontsize=7)

    ax.set_title('Per-Class Confusion Analysis for Swin-T (Best Model)\nValues: Count (Percentage)', fontweight='bold', fontsize=11)
    fig.colorbar(im, ax=ax, label='Proportion')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_11_confusion_analysis.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_11_confusion_analysis.png", format='png')
    plt.close()

    print("Figure 11: Confusion analysis saved.")


def figure_12_model_efficiency(data: ExperimentData, output_dir: str = "results/figures"):
    """
    Figure 12: Model efficiency comparison.
    Shows training time and performance trade-offs.
    """
    baselines = data.get_baseline_experiments()

    # Prepare data with training time
    arch_data = []
    for arch, exp in baselines.items():
        duration = exp.get('duration_seconds', 0)
        results = exp.get('final_results', {})
        arch_data.append({
            'name': ARCHITECTURE_NAMES.get(arch, arch),
            'family': data.get_architecture_family(arch),
            'f1': results.get('micro', {}).get('f1', 0),
            'duration': duration / 60,  # Convert to minutes
            'color': ARCHITECTURE_COLORS.get(arch, COLORS['primary']),
        })

    # Sort by family and then by F1
    family_order = ['Transformer', 'CNN', 'Classic CNN']
    arch_data.sort(key=lambda x: (family_order.index(x['family']), -x['f1']))

    _, ax = plt.subplots(figsize=(10, 6))

    names = [d['name'] for d in arch_data]
    f1_scores = [d['f1'] for d in arch_data]
    durations = [d['duration'] for d in arch_data]
    colors = [d['color'] for d in arch_data]

    # Scatter plot
    ax.scatter(durations, f1_scores, c=colors, s=200,
              edgecolors='black', linewidths=1.5, alpha=0.8)

    # Add labels for each point
    for i, name in enumerate(names):
        ax.annotate(name, (durations[i], f1_scores[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    # Add efficiency zones
    ax.axhline(y=0.78, color=COLORS['secondary'], linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=6, color=COLORS['secondary'], linestyle='--', linewidth=1, alpha=0.5)

    # Zone labels
    ax.text(2, 0.82, 'Fast & Good', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(8, 0.82, 'Slow & Good', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(2, 0.72, 'Fast & Poor', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax.set_xlabel('Training Time (minutes)', fontweight='bold')
    ax.set_ylabel('Micro F1 Score', fontweight='bold')
    ax.set_title('Model Efficiency: Performance vs Training Time', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(durations) + 1)
    ax.set_ylim(0.68, 0.85)

    # Add legend for families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['transformer'], label='Transformer'),
        Patch(facecolor=COLORS['cnn'], label='CNN'),
        Patch(facecolor=COLORS['classic'], label='Classic CNN'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_12_model_efficiency.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_12_model_efficiency.png", format='png')
    plt.close()

    print("Figure 12: Model efficiency saved.")


def generate_all_figures():
    """Generate all research paper figures."""
    print("Loading experiment data...")
    data = ExperimentData("results/experiments")
    print(f"Loaded {len(data.experiments)} experiments.")

    print("\nGenerating figures...")
    print("-" * 50)

    figure_1_architecture_comparison(data)
    figure_2_per_class_performance(data)
    figure_3_training_curves(data)
    figure_4_hyperparameter_tuning(data)
    figure_5_phase_comparison(data)
    figure_6_class_imbalance(data)
    figure_7_augmentation_comparison(data)
    figure_8_model_family_comparison(data)
    figure_9_confusion_matrix_style(data)
    figure_10_label_distribution(data)
    figure_11_confusion_matrix(data)
    figure_12_model_efficiency(data)

    print("-" * 50)
    print(f"\nAll figures saved to: results/figures/")
    print("Formats: PDF (publication quality) and PNG (preview)")


if __name__ == "__main__":
    generate_all_figures()
