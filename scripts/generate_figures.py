#!/usr/bin/env python3
"""
Generate research paper quality figures for the KIIT-MiTA Multi-Label Classification Report.

Updated to include BattleNet (novel single-backbone architecture) results
and corrected Swin-T baseline (0.8121 on local hardware).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches

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
    'battlenet': '#E63946',    # Red for BattleNet (novel)
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
    'battlenet': COLORS['battlenet'],
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
    'battlenet': 'BattleNet (Ours)',
}

CLASS_NAMES = ['Artilary', 'Missile', 'Radar', 'M. Rocket Launcher', 'Soldier', 'Tank', 'Vehicle']

# ─── Authoritative test results (local MPS, matched hardware) ──────────────
SWINT_RESULTS = {
    'Artilary':           {'precision': 0.8500, 'recall': 0.8947, 'f1': 0.8718,
                           'tp': 17, 'fp': 3, 'fn': 2, 'tn': 148},
    'Missile':            {'precision': 0.7917, 'recall': 0.7600, 'f1': 0.7755,
                           'tp': 19, 'fp': 5, 'fn': 6, 'tn': 140},
    'Radar':              {'precision': 1.0000, 'recall': 0.9630, 'f1': 0.9811,
                           'tp': 26, 'fp': 0, 'fn': 1, 'tn': 143},
    'M. Rocket Launcher': {'precision': 0.8462, 'recall': 0.7857, 'f1': 0.8148,
                           'tp': 22, 'fp': 4, 'fn': 6, 'tn': 138},
    'Soldier':            {'precision': 0.8000, 'recall': 0.8000, 'f1': 0.8000,
                           'tp': 28, 'fp': 7, 'fn': 7, 'tn': 128},
    'Tank':               {'precision': 0.8056, 'recall': 0.7838, 'f1': 0.7945,
                           'tp': 29, 'fp': 7, 'fn': 8, 'tn': 126},
    'Vehicle':            {'precision': 0.7391, 'recall': 0.7234, 'f1': 0.7312,
                           'tp': 34, 'fp': 12, 'fn': 13, 'tn': 111},
    'micro':    {'precision': 0.8216, 'recall': 0.8028, 'f1': 0.8121},
    'macro':    {'precision': 0.8332, 'recall': 0.8158, 'f1': 0.8241},
    'exact_match': 0.6765,
}

BATTLENET_RESULTS = {
    'Artilary':           {'precision': 0.8947, 'recall': 0.8947, 'f1': 0.8947},
    'Missile':            {'precision': 0.8333, 'recall': 0.8000, 'f1': 0.8163},
    'Radar':              {'precision': 0.9615, 'recall': 0.9259, 'f1': 0.9434},
    'M. Rocket Launcher': {'precision': 0.9200, 'recall': 0.8214, 'f1': 0.8679},
    'Soldier':            {'precision': 0.7647, 'recall': 0.7429, 'f1': 0.7536},
    'Tank':               {'precision': 0.8056, 'recall': 0.7838, 'f1': 0.7945},
    'Vehicle':            {'precision': 0.7609, 'recall': 0.7447, 'f1': 0.7527},
    'micro':  {'precision': 0.8333, 'recall': 0.8028, 'f1': 0.8178},
    'macro':  {'precision': 0.8487, 'recall': 0.8162, 'f1': 0.8319},
    'exact_match': 0.6706,
}

# Phase-0 architecture survey (baseline, no weight decay)
PHASE0_RESULTS = {
    'vgg16':            {'micro_f1': 0.7021, 'accuracy': 0.5118},
    'mobilenet_v2':     {'micro_f1': 0.7286, 'accuracy': 0.5588},
    'vit_b_16':         {'micro_f1': 0.7563, 'accuracy': 0.5882},
    'densenet121':      {'micro_f1': 0.7830, 'accuracy': 0.6412},
    'convnext_tiny':    {'micro_f1': 0.7854, 'accuracy': 0.6235},
    'efficientnet_v2_s':{'micro_f1': 0.7926, 'accuracy': 0.6412},
    'swin_t':           {'micro_f1': 0.8083, 'accuracy': 0.6706},
}

# BattleNet training history (from best Kaggle run, 45 epochs, for training curves)
BATTLENET_HISTORY = {
    'train_loss': [
        0.4182, 0.3161, 0.2798, 0.2593, 0.2424, 0.2281, 0.2115, 0.1948,
        0.1804, 0.1695, 0.1598, 0.1414, 0.1320, 0.1228, 0.1220,
        0.1300, 0.1250, 0.1230, 0.1161, 0.1122, 0.1054, 0.1012, 0.0974,
        0.0969, 0.0927, 0.0923, 0.0889, 0.0857, 0.0830, 0.0785, 0.0806,
        0.0740, 0.0776, 0.0739, 0.0740, 0.0717, 0.0715, 0.0723, 0.0680, 0.0705
    ],
    'val_loss': [
        0.2989, 0.3074, 0.2551, 0.2572, 0.2803, 0.2341, 0.2191, 0.2196,
        0.2034, 0.2186, 0.2088, 0.2115, 0.2219, 0.2159, 0.2170,
        0.2239, 0.2205, 0.2348, 0.2536, 0.2249, 0.2696, 0.2543, 0.2491,
        0.2483, 0.2487, 0.2512, 0.2499, 0.2581, 0.2529, 0.2578, 0.2490,
        0.2633, 0.2595, 0.2665, 0.2681, 0.2721, 0.2692, 0.2710, 0.2714, 0.2730
    ],
    'val_f1': [
        0.6327, 0.6462, 0.7221, 0.7118, 0.6798, 0.7541, 0.7835, 0.7488,
        0.8101, 0.7847, 0.8103, 0.8037, 0.7972, 0.8094, 0.8141,
        0.8000, 0.8049, 0.8190, 0.7953, 0.8058, 0.7778, 0.7915, 0.8047,
        0.8066, 0.8132, 0.8115, 0.8084, 0.8019, 0.8029, 0.8028, 0.8255,
        0.7972, 0.7962, 0.8095, 0.8132, 0.8028, 0.8075, 0.8150, 0.8169, 0.8047
    ],
    'epochs_head': 15,
}


class ExperimentData:
    """Load and manage experiment data."""

    def __init__(self, experiments_dir: str = "results/experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments = {}
        self._load_experiments()

    def _load_experiments(self):
        for json_file in self.experiments_dir.glob("*.json"):
            if json_file.name in ("index.json", "SUMMARY.md"):
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    exp_id = data.get('experiment_id', json_file.stem)
                    self.experiments[exp_id] = data
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

    def get_architecture_family(self, arch: str) -> str:
        if arch in ['vit_b_16', 'swin_t']:
            return 'Transformer'
        elif arch in ['vgg16']:
            return 'Classic CNN'
        elif arch == 'battlenet':
            return 'Novel'
        else:
            return 'CNN'


def _save(fig, stem, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/{stem}.pdf", format='pdf')
    fig.savefig(f"{output_dir}/{stem}.png", format='png')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Architecture Comparison (Phase 0 + BattleNet)
# ──────────────────────────────────────────────────────────────────────────────
def figure_1_architecture_comparison(output_dir: str = "results/figures"):
    arch_order = [
        ('vgg16',             'Classic CNN'),
        ('mobilenet_v2',      'CNN'),
        ('vit_b_16',          'Transformer'),
        ('densenet121',       'CNN'),
        ('convnext_tiny',     'CNN'),
        ('efficientnet_v2_s', 'CNN'),
        ('swin_t',            'Transformer'),
        ('battlenet',         'Novel'),
    ]

    names  = [ARCHITECTURE_NAMES[a] for a, _ in arch_order]
    f1s    = [PHASE0_RESULTS[a]['micro_f1'] if a != 'battlenet' else BATTLENET_RESULTS['micro']['f1']
              for a, _ in arch_order]
    accs   = [PHASE0_RESULTS[a]['accuracy'] if a != 'battlenet' else BATTLENET_RESULTS['exact_match']
              for a, _ in arch_order]
    colors = [ARCHITECTURE_COLORS[a] for a, _ in arch_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bars1 = ax1.barh(names, f1s, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Micro F1 Score', fontweight='bold')
    ax1.set_title('(a) Micro F1 Score by Architecture', fontweight='bold')
    ax1.set_xlim(0.65, 0.87)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    # Mark Swin-T and BattleNet
    ax1.axvline(x=SWINT_RESULTS['micro']['f1'], color=COLORS['transformer'],
                linestyle='--', linewidth=1.2, alpha=0.7, label='Swin-T baseline')
    ax1.axvline(x=BATTLENET_RESULTS['micro']['f1'], color=COLORS['battlenet'],
                linestyle='--', linewidth=1.2, alpha=0.7, label='BattleNet (Ours)')
    ax1.legend(fontsize=8)
    for bar, val in zip(bars1, f1s):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=8)

    bars2 = ax2.barh(names, accs, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Exact Match Accuracy', fontweight='bold')
    ax2.set_title('(b) Exact Match Accuracy by Architecture', fontweight='bold')
    ax2.set_xlim(0.45, 0.72)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    for bar, val in zip(bars2, accs):
        ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=8)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS['battlenet'],   label='Novel (BattleNet)'),
        mpatches.Patch(color=COLORS['transformer'], label='Transformer'),
        mpatches.Patch(color=COLORS['cnn'],         label='CNN'),
        mpatches.Patch(color=COLORS['classic'],     label='Classic CNN'),
    ]
    ax1.legend(handles=legend_patches, loc='lower right', fontsize=8)

    plt.tight_layout()
    _save(fig, 'figure_1_architecture_comparison', output_dir)
    print("Figure 1: Architecture comparison saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-Class Performance (BattleNet vs Swin-T)
# ──────────────────────────────────────────────────────────────────────────────
def figure_2_per_class_performance(output_dir: str = "results/figures"):
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    bn_f1   = [BATTLENET_RESULTS[c]['f1'] for c in CLASS_NAMES]
    swt_f1  = [SWINT_RESULTS[c]['f1']     for c in CLASS_NAMES]
    delta   = [b - s for b, s in zip(bn_f1, swt_f1)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    bars1 = ax1.bar(x - width/2, swt_f1, width, label='Swin-T (baseline)',
                    color=COLORS['transformer'], edgecolor='black', linewidth=0.7, alpha=0.85)
    bars2 = ax1.bar(x + width/2, bn_f1, width, label='BattleNet (proposed)',
                    color=COLORS['battlenet'],   edgecolor='black', linewidth=0.7, alpha=0.85)

    for bar, val in zip(bars1, swt_f1):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                 f'{val:.3f}', ha='center', fontsize=7.5)
    for bar, val in zip(bars2, bn_f1):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                 f'{val:.3f}', ha='center', fontsize=7.5)

    ax1.set_ylabel('F1 Score', fontweight='bold')
    ax1.set_title('(a) Per-Class F1: BattleNet vs Swin-T', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES, rotation=20, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0.55, 1.05)

    # Delta plot
    bar_colors = [COLORS['tertiary'] if d >= 0 else COLORS['secondary'] for d in delta]
    ax2.bar(x, delta, color=bar_colors, edgecolor='black', linewidth=0.7)
    ax2.axhline(0, color='black', linewidth=1)
    for i, (d, c) in enumerate(zip(delta, CLASS_NAMES)):
        ax2.text(i, d + (0.003 if d >= 0 else -0.006),
                 f'{d:+.3f}', ha='center', fontsize=8, fontweight='bold')
    ax2.set_ylabel('ΔF1 (BattleNet − Swin-T)', fontweight='bold')
    ax2.set_title('(b) F1 Improvement of BattleNet over Swin-T', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASS_NAMES, rotation=20, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    overall_delta = BATTLENET_RESULTS['micro']['f1'] - SWINT_RESULTS['micro']['f1']
    ax2.text(0.98, 0.95, f'Overall Micro ΔF1 = {overall_delta:+.4f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    _save(fig, 'figure_2_per_class_performance', output_dir)
    print("Figure 2: Per-class performance saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: BattleNet Training Curves
# ──────────────────────────────────────────────────────────────────────────────
def figure_3_training_curves(output_dir: str = "results/figures"):
    h = BATTLENET_HISTORY
    n = len(h['train_loss'])
    epochs = list(range(1, n + 1))
    ep_head = h['epochs_head']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(epochs[:ep_head], h['train_loss'][:ep_head],
             color=COLORS['secondary'], marker='o', markersize=3, linewidth=1.8, label='Train (Phase 1)')
    ax1.plot(epochs[ep_head:], h['train_loss'][ep_head:],
             color=COLORS['quaternary'], marker='o', markersize=3, linewidth=1.8, label='Train (Phase 2)')
    ax1.plot(epochs[:ep_head], h['val_loss'][:ep_head],
             color=COLORS['secondary'], marker='s', markersize=3, linewidth=1.8,
             linestyle='--', label='Val (Phase 1)')
    ax1.plot(epochs[ep_head:], h['val_loss'][ep_head:],
             color=COLORS['quaternary'], marker='s', markersize=3, linewidth=1.8,
             linestyle='--', label='Val (Phase 2)')
    ax1.axvline(x=ep_head + 0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.text(ep_head/2, max(h['train_loss'])*0.98, 'Head Training\n(backbone frozen)',
             ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax1.text(ep_head + (n - ep_head)/2, max(h['train_loss'])*0.98, 'Fine-tuning\n(all layers)',
             ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('(a) Training and Validation Loss', fontweight='bold')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # F1
    ax2.plot(epochs[:ep_head], h['val_f1'][:ep_head],
             color=COLORS['secondary'], marker='o', markersize=3, linewidth=1.8, label='Phase 1 (head)')
    ax2.plot(epochs[ep_head:], h['val_f1'][ep_head:],
             color=COLORS['quaternary'], marker='o', markersize=3, linewidth=1.8, label='Phase 2 (finetune)')
    best_f1 = max(h['val_f1'])
    best_ep = h['val_f1'].index(best_f1) + 1
    ax2.axhline(y=best_f1, color=COLORS['battlenet'], linestyle='--', linewidth=1.5,
                label=f'Best val F1: {best_f1:.4f} (ep {best_ep})')
    ax2.axhline(y=BATTLENET_RESULTS['micro']['f1'], color=COLORS['primary'], linestyle='-.',
                linewidth=1.5, label=f'Test Micro F1: {BATTLENET_RESULTS["micro"]["f1"]:.4f}')
    ax2.axvline(x=ep_head + 0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Validation Micro F1', fontweight='bold')
    ax2.set_title('(b) Validation F1 Over Training', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0.6, 0.87)

    plt.tight_layout()
    _save(fig, 'figure_3_training_curves', output_dir)
    print("Figure 3: Training curves saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Hyperparameter Tuning (Swin-T & EfficientNetV2-S)
# ──────────────────────────────────────────────────────────────────────────────
def figure_4_hyperparameter_tuning(output_dir: str = "results/figures"):
    # Swin-T Phase 2 results (Adam, local MPS)
    configs  = ['Baseline\n(WD=0)', 'WD=1e-4', 'WD=1e-3\n+Cosine', 'Label\nSmooth 0.1', 'AdamW', 'Lower\nFinetune LR']
    swin_f1  = [0.8083, 0.8000, 0.8121, 0.7954, 0.8037, 0.8046]
    eff_f1   = [0.7926, 0.7925, 0.8084, 0.7825, 0.7896, 0.7720]

    x = np.arange(len(configs))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    bars1 = ax1.bar(x - width/2, swin_f1, width, label='Swin-T',
                    color=COLORS['transformer'], edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, eff_f1, width, label='EfficientNetV2-S',
                    color=COLORS['cnn'], edgecolor='black', linewidth=0.8)
    ax1.axhline(y=BATTLENET_RESULTS['micro']['f1'], color=COLORS['battlenet'],
                linestyle='--', linewidth=1.5, label=f'BattleNet: {BATTLENET_RESULTS["micro"]["f1"]:.4f}')
    ax1.set_xlabel('Configuration', fontweight='bold')
    ax1.set_ylabel('Micro F1 Score', fontweight='bold')
    ax1.set_title('(a) Regularization Techniques', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0.75, 0.84)
    for bars in [bars1, bars2]:
        for bar in bars:
            h_ = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h_ + 0.001,
                     f'{h_:.4f}', ha='center', va='bottom', fontsize=6.5)

    # LR Schedulers
    configs2 = ['No Scheduler\n(baseline)', 'Step LR', 'Cosine\nAnnealing', 'OneCycle']
    f1_sched = [0.8083, 0.8112, 0.8121, 0.7599]
    bar_cols  = [COLORS['transformer'] if v == max(f1_sched) else '#93C6F4' for v in f1_sched]

    bars3 = ax2.bar(configs2, f1_sched, color=bar_cols, edgecolor='black', linewidth=0.8)
    ax2.axhline(y=BATTLENET_RESULTS['micro']['f1'], color=COLORS['battlenet'],
                linestyle='--', linewidth=1.5, label=f'BattleNet: {BATTLENET_RESULTS["micro"]["f1"]:.4f}')
    ax2.set_xlabel('Learning Rate Schedule', fontweight='bold')
    ax2.set_ylabel('Micro F1 Score', fontweight='bold')
    ax2.set_title('(b) Learning Rate Schedules (Swin-T)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0.73, 0.84)
    ax2.legend(fontsize=8)
    for bar in bars3:
        h_ = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h_ + 0.002,
                 f'{h_:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    _save(fig, 'figure_4_hyperparameter_tuning', output_dir)
    print("Figure 4: Hyperparameter tuning saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Phase Progression Including BattleNet
# ──────────────────────────────────────────────────────────────────────────────
def figure_5_phase_comparison(output_dir: str = "results/figures"):
    phases = [
        'Phase 0\nArchitecture\nSurvey\n(Swin-T)',
        'Phase 2\nRegularization\n(WD=1e-3\n+Cosine)',
        'BattleNet\n(Proposed\nArchitecture)',
    ]
    f1s  = [0.8083, 0.8121, 0.8178]
    accs = [0.6706, 0.6765, 0.6706]
    bar_colors_f1  = [COLORS['transformer'], COLORS['transformer'], COLORS['battlenet']]
    bar_colors_acc = [COLORS['tertiary'],    COLORS['tertiary'],    COLORS['quaternary']]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(phases))
    w = 0.35

    bars1 = ax.bar(x - w/2, f1s, w, color=bar_colors_f1,
                   edgecolor='black', linewidth=0.8, label='Micro F1 Score')
    bars2 = ax.bar(x + w/2, accs, w, color=bar_colors_acc,
                   edgecolor='black', linewidth=0.8, label='Exact Match Accuracy', alpha=0.85)

    # Improvement annotations
    ax.annotate('', xy=(1 - w/2, f1s[1] + 0.005), xytext=(0 - w/2, f1s[0] + 0.005),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.5))
    ax.text(0.5 - w/2, max(f1s[0], f1s[1]) + 0.012,
            f'+{f1s[1]-f1s[0]:.4f}', ha='center', fontsize=9,
            color=COLORS['secondary'], fontweight='bold')

    ax.annotate('', xy=(2 - w/2, f1s[2] + 0.005), xytext=(1 - w/2, f1s[1] + 0.005),
                arrowprops=dict(arrowstyle='->', color=COLORS['battlenet'], lw=1.5))
    ax.text(1.5 - w/2, max(f1s[1], f1s[2]) + 0.012,
            f'+{f1s[2]-f1s[1]:.4f}', ha='center', fontsize=9,
            color=COLORS['battlenet'], fontweight='bold')

    for bar, val in zip(bars1, f1s):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, accs):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Optimization Phase', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Progression: Architecture Survey → Optimization → BattleNet',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=9)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.6, 0.88)

    plt.tight_layout()
    _save(fig, 'figure_5_phase_comparison', output_dir)
    print("Figure 5: Phase comparison saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: Class Imbalance Handling
# ──────────────────────────────────────────────────────────────────────────────
def figure_6_class_imbalance(output_dir: str = "results/figures"):
    configs = ['Baseline\n(BCE)', 'Pos Weight\n2.0', 'Pos Weight\n3.0',
               'Focal Loss\n(γ=2)', 'Auto Class\nWeights']
    overall_f1  = [0.8121, 0.7974, 0.7919, 0.7981, 0.7907]
    vehicle_f1  = [0.7312, 0.7500, 0.6700, 0.7400, 0.6800]

    x = np.arange(len(configs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, overall_f1, w, label='Overall Micro F1',
                   color=COLORS['primary'], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + w/2, vehicle_f1, w, label='Vehicle Class F1',
                   color=COLORS['secondary'], edgecolor='black', linewidth=0.8)
    for bars in [bars1, bars2]:
        for bar in bars:
            h_ = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h_ + 0.005,
                    f'{h_:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Class Imbalance Technique', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Impact of Class Imbalance Techniques on Swin-T (WD=1e-3 + Cosine)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.60, 0.88)

    plt.tight_layout()
    _save(fig, 'figure_6_class_imbalance', output_dir)
    print("Figure 6: Class imbalance saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7: Data Augmentation Comparison
# ──────────────────────────────────────────────────────────────────────────────
def figure_7_augmentation_comparison(output_dir: str = "results/figures"):
    aug_labels = ['Baseline', 'None', 'Light', 'Strong', 'Color', 'Aggressive']
    f1s  = [0.8121, 0.8083, 0.7739, 0.7981, 0.8073, 0.8083]
    accs = [0.6765, 0.6529, 0.6353, 0.6353, 0.6882, 0.6471]
    veh  = [0.7312, 0.7400, 0.7000, 0.6800, 0.7500, 0.7300]

    x = np.arange(len(aug_labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(x, f1s,  marker='o', label='Micro F1',  color=COLORS['primary'],   linewidth=2)
    ax1.plot(x, accs, marker='s', label='Exact Match', color=COLORS['tertiary'], linewidth=2)
    ax1.set_xlabel('Augmentation Strategy', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('(a) Micro F1 and Accuracy', fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(aug_labels, fontsize=9, rotation=15)
    ax1.legend(); ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim(0.60, 0.88)
    for i, (f, a) in enumerate(zip(f1s, accs)):
        ax1.text(i, f + 0.005, f'{f:.4f}', ha='center', fontsize=7.5, color=COLORS['primary'])

    bars = ax2.bar(x, veh, color=COLORS['secondary'], edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Augmentation Strategy', fontweight='bold')
    ax2.set_ylabel('Vehicle F1 Score', fontweight='bold')
    ax2.set_title('(b) Vehicle Class Performance', fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(aug_labels, fontsize=9, rotation=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0.60, 0.82)
    for bar in bars:
        h_ = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h_ + 0.008,
                 f'{h_:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    _save(fig, 'figure_7_augmentation_comparison', output_dir)
    print("Figure 7: Augmentation comparison saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8: Model Family Comparison
# ──────────────────────────────────────────────────────────────────────────────
def figure_8_model_family_comparison(output_dir: str = "results/figures"):
    families = {
        'Classic CNN': [('VGG16', 0.7021)],
        'CNN':         [('ResNet-18', 0.7189), ('ResNet-34', 0.7583), ('ResNet-50', 0.7598),
                        ('MobileNetV2', 0.7286), ('EfficientNet-B0', 0.7653),
                        ('EfficientNetV2-S', 0.7926), ('DenseNet-121', 0.7830),
                        ('ConvNeXt-T', 0.7854)],
        'Transformer': [('ViT-B/16', 0.7563), ('Swin-T', 0.8121)],
        'Novel':       [('BattleNet', 0.8178)],
    }
    fam_colors = {'Classic CNN': COLORS['classic'], 'CNN': COLORS['cnn'],
                  'Transformer': COLORS['transformer'], 'Novel': COLORS['battlenet']}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Family means
    fam_names = list(families.keys())
    fam_means = [np.mean([v for _, v in families[f]]) for f in fam_names]
    fam_maxes = [max(v for _, v in families[f]) for f in fam_names]
    col = [fam_colors[f] for f in fam_names]
    x = np.arange(len(fam_names))
    w = 0.35
    ax1.bar(x - w/2, fam_means, w, label='Mean F1', color=col, edgecolor='black', linewidth=0.8)
    ax1.bar(x + w/2, fam_maxes, w, label='Best F1', color=col, edgecolor='black',
            linewidth=0.8, alpha=0.6, hatch='//')
    ax1.set_ylabel('Micro F1 Score', fontweight='bold')
    ax1.set_title('(a) Mean & Best Micro F1 per Family', fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(fam_names)
    ax1.legend(); ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0.65, 0.85)

    # Individual models
    all_models, all_f1s, all_cols = [], [], []
    for fam in ['Classic CNN', 'CNN', 'Transformer', 'Novel']:
        for name, f1 in families[fam]:
            all_models.append(f'{name}\n({fam})')
            all_f1s.append(f1)
            all_cols.append(fam_colors[fam])

    y_pos = np.arange(len(all_models))
    ax2.barh(y_pos, all_f1s, color=all_cols, edgecolor='black', linewidth=0.6)
    ax2.set_xlabel('Micro F1 Score', fontweight='bold')
    ax2.set_title('(b) Individual Model Performance', fontweight='bold')
    ax2.set_yticks(y_pos); ax2.set_yticklabels(all_models, fontsize=8)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.axvline(x=0.8121, color=COLORS['transformer'], linestyle='--',
                linewidth=1, alpha=0.6, label='Swin-T')
    ax2.axvline(x=0.8178, color=COLORS['battlenet'], linestyle='--',
                linewidth=1, alpha=0.8, label='BattleNet')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, 'figure_8_model_family_comparison', output_dir)
    print("Figure 8: Model family comparison saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 9: Per-Class Metrics Heatmap (BattleNet vs Swin-T side by side)
# ──────────────────────────────────────────────────────────────────────────────
def figure_9_per_class_heatmap(output_dir: str = "results/figures"):
    metrics = ['Precision', 'Recall', 'F1']
    bn_data  = np.array([[BATTLENET_RESULTS[c][m.lower()] for m in metrics] for c in CLASS_NAMES])
    swt_data = np.array([[SWINT_RESULTS[c][m.lower()] for m in metrics] for c in CLASS_NAMES])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title in [(ax1, bn_data, 'BattleNet (Proposed)'),
                             (ax2, swt_data, 'Swin-T (Baseline)')]:
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.65, vmax=1.0)
        ax.set_xticks(np.arange(3)); ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(CLASS_NAMES)
        for i in range(len(CLASS_NAMES)):
            for j in range(3):
                ax.text(j, i, f'{data[i, j]:.3f}',
                        ha='center', va='center', fontsize=9, fontweight='bold')
        ax.set_title(f'Per-Class Metrics: {title}', fontweight='bold', fontsize=11)
        fig.colorbar(im, ax=ax, label='Score')

    plt.tight_layout()
    _save(fig, 'figure_9_per_class_heatmap', output_dir)
    print("Figure 9: Per-class heatmap saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 10: Label Distribution (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def figure_10_label_distribution(output_dir: str = "results/figures"):
    label_counts = {'Radar': 47, 'Artilary': 36, 'M. Rocket Launcher': 28,
                    'Missile': 25, 'Tank': 27, 'Soldier': 35, 'Vehicle': 47}
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1])
    classes = [i[0] for i in sorted_items]
    counts  = [i[1] for i in sorted_items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors_bar = [COLORS['tertiary'] if c < 30 else COLORS['primary'] for c in counts]
    ax1.barh(classes, counts, color=colors_bar, edgecolor='black', linewidth=0.8)
    for i, c in enumerate(counts):
        ax1.text(c + 0.5, i, f'{c}', va='center', fontsize=9, fontweight='bold')
    ax1.axvline(x=np.mean(counts), color=COLORS['secondary'], linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {np.mean(counts):.1f}')
    ax1.set_xlabel('Number of Training Samples', fontweight='bold')
    ax1.set_title('(a) Class Distribution in Training Set', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.legend()

    pie_colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
                  COLORS['quaternary'], COLORS['quinary'], COLORS['senary'], COLORS['cnn']]
    _, _, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%',
                              colors=pie_colors, startangle=90,
                              wedgeprops=dict(edgecolor='black', linewidth=0.8))
    for at in autotexts:
        at.set_fontsize(8); at.set_fontweight('bold')
    ax2.set_title('(b) Class Proportion', fontweight='bold')

    plt.tight_layout()
    _save(fig, 'figure_10_label_distribution', output_dir)
    print("Figure 10: Label distribution saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 11: Confusion Analysis (Swin-T, for reference)
# ──────────────────────────────────────────────────────────────────────────────
def figure_11_confusion_analysis(output_dir: str = "results/figures"):
    conf = np.array([[SWINT_RESULTS[c]['tp'], SWINT_RESULTS[c]['fp'],
                      SWINT_RESULTS[c]['fn'], SWINT_RESULTS[c]['tn']]
                     for c in CLASS_NAMES], dtype=float)
    conf_norm = conf / conf.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(conf_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(4)); ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(['True Pos', 'False Pos', 'False Neg', 'True Neg'])
    ax.set_yticklabels(CLASS_NAMES)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    for i in range(len(CLASS_NAMES)):
        for j in range(4):
            tc = 'white' if conf_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{int(conf[i,j])}\n({conf_norm[i,j]:.1%})',
                    ha='center', va='center', color=tc, fontsize=7)
    ax.set_title('Per-Class Confusion Analysis — Swin-T Baseline\n(Values: Count / Row %)',
                 fontweight='bold', fontsize=11)
    fig.colorbar(im, ax=ax, label='Proportion')

    plt.tight_layout()
    _save(fig, 'figure_11_confusion_analysis', output_dir)
    print("Figure 11: Confusion analysis saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 12: Model Efficiency Scatter
# ──────────────────────────────────────────────────────────────────────────────
def figure_12_model_efficiency(output_dir: str = "results/figures"):
    # Approximate training times (minutes, 30 epochs on local MPS)
    models = [
        ('VGG16',          7.02, 0.7021, COLORS['classic']),
        ('MobileNetV2',    2.30, 0.7286, COLORS['cnn']),
        ('ViT-B/16',       6.80, 0.7563, COLORS['transformer']),
        ('DenseNet-121',   4.20, 0.7830, COLORS['cnn']),
        ('ConvNeXt-T',     4.80, 0.7854, COLORS['cnn']),
        ('EfficientNetV2', 4.50, 0.7926, COLORS['cnn']),
        ('Swin-T',         7.50, 0.8121, COLORS['transformer']),
        ('BattleNet',      8.20, 0.8178, COLORS['battlenet']),
    ]
    names, times, f1s, cols = zip(*models)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(times, f1s, c=cols, s=200, edgecolors='black', linewidths=1.5, alpha=0.85, zorder=3)
    for name, t, f in zip(names, times, f1s):
        ax.annotate(name, (t, f), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.axhline(y=0.78, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=6, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2.5, 0.822, 'Fast & Good',  fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.45))
    ax.text(7.0, 0.822, 'Slow & Good', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.55))
    ax.text(2.5, 0.724, 'Fast & Poor', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.45))

    legend_patches = [
        mpatches.Patch(color=COLORS['battlenet'],   label='Novel (BattleNet)'),
        mpatches.Patch(color=COLORS['transformer'], label='Transformer'),
        mpatches.Patch(color=COLORS['cnn'],         label='CNN'),
        mpatches.Patch(color=COLORS['classic'],     label='Classic CNN'),
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    ax.set_xlabel('Training Time (minutes, 30 epochs)', fontweight='bold')
    ax.set_ylabel('Test Micro F1 Score', fontweight='bold')
    ax.set_title('Model Efficiency: Performance vs. Training Time', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1.5, 9.5)
    ax.set_ylim(0.70, 0.84)

    plt.tight_layout()
    _save(fig, 'figure_12_model_efficiency', output_dir)
    print("Figure 12: Model efficiency saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 13: BattleNet Architecture Diagram
# ──────────────────────────────────────────────────────────────────────────────
def figure_13_battlenet_architecture(output_dir: str = "results/figures"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_facecolor('#FAFAFA')

    def box(x, y, w, h, label, sublabel='', color='#E8F4FD', ec='#2E5EAA', fontsize=10):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.05", facecolor=color,
                               edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold')
        if sublabel:
            ax.text(x, y - 0.25, sublabel, ha='center', va='center',
                    fontsize=fontsize - 1.5, color='#444')

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.5))

    # Input
    box(1.0, 3.0, 1.4, 0.7, 'Input Image', '224×224×3', color='#FFF8E1', ec='#F9A825')
    # Swin-T backbone
    box(3.2, 3.0, 2.0, 1.2, 'Swin-T Backbone', 'ImageNet pretrained\n768-d features',
        color='#EDE7F6', ec='#7B1FA2')
    # Projection
    box(5.8, 3.0, 1.6, 0.8, 'Linear Proj.', '→ 768-d', color='#E8F5E9', ec='#388E3C')
    # SE + BN
    box(7.6, 3.0, 1.6, 0.8, 'SE + BN', 'Channel recalib.', color='#FFF3E0', ec='#E65100')
    # Residual MLP
    box(9.4, 3.0, 1.6, 0.8, 'Residual MLP', '768→768', color='#E8EAF6', ec='#3949AB')
    # Outputs
    box(11.0, 4.1, 1.4, 0.65, 'LabelCoOccur', '7×7 matrix', color='#FCE4EC', ec='#C62828', fontsize=9)
    box(11.0, 3.0, 1.4, 0.65, 'Output', '7 logits', color='#E8F5E9', ec='#1B5E20', fontsize=9)

    # Arrows
    arrow(1.7, 3.0, 2.2, 3.0)
    arrow(4.2, 3.0, 4.8, 3.0)
    arrow(6.6, 3.0, 6.8, 3.0)
    arrow(8.4, 3.0, 8.6, 3.0)
    arrow(10.2, 3.0, 10.3, 3.0)
    ax.annotate('', xy=(10.3, 4.1), xytext=(10.0, 3.4),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.2, connectionstyle='arc3,rad=-0.3'))

    # Bracket for BattleNetClassifier
    ax.annotate('', xy=(10.8, 2.0), xytext=(4.8, 2.0),
                arrowprops=dict(arrowstyle='<->', color=COLORS['battlenet'], lw=1.5))
    ax.text(7.8, 1.75, 'BattleNetClassifier Head (novel)',
            ha='center', fontsize=10, color=COLORS['battlenet'], fontweight='bold')

    ax.set_title('BattleNet Architecture: Swin-T Backbone + Specialized Classification Head',
                 fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    _save(fig, 'figure_13_battlenet_architecture', output_dir)
    print("Figure 13: BattleNet architecture saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 14: BattleNet Training Detail (val F1 + loss per epoch)
# ──────────────────────────────────────────────────────────────────────────────
def figure_14_battlenet_training_detail(output_dir: str = "results/figures"):
    h = BATTLENET_HISTORY
    n = len(h['train_loss'])
    epochs = list(range(1, n + 1))
    ep_h = h['epochs_head']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    ax.plot(epochs, h['val_f1'],   color=COLORS['primary'],    linewidth=2, label='Val F1', zorder=3)
    ax2.plot(epochs, h['val_loss'], color=COLORS['secondary'], linewidth=1.5,
             linestyle='--', label='Val Loss', zorder=2)
    ax.axvline(x=ep_h + 0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axhline(y=max(h['val_f1']), color=COLORS['primary'], linestyle='-.', linewidth=1,
               alpha=0.5, label=f'Peak val F1 = {max(h["val_f1"]):.4f}')
    ax.axhline(y=BATTLENET_RESULTS['micro']['f1'], color=COLORS['battlenet'],
               linestyle='--', linewidth=1.5, label=f'Test F1 = {BATTLENET_RESULTS["micro"]["f1"]:.4f}')

    ax.text(ep_h / 2, 0.62, 'Phase 1\n(Head)', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.text(ep_h + (n - ep_h) / 2, 0.62, 'Phase 2\n(Fine-tune)', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation Micro F1', fontweight='bold', color=COLORS['primary'])
    ax2.set_ylabel('Validation Loss', fontweight='bold', color=COLORS['secondary'])
    ax.set_title('BattleNet: Validation F1 and Loss per Epoch (40-epoch training)',
                 fontweight='bold')
    ax.set_ylim(0.58, 0.87)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    _save(fig, 'figure_14_battlenet_training_detail', output_dir)
    print("Figure 14: BattleNet training detail saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 15: BattleNet vs Swin-T Head-to-Head
# ──────────────────────────────────────────────────────────────────────────────
def figure_15_battlenet_vs_backbones(output_dir: str = "results/figures"):
    metrics_labels = ['Micro F1', 'Macro F1', 'Exact Match\nAccuracy',
                      'Micro Prec.', 'Micro Recall']
    bn_vals  = [BATTLENET_RESULTS['micro']['f1'], BATTLENET_RESULTS['macro']['f1'],
                BATTLENET_RESULTS['exact_match'],
                BATTLENET_RESULTS['micro']['precision'], BATTLENET_RESULTS['micro']['recall']]
    swt_vals = [SWINT_RESULTS['micro']['f1'], SWINT_RESULTS['macro']['f1'],
                SWINT_RESULTS['exact_match'],
                SWINT_RESULTS['micro']['precision'], SWINT_RESULTS['micro']['recall']]

    x = np.arange(len(metrics_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w/2, swt_vals, w, label='Swin-T (baseline, WD=1e-3+Cosine)',
                   color=COLORS['transformer'], edgecolor='black', linewidth=0.8, alpha=0.85)
    bars2 = ax.bar(x + w/2, bn_vals, w, label='BattleNet (proposed)',
                   color=COLORS['battlenet'],   edgecolor='black', linewidth=0.8, alpha=0.85)

    for bars, vals in [(bars1, swt_vals), (bars2, bn_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., v + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Improvement arrows on F1
    delta_micro = bn_vals[0] - swt_vals[0]
    ax.annotate(f'+{delta_micro:.4f}', xy=(0 + w/2, bn_vals[0] + 0.025),
                fontsize=10, ha='center', color=COLORS['battlenet'], fontweight='bold')

    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('BattleNet vs. Swin-T Baseline: All Metrics Comparison',
                 fontweight='bold', fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(metrics_labels, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.60, 0.92)

    plt.tight_layout()
    _save(fig, 'figure_15_battlenet_vs_backbones', output_dir)
    print("Figure 15: BattleNet vs Swin-T saved.")


# ──────────────────────────────────────────────────────────────────────────────
def generate_all_figures():
    print("Generating all figures with updated BattleNet results...")
    print(f"  Swin-T baseline:  Micro F1 = {SWINT_RESULTS['micro']['f1']:.4f}")
    print(f"  BattleNet:        Micro F1 = {BATTLENET_RESULTS['micro']['f1']:.4f}")
    print("-" * 55)

    figure_1_architecture_comparison()
    figure_2_per_class_performance()
    figure_3_training_curves()
    figure_4_hyperparameter_tuning()
    figure_5_phase_comparison()
    figure_6_class_imbalance()
    figure_7_augmentation_comparison()
    figure_8_model_family_comparison()
    figure_9_per_class_heatmap()
    figure_10_label_distribution()
    figure_11_confusion_analysis()
    figure_12_model_efficiency()
    figure_13_battlenet_architecture()
    figure_14_battlenet_training_detail()
    figure_15_battlenet_vs_backbones()

    print("-" * 55)
    print("All 15 figures saved to: results/figures/")


if __name__ == "__main__":
    generate_all_figures()
