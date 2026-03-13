#!/usr/bin/env python3
"""
Fix broken figures 1, 2, and 8 in the BattleNet research project.
These figures should show ALL models, but were only showing BattleNet and ResNet18.

This script loads all model data from test_results.json files and regenerates the 3 figures.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    'hybrid': '#DC2626',       # Red for Hybrid/BattleNet
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
    'battlenet': COLORS['hybrid'],
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
    'battlenet': 'BattleNet',
}

CLASS_NAMES = ['Artilary', 'Missile', 'Radar', 'M. Rocket Launcher', 'Soldier', 'Tank', 'Vehicle']


def get_architecture_family(arch: str) -> str:
    """Get the family (CNN, Transformer, Hybrid, Classic) for an architecture."""
    if arch == 'battlenet':
        return 'Hybrid'
    elif arch in ['vit_b_16', 'swin_t']:
        return 'Transformer'
    elif arch in ['vgg16']:
        return 'Classic CNN'
    else:
        return 'CNN'


def load_best_test_results() -> Dict[str, Dict]:
    """
    Load best test results for each architecture from test_results.json files.
    Returns dict mapping architecture name to best test result data.
    """
    results_dir = Path('results')
    architectures = {}

    # Get all architectures
    for test_file in results_dir.glob('*_test_results.json'):
        # Extract architecture name from filename
        # Format: {architecture}_imagenet_*.json
        stem = test_file.stem
        parts = stem.split('_imagenet_')
        if len(parts) != 2:
            continue
        arch = parts[0]

        if arch not in architectures:
            architectures[arch] = None

    # For each architecture, find the best result
    for arch in architectures:
        test_files = list(results_dir.glob(f'{arch}_imagenet_*_test_results.json'))

        if not test_files:
            print(f"Warning: No test results found for {arch}")
            continue

        best_f1 = 0
        best_data = None
        for f in test_files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    f1 = data['metrics']['micro']['f1']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_data = data
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")

        if best_data:
            architectures[arch] = best_data
            print(f"Loaded {arch}: micro_f1={best_f1:.4f}")
        else:
            print(f"Warning: No valid test data found for {arch}")

    return architectures


def figure_1_architecture_comparison(all_results: Dict[str, Dict], output_dir: str = "results/figures"):
    """
    Figure 1: Architecture comparison showing Micro F1 and Exact Match Accuracy.
    Includes all 12 models sorted by family and F1 score.
    """
    print("\nGenerating Figure 1: Architecture comparison...")

    # Prepare data sorted by family and F1 score
    arch_data = []
    for arch, result_data in all_results.items():
        if result_data is None:
            continue

        metrics = result_data['metrics']
        arch_data.append({
            'architecture': ARCHITECTURE_NAMES.get(arch, arch),
            'family': get_architecture_family(arch),
            'micro_f1': metrics.get('micro', {}).get('f1', 0),
            'accuracy': metrics.get('exact_match_accuracy', 0),
            'color': ARCHITECTURE_COLORS.get(arch, COLORS['primary']),
        })

    # Sort by family and then by F1 score
    family_order = ['Transformer', 'Hybrid', 'CNN', 'Classic CNN']
    arch_data.sort(key=lambda x: (family_order.index(x['family']), -x['micro_f1']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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

    print(f"✓ Figure 1 saved to {output_dir}/")


def figure_2_per_class_performance(all_results: Dict[str, Dict], output_dir: str = "results/figures"):
    """
    Figure 2: Per-class F1 scores for top 5 architectures.
    Shows performance variation across classes for best performing models.
    """
    print("Generating Figure 2: Per-class performance...")

    # Get top 5 architectures by Micro F1
    sorted_archs = sorted(
        [(arch, result_data['metrics']['micro']['f1'])
         for arch, result_data in all_results.items() if result_data is not None],
        key=lambda x: -x[1]
    )[:5]

    top_archs = [arch for arch, _ in sorted_archs]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(CLASS_NAMES))
    width = 0.15

    for i, arch in enumerate(top_archs):
        result_data = all_results[arch]
        metrics = result_data['metrics']
        f1_scores = [metrics.get(cls, {}).get('f1', 0) for cls in CLASS_NAMES]

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
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_2_per_class_performance.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_2_per_class_performance.png", format='png')
    plt.close()

    print(f"✓ Figure 2 saved to {output_dir}/")


def figure_8_model_family_comparison(all_results: Dict[str, Dict], output_dir: str = "results/figures"):
    """
    Figure 8: Model family comparison showing Transformer vs Hybrid vs CNN vs Classic CNN.
    Two subplots: (a) mean performance by family, (b) individual model performance.
    """
    print("Generating Figure 8: Model family comparison...")

    # Group by family
    families = {}
    for arch, result_data in all_results.items():
        if result_data is None:
            continue

        family = get_architecture_family(arch)
        metrics = result_data['metrics']
        f1 = metrics.get('micro', {}).get('f1', 0)
        accuracy = metrics.get('exact_match_accuracy', 0)

        if family not in families:
            families[family] = []
        families[family].append({
            'arch': arch,
            'arch_name': ARCHITECTURE_NAMES.get(arch, arch),
            'f1': f1,
            'accuracy': accuracy,
            'color': ARCHITECTURE_COLORS.get(arch, COLORS['primary'])
        })

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    family_colors = {
        'Transformer': COLORS['transformer'],
        'Hybrid': COLORS['hybrid'],
        'CNN': COLORS['cnn'],
        'Classic CNN': COLORS['classic'],
    }

    # --- Subplot (a): Mean performance by family ---
    family_order = ['Transformer', 'Hybrid', 'CNN', 'Classic CNN']
    family_means_f1 = []
    family_means_acc = []
    family_labels = []

    for family in family_order:
        if family in families:
            f1_vals = [m['f1'] for m in families[family]]
            acc_vals = [m['accuracy'] for m in families[family]]
            family_means_f1.append(np.mean(f1_vals))
            family_means_acc.append(np.mean(acc_vals))
            family_labels.append(family)

    x_pos = np.arange(len(family_labels))
    width = 0.35

    bars1_f1 = ax1.bar(x_pos - width/2, family_means_f1, width, label='Micro F1',
                       color=COLORS['primary'], edgecolor='black', linewidth=0.8)
    bars1_acc = ax1.bar(x_pos + width/2, family_means_acc, width, label='Exact Match Accuracy',
                        color=COLORS['secondary'], edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('(a) Mean Performance by Model Family', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(family_labels)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0.65, 0.9)

    # Add value labels
    for bars in [bars1_f1, bars1_acc]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    # --- Subplot (b): Individual model performance ---
    positions = []
    labels = []
    colors_list = []
    f1_list = []
    pos = 0

    for family in family_order:
        if family in families:
            # Sort by F1 within family
            sorted_archs = sorted(families[family], key=lambda x: -x['f1'])

            for arch_data in sorted_archs:
                positions.append(pos)
                labels.append(arch_data['arch_name'])
                colors_list.append(arch_data['color'])
                f1_list.append(arch_data['f1'])
                pos += 1

            # Add spacing between families
            pos += 1

    bars2 = ax2.barh(positions, f1_list, color=colors_list, edgecolor='black', linewidth=0.8)

    ax2.set_yticks(positions)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Micro F1 Score', fontweight='bold')
    ax2.set_title('(b) Individual Model Performance', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0.65, 0.85)

    # Add value labels
    for i, (pos_val, f1_val) in enumerate(zip(positions, f1_list)):
        ax2.text(f1_val + 0.003, pos_val, f'{f1_val:.4f}', va='center', fontsize=8)

    # Add legend for families
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['transformer'], edgecolor='black', label='Transformer'),
        mpatches.Patch(facecolor=COLORS['hybrid'], edgecolor='black', label='Hybrid'),
        mpatches.Patch(facecolor=COLORS['cnn'], edgecolor='black', label='CNN'),
        mpatches.Patch(facecolor=COLORS['classic'], edgecolor='black', label='Classic CNN'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/figure_8_model_family_comparison.pdf", format='pdf')
    plt.savefig(f"{output_dir}/figure_8_model_family_comparison.png", format='png')
    plt.close()

    print(f"✓ Figure 8 saved to {output_dir}/")


def main():
    print("=" * 60)
    print("Fixing broken figures 1, 2, and 8")
    print("=" * 60)

    # Load all test results
    print("\nLoading test results for all architectures...")
    all_results = load_best_test_results()

    print(f"\nLoaded results for {sum(1 for r in all_results.values() if r is not None)} architectures:")
    for arch, result in sorted(all_results.items()):
        if result:
            f1 = result['metrics']['micro']['f1']
            acc = result['metrics']['exact_match_accuracy']
            print(f"  {ARCHITECTURE_NAMES.get(arch, arch):20s}: F1={f1:.4f}, Accuracy={acc:.4f}")

    # Generate the 3 figures
    output_dir = "results/figures"
    figure_1_architecture_comparison(all_results, output_dir)
    figure_2_per_class_performance(all_results, output_dir)
    figure_8_model_family_comparison(all_results, output_dir)

    # Verify outputs
    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)
    for fig_num in [1, 2, 8]:
        pdf_file = f"{output_dir}/figure_{fig_num}_*.pdf"
        png_file = f"{output_dir}/figure_{fig_num}_*.png"

        # Get actual filenames
        import glob
        pdf_files = glob.glob(f"{output_dir}/figure_{fig_num}_*.pdf")
        png_files = glob.glob(f"{output_dir}/figure_{fig_num}_*.png")

        if pdf_files and png_files:
            pdf_size = os.path.getsize(pdf_files[0])
            png_size = os.path.getsize(png_files[0])
            print(f"Figure {fig_num}:")
            print(f"  PDF: {pdf_files[0]} ({pdf_size:,} bytes)")
            print(f"  PNG: {png_files[0]} ({png_size:,} bytes)")
            if pdf_size > 50000 and png_size > 50000:
                print(f"  ✓ Both files generated successfully")
            else:
                print(f"  ⚠ Warning: Files may be too small")
        else:
            print(f"Figure {fig_num}: ✗ NOT FOUND")

    print("\n" + "=" * 60)
    print("Done! All figures regenerated with ALL model data.")
    print("=" * 60)


if __name__ == '__main__':
    main()
