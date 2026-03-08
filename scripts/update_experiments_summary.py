#!/usr/bin/env python
"""
Generate a human-readable markdown summary of experiments for git tracking.

This script creates a summary table that can be committed to git for easy
reference of experiment results without needing to query the JSON files.

Usage:
    python scripts/update_experiments_summary.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kiit_mita.experiment_logger import ExperimentLogger


def generate_markdown_summary():
    """Generate markdown summary of all experiments."""
    experiments = ExperimentLogger.list_experiments()

    if not experiments:
        return "# Experiments Summary\n\nNo experiments found.\n"

    md = "# Experiments Summary\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += f"Total Experiments: {len(experiments)}\n\n"

    # Summary table
    md += "## Quick Reference\n\n"
    md += "| Date | Experiment | Architecture | Pretrained | Micro F1 | Acc | Tags |\n"
    md += "|------|------------|--------------|------------|----------|-----|------|\n"

    for exp in experiments:
        date = exp['start_time'][:10]  # YYYY-MM-DD
        name = exp['experiment_name'][:40]
        arch = exp.get('config', {}).get('architecture', 'N/A')
        pretrained = exp.get('config', {}).get('pretrained', 'N/A')

        results = exp.get('final_results', {})
        micro = results.get('micro', {})
        f1 = micro.get('f1', 0)
        acc = results.get('exact_match_accuracy', 0)

        if isinstance(f1, float):
            f1_str = f"{f1:.4f}"
        else:
            f1_str = "N/A"

        if isinstance(acc, float):
            acc_str = f"{acc:.4f}"
        else:
            acc_str = "N/A"

        tags = ",".join(exp.get('tags', []))[:20]

        md += f"| {date} | {name} | {arch} | {pretrained} | {f1_str} | {acc_str} | {tags} |\n"

    # Detailed results
    md += "\n## Detailed Results\n\n"

    for exp in experiments:
        md += f"### {exp['experiment_name']}\n\n"
        md += f"**ID:** `{exp['experiment_id']}`\n\n"
        md += f"**Date:** {exp['start_time']}\n\n"
        md += f"**Duration:** {exp.get('duration_seconds', 0):.1f}s\n\n"

        # Tags
        tags = exp.get('tags', [])
        if tags:
            md += f"**Tags:** {', '.join([f'`{t}`' for t in tags])}\n\n"

        # Notes
        # We'd need to load the full experiment for notes
        md += f"**Notes:** See full experiment file for details\n\n"

        # Configuration
        config = exp.get('config', {})
        md += "**Configuration:**\n"
        md += f"- Architecture: `{config.get('architecture', 'N/A')}`\n"
        md += f"- Pretrained: `{config.get('pretrained', 'N/A')}`\n"
        md += f"- Epochs: {config.get('epochs_total', 'N/A')}\n"
        md += f"- LR Head: {config.get('lr_head', 'N/A')}\n"
        md += f"- LR Finetune: {config.get('lr_finetune', 'N/A')}\n"
        md += f"- Batch Size: {config.get('batch_size', 'N/A')}\n"
        md += f"- Optimizer: `{config.get('optimizer', 'N/A')}`\n"
        md += "\n"

        # Results
        results = exp.get('final_results', {})
        if results:
            md += "**Results:**\n"
            md += f"- Exact Match Accuracy: {results.get('exact_match_accuracy', 'N/A')}\n"

            micro = results.get('micro', {})
            md += f"- Micro F1: {micro.get('f1', 'N/A')}\n"
            md += f"- Micro Precision: {micro.get('precision', 'N/A')}\n"
            md += f"- Micro Recall: {micro.get('recall', 'N/A')}\n"

            macro = results.get('macro', {})
            md += f"- Macro F1: {macro.get('f1', 'N/A')}\n"

            md += "\n**Per-Class F1:**\n"
            for class_name in ['Artilary', 'Missile', 'Radar', 'M. Rocket Launcher', 'Soldier', 'Tank', 'Vehicle']:
                if class_name in results:
                    f1 = results[class_name].get('f1', 'N/A')
                    md += f"- {class_name}: {f1}\n"

        md += "\n---\n\n"

    return md


def main():
    # Generate summary
    summary = generate_markdown_summary()

    # Write to file
    summary_path = Path("results/experiments/SUMMARY.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"Experiment summary written to: {summary_path}")
    print(f"\nTo commit to git:")
    print(f"  git add results/experiments/SUMMARY.md")
    print(f"  git commit -m 'Update experiment summary'")

    # Also update the README table
    update_readme_table()


def update_readme_table():
    """Update the experiments table in README.md."""
    experiments = ExperimentLogger.list_experiments()

    readme_path = Path("results/experiments/README.md")
    if not readme_path.exists():
        return

    # Read existing README
    with open(readme_path, "r") as f:
        content = f.read()

    # Generate table
    table = "\n| Date | Experiment | Architecture | F1 | Accuracy | Notes |\n"
    table += "|------|------------|--------------|-----|----------|-------|\n"

    for exp in experiments[:20]:  # Last 20 experiments
        date = exp['start_time'][:10]
        name = exp['experiment_name'][:30]
        arch = exp.get('config', {}).get('architecture', 'N/A')

        results = exp.get('final_results', {})
        micro = results.get('micro', {})
        f1 = micro.get('f1', 0)
        acc = results.get('exact_match_accuracy', 0)

        f1_str = f"{f1:.4f}" if isinstance(f1, float) else "N/A"
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else "N/A"

        notes = ",".join(exp.get('tags', []))[:30] if exp.get('tags') else "-"

        table += f"| {date} | {name} | {arch} | {f1_str} | {acc_str} | {notes} |\n"

    # Replace table section in README
    marker = "<!-- Auto-generated summary will be added below -->"
    if marker in content:
        content = content.split(marker)[0] + marker + table
    else:
        content = content.replace(
            "|------|------------|--------------|-----|----------|-------|",
            "|------|------------|--------------|-----|----------|-------|" + table.split("\n")[-2]
        )

    with open(readme_path, "w") as f:
        f.write(content)

    print(f"README table updated: {readme_path}")


if __name__ == "__main__":
    main()
