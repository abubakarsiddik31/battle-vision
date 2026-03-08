#!/usr/bin/env python
"""
CLI utility to view and search experiments.

Usage:
    # List all experiments
    python scripts/view_experiments.py list

    # Show experiment details
    python scripts/view_experiments.py show <experiment_id>

    # Search by tag
    python scripts/view_experiments.py search --tag baseline

    # Search by model
    python scripts/view_experiments.py search --model resnet18

    # Compare experiments
    python scripts/view_experiments.py compare <exp_id1> <exp_id2>
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kiit_mita.experiment_logger import ExperimentLogger


def list_experiments(args):
    """List all experiments."""
    experiments = ExperimentLogger.list_experiments()

    if not experiments:
        print("No experiments found.")
        return

    print(f"\nFound {len(experiments)} experiments:\n")
    print("=" * 100)

    for exp in experiments:
        print(f"\nID: {exp['experiment_id']}")
        print(f"Name: {exp['experiment_name']}")
        print(f"Model: {exp.get('config', {}).get('model', 'N/A')}")
        print(f"Duration: {exp.get('duration_seconds', 0):.1f}s")
        print(f"Tags: {', '.join(exp.get('tags', []))}")

        results = exp.get('final_results', {})
        if results:
            micro = results.get('micro', {})
            print(f"Test F1: {micro.get('f1', 'N/A'):.4f}" if isinstance(micro.get('f1'), float) else f"Test F1: {micro.get('f1', 'N/A')}")
            print(f"Test Acc: {results.get('exact_match_accuracy', 'N/A')}")

        print("-" * 100)


def show_experiment(args):
    """Show detailed experiment info."""
    experiments = ExperimentLogger.list_experiments()

    # Find by ID or name
    exp = None
    for e in experiments:
        if e['experiment_id'] == args.experiment or e['experiment_name'] == args.experiment:
            exp = e
            break

    if not exp:
        print(f"Experiment '{args.experiment}' not found.")
        return

    # Load full experiment details
    exp_path = Path("results/experiments") / f"{exp['experiment_name']}_{exp['experiment_id']}.json"
    if exp_path.exists():
        full_exp = ExperimentLogger.load_experiment(str(exp_path))
    else:
        full_exp = exp

    print("\n" + "=" * 100)
    print(f"EXPERIMENT: {full_exp['experiment_name']}")
    print("=" * 100)
    print(f"ID: {full_exp['experiment_id']}")
    print(f"Project: {full_exp['project']}")
    print(f"Start: {full_exp['start_time']}")
    print(f"End: {full_exp.get('end_time', 'In progress')}")
    print(f"Duration: {full_exp.get('duration_seconds', 0):.1f}s")
    print(f"\nTags: {', '.join(full_exp.get('tags', []))}")
    print(f"\nConfiguration:")
    for key, value in full_exp.get('config', {}).items():
        print(f"  {key}: {value}")
    print(f"\nNotes:")
    print(full_exp.get('notes', 'No notes'))
    print(f"\nIterations: {full_exp.get('total_iterations', 0)}")

    for it in full_exp.get('iterations', []):
        print(f"\n  Iteration {it['iteration']}: {it['name']}")
        print(f"    Tweaks: {it.get('tweaks', {})}")
        if it.get('final_metrics'):
            metrics = it['final_metrics']
            micro = metrics.get('micro', {})
            print(f"    F1: {micro.get('f1', 'N/A'):.4f}" if isinstance(micro.get('f1'), float) else f"    F1: {micro.get('f1', 'N/A')}")

    print(f"\nFinal Results:")
    results = full_exp.get('final_results', {})
    if results:
        micro = results.get('micro', {})
        print(f"  Micro F1: {micro.get('f1', 'N/A')}")
        print(f"  Micro Precision: {micro.get('precision', 'N/A')}")
        print(f"  Micro Recall: {micro.get('recall', 'N/A')}")
        print(f"  Exact Match Accuracy: {results.get('exact_match_accuracy', 'N/A')}")

        print(f"\n  Per-class F1:")
        for class_name in ['Artilary', 'Missile', 'Radar', 'M. Rocket Launcher', 'Soldier', 'Tank', 'Vehicle']:
            if class_name in results:
                print(f"    {class_name}: {results[class_name].get('f1', 'N/A')}")

    print("=" * 100)


def search_experiments(args):
    """Search experiments by tag or model."""
    results = ExperimentLogger.search_experiments(
        tag=args.tag,
        model=args.model,
    )

    if not results:
        print("No experiments found matching criteria.")
        return

    print(f"\nFound {len(results)} experiments:\n")
    print("=" * 100)

    for exp in results:
        print(f"\nID: {exp['experiment_id']}")
        print(f"Name: {exp['experiment_name']}")
        print(f"Model: {exp.get('config', {}).get('model', 'N/A')}")
        print(f"Tags: {', '.join(exp.get('tags', []))}")

        final_results = exp.get('final_results', {})
        if final_results:
            micro = final_results.get('micro', {})
            print(f"F1: {micro.get('f1', 'N/A')}")
            print(f"Acc: {final_results.get('exact_match_accuracy', 'N/A')}")

        print("-" * 100)


def compare_experiments(args):
    """Compare multiple experiments."""
    experiments = ExperimentLogger.list_experiments()

    exps_to_compare = []
    for exp_id in args.experiments:
        for e in experiments:
            if e['experiment_id'] == exp_id or e['experiment_name'] == exp_id:
                exps_to_compare.append(e)
                break

    if len(exps_to_compare) < 2:
        print("Need at least 2 experiments to compare.")
        return

    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)

    # Header
    print(f"\n{'Experiment':<25} {'Model':<15} {'F1':<10} {'Accuracy':<10} {'Tags'}")
    print("-" * 100)

    for exp in exps_to_compare:
        name = exp['experiment_name'][:24]
        model = exp.get('config', {}).get('model', 'N/A')[:14]
        results = exp.get('final_results', {})
        micro = results.get('micro', {})
        f1 = f"{micro.get('f1', 0):.4f}" if isinstance(micro.get('f1'), float) else "N/A"
        acc = f"{results.get('exact_match_accuracy', 0):.4f}" if isinstance(results.get('exact_match_accuracy'), float) else "N/A"
        tags = ','.join(exp.get('tags', []))[:30]

        print(f"{name:<25} {model:<15} {f1:<10} {acc:<10} {tags}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="View and search experiments")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List all experiments")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("experiment", help="Experiment ID or name")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search experiments")
    search_parser.add_argument("--tag", help="Filter by tag")
    search_parser.add_argument("--model", help="Filter by model")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiments", nargs="+", help="Experiment IDs to compare")

    args = parser.parse_args()

    if args.command == "list":
        list_experiments(args)
    elif args.command == "show":
        show_experiment(args)
    elif args.command == "search":
        search_experiments(args)
    elif args.command == "compare":
        compare_experiments(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
