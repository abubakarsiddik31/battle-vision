"""
Experiment logging system for tracking all runs, configurations, and results.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """
    Comprehensive experiment logger that tracks:
    - Experiment ID and metadata
    - Configuration/hyperparameters
    - Notes and observations
    - Results and metrics
    - Iterations and tweaks
    """

    def __init__(
        self,
        project: str,
        experiment_name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        results_dir: str = "results/experiments",
    ):
        self.project = project
        self.experiment_id = str(uuid.uuid4())[:8]
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.notes = notes or ""
        self.tags = tags or []
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Experiment data
        self.config = {}
        self.metrics_history = []
        self.iterations = []  # Track tweaks/iterations
        self.start_time = datetime.now()
        self.end_time = None

        # Current iteration tracking
        self.current_iteration = 0
        self.tweaks_applied = []

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config.update(config)

    def add_note(self, note: str):
        """Add a note to the experiment."""
        if self.notes:
            self.notes += "\n" + note
        else:
            self.notes = note

    def add_tag(self, tag: str):
        """Add a tag to the experiment."""
        if tag not in self.tags:
            self.tags.append(tag)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics at a specific step."""
        entry = {
            "step": step if step is not None else len(self.metrics_history),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        self.metrics_history.append(entry)

    def start_iteration(self, iteration_name: str, tweaks: Dict[str, Any]):
        """
        Start a new iteration (e.g., after making tweaks).

        Args:
            iteration_name: Name/description of this iteration
            tweaks: Dictionary of tweaks applied
        """
        self.current_iteration += 1
        iteration = {
            "iteration": self.current_iteration,
            "name": iteration_name,
            "start_time": datetime.now().isoformat(),
            "tweaks": tweaks,
            "metrics": [],
        }
        self.iterations.append(iteration)
        self.tweaks_applied.append(tweaks)

        # Log tweaks as notes
        tweaks_str = ", ".join([f"{k}={v}" for k, v in tweaks.items()])
        self.add_note(f"[Iteration {self.current_iteration}] {iteration_name}: {tweaks_str}")

    def log_iteration_metrics(self, metrics: Dict[str, Any]):
        """Log metrics for the current iteration."""
        if self.iterations:
            self.iterations[-1]["metrics"].append({
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
            })

    def finish_iteration(self, final_metrics: Dict[str, Any]):
        """Finish the current iteration with final metrics."""
        if self.iterations:
            self.iterations[-1]["end_time"] = datetime.now().isoformat()
            self.iterations[-1]["final_metrics"] = final_metrics

    def finish(self, final_results: Optional[Dict[str, Any]] = None):
        """Finish the experiment and save results."""
        self.end_time = datetime.now()

        # Compile experiment summary
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "project": self.project,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "notes": self.notes,
            "tags": self.tags,
            "config": self.config,
            "final_results": final_results or {},
            "metrics_history": self.metrics_history,
            "iterations": self.iterations,
            "total_iterations": len(self.iterations),
        }

        # Save experiment summary
        exp_file = self.results_dir / f"{self.experiment_name}_{self.experiment_id}.json"
        with open(exp_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save to index
        self._update_index(summary)

        return summary, exp_file

    def _update_index(self, summary: Dict):
        """Update the experiments index file."""
        index_file = self.results_dir / "index.json"

        # Load existing index
        if index_file.exists():
            with open(index_file, "r") as f:
                index = json.load(f)
        else:
            index = {"experiments": [], "last_updated": None}

        # Add new experiment
        index["experiments"].append({
            "experiment_id": summary["experiment_id"],
            "experiment_name": summary["experiment_name"],
            "start_time": summary["start_time"],
            "end_time": summary["end_time"],
            "duration_seconds": summary["duration_seconds"],
            "tags": summary["tags"],
            "config": summary["config"],
            "final_results": summary.get("final_results", {}),
        })
        index["last_updated"] = datetime.now().isoformat()

        # Sort by start time (newest first)
        index["experiments"].sort(key=lambda x: x["start_time"], reverse=True)

        # Save index
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

    @classmethod
    def load_experiment(cls, experiment_path: str) -> Dict:
        """Load an experiment from file."""
        with open(experiment_path, "r") as f:
            return json.load(f)

    @classmethod
    def get_experiment_index(cls, results_dir: str = "results/experiments") -> Dict:
        """Get the experiments index."""
        index_file = Path(results_dir) / "index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                return json.load(f)
        return {"experiments": [], "last_updated": None}

    @classmethod
    def list_experiments(cls, results_dir: str = "results/experiments") -> List[Dict]:
        """List all experiments."""
        index = cls.get_experiment_index(results_dir)
        return index.get("experiments", [])

    @classmethod
    def search_experiments(
        cls,
        tag: Optional[str] = None,
        model: Optional[str] = None,
        results_dir: str = "results/experiments",
    ) -> List[Dict]:
        """Search experiments by tag or model."""
        experiments = cls.list_experiments(results_dir)
        filtered = []

        for exp in experiments:
            if tag and tag not in exp.get("tags", []):
                continue
            if model and exp.get("config", {}).get("model") != model:
                continue
            filtered.append(exp)

        return filtered

    def print_summary(self):
        """Print a summary of the experiment."""
        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"ID: {self.experiment_id}")
        print(f"Project: {self.project}")
        print("=" * 60)
        print(f"Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print(f"\nNotes: {self.notes}")
        print(f"Tags: {', '.join(self.tags)}")
        print(f"Iterations: {len(self.iterations)}")
        if self.iterations:
            for it in self.iterations:
                print(f"  - {it['name']}: {it.get('tweaks', {})}")
        print("=" * 60)


def format_results_for_logging(metrics: Dict) -> Dict[str, float]:
    """Format metrics dict for logging to trackio."""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float)):
                    formatted[f"{key}_{subkey}"] = subvalue
        elif isinstance(value, (int, float)):
            formatted[key] = value
    return formatted
