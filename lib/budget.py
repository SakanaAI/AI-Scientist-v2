"""
Simple time and resource budget tracking for experiments.
"""

import json
import os
import time
from datetime import datetime
from typing import Optional


class BudgetTracker:
    """Track experiment time budget."""

    def __init__(self, max_hours: float = 8.0, budget_file: Optional[str] = None):
        self.max_seconds = max_hours * 3600
        self.start_time = time.time()
        self.budget_file = budget_file
        self._experiments = []

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since tracker was created."""
        return time.time() - self.start_time

    @property
    def remaining(self) -> float:
        """Seconds remaining in budget."""
        return max(0, self.max_seconds - self.elapsed)

    @property
    def is_over_budget(self) -> bool:
        """Whether we've exceeded the time budget."""
        return self.elapsed > self.max_seconds

    def log_experiment(self, name: str, duration: float, success: bool) -> None:
        """Log a completed experiment run."""
        self._experiments.append({
            "name": name,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })
        if self.budget_file:
            self.save(self.budget_file)

    def summary(self) -> str:
        """Human-readable budget summary."""
        hours = self.elapsed / 3600
        max_h = self.max_seconds / 3600
        n = len(self._experiments)
        return f"Budget: {hours:.1f}h / {max_h:.1f}h used ({n} experiments run)"

    def save(self, path: str) -> None:
        """Save budget state to JSON."""
        data = {
            "max_hours": self.max_seconds / 3600,
            "elapsed_seconds": self.elapsed,
            "experiments": self._experiments,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BudgetTracker":
        """Load budget state from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        tracker = cls(max_hours=data["max_hours"], budget_file=path)
        tracker._experiments = data.get("experiments", [])
        # Adjust start time to account for previously elapsed time
        tracker.start_time = time.time() - data.get("elapsed_seconds", 0)
        return tracker
