"""
Research state checkpoint management.
Simple JSON-based persistence for resuming research sessions.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def load_checkpoint(experiment_dir: str) -> Dict[str, Any]:
    """Load checkpoint from experiment directory. Returns empty dict if none exists."""
    path = os.path.join(experiment_dir, "checkpoint.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(experiment_dir: str, **kwargs) -> None:
    """Save/update checkpoint. Merges kwargs with existing checkpoint."""
    checkpoint = load_checkpoint(experiment_dir)
    checkpoint.update(kwargs)
    checkpoint["last_updated"] = datetime.now().isoformat()

    path = os.path.join(experiment_dir, "checkpoint.json")
    os.makedirs(experiment_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def get_phase(experiment_dir: str) -> Optional[str]:
    """Get the current phase from checkpoint."""
    return load_checkpoint(experiment_dir).get("phase")


def is_phase_complete(experiment_dir: str, phase: str) -> bool:
    """Check if a specific phase is marked as complete."""
    checkpoint = load_checkpoint(experiment_dir)
    completed = checkpoint.get("completed_phases", [])
    return phase in completed


def mark_phase_complete(experiment_dir: str, phase: str) -> None:
    """Mark a phase as complete in the checkpoint."""
    checkpoint = load_checkpoint(experiment_dir)
    completed = checkpoint.get("completed_phases", [])
    if phase not in completed:
        completed.append(phase)
    save_checkpoint(experiment_dir, completed_phases=completed, phase=phase)


def append_to_research_log(experiment_dir: str, entry: str) -> None:
    """Append an entry to the research log."""
    log_path = os.path.join(experiment_dir, "research-log.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(log_path, "a") as f:
        f.write(f"\n## {timestamp}\n{entry}\n")
