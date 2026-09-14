"""Run manifests: everything needed to reproduce a run.

Recorded per run, written next to the checkpoint, and embedded in the experiment
report. If a fact cannot be obtained (no git repository, for instance) the manifest
says so rather than omitting the field.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["environment_report", "git_commit", "RunManifest"]


def git_commit(root: Path | None = None) -> str:
    """Current commit, or an explicit statement that there is none."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - environment specific
        return f"unavailable: {exc}"
    if result.returncode != 0:
        return "unavailable: not a git repository"
    return result.stdout.strip()


def environment_report() -> dict[str, Any]:
    """Versions and hardware facts for the manifest."""
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


@dataclass(slots=True)
class RunManifest:
    """One training run, fully described."""

    run_id: str
    arm: str
    seed: int
    config: Mapping[str, Any]
    dataset: Mapping[str, Any]
    parameters: Mapping[str, int]
    environment: Mapping[str, Any] = field(default_factory=environment_report)
    commit: str = field(default_factory=git_commit)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    finished_at: str | None = None
    runtime_seconds: float | None = None
    steps_completed: int = 0
    train_history: list[Mapping[str, float]] = field(default_factory=list)
    validation_history: list[Mapping[str, float]] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "seed": self.seed,
            "commit": self.commit,
            "config": dict(self.config),
            "dataset": dict(self.dataset),
            "parameters": dict(self.parameters),
            "environment": dict(self.environment),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_seconds": self.runtime_seconds,
            "steps_completed": self.steps_completed,
            "train_history": [dict(entry) for entry in self.train_history],
            "validation_history": [dict(entry) for entry in self.validation_history],
            "results": dict(self.results),
            "notes": list(self.notes),
        }

    def save(self, path: str | Path) -> Path:
        """Write the manifest as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Read a manifest back as plain data."""
        return dict(json.loads(Path(path).read_text(encoding="utf-8")))
