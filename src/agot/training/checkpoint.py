"""Resumable training state and a run log.

A long self-play run has to survive a machine restart. The network weights live
in the ``--checkpoint`` file (see :meth:`agot.nn.model.PolicyValueNet.save`); the
rest of what the loop needs *to continue rather than restart* -- which iteration
is next, the best evaluation seen so far and the optimiser's momentum -- lives in
a small companion file managed here. A JSONL run log records one line per
iteration, so a run leaves a greppable history instead of only stdout.

The replay buffer is deliberately not saved: it is large and refills within a few
iterations, so a resumed run rebuilds it from fresh self-play.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

#: Suffix appended to a checkpoint path to find its companion state file.
STATE_SUFFIX = ".state.pt"
#: Suffix appended to a checkpoint path to find its JSONL run log.
LOG_SUFFIX = ".log.jsonl"


def state_path(checkpoint: str | Path) -> Path:
    """Path of the companion state file for a checkpoint."""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.stem + STATE_SUFFIX)


def log_path(checkpoint: str | Path) -> Path:
    """Path of the JSONL run log for a checkpoint."""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.stem + LOG_SUFFIX)


@dataclass
class RunState:
    """Everything needed to resume a run, except the network weights."""

    iteration: int = 0
    best_rate: float = -1.0
    trainer: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": self.iteration,
                "best_rate": self.best_rate,
                "trainer": self.trainer,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> RunState | None:
        path = Path(path)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            iteration=int(payload.get("iteration", 0)),
            best_rate=float(payload.get("best_rate", -1.0)),
            trainer=payload.get("trainer", {}) or {},
        )


class RunLog:
    """Append-only JSONL history of a training run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
