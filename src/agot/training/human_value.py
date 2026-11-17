"""Value-only batches from the human-game archive.

The dataset is produced by ``tools/export_human_value.py``: one row per
``orders-revealed`` snapshot, with the encoded state, the legal-action rows at
the first decision, and the ±1 per-House outcome. The policy target is absent on
purpose — the archive does not record the orders during planning — so these rows
train the value head only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class HumanValueSet:
    """Parallel arrays of human snapshots, shaped for ``forward_packed``."""

    states: np.ndarray  # (N, STATE_DIM)
    actions: np.ndarray  # (T, ACTION_DIM), concatenated
    offsets: np.ndarray  # (N + 1,), where each row's actions start and end
    outcomes: np.ndarray  # (N, N_HOUSES), +1 winner / -1 rest
    games: np.ndarray  # (N,), the source game id
    rounds: np.ndarray  # (N,), the round of the snapshot

    @classmethod
    def load(cls, path: str | Path) -> HumanValueSet:
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            states=data["states"],
            actions=data["actions"],
            offsets=data["offsets"],
            outcomes=data["outcomes"],
            games=data["games"],
            rounds=data["rounds"],
        )

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def select(self, indices) -> HumanValueSet:
        """A new set with only the selected rows, actions and offsets rebuilt."""
        indices = np.asarray(indices, dtype=np.int64)
        rows = []
        offsets = [0]
        for row in indices:
            start = int(self.offsets[row])
            end = int(self.offsets[row + 1])
            rows.append(self.actions[start:end])
            offsets.append(offsets[-1] + end - start)
        return HumanValueSet(
            states=self.states[indices],
            actions=np.concatenate(rows, axis=0),
            offsets=np.asarray(offsets, dtype=np.int64),
            outcomes=self.outcomes[indices],
            games=self.games[indices],
            rounds=self.rounds[indices],
        )

    def batch(self, indices: np.ndarray) -> tuple[torch.Tensor, ...]:
        """Return ``(state, actions, index, outcome)`` for the selected rows."""
        rows = []
        positions = []
        for local, row in enumerate(indices):
            start = int(self.offsets[row])
            end = int(self.offsets[row + 1])
            rows.append(self.actions[start:end])
            positions.append(np.full(end - start, local, dtype=np.int64))
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(np.concatenate(rows, axis=0)),
            torch.from_numpy(np.concatenate(positions)),
            torch.from_numpy(self.outcomes[indices]),
        )

    def batches(self, batch_size: int, *, shuffle: bool = True, seed: int = 0):
        order = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(order), max(1, batch_size)):
            yield self.batch(order[start : start + max(1, batch_size)])
