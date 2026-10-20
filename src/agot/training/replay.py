"""Self-play data storage.

One :class:`Example` is a single decision point: the encoded position, the
encoded candidate actions, the search's target distribution over them, which
house was acting, and the final game outcome (a per-house reward vector).

Because the number of candidate actions ``N`` varies from position to position,
the buffer stores examples individually and pads to a common ``N`` only when a
batch is collated for the network. :meth:`ReplayBuffer.save` writes a compressed
``.npz`` with explicit offsets so a dataset round-trips exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agot.engine.observation import N_HOUSES


@dataclass(slots=True)
class Example:
    """A single (state, actions, target, outcome) training tuple."""

    state: np.ndarray  # (STATE_DIM,)
    actions: np.ndarray  # (N, ACTION_DIM)
    policy: np.ndarray  # (N,)
    player: int  # index into HOUSE_ORDER
    outcome: np.ndarray  # (N_HOUSES,)

    @property
    def value(self) -> float:
        """The acting player's own share of the outcome."""
        return float(self.outcome[self.player])


class ReplayBuffer:
    """A bounded reservoir of :class:`Example` objects."""

    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self._items: list[Example] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, example: Example) -> None:
        if len(self._items) >= self.capacity:
            # Drop the oldest quarter at once, so we do not shift on every add.
            drop = max(1, self.capacity // 4)
            del self._items[:drop]
        self._items.append(example)

    def extend(self, examples) -> None:
        for example in examples:
            self.add(example)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Example]:
        if not self._items:
            raise ValueError("cannot sample from an empty buffer")
        size = min(batch_size, len(self._items))
        indices = rng.choice(len(self._items), size=size, replace=False)
        return [self._items[int(index)] for index in indices]

    def all(self) -> list[Example]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        states = np.stack([item.state for item in self._items]) if self._items else np.zeros((0, 0))
        policies = np.array([item.policy for item in self._items], dtype=object)
        outcomes = (
            np.stack([item.outcome for item in self._items])
            if self._items
            else np.zeros((0, N_HOUSES))
        )
        players = np.array([item.player for item in self._items], dtype=np.int64)
        offsets = np.zeros(len(self._items) + 1, dtype=np.int64)
        flat_actions: list[np.ndarray] = []
        for index, item in enumerate(self._items):
            flat_actions.append(item.actions)
            offsets[index + 1] = offsets[index] + len(item.actions)
        action_matrix = (
            np.concatenate(flat_actions).astype(np.float32)
            if flat_actions
            else np.zeros((0, 0), dtype=np.float32)
        )
        np.savez_compressed(
            path,
            states=states,
            actions=action_matrix,
            offsets=offsets,
            policies=policies,
            players=players,
            outcomes=outcomes,
        )

    @classmethod
    def load(cls, path: str) -> ReplayBuffer:
        with np.load(path, allow_pickle=True) as data:
            states = data["states"]
            actions = data["actions"]
            offsets = data["offsets"]
            policies = data["policies"]
            players = data["players"]
            outcomes = data["outcomes"]
            buffer = cls(capacity=max(1, len(states)))
            for index in range(len(states)):
                start, stop = int(offsets[index]), int(offsets[index + 1])
                buffer.add(
                    Example(
                        state=states[index],
                        actions=actions[start:stop],
                        policy=np.asarray(policies[index], dtype=np.float32),
                        player=int(players[index]),
                        outcome=outcomes[index],
                    )
                )
            return buffer
