"""Self-play data storage.

One :class:`Example` is a single decision point: the encoded position, the
encoded candidate actions, the search's target distribution over them, which
house was acting, and the final game outcome (a per-house reward vector). It
optionally carries the index of the action that was actually played, which the
Q head trains on; a buffer refuses to mix examples with and without it.

The buffer is a *structure of arrays*, not a list of objects: positions,
players and outcomes are dense rows, while the variable-length candidates of
every example share one concatenated action matrix and one policy vector with
an offset table. :meth:`ReplayBuffer.all` and :meth:`ReplayBuffer.sample` hand
back :class:`Example` objects whose arrays are **views** into that storage, so
the training loop still sees the same API without paying a copy per example.
Views are valid until the buffer is mutated (an add compacts only when full).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM


@dataclass(slots=True)
class Example:
    """A single (state, actions, target, outcome) training tuple."""

    state: np.ndarray  # (STATE_DIM,)
    actions: np.ndarray  # (N, ACTION_DIM)
    policy: np.ndarray  # (N,)
    player: int  # index into HOUSE_ORDER
    outcome: np.ndarray  # (N_HOUSES,)
    #: Index into ``actions`` of the action that was actually played, or
    #: ``None`` when the producer did not record it. The optional Q head needs
    #: it to train on the played action's outcome rather than the whole set.
    chosen: int | None = None

    @property
    def value(self) -> float:
        """The acting player's own share of the outcome."""
        return float(self.outcome[self.player])


class ReplayBuffer:
    """A bounded reservoir of examples, stored as parallel arrays."""

    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = max(1, capacity)
        self._size = 0
        self._rows = 0
        # Start small and grow; a 200k buffer is the cap, not the allocation.
        self._states = np.zeros((0, STATE_DIM), dtype=np.float32)
        self._actions = np.zeros((0, ACTION_DIM), dtype=np.float32)
        self._policies = np.zeros((0,), dtype=np.float32)
        self._players = np.zeros((0,), dtype=np.int64)
        self._outcomes = np.zeros((0, N_HOUSES), dtype=np.float32)
        self._offsets = np.zeros((1,), dtype=np.int64)
        #: Played-action indices, valid only when :attr:`_has_chosen` is set.
        #: A buffer never mixes the two modes: a missing index would silently
        #: train the Q head on a different action than the one played.
        self._chosen = np.zeros((0,), dtype=np.int64)
        self._has_chosen = False

    # -- sizing ----------------------------------------------------------

    def _ensure_examples(self, count: int) -> None:
        if count <= len(self._states):
            return
        target = min(self.capacity, max(count, max(1024, len(self._states) * 2)))
        if target <= len(self._states):
            return
        for name in ("_states", "_players", "_outcomes", "_chosen"):
            array = getattr(self, name)
            grown = np.zeros((target, *array.shape[1:]), dtype=array.dtype)
            grown[: len(array)] = array
            setattr(self, name, grown)
        if len(self._offsets) < target + 1:
            grown_offsets = np.zeros((target + 1,), dtype=np.int64)
            grown_offsets[: len(self._offsets)] = self._offsets
            self._offsets = grown_offsets

    def _ensure_rows(self, count: int) -> None:
        if count <= len(self._actions):
            return
        target = max(count, max(16384, len(self._actions) * 2))
        actions = np.zeros((target, ACTION_DIM), dtype=np.float32)
        actions[: self._rows] = self._actions[: self._rows]
        self._actions = actions
        policies = np.zeros((target,), dtype=np.float32)
        policies[: self._rows] = self._policies[: self._rows]
        self._policies = policies

    def _drop_oldest(self) -> None:
        drop = max(1, self.capacity // 4)
        drop = min(drop, self._size)
        keep = self._size - drop
        row_start = int(self._offsets[drop])
        row_end = self._rows
        self._states[:keep] = self._states[drop : self._size]
        self._players[:keep] = self._players[drop : self._size]
        self._outcomes[:keep] = self._outcomes[drop : self._size]
        self._chosen[:keep] = self._chosen[drop : self._size]
        self._actions[: row_end - row_start] = self._actions[row_start:row_end]
        self._policies[: row_end - row_start] = self._policies[row_start:row_end]
        self._offsets[1 : keep + 1] = self._offsets[drop + 1 : self._size + 1] - row_start
        self._size = keep
        self._rows = row_end - row_start
        if keep == 0:
            self._has_chosen = False

    # -- basic API -------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Example:
        """One example as views, without building the whole list."""
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(index)
        return self._example(index)

    def add(self, example: Example) -> None:
        count = len(example.actions)
        chosen = example.chosen
        if chosen is not None:
            if not 0 <= chosen < count:
                raise ValueError(
                    f"chosen index {chosen} is outside an action set of {count}"
                )
            if self._size and not self._has_chosen:
                raise ValueError(
                    "cannot add an example with a chosen action to a buffer "
                    "that already holds examples without one"
                )
        elif self._size and self._has_chosen:
            raise ValueError(
                "cannot add an example without a chosen action to a buffer "
                "that already holds examples with one"
            )
        if self._size >= self.capacity:
            self._drop_oldest()
        self._ensure_examples(self._size + 1)
        self._ensure_rows(self._rows + count)

        row = self._size
        start = self._rows
        self._states[row] = example.state
        self._actions[start : start + count] = example.actions
        self._policies[start : start + count] = example.policy
        self._players[row] = example.player
        self._outcomes[row] = example.outcome
        if chosen is not None:
            self._has_chosen = True
            self._chosen[row] = chosen
        elif self._has_chosen:  # pragma: no cover - rejected above
            self._chosen[row] = -1
        self._offsets[row + 1] = start + count
        self._size += 1
        self._rows = start + count

    def extend(self, examples) -> None:
        for example in examples:
            self.add(example)

    def _example(self, index: int) -> Example:
        start = int(self._offsets[index])
        stop = int(self._offsets[index + 1])
        return Example(
            state=self._states[index],
            actions=self._actions[start:stop],
            policy=self._policies[start:stop],
            player=int(self._players[index]),
            outcome=self._outcomes[index],
            chosen=int(self._chosen[index]) if self._has_chosen else None,
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Example]:
        if not self._size:
            raise ValueError("cannot sample from an empty buffer")
        size = min(batch_size, self._size)
        indices = rng.choice(self._size, size=size, replace=False)
        return [self._example(int(index)) for index in indices]

    def all(self) -> list[Example]:
        return [self._example(index) for index in range(self._size)]

    def clear(self) -> None:
        self._size = 0
        self._rows = 0
        self._offsets[0] = 0
        self._has_chosen = False

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        states = (
            self._states[: self._size]
            if self._size
            else np.zeros((0, STATE_DIM), dtype=np.float32)
        )
        outcomes = (
            self._outcomes[: self._size]
            if self._size
            else np.zeros((0, N_HOUSES), dtype=np.float32)
        )
        arrays = {
            "states": states,
            "actions": np.asarray(self._actions[: self._rows], dtype=np.float32),
            "offsets": self._offsets[: self._size + 1].astype(np.int64),
            "policies": np.asarray(self._policies[: self._rows], dtype=np.float32),
            "players": self._players[: self._size].astype(np.int64),
            "outcomes": outcomes.astype(np.float32),
        }
        if self._has_chosen:
            # Omitted entirely when the run does not train a Q head, so its
            # snapshot stays byte-compatible with earlier files.
            arrays["chosen"] = self._chosen[: self._size].astype(np.int64)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str, capacity: int | None = None) -> ReplayBuffer:
        with np.load(path, allow_pickle=True) as data:
            states = data["states"]
            actions = data["actions"]
            offsets = data["offsets"]
            policies = data["policies"]
            players = data["players"]
            outcomes = data["outcomes"]
            chosen = data["chosen"] if "chosen" in data.files else None
            # Older files stored one object array per example in `policies`;
            # current files store one flat float32 vector.
            legacy = policies.dtype == object
            buffer = cls(capacity=max(1, capacity or len(states)))
            for index in range(len(states)):
                start, stop = int(offsets[index]), int(offsets[index + 1])
                policy = (
                    policies[index] if legacy else policies[start:stop]
                )
                buffer.add(
                    Example(
                        state=states[index],
                        actions=actions[start:stop],
                        policy=np.asarray(policy, dtype=np.float32),
                        player=int(players[index]),
                        outcome=outcomes[index],
                        chosen=(
                            int(chosen[index]) if chosen is not None else None
                        ),
                    )
                )
            return buffer
