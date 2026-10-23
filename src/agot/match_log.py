"""Structured logs for human-versus-bot games.

Every match gets a JSONL chronicle and, when the bot seats are MCTS, a
``ReplayBuffer``-compatible ``.npz`` next to it. The JSONL is for analysis and
for replaying the human's decisions. The NPZ contains only MCTS decisions, with
the search's visit distribution as the policy target and the final per-house
outcome as the value target, so it can be inspected or merged into a training
buffer without teaching the network to imitate the human's one-hot moves.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agot.engine.actions import Action
from agot.engine.observation import HOUSE_ORDER, Observation
from agot.training.replay import Example, ReplayBuffer


def _action_record(action: Action) -> dict:
    return {
        "kind": action.kind.value,
        "house": action.house,
        "region": action.region,
        "target": action.target,
        "order": action.order,
        "card": action.card,
        "unit": action.unit,
        "side": action.side,
        "amount": action.amount,
        "units": None if action.units is None else list(action.units),
        "establish": action.establish,
    }


class MatchLog:
    """One local match log. Logging failures are reported to the caller."""

    def __init__(
        self,
        directory: str | Path = "checkpoints/play-logs",
        *,
        human_house: str,
        opponents: str,
        checkpoint: str | None,
        simulations: int,
        determinizations: int,
        seed: int,
        max_rounds: int,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = f"play-{stamp}-{human_house}-{uuid.uuid4().hex[:6]}"
        self.jsonl_path = self.directory / f"{base}.jsonl"
        self.npz_path = self.directory / f"{base}.npz"
        self._handle = self.jsonl_path.open("w", encoding="utf-8")
        self._examples: list[Example] = []
        self._closed = False
        self._write(
            {
                "type": "start",
                "format": "agot-match-v1",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "human_house": human_house,
                "opponents": opponents,
                "checkpoint": checkpoint,
                "simulations": simulations,
                "determinizations": determinizations,
                "seed": seed,
                "max_rounds": max_rounds,
            }
        )

    @property
    def example_count(self) -> int:
        return len(self._examples)

    def _write(self, record: dict) -> None:
        if self._closed:
            return
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def decision(
        self,
        *,
        step: int,
        round_number: int,
        phase: str,
        action_step: str | None,
        house: str,
        actor: str,
        source: str,
        action: Action,
        legal_actions: list[Action],
        chosen_index: int,
        target: np.ndarray,
        observation: Observation | None,
    ) -> None:
        """Append one decision and optionally add its MCTS example."""
        target = np.asarray(target, dtype=np.float64)
        total = float(target.sum())
        if total > 0:
            normalised = target / total
        else:
            normalised = np.full(len(target), 1.0 / max(1, len(target)))
        clipped = np.clip(normalised, 1e-12, 1.0)
        entropy = float(-(clipped * np.log(clipped)).sum())
        chosen_prob = (
            float(normalised[chosen_index])
            if 0 <= chosen_index < len(normalised)
            else None
        )
        self._write(
            {
                "type": "decision",
                "step": step,
                "round": round_number,
                "phase": phase,
                "action_step": action_step,
                "house": house,
                "actor": actor,
                "source": source,
                "action": _action_record(action),
                "chosen_index": chosen_index,
                "legal_count": len(legal_actions),
                "legal_actions": [
                    str(candidate) for candidate in legal_actions
                ],
                "target_entropy": entropy,
                "target_max": float(normalised.max()) if len(normalised) else 0.0,
                "chosen_probability": chosen_prob,
            }
        )

        if source != "mcts" or observation is None:
            return
        if len(observation.actions) != len(normalised):
            return
        # The encoded action rows and the visit vector must be in the same
        # order. They are in the GUI, but never store a misaligned example.
        if list(observation.actions) != list(legal_actions):
            self._write(
                {
                    "type": "warning",
                    "step": step,
                    "message": "action order differed between caller and observation",
                }
            )
            return
        if action not in observation.actions:
            return
        self._examples.append(
            Example(
                state=observation.state_features,
                actions=observation.action_features,
                policy=normalised.astype(np.float32),
                player=HOUSE_ORDER.index(house),
                outcome=np.zeros(len(HOUSE_ORDER), dtype=np.float32),
            )
        )

    def undo(self, *, step: int) -> None:
        self._write({"type": "undo", "step": step})

    def truncate_examples(self, count: int) -> None:
        """Drop examples after an undo; the JSONL keeps the undo event."""
        del self._examples[count:]

    def finish(
        self,
        *,
        winner: str | None,
        rounds: int,
        steps: int,
        aborted: bool,
        note: str | None = None,
        outcome: np.ndarray | None = None,
    ) -> None:
        """Write the end record and save the MCTS examples when complete."""
        if self._closed:
            return
        saved_npz: str | None = None
        if not aborted and outcome is not None and self._examples:
            outcome = np.asarray(outcome, dtype=np.float32)
            if outcome.shape == (len(HOUSE_ORDER),):
                for example in self._examples:
                    example.outcome = outcome.copy()
                buffer = ReplayBuffer(capacity=len(self._examples))
                for example in self._examples:
                    buffer.add(example)
                buffer.save(str(self.npz_path))
                saved_npz = str(self.npz_path)
        self._write(
            {
                "type": "end",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "winner": winner,
                "rounds": rounds,
                "steps": steps,
                "aborted": aborted,
                "note": note,
                "training_examples": len(self._examples) if saved_npz else 0,
                "npz": saved_npz,
                "jsonl": str(self.jsonl_path),
            }
        )
        self._closed = True
        self._handle.close()

    def close(self) -> None:
        if not self._closed:
            self.finish(
                winner=None,
                rounds=0,
                steps=0,
                aborted=True,
                note="logger closed",
            )
