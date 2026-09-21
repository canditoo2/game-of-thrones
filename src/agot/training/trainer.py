"""Train the policy/value network on self-play data (requires PyTorch).

The training objective is the AlphaZero one, generalised to six players:

* policy: cross-entropy between the network's masked softmax over candidate
  actions and the search's target distribution;
* value: mean-squared error between the network's per-house prediction and the
  final outcome vector, masked to the houses actually in the game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM
from agot.nn.model import PolicyValueNet
from agot.training.replay import Example, ReplayBuffer


@dataclass
class TrainConfig:
    """Hyper-parameters for one training run."""

    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 1
    value_weight: float = 1.0
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"


@dataclass
class Batch:
    """A collated, padded mini-batch of examples."""

    state: torch.Tensor
    actions: torch.Tensor
    mask: torch.Tensor
    policy: torch.Tensor
    outcome: torch.Tensor
    outcome_mask: torch.Tensor

    def to(self, device: str) -> Batch:
        return Batch(
            state=self.state.to(device),
            actions=self.actions.to(device),
            mask=self.mask.to(device),
            policy=self.policy.to(device),
            outcome=self.outcome.to(device),
            outcome_mask=self.outcome_mask.to(device),
        )


def collate(examples: list[Example]) -> Batch:
    """Pad a list of variable-length examples into dense tensors."""
    size = len(examples)
    max_actions = max(len(example.actions) for example in examples)
    state = torch.zeros(size, STATE_DIM, dtype=torch.float32)
    actions = torch.zeros(size, max_actions, ACTION_DIM, dtype=torch.float32)
    mask = torch.zeros(size, max_actions, dtype=torch.float32)
    policy = torch.zeros(size, max_actions, dtype=torch.float32)
    outcome = torch.zeros(size, N_HOUSES, dtype=torch.float32)
    outcome_mask = torch.zeros(size, N_HOUSES, dtype=torch.float32)

    for row, example in enumerate(examples):
        count = len(example.actions)
        state[row] = torch.from_numpy(np.asarray(example.state, dtype=np.float32))
        actions[row, :count] = torch.from_numpy(
            np.asarray(example.actions, dtype=np.float32)
        )
        mask[row, :count] = 1.0
        target = np.asarray(example.policy, dtype=np.float32)
        total = target.sum()
        policy[row, :count] = target / total if total > 0 else 1.0 / count
        outcome[row] = torch.from_numpy(np.asarray(example.outcome, dtype=np.float32))
        outcome_mask[row] = torch.from_numpy(
            (np.asarray(example.outcome) != 0).astype(np.float32)
        )

    return Batch(state, actions, mask, policy, outcome, outcome_mask)


@dataclass
class TrainStats:
    """Losses averaged over an epoch."""

    batches: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        denom = max(1, self.batches)
        return {
            "batches": self.batches,
            "policy_loss": self.policy_loss / denom,
            "value_loss": self.value_loss / denom,
            "total_loss": self.total_loss / denom,
        }


class Trainer:
    """Wraps an optimizer around a :class:`PolicyValueNet`."""

    def __init__(
        self,
        model: PolicyValueNet,
        config: TrainConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainConfig()
        self.device = self.config.device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.rng = np.random.default_rng(0)

    def train_epoch(self, examples: list[Example]) -> TrainStats:
        if not examples:
            return TrainStats()
        self.model.train()
        stats = TrainStats()
        order = self.rng.permutation(len(examples))
        batch_size = self.config.batch_size
        for start in range(0, len(order), batch_size):
            chunk = [
                examples[int(index)]
                for index in order[start : start + batch_size]
            ]
            batch = collate(chunk).to(self.device)
            policy_loss, value_loss, total = self._step(batch)
            stats.batches += 1
            stats.policy_loss += policy_loss
            stats.value_loss += value_loss
            stats.total_loss += total
        return stats

    def _step(self, batch: Batch) -> tuple[float, float, float]:
        logits, values = self.model(batch.state, batch.actions, batch.mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(batch.policy * log_probs).sum(dim=-1).mean()

        value_mask = batch.outcome_mask
        squared = (values - batch.outcome) ** 2
        value_loss = (squared * value_mask).sum() / value_mask.sum().clamp(min=1.0)

        total = policy_loss + self.config.value_weight * value_loss

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.config.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return float(policy_loss), float(value_loss), float(total)

    def train(
        self,
        buffer: ReplayBuffer,
        epochs: int | None = None,
        *,
        callback=None,
    ) -> list[dict]:
        """Train for ``epochs`` passes over the current buffer contents."""
        examples = buffer.all()
        history: list[dict] = []
        for epoch in range(epochs if epochs is not None else self.config.epochs):
            stats = self.train_epoch(examples)
            record = {"epoch": epoch, **stats.as_dict()}
            history.append(record)
            if callback is not None:
                callback(record)
        return history
