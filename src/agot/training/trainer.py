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
from agot.nn.model import PolicyValueNet, segment_log_softmax
from agot.training.replay import Example, ReplayBuffer


@dataclass
class TrainConfig:
    """Hyper-parameters for one training run."""

    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 1
    value_weight: float = 1.0
    #: Decoupled (AdamW) decay on weight matrices only; biases and other 1-D
    #: parameters are left alone. See :func:`_parameter_groups`.
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"


@dataclass
class PackedBatch:
    """A collated mini-batch with no padding.

    ``actions`` and ``policy`` hold the concatenated candidates of every
    example; ``index[i]`` names the example row that candidate ``i`` belongs
    to. The scorer and the policy loss run over exactly ``T = sum N_i`` rows
    instead of ``B * max N_i``.
    """

    state: torch.Tensor
    actions: torch.Tensor
    index: torch.Tensor
    policy: torch.Tensor
    outcome: torch.Tensor
    outcome_mask: torch.Tensor
    rows: int

    def to(self, device: str) -> PackedBatch:
        return PackedBatch(
            state=self.state.to(device),
            actions=self.actions.to(device),
            index=self.index.to(device),
            policy=self.policy.to(device),
            outcome=self.outcome.to(device),
            outcome_mask=self.outcome_mask.to(device),
            rows=self.rows,
        )


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


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decayed and undecayed groups.

    Biases and other one-dimensional parameters are never decayed. Decay is
    *decoupled* (``AdamW``), so it shrinks a weight geometrically rather than by a
    constant step. Plain ``Adam`` with an L2 penalty adds ``wd * parameter`` to
    the gradient, which adaptive scaling turns into a roughly constant pull
    toward zero -- strong enough to erase a parameter whose task gradient is
    weak, which is how the original policy head died.
    """
    decayed: list[nn.Parameter] = []
    undecayed: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias"):
            undecayed.append(parameter)
        else:
            decayed.append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]


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
        if total > 0:
            normalised = target / total
        else:
            normalised = np.full(count, 1.0 / count, dtype=np.float32)
        policy[row, :count] = torch.from_numpy(normalised)
        outcome[row] = torch.from_numpy(np.asarray(example.outcome, dtype=np.float32))
        outcome_mask[row] = torch.from_numpy(
            (np.asarray(example.outcome) != 0).astype(np.float32)
        )

    return Batch(state, actions, mask, policy, outcome, outcome_mask)


def collate_packed(examples: list[Example]) -> PackedBatch:
    """Collate variable-length examples into the unpadded packed layout."""
    size = len(examples)
    counts = [len(example.actions) for example in examples]
    total = sum(counts)
    state = torch.zeros(size, STATE_DIM, dtype=torch.float32)
    actions = torch.zeros(total, ACTION_DIM, dtype=torch.float32)
    index = torch.empty(total, dtype=torch.int64)
    policy = torch.zeros(total, dtype=torch.float32)
    outcome = torch.zeros(size, N_HOUSES, dtype=torch.float32)
    outcome_mask = torch.zeros(size, N_HOUSES, dtype=torch.float32)

    cursor = 0
    for row, example in enumerate(examples):
        count = counts[row]
        state[row] = torch.from_numpy(np.asarray(example.state, dtype=np.float32))
        actions[cursor : cursor + count] = torch.from_numpy(
            np.asarray(example.actions, dtype=np.float32)
        )
        index[cursor : cursor + count] = row
        target = np.asarray(example.policy, dtype=np.float32)
        total_target = target.sum()
        if total_target > 0:
            normalised = target / total_target
        else:
            normalised = np.full(count, 1.0 / count, dtype=np.float32)
        policy[cursor : cursor + count] = torch.from_numpy(normalised)
        outcome[row] = torch.from_numpy(np.asarray(example.outcome, dtype=np.float32))
        outcome_mask[row] = torch.from_numpy(
            (np.asarray(example.outcome) != 0).astype(np.float32)
        )
        cursor += count

    return PackedBatch(state, actions, index, policy, outcome, outcome_mask, size)


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
        self.optimizer = torch.optim.AdamW(
            _parameter_groups(self.model, self.config.weight_decay),
            lr=self.config.learning_rate,
        )
        self.rng = np.random.default_rng(0)

    # -- persistence -----------------------------------------------------

    def state_dict(self) -> dict:
        """Optimizer momentum and RNG state, for resuming a run.

        The weights themselves live in the network checkpoint; this is the rest
        of what :meth:`train_epoch` depends on. Adam's per-parameter moments are
        the reason a resumed run does not take a transient hit.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]

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
            batch = collate_packed(chunk).to(self.device)
            policy_loss, value_loss, total = self._step(batch)
            stats.batches += 1
            stats.policy_loss += policy_loss
            stats.value_loss += value_loss
            stats.total_loss += total
        return stats

    def _step(self, batch: Batch | PackedBatch) -> tuple[float, float, float]:
        if isinstance(batch, PackedBatch):
            logits, values = self.model.forward_packed(
                batch.state, batch.actions, batch.index
            )
            log_probs = segment_log_softmax(logits, batch.index, batch.rows)
            policy_loss = -(batch.policy * log_probs).sum() / batch.rows
        else:
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
        return (
            float(policy_loss.detach()),
            float(value_loss.detach()),
            float(total.detach()),
        )

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
