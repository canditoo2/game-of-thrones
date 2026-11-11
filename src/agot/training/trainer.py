"""Train the policy/value network on self-play data (requires PyTorch).

The training objective is the AlphaZero one, generalised to six players:

* policy: cross-entropy between the network's masked softmax over candidate
  actions and the search's target distribution;
* value: mean-squared error between the network's per-house prediction and the
  final outcome vector, masked to the houses actually in the game.

Two optional terms exist for the planning problem, where the state vector does
not describe the board and the value head cannot rank the candidate orders:

* Q: mean-squared error between the optional per-candidate Q head's prediction
  for the *played* action and the final outcome vector (requires a model with
  ``q_head`` and examples that carry ``chosen``);
* teacher: cross-entropy on a separate set of greedy planning examples, mixed
  into each mini-batch at ``auxiliary_rate`` with ``auxiliary_weight``.

Both are off unless their inputs are present, so a default run computes the
same two losses as before.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    #: Weight of the per-candidate Q loss. Only used when the model has a Q head
    #: and the batch carries played-action indices.
    q_weight: float = 0.5
    #: Weight of the planning-teacher cross-entropy on the auxiliary examples.
    auxiliary_weight: float = 0.3


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
    #: Played-action index per example row, or ``None`` when the data was
    #: recorded without one. Never partially filled; see :func:`collate_packed`.
    chosen: torch.Tensor | None = None

    def to(self, device: str) -> PackedBatch:
        return PackedBatch(
            state=self.state.to(device),
            actions=self.actions.to(device),
            index=self.index.to(device),
            policy=self.policy.to(device),
            outcome=self.outcome.to(device),
            outcome_mask=self.outcome_mask.to(device),
            rows=self.rows,
            chosen=None if self.chosen is None else self.chosen.to(device),
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
    chosen: torch.Tensor | None = None

    def to(self, device: str) -> Batch:
        return Batch(
            state=self.state.to(device),
            actions=self.actions.to(device),
            mask=self.mask.to(device),
            policy=self.policy.to(device),
            outcome=self.outcome.to(device),
            outcome_mask=self.outcome_mask.to(device),
            chosen=None if self.chosen is None else self.chosen.to(device),
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


def _chosen_column(examples: list[Example]) -> torch.Tensor | None:
    """The played-action indices as a column, or ``None`` for a legacy batch.

    A batch is never allowed to mix the two: half the rows would train the Q
    head on the played action and half would silently train it on nothing.
    """
    values = [example.chosen for example in examples]
    present = sum(value is not None for value in values)
    if present == 0:
        return None
    if present != len(values):
        raise ValueError(
            "cannot mix examples with and without a chosen action in one batch"
        )
    for index, example in enumerate(examples):
        chosen = example.chosen
        if not 0 <= int(chosen) < len(example.actions):  # type: ignore[arg-type]
            raise ValueError(
                f"example {index}: chosen index {chosen} is outside an action "
                f"set of {len(example.actions)}"
            )
    return torch.tensor(values, dtype=torch.int64)


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

    return Batch(
        state, actions, mask, policy, outcome, outcome_mask, _chosen_column(examples)
    )


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

    return PackedBatch(
        state,
        actions,
        index,
        policy,
        outcome,
        outcome_mask,
        size,
        _chosen_column(examples),
    )


@dataclass
class TrainStats:
    """Losses averaged over an epoch."""

    batches: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    q_loss: float = 0.0
    auxiliary_loss: float = 0.0
    total_loss: float = 0.0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        denom = max(1, self.batches)
        return {
            "batches": self.batches,
            "policy_loss": self.policy_loss / denom,
            "value_loss": self.value_loss / denom,
            "q_loss": self.q_loss / denom,
            "auxiliary_loss": self.auxiliary_loss / denom,
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

    def train_epoch(
        self,
        examples: Sequence[Example],
        *,
        auxiliary: Sequence[Example] | None = None,
        auxiliary_rate: float = 0.0,
    ) -> TrainStats:
        """One pass over ``examples``, optionally mixed with teacher examples.

        ``auxiliary`` is a separate (small) dataset; ``auxiliary_rate`` is the
        fraction of each mini-batch drawn from it and added to the auxiliary
        cross-entropy. A rate of zero disables the term, so the default path is
        the same computation as before.
        """
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
            teacher = self._auxiliary_batch(auxiliary, auxiliary_rate, len(chunk))
            policy_loss, value_loss, q_loss, aux_loss, total = self._step(
                batch, teacher
            )
            stats.batches += 1
            stats.policy_loss += policy_loss
            stats.value_loss += value_loss
            stats.q_loss += 0.0 if q_loss is None else q_loss
            stats.auxiliary_loss += 0.0 if aux_loss is None else aux_loss
            stats.total_loss += total
        return stats

    def _auxiliary_batch(
        self,
        auxiliary: Sequence[Example] | None,
        rate: float,
        width: int,
    ) -> PackedBatch | None:
        """Draw ``rate * width`` teacher examples, without replacement issues.

        Drawing with replacement is fine: the teacher set is small and each
        batch only needs a representative sample, not an epoch over it.
        """
        if (
            auxiliary is None
            or not auxiliary
            or rate <= 0.0
            or self.config.auxiliary_weight <= 0.0
        ):
            return None
        count = int(round(rate * width))
        if count <= 0:
            return None
        picks = self.rng.integers(0, len(auxiliary), size=count)
        examples = [auxiliary[int(index)] for index in picks]
        return collate_packed(examples).to(self.device)

    def _losses(
        self,
        batch: Batch | PackedBatch,
        auxiliary: PackedBatch | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """The policy/value/Q/teacher losses for one batch, without stepping."""
        q_values = None
        if isinstance(batch, PackedBatch):
            output = self.model.forward_packed(
                batch.state, batch.actions, batch.index
            )
            logits, values = output[:2]
            if getattr(self.model, "q_head", False):
                q_values = output[2]
            log_probs = segment_log_softmax(logits, batch.index, batch.rows)
            policy_loss = -(batch.policy * log_probs).sum() / batch.rows
        else:
            output = self.model(batch.state, batch.actions, batch.mask)
            logits, values = output[:2]
            if getattr(self.model, "q_head", False):
                q_values = output[2]
            log_probs = torch.log_softmax(logits, dim=-1)
            policy_loss = -(batch.policy * log_probs).sum(dim=-1).mean()

        value_mask = batch.outcome_mask
        squared = (values - batch.outcome) ** 2
        value_loss = (squared * value_mask).sum() / value_mask.sum().clamp(min=1.0)

        q_loss = None
        if q_values is not None and batch.chosen is not None:
            chosen_values = _gather_chosen(q_values, batch)
            squared_q = (chosen_values - batch.outcome) ** 2
            q_loss = (squared_q * value_mask).sum() / value_mask.sum().clamp(min=1.0)

        total = policy_loss + self.config.value_weight * value_loss
        if q_loss is not None:
            total = total + self.config.q_weight * q_loss

        auxiliary_loss = None
        if auxiliary is not None:
            aux_output = self.model.forward_packed(
                auxiliary.state, auxiliary.actions, auxiliary.index
            )
            aux_log_probs = segment_log_softmax(
                aux_output[0], auxiliary.index, auxiliary.rows
            )
            auxiliary_loss = -(auxiliary.policy * aux_log_probs).sum() / auxiliary.rows
            total = total + self.config.auxiliary_weight * auxiliary_loss

        return policy_loss, value_loss, q_loss, auxiliary_loss, total

    def _step(
        self,
        batch: Batch | PackedBatch,
        auxiliary: PackedBatch | None = None,
    ) -> tuple[
        float, float, float | None, float | None, float
    ]:
        policy_loss, value_loss, q_loss, aux_loss, total = self._losses(
            batch, auxiliary
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.config.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return (
            float(policy_loss.detach()),
            float(value_loss.detach()),
            None if q_loss is None else float(q_loss.detach()),
            None if aux_loss is None else float(aux_loss.detach()),
            float(total.detach()),
        )

    def train(
        self,
        buffer: ReplayBuffer,
        epochs: int | None = None,
        *,
        auxiliary: Sequence[Example] | None = None,
        auxiliary_rate: float = 0.0,
        callback=None,
    ) -> list[dict]:
        """Train for ``epochs`` passes over the current buffer contents.

        The buffer itself is indexed: it implements ``__len__``/``__getitem__``
        over its arrays, so an epoch no longer materialises every example as an
        object before touching it. ``auxiliary``/``auxiliary_rate`` are handed
        to every epoch (see :meth:`train_epoch`).
        """
        examples = buffer
        history: list[dict] = []
        for epoch in range(epochs if epochs is not None else self.config.epochs):
            stats = self.train_epoch(
                examples, auxiliary=auxiliary, auxiliary_rate=auxiliary_rate
            )
            record = {"epoch": epoch, **stats.as_dict()}
            history.append(record)
            if callback is not None:
                callback(record)
        return history


def _gather_chosen(
    q_values: torch.Tensor, batch: Batch | PackedBatch
) -> torch.Tensor:
    """``q_values`` at the played candidate of every example row.

    Q is scored per candidate. The loss may only read the row that was actually
    played; every other candidate's value is trained by its own decisions, not
    by this example's outcome.
    """
    if batch.chosen is None:  # pragma: no cover - guarded by the caller
        raise ValueError("batch has no chosen action indices")
    if isinstance(batch, PackedBatch):
        # ``index`` is sorted, so searchsorted gives each row's first slot.
        rows = torch.arange(batch.rows, device=batch.index.device)
        starts = torch.searchsorted(batch.index, rows)
        return q_values[starts + batch.chosen]
    gather_index = batch.chosen.view(-1, 1, 1).expand(-1, 1, q_values.size(-1))
    return torch.gather(q_values, 1, gather_index).squeeze(1)
