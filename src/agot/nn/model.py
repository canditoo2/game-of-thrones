"""The policy/value network.

Architecture: a *pointer* network, the standard answer to an open-ended action
space. A shared trunk embeds the position into a vector ``h``; a second trunk
embeds each candidate action into a vector ``a_i``; a scorer MLP reads the
concatenation ``[h, a_i]`` and emits one logit per candidate. The value head
emits one number per canonical house, because the game has up to six players and
a single scalar cannot express everybody's prospects.

The scorer is deliberately *not* a bare dot product ``h . a_i``. A product of two
learned tensors has an absorbing state: if either factor reaches zero the
gradient through the other vanishes, and the whole policy branch stops training
for good. That is what happened to the first two runs -- the policy parameters
were ground down to the float32 floor (~1e-37) and emitted a uniform
distribution forever -- so the head now reads ``h`` unconditionally and cannot
shut itself off. See the "Notable fixes" section of ``docs/roadmap.md``.

Requires PyTorch, which is an optional extra::

    pip install -e ".[rl]"

Nothing else in :mod:`agot.engine` or :mod:`agot.agents` imports this module, so
the rules engine keeps working on a machine without torch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from agot.engine.actions import Action
from agot.engine.game import Game
from agot.engine.observation import (
    ACTION_DIM,
    N_HOUSES,
    STATE_DIM,
    encode_actions,
    encode_state,
)


class PolicyValueNet(nn.Module):
    """A scorer MLP over ``[state, action]`` plus a per-house value head."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 256,
        trunk_layers: int = 2,
        n_houses: int = N_HOUSES,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.trunk_layers = trunk_layers
        self.n_houses = n_houses

        self.state_trunk = _mlp(state_dim, hidden, trunk_layers)
        self.action_trunk = _mlp(action_dim, hidden, trunk_layers)
        #: Scorer over ``[h, a_i]``. A dot product here would be a product of two
        #: learned tensors, which has a zero fixed point that kills all gradient.
        self.policy_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_houses),
            nn.Tanh(),
        )

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, values)``.

        ``state`` has shape ``(B, state_dim)``, ``actions`` ``(B, N, action_dim)``
        and ``mask`` ``(B, N)`` with ones for legal actions. Logits for illegal
        actions are pushed to a large negative number.
        """
        h = self.state_trunk(state)  # (B, H)
        a = self.action_trunk(actions)  # (B, N, H)
        context = h.unsqueeze(1).expand(-1, a.size(1), -1)  # (B, N, H)
        logits = self.policy_head(torch.cat([context, a], dim=-1)).squeeze(-1)  # (B, N)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, -1e9)
        values = self.value_head(h)  # (B, n_houses)
        return logits, values

    def forward_packed(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward over a *concatenated* action set, one segment per state row.

        ``state`` is ``(B, state_dim)``, ``actions`` is ``(T, action_dim)`` for
        ``T = sum_i N_i`` candidates, and ``index`` is ``(T,)`` giving the state
        row each action belongs to. This is :meth:`forward` without the padding:
        a leaf with two candidates costs two rows instead of being widened to the
        widest leaf in the batch. On real self-play batches that padding made the
        action trunk and the scorer do 2.5x the rows they needed, and up to 12x on
        the worst batch.

        Returns flat ``(T,)`` logits and ``(B, n_houses)`` values. The caller
        normalises each segment, because the policy is a distribution over one
        leaf's actions only (see :func:`segment_softmax`).
        """
        h = self.state_trunk(state)  # (B, H)
        a = self.action_trunk(actions)  # (T, H)
        logits = self.policy_head(torch.cat([h[index], a], dim=-1)).squeeze(-1)
        return logits, self.value_head(h)

    # -- inference helpers ----------------------------------------------

    @torch.inference_mode()
    def predict(self, game: Game) -> tuple[list[Action], np.ndarray, np.ndarray]:
        """Evaluate one position end to end.

        Returns the legal actions, their probability distribution and the
        predicted value vector.
        """
        actions = game.legal_actions()
        if not actions:
            raise ValueError("cannot predict on a position with no legal actions")
        return self.predict_actions(game, actions)

    @torch.inference_mode()
    def predict_actions(
        self, game: Game, actions: list[Action] | tuple[Action, ...]
    ) -> tuple[list[Action], np.ndarray, np.ndarray]:
        """Like :meth:`predict` but for an action list the caller already has.

        Tree search computes the legal actions to expand a node; re-deriving
        them here would double that cost on every simulation.
        """
        actions = list(actions)
        state = torch.from_numpy(encode_state(game)).unsqueeze(0)
        action_tensor = torch.from_numpy(
            encode_actions(game, actions, game.state.active_house)
        ).unsqueeze(0)
        mask = torch.ones((1, len(actions)), dtype=torch.float32)
        logits, values = self.forward(state, action_tensor, mask)
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        return actions, probs.astype(np.float32), values[0].cpu().numpy()

    @torch.inference_mode()
    def predict_many(
        self, requests: list[tuple[Game, list[Action]]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Score several positions in one forward pass, without padding.

        ``requests`` is a list of ``(game, actions)``. Every candidate of every
        request goes through the action trunk exactly once, and the policy head
        is normalised within each request's own action set, so the two leaves
        with the fewest candidates are not widened to match the widest one in the
        batch. This is what makes batched self-play pay off: one network call
        scores many leaves, and it no longer scores their padding too.
        """
        if not requests:
            return []
        counts = [len(actions) for _, actions in requests]
        state = np.stack([encode_state(game) for game, _ in requests])
        total = sum(counts)
        index = np.empty(total, dtype=np.int64)
        # One call per leaf: the Rust encoder writes its whole candidate matrix
        # in one go, so there is no per-candidate Python work left.
        rows: list[np.ndarray] = []
        cursor = 0
        for row, (game, actions) in enumerate(requests):
            matrix = encode_actions(game, actions, game.state.active_house)
            rows.append(matrix)
            index[cursor : cursor + len(actions)] = row
            cursor += len(actions)
        action_tensor = (
            np.concatenate(rows, axis=0)
            if rows
            else np.zeros((0, ACTION_DIM), dtype=np.float32)
        )
        logits, values = self.forward_packed(
            torch.from_numpy(state),
            torch.from_numpy(action_tensor),
            torch.from_numpy(index),
        )
        flat_index = torch.from_numpy(index)
        probabilities = (
            segment_softmax(logits, flat_index, len(requests)).cpu().numpy()
        )
        value_rows = values.cpu().numpy()
        bounds = np.concatenate(([0], np.cumsum(counts)))
        return [
            (
                probabilities[bounds[row] : bounds[row + 1]].astype(np.float32),
                value_rows[row],
            )
            for row in range(len(requests))
        ]

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "hidden": self.hidden,
                    "trunk_layers": self.trunk_layers,
                    "n_houses": self.n_houses,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> PolicyValueNet:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        state = payload.get("state_dict", {})
        if any(key.startswith("policy_proj.") for key in state):
            raise ValueError(
                f"{path} was written by the old dot-product policy head. That "
                "head collapsed to zero and never trained, so the checkpoint is "
                "not worth loading; start a fresh run."
            )
        config = payload.get("config", {})
        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def segment_softmax(
    logits: torch.Tensor, index: torch.Tensor, rows: int
) -> torch.Tensor:
    """Softmax of ``logits`` within each of ``rows`` contiguous segments.

    ``index`` maps each logit to its segment (``0 <= index[i] < rows``). This is
    ``softmax`` along the last axis of the equivalent padded ``(rows, width)``
    matrix, computed without materialising the padding.
    """
    maximum = torch.full(
        (rows,), float("-inf"), dtype=logits.dtype, device=logits.device
    ).scatter_reduce(0, index, logits, reduce="amax")
    shifted = torch.exp(logits - maximum[index])
    total = torch.zeros(
        rows, dtype=logits.dtype, device=logits.device
    ).index_add_(0, index, shifted)
    return shifted / total[index]


def _mlp(input_dim: int, hidden: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(max(1, layers)):
        modules += [nn.Linear(current, hidden), nn.ReLU()]
        current = hidden
    return nn.Sequential(*modules)


class NeuralEvaluator:
    """Adapter that lets :class:`~agot.agents.mcts.MCTS` drive a neural net."""

    def __init__(self, model: PolicyValueNet, temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = temperature

    def __call__(
        self, game: Game, actions: list[Action]
    ) -> tuple[np.ndarray, np.ndarray]:
        _, probs, values = self.model.predict_actions(game, actions)
        if len(probs) != len(actions):  # pragma: no cover - safety net
            raise RuntimeError("network policy does not match the legal actions")
        return probs, values

    def evaluate_batch(
        self, requests: list[tuple[Game, list[Action]]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Score many ``(game, actions)`` leaves in one forward pass."""
        return self.model.predict_many(requests)
