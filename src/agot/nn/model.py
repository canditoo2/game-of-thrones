"""The policy/value network.

Architecture: a *pointer* network, the standard answer to an open-ended action
space. A shared trunk embeds the position into a vector ``h``; a second trunk
embeds each candidate action into a vector ``a_i``; the policy logit for action
``i`` is their dot product scaled by ``1/sqrt(d)``. The value head emits one
number per canonical house, because the game has up to six players and a single
scalar cannot express everybody's prospects.

Requires PyTorch, which is an optional extra::

    pip install -e ".[rl]"

Nothing else in :mod:`agot.engine` or :mod:`agot.agents` imports this module, so
the rules engine keeps working on a machine without torch.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from agot.engine.actions import Action
from agot.engine.game import Game
from agot.engine.observation import ACTION_DIM, N_HOUSES, STATE_DIM, encode_action, encode_state


class PolicyValueNet(nn.Module):
    """Pointer-style policy head plus a per-house value head."""

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
        self.n_houses = n_houses

        self.state_trunk = _mlp(state_dim, hidden, trunk_layers)
        self.action_trunk = _mlp(action_dim, hidden, trunk_layers)
        self.policy_proj = nn.Linear(hidden, hidden)
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
        query = self.policy_proj(h).unsqueeze(1)  # (B, 1, H)
        logits = (a * query).sum(dim=-1) / math.sqrt(self.hidden)  # (B, N)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, -1e9)
        values = self.value_head(h)  # (B, n_houses)
        return logits, values

    # -- inference helpers ----------------------------------------------

    @torch.no_grad()
    def predict(self, game: Game) -> tuple[list[Action], np.ndarray, np.ndarray]:
        """Evaluate one position end to end.

        Returns the legal actions, their probability distribution and the
        predicted value vector.
        """
        actions = game.legal_actions()
        if not actions:
            raise ValueError("cannot predict on a position with no legal actions")
        state = torch.from_numpy(encode_state(game)).unsqueeze(0)
        action_tensor = torch.from_numpy(
            np.stack([encode_action(game, action) for action in actions])
        ).unsqueeze(0)
        mask = torch.ones((1, len(actions)), dtype=torch.float32)
        logits, values = self.forward(state, action_tensor, mask)
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        return actions, probs.astype(np.float32), values[0].cpu().numpy()

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "hidden": self.hidden,
                    "n_houses": self.n_houses,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> PolicyValueNet:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = payload.get("config", {})
        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


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
        _, probs, values = self.model.predict(game)
        if len(probs) != len(actions):  # pragma: no cover - safety net
            raise RuntimeError("network policy does not match the legal actions")
        return probs, values
