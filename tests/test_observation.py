"""Tests for the observation encoding."""

from __future__ import annotations

import numpy as np

from agot.engine import observation as obs
from agot.engine.actions import ActionKind
from agot.engine.game import Game, GameConfig


def test_dimensions_are_documented_constants() -> None:
    assert obs.STATE_DIM == 97
    assert obs.ACTION_DIM == 136
    assert obs.N_HOUSES == 6


def test_observe_shapes_match_constants() -> None:
    game = Game.new(GameConfig(seed=0))
    observation = obs.observe(game)
    assert observation.state_features.shape == (obs.STATE_DIM,)
    assert observation.action_features.shape == (
        len(observation.actions),
        obs.ACTION_DIM,
    )
    assert len(observation.actions) == len(game.legal_actions())


def test_state_encoding_is_deterministic() -> None:
    game = Game.new(GameConfig(seed=1))
    first = obs.encode_state(game)
    second = obs.encode_state(game)
    np.testing.assert_array_equal(first, second)


def test_action_kind_block_is_one_hot() -> None:
    game = Game.new(GameConfig(seed=2))
    action = next(
        a for a in game.legal_actions() if a.kind is ActionKind.PLACE_ORDER
    )
    vector = obs.encode_action(game, action)
    kind_block = vector[: obs.N_ACTION_KINDS]
    assert kind_block.sum() == 1.0
    assert kind_block[list(ActionKind).index(ActionKind.PLACE_ORDER)] == 1.0


def test_absent_houses_encode_to_zeros() -> None:
    game = Game.new(GameConfig(houses=("stark", "lannister"), seed=3))
    state = obs.encode_state(game)
    block = obs.HOUSE_BLOCK
    for index, house in enumerate(obs.HOUSE_ORDER):
        if house in game.state.houses:
            continue
        chunk = state[index * block : (index + 1) * block]
        assert not chunk.any(), house
