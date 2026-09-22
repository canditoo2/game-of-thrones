"""Tests for the observation encoding."""

from __future__ import annotations

import numpy as np

from agot.engine import observation as obs
from agot.engine.actions import Action, ActionKind
from agot.engine.enums import Phase
from agot.engine.game import Game, GameConfig


def test_dimensions_are_documented_constants() -> None:
    assert obs.STATE_DIM == 97
    assert obs.ACTION_DIM == 141
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


def test_support_side_is_encoded() -> None:
    game = Game.new(GameConfig(seed=4))
    vector = obs.encode_action(
        game, Action.declare_support("stark", "winterfell", "attacker")
    )
    assert vector.shape == (obs.ACTION_DIM,)
    assert list(vector[-obs.N_SIDES :]) == [1.0, 0.0, 0.0]
    refusing = obs.encode_action(
        game, Action.declare_support("stark", "winterfell", "refuse")
    )
    assert list(refusing[-obs.N_SIDES :]) == [0.0, 0.0, 1.0]


def test_absent_houses_encode_to_zeros() -> None:
    game = Game.new(GameConfig(houses=("stark", "lannister"), seed=3))
    state = obs.encode_state(game)
    block = obs.HOUSE_BLOCK
    for index, house in enumerate(obs.HOUSE_ORDER):
        if house in game.state.houses:
            continue
        chunk = state[index * block : (index + 1) * block]
        assert not chunk.any(), house


def test_orders_revealed_tracks_the_phase() -> None:
    game = Game.new(GameConfig(seed=6))
    state = game.state
    state.phase = Phase.PLANNING
    assert not state.orders_revealed
    state.phase = Phase.RAVEN
    assert not state.orders_revealed
    state.phase = Phase.ACTION
    assert state.orders_revealed


def test_opponent_orders_are_hidden_until_the_action_phase() -> None:
    game = Game.new(GameConfig(seed=5))
    state = game.state
    state.regions["kings_landing"].controller = "baratheon"
    state.regions["kings_landing"].order = "march"
    state.regions["kings_landing"].order_house = "baratheon"
    state.regions["winterfell"].controller = "stark"
    state.regions["winterfell"].order = "defend_2"
    state.regions["winterfell"].order_house = "stark"
    action = Action.march(
        "baratheon", "kings_landing", "winterfell", units=(1, 0, 0, 0)
    )
    state.phase = Phase.PLANNING

    # The *identity* of a hidden enemy order is invisible: two different orders
    # encode identically while the Action Phase has not begun.
    hidden_defend = obs.encode_action(game, action, "baratheon")
    state.regions["winterfell"].order = "march_plus"
    hidden_march = obs.encode_action(game, action, "baratheon")
    np.testing.assert_array_equal(hidden_defend, hidden_march)

    # A house always sees the order it placed itself.
    own_changed = obs.encode_action(game, action, "baratheon")
    state.regions["kings_landing"].order = "defend_2"
    assert not np.array_equal(
        own_changed, obs.encode_action(game, action, "baratheon")
    )

    # Once the Action Phase begins, the enemy order's identity is public.
    state.regions["winterfell"].order = "defend_2"
    state.phase = Phase.ACTION
    revealed_defend = obs.encode_action(game, action, "baratheon")
    state.regions["winterfell"].order = "march_plus"
    revealed_march = obs.encode_action(game, action, "baratheon")
    assert not np.array_equal(revealed_defend, revealed_march)
