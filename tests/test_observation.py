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


def test_region_block_initialises_every_slot() -> None:
    """`_region_block` returns an ``np.empty`` array, so no slot may leak garbage.

    Two slots used to be left unwritten - the reserved controller slot and the
    whole Order one-hot while an order is hidden - which occasionally surfaced
    as an enormous (uninitialised-memory) float that blew up the network.
    """
    game = Game.new(GameConfig(seed=0))
    state = game.state
    region = state.regions["the_golden_sound"]
    region.order = "march"
    region.order_house = "baratheon"
    state.phase = Phase.PLANNING  # so the order is hidden from stark

    block = obs._region_block(region, "stark", state)
    base = 6 + obs.N_UNITS
    controller = block[base + 6 : base + 10]
    assert controller.sum() == 1.0, "exactly one controller flag must be set"
    assert controller[1] == 0.0  # reserved
    order_start = base + 10
    assert not block[order_start : order_start + obs.N_ORDERS].any()
    # The order-strength slot must be written even while the order is hidden.
    assert block[order_start + obs.N_ORDERS] == 0.0
    assert np.isfinite(block).all()


def test_encoded_actions_stay_bounded_across_a_game() -> None:
    """No candidate action may carry a runaway feature value.

    A regression guard for the ``np.empty`` leaks in :func:`obs._region_block`.
    """
    game = Game.new(GameConfig(seed=3, max_rounds=3))
    steps = 0
    while not game.is_terminal() and steps < 60:
        actions = game.legal_actions()
        if not actions:
            break
        for action in actions:
            vector = obs.encode_action(game, action)
            assert np.isfinite(vector).all()
            assert np.abs(vector).max() < 100.0, (action, vector.max())
        game.step(actions[0])
        steps += 1


def _positions_across_games(count: int = 3, depth: int = 20) -> list:
    positions = []
    for seed in range(count):
        game = Game.new(GameConfig(seed=seed, max_rounds=10))
        for _ in range(depth):
            actions = game.legal_actions()
            if game.is_terminal() or not actions:
                break
            positions.append((game.clone(), actions))
            game.step(actions[0])
    return positions


def test_region_memo_scope_does_not_change_a_row() -> None:
    """A memo shared across positions must encode the same rows as a fresh one.

    ``PolicyValueNet.predict_many`` shares one :data:`obs.RegionCache` across
    every leaf in a batch and keys it on region *object identity*, because two
    leaves can hold different objects under the same region id. Keying on the id
    string instead would let one leaf's block stand in for another's. This checks
    the invariant from the outside.
    """
    positions = _positions_across_games()
    assert len(positions) > 10

    def encode_all(shared: bool) -> list[np.ndarray]:
        caches: dict[tuple, obs.RegionCache] = {}
        rows: list[np.ndarray] = []
        for game, actions in positions:
            actor = game.state.active_house
            cache = (
                caches.setdefault((actor, game.state.orders_revealed), {})
                if shared
                else {}
            )
            for action in actions:
                out = np.zeros(obs.ACTION_DIM, dtype=np.float32)
                obs.encode_action_into(game, action, out, actor, cache)
                rows.append(out)
        return rows

    fresh = encode_all(shared=False)
    shared = encode_all(shared=True)
    assert len(fresh) == len(shared)
    for index, (alone, together) in enumerate(zip(fresh, shared, strict=True)):
        np.testing.assert_array_equal(
            alone, together, err_msg=f"row {index} differed under a shared memo"
        )
