"""``Game.snapshot()`` is plain Python data with ``json.loads`` semantics.

The snapshot is built directly as Python objects (no JSON text round trip);
these checks pin what the old ``json.loads`` path guaranteed: sorted object
keys, ``bool`` vs ``int`` kept apart, ``None`` for nulls, and a value that
survives a JSON round trip unchanged.
"""

from __future__ import annotations

import json

import pytest

from agot.agents import GreedyPolicy
from agot.engine.game import GameConfig
from agot.engine.rust import RustGame


def _walk(value):
    if isinstance(value, dict):
        assert list(value) == sorted(value), "object keys must be sorted"
        for key, item in value.items():
            assert isinstance(key, str)
            _walk(item)
    elif isinstance(value, list):
        for item in value:
            _walk(item)
    else:
        assert value is None or isinstance(value, (bool, int, float, str))


def _typed(value):
    """The value with every dict's key order and every leaf's exact type."""
    if isinstance(value, dict):
        return ("dict", [(key, _typed(item)) for key, item in value.items()])
    if isinstance(value, list):
        return ("list", [_typed(item) for item in value])
    return (type(value).__name__, value)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_snapshot_equals_the_fingerprint_reference(seed):
    """The direct builder matches the serde fingerprint on every position."""
    config = GameConfig(
        seed=seed,
        max_rounds=6,
        sequential_westeros=True,
        deferred_battles=True,
        doran_choice=True,
        choice_choices=True,
    )
    game = RustGame.new(config)
    policy = GreedyPolicy(seed)
    seen = set()
    compared = 0
    while not game.is_terminal() and compared < 4000:
        snapshot = game._inner.snapshot()
        assert _typed(snapshot) == _typed(game._inner._snapshot_reference())
        compared += 1
        for key in ("pending_march", "westeros_choice", "pending_westeros", "return_phase"):
            if snapshot[key] is not None:
                seen.add(key)
        if snapshot["pending_bids"]:
            seen.add("pending_bids")
        if game.state.westeros_waiting:
            game.continue_westeros()
            continue
        if game.current_player() is None:
            break
        action, _ = policy.decide(game)
        game.step(action)
    assert compared > 100
    assert {"pending_march", "pending_bids"} <= seen


def test_snapshot_round_trips_through_json_unchanged():
    game = RustGame.new(GameConfig(seed=4, max_rounds=2, sequential_westeros=True))
    policy = GreedyPolicy(4)
    seen = 0
    while not game.is_terminal() and seen < 120:
        if game.state.westeros_waiting:
            game.continue_westeros()
            continue
        if game.current_player() is None:
            break
        snapshot = game._inner.snapshot()
        _walk(snapshot)
        assert json.loads(json.dumps(snapshot)) == snapshot
        assert type(snapshot["raven_used"]) is bool
        assert type(snapshot["round"]) is int
        region = next(iter(snapshot["regions"].values()))
        assert type(region["garrison"]) is int
        assert all(type(count) is int for count in region["units"].values())
        action, _ = policy.decide(game)
        game.step(action)
        seen += 1
    assert seen > 20
