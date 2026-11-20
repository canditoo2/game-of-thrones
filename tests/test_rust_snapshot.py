"""``Game.snapshot()`` is plain Python data with ``json.loads`` semantics.

The snapshot is built directly as Python objects (no JSON text round trip);
these checks pin what the old ``json.loads`` path guaranteed: sorted object
keys, ``bool`` vs ``int`` kept apart, ``None`` for nulls, and a value that
survives a JSON round trip unchanged.
"""

from __future__ import annotations

import json

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
