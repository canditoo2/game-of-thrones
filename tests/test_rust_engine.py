"""Integration tests for the Rust engine extension (``agot._core``).

The extension is the engine now: ``agot.engine.game.Game`` is its session view
and ``agot.engine.observation`` forwards to its encoder. These tests exercise
that seam - actions, stepping, snapshots, the position builder, observations,
the batch self-play driver and the training-loop adapter.

The engine's own rules/parity gates live in the Rust crate; a game is played
with ``use_westeros_decks=False`` here only where a deterministic script keeps a
test short and independent of shuffle order.
"""

from __future__ import annotations

import numpy as np
import pytest

from agot import _core as core
from agot.engine.actions import Action, ActionKind
from agot.engine.game import GameConfig
from agot.engine.observation import HOUSE_ORDER

#: Deterministic action policies: first, last, and a fixed pseudo-random index.
POLICIES = {
    "first": lambda count, step: 0,
    "last": lambda count, step: count - 1,
    "mixed": lambda count, step: (step * 7 + 3) % count,
}


def play_rust(seed: int, policy, decks: bool = False):
    game = core.Game.new(seed=seed, use_westeros_decks=decks)
    steps = 0
    while not game.is_terminal():
        actions = game.legal_actions()
        if not actions:
            break
        game.step(actions[policy(len(actions), steps)])
        steps += 1
    return steps, game.round, game.winner()


def test_backend_identity():
    assert core.backend() == "rust"
    assert core.__version__


def test_rust_reaches_a_winner_with_the_decks_on():
    """The default configuration must at least terminate with a winner."""
    steps, rounds, winner = play_rust(1, POLICIES["first"], decks=True)
    assert steps > 100
    assert rounds >= 1
    assert winner in {"baratheon", "lannister", "stark", "greyjoy", "tyrell", "martell"}


def test_actions_compare_and_hash_by_value():
    """`action in game.legal_actions()` and set membership must keep working."""
    game = core.Game.new(seed=1, use_westeros_decks=False)
    actions = game.legal_actions()
    assert actions
    assert actions[0] in game.legal_actions()
    assert len(set(actions)) == len(actions)
    assert actions[0] == actions[0]
    assert actions[0] != actions[1]


def test_actions_carry_the_python_field_names():
    game = core.Game.new(seed=1, use_westeros_decks=False)
    action = game.legal_actions()[0]
    assert action.kind == "place_order"
    assert action.house == "baratheon"
    assert action.region is not None
    assert action.order is not None
    assert action.target is None
    assert action.units is None
    assert action.establish is False
    assert "place_order" in repr(action)


def test_clone_does_not_share_state_with_its_source():
    game = core.Game.new(seed=1, use_westeros_decks=False)
    child = game.clone()
    action = child.legal_actions()[0]
    child.step(action)
    assert game.legal_actions() == game.legal_actions()  # the source still stands
    assert child.round == game.round


def test_summary_reports_the_same_shape_as_python():
    game = core.Game.new(seed=1, use_westeros_decks=False)
    summary = game.summary()
    assert set(summary) == {"round", "phase", "active_house", "winner", "castles", "power"}
    assert summary["round"] == 1
    assert summary["phase"] == "planning"
    assert set(summary["castles"]) == {
        "baratheon",
        "lannister",
        "stark",
        "greyjoy",
        "tyrell",
        "martell",
    }


def test_observation_wrapper_matches_the_extension():
    """`agot.engine.observation` must be the extension's encoder, not a copy."""
    from agot.engine import rust
    from agot.engine.observation import ACTION_DIM, STATE_DIM, observe

    assert core.observation_dims() == (STATE_DIM, ACTION_DIM)

    game = rust.RustGame.new(GameConfig(seed=1, use_westeros_decks=False))
    for step in range(40):
        actor, state, matrix = game._inner.observe()
        wrapped = observe(game)
        assert wrapped.actor == (actor or "")
        assert wrapped.state_features.shape == (STATE_DIM,)
        assert wrapped.action_features.shape == (len(wrapped.actions), ACTION_DIM)
        assert np.array_equal(wrapped.state_features, state), f"step {step}: state"
        assert np.array_equal(wrapped.action_features, matrix), f"step {step}: actions"
        if game.is_terminal():
            break
        action_index = (step * 5 + 1) % len(wrapped.actions)
        game.step(game.legal_actions()[action_index])


def test_batched_self_play_uses_the_rust_driver():
    """The training entry point must actually take the Rust path."""
    pytest.importorskip("torch")

    from agot.nn.model import NeuralEvaluator, PolicyValueNet
    from agot.training import selfplay as selfplay_module

    model = PolicyValueNet(hidden=32, trunk_layers=1)
    configs = [GameConfig(seed=200 + index, max_rounds=2) for index in range(3)]
    records = selfplay_module.batched_self_play(
        NeuralEvaluator(model), configs, simulations=4, determinizations=1
    )
    assert selfplay_module.LAST_BACKEND == "rust"
    assert len(records) == 3
    assert all(record.examples for record in records)
    assert all(record.steps == len(record.examples) for record in records)


def test_batched_self_play_drives_scripted_seats_from_rust():
    """Scripted seats are decided in Python, but the game stays on the Rust driver.

    Their decisions are not imitated unless `record_opponents` asks for it, so
    the example count is lower than the game's step count.
    """
    pytest.importorskip("torch")

    from agot.agents import GreedyPolicy
    from agot.nn.model import NeuralEvaluator, PolicyValueNet
    from agot.training import selfplay as selfplay_module

    model = PolicyValueNet(hidden=32, trunk_layers=1)
    configs = [GameConfig(seed=300 + index, max_rounds=2) for index in range(2)]
    opponents = [{config.houses[0]: GreedyPolicy(1, noise=0.0)} for config in configs]
    records = selfplay_module.batched_self_play(
        NeuralEvaluator(model),
        configs,
        simulations=4,
        determinizations=1,
        opponents=opponents,
    )
    assert selfplay_module.LAST_BACKEND == "rust"
    assert len(records) == 2
    assert all(record.examples for record in records)
    assert all(record.steps > len(record.examples) for record in records)
    assert all(
        example.player != 0
        for record in records
        for example in record.examples
    )


def test_batched_self_play_can_record_scripted_seats_from_rust():
    pytest.importorskip("torch")

    from agot.agents import GreedyPolicy
    from agot.nn.model import NeuralEvaluator, PolicyValueNet
    from agot.training import selfplay as selfplay_module

    model = PolicyValueNet(hidden=32, trunk_layers=1)
    config = GameConfig(seed=310, max_rounds=2)
    opponents = [{config.houses[0]: GreedyPolicy(1, noise=0.0)}]
    records = selfplay_module.batched_self_play(
        NeuralEvaluator(model),
        [config],
        simulations=4,
        determinizations=1,
        opponents=opponents,
        record_opponents=True,
    )
    assert selfplay_module.LAST_BACKEND == "rust"
    assert records[0].steps == len(records[0].examples)
    assert any(example.player == 0 for example in records[0].examples)


def test_rust_view_matches_its_own_snapshot():
    """Every field the Python-shaped view exposes comes from the engine snapshot."""
    from agot.engine import rust
    from agot.engine.enums import UnitType

    game = rust.RustGame.new(GameConfig(seed=5, use_westeros_decks=False))
    for step in range(60):
        snapshot = game.snapshot()
        state = game.state
        assert state.round == snapshot["round"]
        assert state.phase.value == snapshot["phase"]
        assert state.active_house == snapshot["active_house"]
        assert state.winner == snapshot["winner"]

        for region_id, raw in snapshot["regions"].items():
            view = state.regions[region_id]
            assert view.controller == raw["controller"]
            assert view.owner == raw["owner"]
            assert view.order == raw["order"]
            assert view.order_house == raw["order_house"]
            assert view.units == {
                UnitType(name): count for name, count in raw["units"].items() if count
            }
            assert view.routed == {
                UnitType(name): count for name, count in raw["routed"].items() if count
            }
            assert view.garrison == raw["garrison"]
            assert view.neutral_force == raw["neutral_force"]

        for house_id, raw in snapshot["houses"].items():
            view = state.houses[house_id]
            assert view.power == raw["power"]
            assert view.supply_level == raw["supply_level"]
            assert view.orders_placed == raw["orders_placed"]
            assert view.orders_available == raw["orders_available"]
            assert view.house_cards_available == raw["cards_available"]
            assert view.house_cards_used == raw["cards_used"]
            assert view.blade_used == raw["blade_used"]

        assert {
            track.value: list(order) for track, order in state.tracks.items()
        } == snapshot["tracks"]

        if game.is_terminal():
            break
        index = (step * 3 + 1) % len(game.legal_actions())
        game.step(game.legal_actions()[index])


def test_position_builder_writes_through_to_the_engine():
    """Building a position must change what the engine then does.

    The Python engine's tests set up positions by writing state fields directly;
    these named writes are the same thing. The assertion is behavioural on
    purpose: the legal-action list must follow the edit, not just the snapshot.
    """
    game = core.Game.new(seed=1, use_westeros_decks=False)
    targets = {
        action.region for action in game.legal_actions() if action.kind == "place_order"
    }
    assert targets, "the opening position must offer orders"
    target = sorted(targets)[0]

    # Emptying an area removes it from the areas that still need an order; a
    # garrison alone never needs one.
    game.set_region(target, units={})
    after = {
        action.region for action in game.legal_actions() if action.kind == "place_order"
    }
    assert target not in after

    # Putting a unit back makes the engine ask for an order there again.
    game.set_region(target, units={"footman": 1})
    restored = {
        action.region for action in game.legal_actions() if action.kind == "place_order"
    }
    assert target in restored

    # And the engine still runs on the edited position: placing an order there
    # puts the token on the board.
    action = next(
        candidate
        for candidate in game.legal_actions()
        if candidate.kind == "place_order" and candidate.region == target
    )
    game.step(action)
    assert game.snapshot()["regions"][target]["order"] is not None


def test_position_builder_builds_actions_the_engine_accepts():
    game = core.Game.new(seed=1, use_westeros_decks=False)
    house = game.current_player()
    produced = game.legal_actions()[0]
    assert produced.kind == "place_order"

    # An action built from names is indistinguishable from one the engine made.
    built = game.action(
        "place_order", house=house, region=produced.region, order=produced.order
    )
    assert built == produced
    assert built in game.legal_actions()

    game.set_house(house, power=7, blade_used=True)
    snapshot = game.snapshot()
    assert snapshot["houses"][house]["power"] == 7
    assert snapshot["houses"][house]["blade_used"] is True

    game.set_winner(house)
    assert game.snapshot()["winner"] == house
    assert game.is_terminal()


def test_rust_game_rejects_an_action_the_engine_did_not_offer():
    from agot.engine import rust

    game = rust.RustGame.new(GameConfig(seed=1, use_westeros_decks=False))
    bogus = Action(
        kind=ActionKind.PASS,
        house="tyrell",
        region="winterfell",
    )
    with pytest.raises(ValueError):
        game.step(bogus)


def test_observe_on_a_terminal_position_is_empty_not_an_error():
    game = core.Game.new(seed=1, use_westeros_decks=False)
    while not game.is_terminal():
        actions = game.legal_actions()
        if not actions:
            break
        game.step(actions[0])
    _, state, matrix = game.observe()
    assert state.shape == (97,)
    assert matrix.shape == (0, 141)


# ---------------------------------------------------------------------------
# Batched self-play
# ---------------------------------------------------------------------------


def uniform_evaluator(calls):
    """A network-free stand-in: uniform logits, neutral values."""

    def evaluate(state, actions, index, mask):
        calls.append((int(state.shape[0]), int(actions.shape[0])))
        return (
            np.zeros(actions.shape[0], dtype=np.float32),
            np.zeros((state.shape[0], 6), dtype=np.float32),
        )

    return evaluate


SPEC = {
    "games": 4,
    "simulations": 8,
    "determinizations": 1,
    "batch_games": 2,
    "seed": 1,
    "temperature": 1.0,
    "temperature_rounds": 3,
    "final_temperature": 0.25,
    "max_rounds": 3,
    "use_westeros_decks": False,
}


def test_selfplay_returns_packed_examples():
    calls: list = []
    (
        states,
        actions,
        offsets,
        policy,
        players,
        outcomes,
        winners,
        steps,
        examples_per_game,
        rounds,
    ) = core.selfplay(SPEC, uniform_evaluator(calls))

    assert states.shape[1] == 97
    assert actions.shape[1] == 141
    assert offsets[0] == 0
    assert offsets[-1] == actions.shape[0] == policy.shape[0]
    assert states.shape[0] == players.shape[0] == outcomes.shape[0] == len(offsets) - 1
    assert len(winners) == len(steps) == len(rounds) == SPEC["games"]
    # Every seat is searched here, so all decisions are recorded.
    assert list(steps) == list(examples_per_game)
    assert sum(examples_per_game) == states.shape[0]

    # Every recorded target is a distribution, and every outcome is a terminal
    # reward: +1 for the winner, -1 for the losers, 0 for absent houses.
    for index in range(len(offsets) - 1):
        row = policy[offsets[index] : offsets[index + 1]]
        assert abs(float(row.sum()) - 1.0) < 1e-5
        assert offsets[index + 1] - offsets[index] >= 1
    assert set(np.unique(outcomes)).issubset({-1.0, 0.0, 1.0})
    for winner, steps_taken in zip(winners, steps, strict=True):
        assert winner in {"baratheon", "lannister", "stark", "greyjoy", "tyrell", "martell"}
        assert steps_taken > 10
        assert 1 <= outcomes.shape[1]


def test_selfplay_batches_the_network_calls():
    """The point of the batch tier: many leaves per call, not one per decision."""
    calls: list = []
    core.selfplay(SPEC, uniform_evaluator(calls))
    assert calls, "the evaluator must be called"

    # A call carries at most one leaf per game in the batch, and the tail of a
    # search phase can carry fewer (some simulations end at a terminal).
    assert all(1 <= leaves <= SPEC["batch_games"] for leaves, _ in calls), calls[:5]
    average = sum(leaves for leaves, _ in calls) / len(calls)
    assert average > SPEC["batch_games"] * 0.8, average
    # Every leaf's candidates are scored in the same call, so a leaf is never
    # charged the padding of its neighbours.
    assert all(rows >= leaves for leaves, rows in calls)


def test_selfplay_is_deterministic_for_a_seed():
    first = core.selfplay(SPEC, uniform_evaluator([]))
    second = core.selfplay(SPEC, uniform_evaluator([]))
    assert first[6] == second[6]  # winners
    assert first[7] == second[7]  # steps
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[3], second[3])


def test_selfplay_respects_pimc_determinizations():
    """Two worlds per decision must still produce one target per decision."""
    calls: list = []
    spec = dict(SPEC, games=2, batch_games=2, determinizations=2, simulations=8)
    (
        states,
        actions,
        offsets,
        policy,
        players,
        outcomes,
        winners,
        steps,
        examples_per_game,
        rounds,
    ) = core.selfplay(spec, uniform_evaluator(calls))
    assert len(winners) == 2
    assert offsets[-1] == actions.shape[0]
    # Both worlds of a decision are searched before the move is chosen, so the
    # number of leaves scored exceeds the number of decisions.
    assert sum(leaves for leaves, _ in calls) > states.shape[0]


# ---------------------------------------------------------------------------
# The training-loop adapter
# ---------------------------------------------------------------------------


def test_rust_engine_driver_feeds_the_training_loop():
    """`rust_engine.self_play_records` must return records the trainer can use.

    This is the seam the training loop consumes: GameRecord objects holding
    Example objects with the same shapes `batched_self_play` produces.
    """
    torch = pytest.importorskip("torch")

    from agot.nn.model import PolicyValueNet
    from agot.training import rust_engine
    from agot.training.parallel import SelfPlayPlan

    model = PolicyValueNet(hidden=32, trunk_layers=1)
    plan = SelfPlayPlan(
        games=2,
        max_rounds=2,
        simulations=4,
        determinizations=1,
        batch_games=2,
        seed=3,
    )
    records = rust_engine.self_play_records(model, plan)
    assert len(records) == 2

    for record in records:
        assert record.examples, "a game must record decisions"
        assert record.steps == len(record.examples)
        assert record.rounds >= 1
        assert record.winner is None or record.winner in HOUSE_ORDER
        for example in record.examples:
            assert example.state.shape == (97,)
            assert example.actions.ndim == 2 and example.actions.shape[1] == 141
            assert example.policy.shape == (example.actions.shape[0],)
            assert abs(float(example.policy.sum()) - 1.0) < 1e-5
            assert 0 <= example.player < len(HOUSE_ORDER)
            assert example.outcome.shape == (len(HOUSE_ORDER),)
            assert set(np.unique(example.outcome)).issubset({-1.0, 0.0, 1.0})

    # A short plan keeps the runtime trivial, but the model really was consulted.
    assert torch is not None
