"""Integration tests for the Rust engine extension (``agot._core``).

Skipped when the extension is not installed, so the numpy-only CI job and a
checkout without a built wheel still run the rest of the suite.

Why the decks are off
---------------------
The two engines shuffle with different generators, so a game can only be compared
move for move when nothing is *drawn*. With ``use_westeros_decks=False`` both
engines play the identical game for a given seed and action policy, which is what
these tests assert; the card-drawing phases are covered by the Rust unit tests
and by the differential corpus.

Once the port is finished these tests become the acceptance gate for swapping the
Python engine out: the same assertions run against whichever implementation
``agot.engine`` resolves to.
"""

from __future__ import annotations

import pytest

from agot.engine.game import Game, GameConfig

core = pytest.importorskip("agot_core_ext")

SEEDS = (1, 2, 3, 7, 11, 13)

#: Deterministic action policies: first, last, and a fixed pseudo-random index.
POLICIES = {
    "first": lambda count, step: 0,
    "last": lambda count, step: count - 1,
    "mixed": lambda count, step: (step * 7 + 3) % count,
}


def play_python(seed: int, policy, decks: bool = False):
    game = Game.new(GameConfig(seed=seed, use_westeros_decks=decks))
    steps = 0
    while not game.is_terminal():
        actions = game.legal_actions()
        if not actions:
            break
        game.step(actions[policy(len(actions), steps)])
        steps += 1
    return steps, game.state.round, game.winner()


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


@pytest.mark.parametrize("policy_name", sorted(POLICIES))
@pytest.mark.parametrize("seed", SEEDS)
def test_rust_plays_the_same_game_as_python(seed: int, policy_name: str):
    """Same seed, same policy, no card draws: identical game, step for step."""
    policy = POLICIES[policy_name]
    assert play_rust(seed, policy) == play_python(seed, policy)


def test_rust_reaches_a_winner_with_the_decks_on():
    """The default configuration must at least terminate with a winner."""
    steps, rounds, winner = play_rust(1, POLICIES["first"], decks=True)
    assert steps > 100
    assert rounds >= 1
    assert winner in {"baratheon", "lannister", "stark", "greyjoy", "tyrell", "martell"}


def test_legal_actions_compare_and_hash_like_the_python_ones():
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


def test_observation_is_bit_exact_with_the_python_encoder():
    """The network input must not change when the engine does.

    A checkpoint's weights only mean anything against the encoding they were
    trained on, so the two encoders have to agree to the last bit — not
    approximately.
    """
    import numpy as np

    from agot.engine.observation import ACTION_DIM, STATE_DIM, observe

    assert core.observation_dims() == (STATE_DIM, ACTION_DIM)

    rust = core.Game.new(seed=1, use_westeros_decks=False)
    python = Game.new(GameConfig(seed=1, use_westeros_decks=False))

    for step in range(40):
        actor, state, matrix = rust.observe()
        reference = observe(python)
        assert actor == reference.actor
        assert state.shape == (STATE_DIM,)
        assert matrix.shape == (len(reference.actions), ACTION_DIM)
        assert np.array_equal(state, reference.state_features), f"step {step}: state"
        assert np.array_equal(matrix, reference.action_features), f"step {step}: actions"
        if rust.is_terminal():
            break
        action_index = (step * 5 + 1) % len(matrix)
        rust.step(rust.legal_actions()[action_index])
        python.step(python.legal_actions()[action_index])


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
