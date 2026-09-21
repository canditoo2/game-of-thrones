"""End-to-end tests of the round loop and legal-action generation."""

from agot.agents import GreedyPolicy, RandomPolicy
from agot.engine.actions import ActionKind
from agot.engine.enums import Phase
from agot.engine.game import Game, GameConfig
from agot.training.selfplay import play_game


def _random_policies(config: GameConfig):
    return {
        house: RandomPolicy(seed=index)
        for index, house in enumerate(config.houses)
    }


def test_round_one_skips_the_westeros_phase() -> None:
    game = Game.new(GameConfig(seed=0))
    assert game.state.round == 1
    assert game.state.phase is Phase.PLANNING


def test_random_game_reaches_a_winner() -> None:
    config = GameConfig(seed=0)
    record = play_game(_random_policies(config), config)
    assert record.winner in config.houses
    assert record.steps > 0
    assert record.rounds <= config.max_rounds


def test_greedy_game_reaches_a_winner() -> None:
    config = GameConfig(seed=1)
    policies = {
        house: GreedyPolicy(seed=index)
        for index, house in enumerate(config.houses)
    }
    record = play_game(policies, config)
    assert record.winner in config.houses


def test_planning_never_offers_an_occupied_region() -> None:
    game = Game.new(GameConfig(seed=2))
    house = game.current_player()
    actions = game.legal_actions()
    placed = next(a for a in actions if a.kind is ActionKind.PLACE_ORDER)
    region_id = placed.region
    game.step(placed)
    if game.state.phase is Phase.PLANNING and game.current_player() == house:
        for action in game.legal_actions():
            assert action.region != region_id or action.kind is not ActionKind.PLACE_ORDER


def test_legal_actions_are_never_empty_while_not_terminal() -> None:
    game = Game.new(GameConfig(seed=3))
    steps = 0
    while not game.is_terminal() and steps < 400:
        actions = game.legal_actions()
        assert actions, f"no legal actions in phase {game.state.phase}"
        game.step(actions[0])
        steps += 1


def test_clone_is_independent() -> None:
    game = Game.new(GameConfig(seed=4))
    clone = game.clone()
    before = (clone.state.round, clone.state.phase, len(clone.state.log))
    action = game.legal_actions()[0]
    game.step(action)
    after = (clone.state.round, clone.state.phase, len(clone.state.log))
    assert before == after
    assert game.state is not clone.state
    if action.region is not None:
        assert game.state.regions[action.region].order is not None
        assert clone.state.regions[action.region].order is None
    assert clone.legal_actions()
