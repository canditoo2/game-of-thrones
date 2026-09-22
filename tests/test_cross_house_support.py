"""Regression tests for cross-house support (rulebook pages 17-18 and 26).

Support is the first step of combat and is declared in Iron Throne order. Any
player with a Support order adjacent to the embattled area may back either side
(or refuse), except that a player may never support an opponent against their
own units. The Support order is not consumed: it may help several battles in the
same round. A Neutral Force can never receive support, but the attacker against
one can.

Before this change the engine only ever counted a house's *own* Support orders.
"""

from __future__ import annotations

from agot.engine.actions import Action, ActionKind
from agot.engine.combat import support_from_choices, support_orders
from agot.engine.enums import ORDERS_BY_KEY, Phase, Track, UnitType
from agot.engine.game import Game, GameConfig

# Adjacent land regions used by the end-to-end tests. Highgarden is a Stronghold
# with several land neighbours, one of which ("the_reach") is also adjacent to
# King's Landing.
TARGET = "highgarden"
ORIGIN = "the_reach"
SUPPORT_REGION = "searoad_marches"
OTHER_SUPPORT_REGION = "oldtown"

NEUTRAL_TARGET = "kings_landing"
NEUTRAL_ORIGIN = "kingswood"
NEUTRAL_SUPPORT_REGION = "the_reach"


def _clear_board(game: Game) -> None:
    for region in game.state.regions.values():
        region.units.clear()
        region.routed.clear()
        region.controller = None
        region.order = None
        region.order_house = None
        region.garrison = 0
        region.influence_token = None
        region.mustering_points = 0
        region.neutral_force = 0
    for house in game.state.houses.values():
        house.orders_placed = []
        house.orders_available = [spec.key for spec in ORDERS_BY_KEY.values()]


def _stage(game: Game) -> str:
    battle = game.state.pending_battle
    return "" if battle is None else battle.get("stage", "")


def _place_support(game: Game, house: str, region: str, units, order: str) -> None:
    support = game.state.regions[region]
    support.controller = house
    support.units = dict(units)
    support.order = order
    support.order_house = house


def _setup_battle(
    attacker: str,
    defender: str,
    attacker_units: dict[UnitType, int],
    defender_units: dict[UnitType, int],
    *,
    supports: tuple[tuple[str, str, dict[UnitType, int], str], ...] = (),
) -> Game:
    """Start a battle at the support stage, with cards and Blade pre-decided.

    This mirrors what :meth:`Game._apply_march` sets up for an enemy-held area.
    """
    game = Game.new(GameConfig(seed=0))
    _clear_board(game)
    state = game.state
    state.regions[ORIGIN].controller = attacker
    state.regions[ORIGIN].units = dict(attacker_units)
    state.regions[TARGET].controller = defender
    state.regions[TARGET].units = dict(defender_units)
    for house, region, units, order in supports:
        _place_support(game, house, region, units, order)
    state.phase = Phase.COMBAT
    state.active_house = None
    state.pending_battle = {
        "region": TARGET,
        "attacker": attacker,
        "defender": defender,
        "neutral": False,
        "origin": ORIGIN,
        "attacker_order": None,
        "attacker_card": None,
        "defender_card": None,
        "attacker_card_chosen": True,
        "defender_card_chosen": True,
        "attacker_units": dict(attacker_units),
        "defender_units": dict(defender_units),
        "attacker_order_strength": 0,
        "defender_order_strength": 0,
        "blade_done": True,
        "attacker_blade": False,
        "defender_blade": False,
        "stage": "support",
        "support_queue": [region for region, _ in support_orders(state, TARGET)],
        "support_index": 0,
        "support_choices": {},
    }
    game._advance()
    return game


def _finish(game: Game) -> None:
    """Step through any remaining casualties and retreats."""
    guard = 0
    while game.state.pending_battle is not None and guard < 20:
        guard += 1
        actions = game.legal_actions()
        if not actions:  # pragma: no cover - defensive
            break
        game.step(actions[0])


def test_third_house_may_back_either_side_or_refuse() -> None:
    game = _setup_battle(
        "stark",
        "baratheon",
        {UnitType.FOOTMAN: 2},
        {UnitType.FOOTMAN: 4},
        supports=(("tyrell", SUPPORT_REGION, {UnitType.KNIGHT: 2}, "support_a"),),
    )
    assert _stage(game) == "support"
    assert game.state.active_house == "tyrell"
    sides = {action.side for action in game.legal_actions()}
    assert sides == {"attacker", "defender", "refuse"}


def test_a_house_may_not_support_against_its_own_units() -> None:
    # The attacker itself may only back the attacker; likewise for the defender.
    game = _setup_battle(
        "stark",
        "baratheon",
        {UnitType.FOOTMAN: 2},
        {UnitType.FOOTMAN: 4},
        supports=(
            ("stark", OTHER_SUPPORT_REGION, {UnitType.KNIGHT: 1}, "support_a"),
        ),
    )
    assert game.state.active_house == "stark"
    assert {action.side for action in game.legal_actions()} == {
        "attacker",
        "refuse",
    }


def test_third_house_support_flips_a_battle() -> None:
    # Without help the attacker (2) loses to the defender (4); two supporting
    # Knights add 4 and win it 6-4.
    game = _setup_battle(
        "stark",
        "baratheon",
        {UnitType.FOOTMAN: 2},
        {UnitType.FOOTMAN: 4},
        supports=(("tyrell", SUPPORT_REGION, {UnitType.KNIGHT: 2}, "support_a"),),
    )
    game.step(Action.declare_support("tyrell", SUPPORT_REGION, "attacker"))
    result = game.state.pending_battle
    if result is not None:
        assert result["result"].winner == "stark"
    _finish(game)
    assert game.state.regions[TARGET].controller == "stark"


def test_refused_support_leaves_the_attacker_short() -> None:
    game = _setup_battle(
        "stark",
        "baratheon",
        {UnitType.FOOTMAN: 2},
        {UnitType.FOOTMAN: 4},
        supports=(("tyrell", SUPPORT_REGION, {UnitType.KNIGHT: 2}, "support_a"),),
    )
    game.step(Action.declare_support("tyrell", SUPPORT_REGION, "refuse"))
    _finish(game)
    # The attack failed; the defender keeps the area.
    assert game.state.regions[TARGET].controller == "baratheon"


def test_support_order_is_not_consumed_by_a_battle() -> None:
    game = _setup_battle(
        "stark",
        "baratheon",
        {UnitType.FOOTMAN: 2},
        {UnitType.FOOTMAN: 4},
        supports=(("tyrell", SUPPORT_REGION, {UnitType.KNIGHT: 2}, "support_a"),),
    )
    game.step(Action.declare_support("tyrell", SUPPORT_REGION, "attacker"))
    # While the battle is still resolving, the supporting order is untouched.
    assert game.state.regions[SUPPORT_REGION].order == "support_a"
    assert game.state.regions[SUPPORT_REGION].order_house == "tyrell"


def test_declarations_follow_iron_throne_order() -> None:
    game = _setup_battle(
        "stark",
        "baratheon",
        {UnitType.FOOTMAN: 2},
        {UnitType.FOOTMAN: 4},
        supports=(
            ("tyrell", SUPPORT_REGION, {UnitType.KNIGHT: 1}, "support_a"),
            ("martell", OTHER_SUPPORT_REGION, {UnitType.KNIGHT: 1}, "support_a"),
        ),
    )
    throne = game.state.tracks[Track.IRON_THRONE]
    expected = min(("tyrell", "martell"), key=throne.index)
    assert game.state.active_house == expected


def test_support_from_choices_adds_all_declared_areas() -> None:
    game = Game.new(GameConfig(seed=0))
    _clear_board(game)
    state = game.state
    _place_support(game, "tyrell", SUPPORT_REGION, {UnitType.KNIGHT: 2}, "support_a")
    _place_support(
        game, "martell", OTHER_SUPPORT_REGION, {UnitType.FOOTMAN: 1}, "support_plus"
    )
    attacker_total, _a_units, defender_total, _d_units = support_from_choices(
        state,
        TARGET,
        {SUPPORT_REGION: "attacker", OTHER_SUPPORT_REGION: "defender"},
    )
    assert attacker_total == 4  # 2 Knights
    assert defender_total == 2  # 1 Footman + the starred Support order


def test_siege_support_needs_a_fortified_target() -> None:
    game = Game.new(GameConfig(seed=0))
    _clear_board(game)
    _place_support(game, "tyrell", SUPPORT_REGION, {UnitType.SIEGE: 1}, "support_a")
    state = game.state
    # Against Highgarden (a Stronghold) the Siege Engine adds its 4.
    attacker_total, _u, _d, _du = support_from_choices(
        state, TARGET, {SUPPORT_REGION: "attacker"}
    )
    assert attacker_total == 4
    # Against a plain land area, and when backing the defender, it adds nothing.
    against_open, _u, _d, _du = support_from_choices(
        state, "blackwater", {SUPPORT_REGION: "attacker"}
    )
    assert against_open == 0
    backing_defender, _u, _d, _du = support_from_choices(
        state, TARGET, {SUPPORT_REGION: "defender"}
    )
    assert backing_defender == 0


def _setup_neutral_battle(
    attacker_units: dict[UnitType, int],
    *,
    supports: tuple[tuple[str, str, dict[UnitType, int], str], ...] = (),
) -> Game:
    game = Game.new(GameConfig(seed=0))
    _clear_board(game)
    state = game.state
    state.regions[NEUTRAL_ORIGIN].controller = "stark"
    state.regions[NEUTRAL_ORIGIN].units = {}
    state.regions[NEUTRAL_TARGET].neutral_force = 5
    for house, region, units, order in supports:
        _place_support(game, house, region, units, order)
    state.phase = Phase.COMBAT
    state.active_house = None
    state.pending_battle = {
        "region": NEUTRAL_TARGET,
        "attacker": "stark",
        "defender": None,
        "neutral": True,
        "origin": NEUTRAL_ORIGIN,
        "attacker_order": None,
        "attacker_units": dict(attacker_units),
        "defender_units": {},
        "attacker_order_strength": 0,
        "defender_order_strength": 0,
        "stage": "support",
        "support_queue": [
            region for region, _ in support_orders(state, NEUTRAL_TARGET)
        ],
        "support_index": 0,
        "support_choices": {},
    }
    game._advance()
    return game


def test_neutral_force_can_never_be_supported() -> None:
    game = _setup_neutral_battle(
        {UnitType.KNIGHT: 1, UnitType.FOOTMAN: 1},
        supports=(
            (
                "tyrell",
                NEUTRAL_SUPPORT_REGION,
                {UnitType.KNIGHT: 2},
                "support_a",
            ),
        ),
    )
    assert _stage(game) == "support"
    assert game.state.active_house == "tyrell"
    # Only the attacker side, or refusing; the Neutral Force has no allies.
    assert {action.side for action in game.legal_actions()} == {
        "attacker",
        "refuse",
    }


def test_third_house_support_can_destroy_a_neutral_force() -> None:
    game = _setup_neutral_battle(
        {UnitType.KNIGHT: 1, UnitType.FOOTMAN: 1},  # 3
        supports=(
            (
                "tyrell",
                NEUTRAL_SUPPORT_REGION,
                {UnitType.KNIGHT: 2},
                "support_a",
            ),
        ),
    )
    game.step(Action.declare_support("tyrell", NEUTRAL_SUPPORT_REGION, "attacker"))
    assert game.state.regions[NEUTRAL_TARGET].neutral_force == 0
    assert game.state.regions[NEUTRAL_TARGET].controller == "stark"


def test_neutral_attack_fails_without_support() -> None:
    game = _setup_neutral_battle(
        {UnitType.KNIGHT: 1, UnitType.FOOTMAN: 1},  # 3 < 5
        supports=(
            (
                "tyrell",
                NEUTRAL_SUPPORT_REGION,
                {UnitType.KNIGHT: 2},
                "support_a",
            ),
        ),
    )
    game.step(Action.declare_support("tyrell", NEUTRAL_SUPPORT_REGION, "refuse"))
    assert game.state.regions[NEUTRAL_TARGET].neutral_force == 5
    assert game.state.regions[NEUTRAL_ORIGIN].controller == "stark"


def test_march_against_a_neutral_force_offers_support_stage() -> None:
    # A real march into King's Landing must pause for an adjacent Support order
    # rather than resolving immediately.
    game = Game.new(GameConfig(seed=0))
    _clear_board(game)
    state = game.state
    state.regions[NEUTRAL_ORIGIN].controller = "stark"
    state.regions[NEUTRAL_ORIGIN].order = "march"
    state.regions[NEUTRAL_ORIGIN].order_house = "stark"
    state.regions[NEUTRAL_ORIGIN].units = {UnitType.KNIGHT: 3}
    state.regions[NEUTRAL_TARGET].neutral_force = 5
    _place_support(
        game, "tyrell", NEUTRAL_SUPPORT_REGION, {UnitType.KNIGHT: 2}, "support_a"
    )
    actions = [
        action
        for action in game._land_march_actions(
            "stark", NEUTRAL_ORIGIN, NEUTRAL_TARGET
        )
        if action.kind is ActionKind.MARCH
    ]
    assert actions
    game._apply_march(actions[0])
    assert game.state.phase is Phase.COMBAT
    assert _stage(game) == "support"
