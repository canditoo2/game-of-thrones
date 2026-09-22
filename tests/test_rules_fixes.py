"""Regression tests for the rulebook fixes made after the rules audit.

Each test targets one rule that the engine previously got wrong or did not
model at all. They drive the engine's internal helpers directly where that
gives a deterministic setup, and use ``step``/``_advance`` where the phase
machine itself is what is being tested.
"""

from __future__ import annotations

from agot.engine.actions import Action, ActionKind
from agot.engine.combat import resolve_and_apply, resolve_battle
from agot.engine.enums import (
    ORDERS_BY_KEY,
    ActionStep,
    OrderType,
    Phase,
    Track,
    UnitType,
)
from agot.engine.game import Game, GameConfig
from agot.engine.map import load_house_cards, load_houses


def _clean_game() -> Game:
    """A game with an empty board so tests can place exactly what they need."""
    game = Game.new(GameConfig(seed=0))
    for region in game.state.regions.values():
        region.units.clear()
        region.routed.clear()
        region.controller = None
        region.order = None
        region.order_house = None
        region.garrison = 0
        region.influence_token = None
        region.mustering_points = 0
    for house in game.state.houses.values():
        house.power = 5
        house.orders_placed = []
        house.orders_available = [spec.key for spec in ORDERS_BY_KEY.values()]
    return game


def _adjacent_land_pair() -> tuple[str, str]:
    game_map = Game.new(GameConfig(seed=0)).map
    for region_id in game_map.land_region_ids:
        for neighbour in game_map.land_neighbours_of(region_id):
            return region_id, neighbour
    raise AssertionError("no adjacent land regions")  # pragma: no cover


def _first_house(game: Game) -> str:
    return game.state.tracks[Track.IRON_THRONE][0]


# -- turn order --------------------------------------------------------------


def test_action_step_cycles_one_order_per_house() -> None:
    game = _clean_game()
    state = game.state
    state.phase = Phase.ACTION
    state.action_step = ActionStep.RAID
    state.action_house_last = None

    chosen = state.tracks[Track.IRON_THRONE][:2]
    regions = list(state.regions)
    for index, house in enumerate(chosen):
        for region_id in regions[2 * index : 2 * index + 2]:
            state.regions[region_id].order = "raid_a"
            state.regions[region_id].order_house = house

    sequence = [game._next_action_house() for _ in range(4)]
    assert set(sequence) == set(chosen)
    # One order each, alternating, instead of a house resolving both in a row.
    assert sequence[0] != sequence[1]
    assert sequence[1] != sequence[2]
    assert sequence[0] == sequence[2]
    assert sequence[1] == sequence[3]


# -- consolidate power -------------------------------------------------------


def test_consolidate_on_a_sea_area_has_no_effect() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    sea = sorted(game.map.sea_region_ids)[0]
    region = state.regions[sea]
    region.controller = house
    region.units = {UnitType.SHIP: 1}
    region.order = "consolidate_a"
    region.order_house = house

    before = state.houses[house].power
    game._apply_consolidate(Action.resolve_consolidate(house, sea))
    assert state.houses[house].power == before
    assert region.order is None


# -- mustering ---------------------------------------------------------------


def test_ships_may_muster_into_a_friendly_sea_but_not_an_enemy_one() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    source, sea_id = None, None
    for land in game.map.land_region_ids:
        seas = sorted(game.map.sea_neighbours_of(land))
        if seas:
            source, sea_id = land, seas[0]
            break
    assert source is not None and sea_id is not None
    state.regions[source].controller = house
    sea = state.regions[sea_id]
    sea.controller = house
    sea.units = {UnitType.SHIP: 1}
    assert sea_id in game._ship_muster_targets(house, source)

    enemy = next(h for h in state.houses if h != house)
    sea.controller = enemy
    assert sea_id not in game._ship_muster_targets(house, source)


def test_marching_into_a_port_respects_the_three_ship_cap() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    port_id, host = next(iter(game.map.port_hosts.items()))
    sea_id = game.map.regions[port_id].sea
    assert sea_id is not None
    state.regions[host].controller = house
    state.regions[port_id].controller = house
    state.regions[port_id].units = {UnitType.SHIP: 2}
    state.regions[sea_id].controller = house
    state.regions[sea_id].units = {UnitType.SHIP: 1}

    actions = game._naval_march_actions(house, sea_id, port_id)
    assert actions
    assert all(action.units[3] == 1 for action in actions)  # one free slot

    state.regions[port_id].units = {UnitType.SHIP: 3}
    assert game._naval_march_actions(house, sea_id, port_id) == []


# -- ship transport ----------------------------------------------------------


def test_routed_ships_can_still_transport() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    sea_id = sorted(game.map.sea_region_ids)[0]
    land_neighbours = sorted(game.map.land_neighbours_of(sea_id))
    assert land_neighbours
    origin = land_neighbours[0]
    state.regions[origin].controller = house
    state.regions[origin].units = {UnitType.FOOTMAN: 1}
    sea = state.regions[sea_id]
    sea.controller = house
    sea.routed = {UnitType.SHIP: 1}

    assert game._transport_targets(house, origin)


# -- supply ------------------------------------------------------------------


def test_a_march_that_exceeds_supply_is_not_offered() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    origin, target = _adjacent_land_pair()
    state.regions[origin].controller = house
    state.regions[origin].units = {UnitType.FOOTMAN: 2}
    state.regions[target].controller = house
    state.regions[target].units = {UnitType.FOOTMAN: 2}
    state.houses[house].supply_level = 0  # limits (2, 2)

    assert game._land_march_actions(house, origin, target) == []

    state.houses[house].supply_level = 6  # limits (4, 3, 2, 2, 2)
    actions = game._land_march_actions(house, origin, target)
    assert any(action.units == (2, 0, 0, 0) for action in actions)


# -- march subsets and establishing control ----------------------------------


def test_a_march_may_leave_units_behind() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    origin, target = _adjacent_land_pair()
    region = state.regions[origin]
    region.controller = house
    region.units = {UnitType.FOOTMAN: 2}
    region.order = "march"
    region.order_house = house
    state.regions[target].controller = house

    actions = game._land_march_actions(house, origin, target)
    assert any(action.units == (1, 0, 0, 0) for action in actions)
    assert any(action.units == (2, 0, 0, 0) for action in actions)

    one = next(action for action in actions if action.units == (1, 0, 0, 0))
    game._apply_march(one)
    assert region.units.get(UnitType.FOOTMAN, 0) == 1
    assert state.regions[target].units.get(UnitType.FOOTMAN, 0) == 1


def test_a_march_may_establish_control_with_a_power_token() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    origin, target = _adjacent_land_pair()
    region = state.regions[origin]
    region.controller = house
    region.owner = None
    region.units = {UnitType.FOOTMAN: 1}
    region.order = "march"
    region.order_house = house
    state.regions[target].controller = house
    state.houses[house].power = 3

    actions = game._land_march_actions(house, origin, target)
    establish = [action for action in actions if action.establish]
    assert establish

    before = state.houses[house].power
    game._apply_march(establish[0])
    assert region.influence_token == house
    assert region.controller == house
    assert state.houses[house].power == before - 1


# -- special consolidate power ----------------------------------------------


def test_special_consolidate_may_muster_in_its_own_area() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    fort = sorted(
        region_id
        for region_id in state.fortifications
        if game.map.regions[region_id].mustering >= 1
    )[0]
    region = state.regions[fort]
    region.controller = house
    region.units = {UnitType.FOOTMAN: 1}
    region.order = "consolidate_star"
    region.order_house = house

    options = game._order_actions(house, fort, OrderType.CONSOLIDATE_POWER)
    assert any(a.kind is ActionKind.CONSOLIDATE_MUSTER for a in options)

    game._apply_consolidate_muster(Action.consolidate_muster(house, fort))
    game._advance()
    assert state.phase is Phase.MUSTERING
    assert state.return_phase is Phase.ACTION
    assert game.current_player() == house
    assert region.mustering_points == game.map.regions[fort].mustering

    # Finishing the local mustering resumes the Action Phase.
    game._apply_pass(Action.pass_(house))
    game._advance()
    assert state.return_phase is None


# -- Messenger Raven ---------------------------------------------------------


def test_the_messenger_raven_can_swap_an_order() -> None:
    game = _clean_game()
    state = game.state
    holder = state.raven_holder()
    state.phase = Phase.RAVEN
    state.active_house = holder
    state.raven_used = False
    region = next(iter(state.regions.values()))
    region.controller = holder
    region.units = {UnitType.FOOTMAN: 1}
    region.order = "raid_a"
    region.order_house = holder
    house = state.houses[holder]
    house.orders_available.remove("raid_a")
    house.orders_placed.append("raid_a")

    swap = next(
        action
        for action in game._raven_actions(holder)
        if action.kind is ActionKind.RAVEN_SWAP and action.order == "defend_1a"
    )
    game._apply_raven_swap(swap)
    assert region.order == "defend_1a"
    assert state.raven_used


def test_the_messenger_raven_can_bury_the_wildling_card() -> None:
    game = _clean_game()
    state = game.state
    holder = state.raven_holder()
    state.phase = Phase.RAVEN
    state.active_house = holder
    state.raven_used = False
    state.wildling_deck = ["a", "b", "c"]

    game._apply_raven_deck(Action.raven_deck(holder, to_bottom=True), to_bottom=True)
    assert state.wildling_deck == ["b", "c", "a"]
    assert state.raven_used


# -- Valyrian Steel Blade ----------------------------------------------------


def test_using_the_valyrian_steel_blade_is_the_holders_choice() -> None:
    game = _clean_game()
    state = game.state
    attacker = state.blade_holder()
    defender = next(h for h in state.houses if h != attacker)
    region_id = sorted(game.map.land_region_ids)[0]
    state.phase = Phase.COMBAT
    state.active_house = attacker
    state.houses[attacker].blade_used = False
    state.pending_battle = {
        "region": region_id,
        "attacker": attacker,
        "defender": defender,
        "origin": region_id,
        "attacker_card": "chosen",
        "defender_card": "chosen",
        "attacker_card_chosen": True,
        "defender_card_chosen": True,
        "attacker_units": {UnitType.FOOTMAN: 1},
        "defender_units": {UnitType.FOOTMAN: 1},
        "attacker_order_strength": 0,
        "defender_order_strength": 0,
        "blade_done": False,
        "attacker_blade": False,
        "defender_blade": False,
    }

    assert game._next_combat_chooser() == attacker
    actions = game._combat_actions(attacker)
    assert any(action.kind is ActionKind.USE_BLADE for action in actions)
    assert any(action.kind is ActionKind.PASS for action in actions)

    game._apply_use_blade(Action.use_blade(attacker))
    assert state.pending_battle["attacker_blade"] is True
    assert state.pending_battle["blade_done"] is True


# -- ports -------------------------------------------------------------------


def test_capturing_a_land_area_replaces_enemy_port_ships() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    port_id, host = next(iter(game.map.port_hosts.items()))
    enemy = next(h for h in state.houses if h != house)
    port = state.regions[port_id]
    port.controller = enemy
    port.units = {UnitType.SHIP: 2}
    state.regions[host].controller = house
    state.houses[house].supply_level = 6

    game._capture_port(house, host)
    assert port.controller == house
    assert port.units.get(UnitType.SHIP, 0) == 2


# -- wildlings ---------------------------------------------------------------


def test_wildling_icons_advance_before_card_effects_resolve() -> None:
    game = _clean_game()
    state = game.state
    state.round = 2
    state.phase = Phase.WESTEROS
    state.wildling_threat = 10
    state.westeros_decks = {
        "deck_i": ["wildlings_attack"],
        "deck_ii": ["last_days_of_summer"],
        "deck_iii": ["sea_of_storms"],
    }
    state.westeros_deck_pos = {name: 0 for name in state.westeros_decks}

    resolutions: list[int] = []
    original = game._resolve_wildlings

    def spy() -> None:
        resolutions.append(game.state.wildling_threat)
        original()

    game._resolve_wildlings = spy  # type: ignore[method-assign]
    game._run_westeros_phase()

    # The two icons push the track to 12 before the "Wildlings Attack" card
    # resolves, so the first attack happens at 12, not at the starting 10.
    assert resolutions
    assert resolutions[0] == 12


# -- end-of-game tiebreakers -------------------------------------------------


def test_tiebreakers_prefer_strongholds_over_plain_castles() -> None:
    game = _clean_game()
    state = game.state
    first, second = state.tracks[Track.IRON_THRONE][:2]
    castle = sorted(
        r for r in state.fortifications if game.map.regions[r].mustering == 1
    )[0]
    stronghold = sorted(
        r for r in state.fortifications if game.map.regions[r].mustering == 2
    )[0]
    state.regions[castle].controller = first
    state.regions[stronghold].controller = second

    assert state.castle_count(first) == state.castle_count(second) == 1
    assert game._final_ranking_key(second) > game._final_ranking_key(first)


def test_tiebreakers_use_supply_power_then_the_iron_throne() -> None:
    game = _clean_game()
    state = game.state
    first, second = state.tracks[Track.IRON_THRONE][:2]

    state.houses[first].supply_level = 5
    state.houses[second].supply_level = 4
    assert game._final_ranking_key(first) > game._final_ranking_key(second)

    state.houses[second].supply_level = 5
    state.houses[first].power = 3
    state.houses[second].power = 8
    assert game._final_ranking_key(second) > game._final_ranking_key(first)

    state.houses[first].power = state.houses[second].power = 5
    # ``first`` is earlier on the Iron Throne track, i.e. a better position.
    assert game._final_ranking_key(first) > game._final_ranking_key(second)


# -- neutral forces ----------------------------------------------------------


def test_neutral_forces_are_placed_at_setup() -> None:
    game = Game.new(GameConfig(seed=0))
    assert game.map.neutral_forces == {"kings_landing": 5, "the_eyrie": 6}
    assert game.state.regions["kings_landing"].neutral_force == 5
    assert game.state.regions["the_eyrie"].neutral_force == 6


def test_too_weak_an_army_cannot_attack_a_neutral_force() -> None:
    game = _clean_game()
    house = _first_house(game)
    origin, target = "kingswood", "kings_landing"
    game.state.regions[target].neutral_force = 5
    region = game.state.regions[origin]
    region.controller = house
    region.order = "march"
    region.order_house = house

    # Three Footmen have Combat Strength 3, below the token's 5.
    region.units = {UnitType.FOOTMAN: 3}
    assert game._land_march_actions(house, origin, target) == []

    # Three Knights have Combat Strength 6 and may attack.
    region.units = {UnitType.KNIGHT: 3}
    assert game._land_march_actions(house, origin, target)


def test_marching_onto_a_neutral_force_takes_the_area() -> None:
    game = _clean_game()
    house = _first_house(game)
    origin, target = "kingswood", "kings_landing"
    game.state.regions[target].neutral_force = 5
    region = game.state.regions[origin]
    region.controller = house
    region.order = "march"
    region.order_house = house
    region.units = {UnitType.KNIGHT: 3}

    actions = game._land_march_actions(house, origin, target)
    assert actions
    game._apply_march(actions[0])

    target_state = game.state.regions[target]
    assert target_state.neutral_force == 0
    assert target_state.controller == house
    assert target_state.units.get(UnitType.KNIGHT, 0) == 3


def test_a_neutral_force_blocks_a_retreat() -> None:
    from agot.engine.combat import _enemy_presence

    game = _clean_game()
    house = _first_house(game)
    game.state.regions["kings_landing"].neutral_force = 5
    assert _enemy_presence(game.state, "kings_landing", house)


# -- wildling attacks --------------------------------------------------------


def _begin_wildling_attack(game: Game, *, card: str, strength: int) -> None:
    """Drop a game into the wildling bidding step with one known card."""
    state = game.state
    state.wildling_deck = [card]
    state.wildling_discard = []
    state.wildling_threat = strength
    state.phase = Phase.WESTEROS
    game._resolve_wildlings()
    game._advance()


def test_wildling_attack_bids_in_iron_throne_order() -> None:
    game = _clean_game()
    _begin_wildling_attack(game, card="silence", strength=6)
    state = game.state
    order = state.tracks[Track.IRON_THRONE]

    seen: list[str | None] = []
    while state.phase is Phase.WILDLING_ATTACK:
        seen.append(state.active_house)
        game.step(Action.bid(state.active_house, 1))

    assert seen == order
    # Six bids of 1 meet the strength of 6: the Night's Watch holds.
    assert state.wildling_threat == 0
    # Every house spent the Power it bid, win or lose.
    assert all(state.houses[house].power == 4 for house in order)
    assert state.wildling_discard == ["silence"]


def test_wildlings_win_moves_the_track_back_two() -> None:
    game = _clean_game()
    _begin_wildling_attack(game, card="silence", strength=8)
    state = game.state
    while state.phase is Phase.WILDLING_ATTACK:
        game.step(Action.bid(state.active_house, 0))
    assert state.wildling_threat == 6


def test_wildling_lowest_bidder_loses_units() -> None:
    game = _clean_game()
    state = game.state
    house = _first_house(game)
    region = state.regions["kingswood"]
    region.controller = house
    region.units = {UnitType.FOOTMAN: 4}

    _begin_wildling_attack(game, card="mammoth_riders", strength=8)
    # Everyone bids zero. The wildlings win and the tied lowest bidders are
    # broken by Iron Throne position, so ``house`` is the lowest bidder.
    while state.phase is Phase.WILDLING_ATTACK:
        game.step(Action.bid(state.active_house, 0))

    assert state.regions["kingswood"].units.get(UnitType.FOOTMAN, 0) == 1


def test_preemptive_raid_starts_a_second_attack() -> None:
    game = _clean_game()
    _begin_wildling_attack(game, card="preemptive_raid", strength=4)
    state = game.state

    second_seen = False
    guard = 0
    while state.phase is Phase.WILDLING_ATTACK and guard < 20:
        attack = state.wildling_attack or {}
        if attack.get("strength") == 6 and attack.get("excluded"):
            second_seen = True
            break
        game.step(Action.bid(state.active_house, 1))
        guard += 1

    assert second_seen


def test_westeros_cards_resolve_after_an_interrupting_wildling_attack() -> None:
    game = _clean_game()
    state = game.state
    state.round = 2
    state.phase = Phase.WESTEROS
    state.pending_westeros = None
    state.wildling_threat = 12
    state.order_restrictions = []
    state.westeros_decks = {
        "deck_i": ["wildlings_attack"],
        "deck_ii": ["last_days_of_summer"],
        "deck_iii": ["sea_of_storms"],
    }
    state.westeros_deck_pos = {name: 0 for name in state.westeros_decks}
    state.wildling_deck = ["silence"]
    state.wildling_discard = []

    game._run_westeros_phase()
    # The two Wildling icons pushed the track to 12, so the attack interrupts
    # the Westeros phase before the cards resolve.
    assert state.phase is Phase.WILDLING_ATTACK

    guard = 0
    while state.phase is Phase.WILDLING_ATTACK and guard < 50:
        game.step(Action.bid(state.active_house, 5))
        guard += 1

    # The queue drains instead of stalling, and the round is ready to continue.
    assert state.pending_westeros is None
    assert state.phase not in (Phase.WESTEROS, Phase.WILDLING_ATTACK)


# -- House Stark starting supply ---------------------------------------------


def test_stark_starts_with_one_supply() -> None:
    data = load_houses()
    assert data.houses["stark"]["initial_supply"] == 1
    assert all(
        payload["initial_supply"] == 2
        for house_id, payload in data.houses.items()
        if house_id != "stark"
    )
    game = Game.new(GameConfig(seed=0))
    assert game.state.houses["stark"].supply_level == 1
    assert game.state.houses["lannister"].supply_level == 2


# -- holders'-choice Westeros cards ------------------------------------------


def _westeros_choice_game(cards: dict[str, list[str]]) -> Game:
    game = _clean_game()
    state = game.state
    state.round = 2
    state.phase = Phase.WESTEROS
    state.pending_westeros = None
    state.order_restrictions = []
    state.westeros_decks = {name: list(deck) for name, deck in cards.items()}
    state.westeros_deck_pos = {name: 0 for name in state.westeros_decks}
    return game


def test_throne_of_blades_holder_chooses_mustering() -> None:
    game = _westeros_choice_game(
        {
            "deck_i": ["throne_of_blades"],
            "deck_ii": ["game_of_thrones"],
            "deck_iii": ["sea_of_storms"],
        }
    )
    state = game.state
    holder = state.tracks[Track.IRON_THRONE][0]
    fort = sorted(state.fortifications)[0]
    state.regions[fort].controller = holder
    state.regions[fort].mustering_points = 0

    game._run_westeros_phase()
    game._advance()
    assert state.phase is Phase.WESTEROS_CHOICE
    assert state.active_house == holder
    assert {action.card for action in game.legal_actions()} == {
        "mustering",
        "supply",
        "none",
    }

    game.step(Action.westeros_choice(holder, "mustering"))
    assert state.regions[fort].mustering_points == game.map.regions[fort].mustering
    assert state.westeros_choice is None
    assert state.phase in (Phase.MUSTERING, Phase.PLANNING)


def test_dark_wings_holder_chooses_clash_of_kings_and_the_queue_resumes() -> None:
    game = _westeros_choice_game(
        {
            "deck_i": ["supply"],
            "deck_ii": ["dark_wings_dark_words"],
            "deck_iii": ["sea_of_storms"],
        }
    )
    state = game.state
    holder = state.raven_holder()

    game._run_westeros_phase()
    game._advance()
    assert state.phase is Phase.WESTEROS_CHOICE
    assert state.active_house == holder

    game.step(Action.westeros_choice(holder, "clash_of_kings"))
    assert state.phase is Phase.BIDDING
    assert state.bid_tracks == ["iron_throne", "fiefdoms", "kings_court"]

    guard = 0
    while state.phase is Phase.BIDDING and guard < 100:
        game.step(Action.bid(state.active_house, 0))
        guard += 1

    # The queued "Sea of Storms" still resolves after the bidding.
    assert "raid" in state.order_restrictions
    assert state.pending_westeros is None


def test_put_to_the_sword_holder_chooses_a_restriction() -> None:
    game = _clean_game()
    state = game.state
    holder = state.blade_holder()
    state.phase = Phase.WESTEROS_CHOICE
    state.pending_westeros = []
    state.order_restrictions = []
    state.westeros_choice = {
        "card": "put_to_the_sword",
        "holder": holder,
        "options": ["rains_of_autumn", "storm_of_swords", "none"],
    }

    game._advance()
    assert state.active_house == holder
    assert {action.card for action in game.legal_actions()} == {
        "rains_of_autumn",
        "storm_of_swords",
        "none",
    }

    game.step(Action.westeros_choice(holder, "rains_of_autumn"))
    assert "march_plus" in state.order_restrictions
    assert state.westeros_choice is None


def test_a_holders_choice_can_be_no_effect() -> None:
    game = _clean_game()
    state = game.state
    holder = state.blade_holder()
    state.phase = Phase.WESTEROS_CHOICE
    state.pending_westeros = []
    state.order_restrictions = []
    state.westeros_choice = {
        "card": "put_to_the_sword",
        "holder": holder,
        "options": ["rains_of_autumn", "storm_of_swords", "none"],
    }
    game.step(Action.westeros_choice(holder, "none"))
    assert state.order_restrictions == []
    assert state.westeros_choice is None


# -- House card data ---------------------------------------------------------


def test_every_house_card_ability_is_implemented() -> None:
    from agot.engine import house_cards

    declared = {
        card["ability"]
        for cards in load_house_cards().cards.values()
        for card in cards
        if card.get("ability", "none") != "none"
    }
    assert declared <= house_cards.IMPLEMENTED_ABILITIES


def test_martell_deck_matches_the_reference_sheet() -> None:
    deck = {card["id"]: card for card in load_house_cards().deck("martell")}
    assert deck["the_red_viper"]["strength"] == 4
    assert deck["the_red_viper"]["swords"] == 2
    assert deck["the_red_viper"]["forts"] == 1
    assert deck["areo_hotah"]["strength"] == 3
    assert deck["obara_sand"]["strength"] == 2
    assert deck["darkstar"]["strength"] == 2
    assert deck["nymeria_sand"]["strength"] == 1
    assert deck["arianne_martell"]["strength"] == 1
    assert deck["doran_martell"]["strength"] == 0
    assert "ellaria_sand" not in deck


# -- House card abilities in battle ------------------------------------------


def _battle_pair(game: Game) -> tuple[str, str]:
    return _adjacent_land_pair()


def test_stannis_gains_strength_when_the_opponent_holds_the_throne() -> None:
    game = _clean_game()
    state = game.state
    _, target = _battle_pair(game)
    state.tracks[Track.IRON_THRONE] = [
        "lannister",
        "baratheon",
        "stark",
        "greyjoy",
        "tyrell",
        "martell",
    ]
    result = resolve_battle(
        state,
        target,
        "baratheon",
        "lannister",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        "stannis_baratheon",
        None,
    )
    assert result.attacker.card_strength == 5  # printed 4, +1

    state.tracks[Track.IRON_THRONE] = [
        "baratheon",
        "lannister",
        "stark",
        "greyjoy",
        "tyrell",
        "martell",
    ]
    result = resolve_battle(
        state,
        target,
        "baratheon",
        "lannister",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        "stannis_baratheon",
        None,
    )
    assert result.attacker.card_strength == 4


def test_davos_needs_stannis_in_the_discard_pile() -> None:
    game = _clean_game()
    state = game.state
    _, target = _battle_pair(game)
    state.houses["baratheon"].house_cards_used = ["stannis_baratheon"]
    result = resolve_battle(
        state,
        target,
        "baratheon",
        "stark",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        "davos_seaworth",
        None,
    )
    assert result.attacker.card_strength == 3  # printed 2, +1
    assert result.attacker.swords == 1


def test_balon_zeroes_the_opponents_printed_strength() -> None:
    game = _clean_game()
    _, target = _battle_pair(game)
    result = resolve_battle(
        game.state,
        target,
        "greyjoy",
        "stark",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        "balon_greyjoy",
        "eddard_stark",
    )
    assert result.defender.card_strength == 0


def test_asha_gains_icons_when_not_supported() -> None:
    game = _clean_game()
    _, target = _battle_pair(game)
    result = resolve_battle(
        game.state,
        target,
        "greyjoy",
        "stark",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        "asha_greyjoy",
        None,
    )
    assert result.attacker.swords == 2
    assert result.attacker.forts == 1


def test_nymeria_switches_icon_by_role() -> None:
    game = _clean_game()
    _, target = _battle_pair(game)
    attacking = resolve_battle(
        game.state,
        target,
        "martell",
        "stark",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        "nymeria_sand",
        None,
    )
    assert attacking.attacker.swords == 1
    assert attacking.attacker.forts == 0

    defending = resolve_battle(
        game.state,
        target,
        "stark",
        "martell",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        None,
        "nymeria_sand",
    )
    assert defending.defender.forts == 1
    assert defending.defender.swords == 0


def test_theon_needs_a_fortification_when_defending() -> None:
    game = _clean_game()
    state = game.state
    fort = sorted(state.fortifications)[0]
    result = resolve_battle(
        state,
        fort,
        "stark",
        "greyjoy",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        None,
        "theon_greyjoy",
    )
    assert result.defender.card_strength == 3  # printed 2, +1
    assert result.defender.swords == 1


def test_kevan_doubles_attacking_footmen() -> None:
    game = _clean_game()
    _, target = _battle_pair(game)
    result = resolve_battle(
        game.state,
        target,
        "lannister",
        "stark",
        {UnitType.FOOTMAN: 3},
        {UnitType.FOOTMAN: 1},
        "kevan_lannister",
        None,
    )
    assert result.attacker.units_strength == 6


def test_victarion_doubles_attacking_ships() -> None:
    game = _clean_game()
    sea = sorted(game.map.sea_region_ids)[0]
    result = resolve_battle(
        game.state,
        sea,
        "greyjoy",
        "stark",
        {UnitType.SHIP: 3},
        {UnitType.SHIP: 1},
        "victarion_greyjoy",
        None,
    )
    assert result.attacker.units_strength == 6


def test_catelyn_doubles_a_defend_order() -> None:
    game = _clean_game()
    state = game.state
    _, target = _battle_pair(game)
    state.regions[target].order = "defend_2"
    state.regions[target].order_house = "stark"
    result = resolve_battle(
        state,
        target,
        "lannister",
        "stark",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 1},
        None,
        "catelyn_stark",
        defender_order_strength=2,
    )
    assert result.defender.order_strength == 4


def test_blackfish_ignores_casualties() -> None:
    game = _clean_game()
    _, target = _battle_pair(game)
    result = resolve_battle(
        game.state,
        target,
        "lannister",
        "stark",
        {UnitType.FOOTMAN: 3},
        {UnitType.FOOTMAN: 1},
        "gregor_clegane",
        "blackfish",
    )
    assert result.winner == "lannister"
    assert result.casualties == 0


def test_mace_destroys_an_opposing_footman() -> None:
    game = _clean_game()
    _, target = _battle_pair(game)
    result = resolve_battle(
        game.state,
        target,
        "tyrell",
        "stark",
        {UnitType.FOOTMAN: 1},
        {UnitType.FOOTMAN: 2},
        "mace_tyrell",
        None,
    )
    assert result.defender.units_strength == 1
    assert result.context is not None
    assert result.context.defender_units[UnitType.FOOTMAN] == 1


def test_salladhor_reduces_non_baratheon_ships() -> None:
    game = _clean_game()
    state = game.state
    sea = None
    neighbour = None
    for candidate in sorted(game.map.sea_region_ids):
        seas = sorted(game.map.sea_neighbours_of(candidate))
        if seas:
            sea, neighbour = candidate, seas[0]
            break
    assert sea is not None and neighbour is not None
    state.regions[neighbour].controller = "baratheon"
    state.regions[neighbour].units = {UnitType.SHIP: 1}
    state.regions[neighbour].order = "support_a"
    state.regions[neighbour].order_house = "baratheon"

    result = resolve_battle(
        state,
        sea,
        "baratheon",
        "lannister",
        {UnitType.SHIP: 1},
        {UnitType.SHIP: 2},
        "salladhor_saan",
        None,
    )
    assert result.defender.units_strength == 0
    assert result.attacker.units_strength == 1


def test_tywin_gains_two_power_when_he_wins() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "lannister"
    state.regions[origin].units = {}
    before = state.houses["lannister"].power
    result = resolve_and_apply(
        state,
        target,
        "lannister",
        "stark",
        {UnitType.FOOTMAN: 3},
        {UnitType.FOOTMAN: 1},
        "tywin_lannister",
        None,
        attacker_origin=origin,
    )
    assert result.winner == "lannister"
    assert state.houses["lannister"].power == before + 2


def test_roose_returns_the_discard_pile_on_a_loss() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "lannister"
    state.regions[origin].units = {}
    stark = state.houses["stark"]
    stark.house_cards_available = ["robb_stark", "eddard_stark"]
    stark.house_cards_used = ["greatjon_umber", "roose_bolton"]

    resolve_and_apply(
        state,
        target,
        "lannister",
        "stark",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        "gregor_clegane",
        "roose_bolton",
        attacker_origin=origin,
    )
    assert set(stark.house_cards_available) == {
        "robb_stark",
        "eddard_stark",
        "greatjon_umber",
        "roose_bolton",
    }
    assert stark.house_cards_used == []


def test_arianne_blocks_the_conquest() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "lannister"
    state.regions[origin].units = {}
    result = resolve_and_apply(
        state,
        target,
        "lannister",
        "martell",
        {UnitType.FOOTMAN: 3},
        {UnitType.FOOTMAN: 1},
        None,
        "arianne_martell",
        attacker_origin=origin,
    )
    assert result.winner == "lannister"
    # The attacker may not occupy the area; it returns to where it marched from.
    assert state.regions[target].controller != "lannister"
    assert state.regions[origin].units.get(UnitType.FOOTMAN, 0) == 3


def test_loras_replaces_the_march_order_in_the_conquered_area() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "tyrell"
    state.regions[origin].units = {}
    resolve_and_apply(
        state,
        target,
        "tyrell",
        "stark",
        {UnitType.FOOTMAN: 3},
        {UnitType.FOOTMAN: 1},
        "loras_tyrell",
        None,
        attacker_origin=origin,
        attacker_order="march",
    )
    assert state.regions[target].order == "march"
    assert state.regions[target].order_house == "tyrell"


def test_cersei_removes_an_opponent_order() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "lannister"
    state.regions[origin].units = {}
    victim = next(region for region in sorted(state.regions) if region != target)
    state.regions[victim].order = "march"
    state.regions[victim].order_house = "stark"
    resolve_and_apply(
        state,
        target,
        "lannister",
        "stark",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        "cersei_lannister",
        None,
        attacker_origin=origin,
    )
    assert state.regions[victim].order is None


def test_queen_of_thorns_removes_an_adjacent_order() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "tyrell"
    state.regions[origin].units = {}
    adjacent = next(
        neighbour
        for neighbour in sorted(game.map.neighbours_of(target))
        if neighbour != origin
    )
    state.regions[adjacent].order = "support_a"
    state.regions[adjacent].order_house = "stark"
    resolve_and_apply(
        state,
        target,
        "tyrell",
        "stark",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        "queen_of_thorns",
        None,
        attacker_origin=origin,
    )
    assert state.regions[adjacent].order is None


def test_doran_moves_the_opponent_to_the_bottom_of_a_track() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "martell"
    state.regions[origin].units = {}
    state.tracks[Track.FIEFDOMS] = [
        "stark",
        "baratheon",
        "lannister",
        "greyjoy",
        "tyrell",
        "martell",
    ]
    resolve_and_apply(
        state,
        target,
        "martell",
        "stark",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        "doran_martell",
        None,
        attacker_origin=origin,
    )
    assert state.tracks[Track.FIEFDOMS][-1] == "stark"


def test_tyrion_returns_the_opponents_card_and_forces_a_swap() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "lannister"
    state.regions[origin].units = {}
    stark = state.houses["stark"]
    stark.house_cards_available = ["catelyn_stark"]
    stark.house_cards_used = ["eddard_stark"]

    result = resolve_and_apply(
        state,
        target,
        "lannister",
        "stark",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        "tyrion_lannister",
        "eddard_stark",
        attacker_origin=origin,
    )
    assert result.defender.card_strength == 0  # forced onto Catelyn (0 strength)
    assert "eddard_stark" in stark.house_cards_available
    assert "catelyn_stark" in stark.house_cards_used


def test_tyrion_leaves_an_empty_handed_opponent_with_no_card() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "lannister"
    state.regions[origin].units = {}
    stark = state.houses["stark"]
    stark.house_cards_available = []
    stark.house_cards_used = ["eddard_stark"]

    result = resolve_and_apply(
        state,
        target,
        "lannister",
        "stark",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        "tyrion_lannister",
        "eddard_stark",
        attacker_origin=origin,
    )
    assert result.defender.card_strength == 0
    assert stark.house_cards_available == ["eddard_stark"]
    assert stark.house_cards_used == []


def test_aeron_pays_power_to_swap_itself_out() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "greyjoy"
    state.regions[origin].units = {}
    greyjoy = state.houses["greyjoy"]
    greyjoy.power = 5
    greyjoy.house_cards_available = ["balon_greyjoy"]
    greyjoy.house_cards_used = ["aeron_damphair"]

    result = resolve_and_apply(
        state,
        target,
        "greyjoy",
        "stark",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        "aeron_damphair",
        None,
        attacker_origin=origin,
    )
    assert greyjoy.power == 3
    assert result.attacker.card_strength == 2  # Balon Greyjoy
    assert "balon_greyjoy" in greyjoy.house_cards_used


def test_patchface_discards_the_opponents_best_card() -> None:
    game = _clean_game()
    state = game.state
    origin, target = _battle_pair(game)
    state.regions[origin].controller = "baratheon"
    state.regions[origin].units = {}
    stark = state.houses["stark"]
    stark.house_cards_available = ["greatjon_umber", "robb_stark"]
    stark.house_cards_used = ["eddard_stark"]

    resolve_and_apply(
        state,
        target,
        "baratheon",
        "stark",
        {UnitType.FOOTMAN: 5},
        {UnitType.FOOTMAN: 1},
        "patchface",
        "eddard_stark",
        attacker_origin=origin,
    )
    assert "robb_stark" in stark.house_cards_used
    assert "robb_stark" not in stark.house_cards_available

