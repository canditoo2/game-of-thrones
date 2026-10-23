//! Rule-fidelity regressions, ported from `tests/test_rules_fixes.py`.
//!
//! The Python file holds one regression per audited rule fix — it is the rulebook
//! audit's safety net, and it moves here because Rust is becoming the engine.
//! Every test below is named after its Python original so the audit trail in
//! `docs/rulebook-audit.md` can be followed across the port.
//!
//! They live inside the crate rather than in `tests/` because several need
//! `pub(crate)` internals (the action generators, the apply steps), which an
//! integration test cannot reach. Positions are built directly on the state: a
//! rule fix is about one mechanism, and going through the phase machine would
//! make each regression depend on every phase before it.
//!
//! Status: a first batch. The remaining tests are tracked in the roadmap.

#![cfg(test)]

use std::sync::Arc;

use smallvec::SmallVec;

use crate::action::{Action, ActionKind};
use crate::combat::{resolve_and_apply, resolve_battle, BattleRequest};
use crate::enums::{
    ActionStep, CardId, HouseId, OrderKey, Phase, RegionId, Terrain, Track, UnitType,
};
use crate::game::Game;
use crate::map::DataSet;
use crate::setup::GameConfig;
use crate::state::{BattleStage, DeckState, PendingBattle, State, WesterosChoice, NONE_OPTION};

fn data() -> Arc<DataSet> {
    Arc::new(DataSet::load_default().expect("bundled data must load"))
}

fn game(seed: u64) -> Game {
    Game::new(data(), GameConfig::with_seed(seed)).expect("game must start")
}

/// A fresh game whose board the caller rewrites: the equivalent of the Python
/// tests assigning `game.state.regions[...]` directly.
fn blank(seed: u64) -> (Game, Arc<DataSet>) {
    let game = game(seed);
    let data = Arc::clone(&game.data);
    (game, data)
}

fn region_of(data: &DataSet, id: &str) -> RegionId {
    data.map
        .lookup(id)
        .unwrap_or_else(|| panic!("unknown region {id}"))
}

/// Puts a march order on an area and returns it, so the generators offer that
/// area's orders.
fn with_march_order(game: &mut Game, region: RegionId, house: HouseId) {
    let order = OrderKey::from_name("march").expect("march exists");
    {
        let state = game.state.region_mut(region);
        state.controller = Some(house);
        state.order = Some(order);
        state.order_house = Some(house);
    }
    if let Some(house_state) = game.state.house_mut(house) {
        house_state.take_order(order);
    }
}

/// Python's `_clean_game`: the real opening position with every region emptied
/// of units, orders and tokens, and every house reset to 5 Power.
///
/// The neutral forces and the printed owners are deliberately left in place,
/// exactly as the Python helper does.
fn clean(seed: u64) -> (Game, Arc<DataSet>) {
    let mut game = game(seed);
    let data = Arc::clone(&game.data);
    let ids: Vec<RegionId> = game.state.region_ids().collect();
    for id in ids {
        let region = game.state.region_mut(id);
        region.units = [0; 4];
        region.routed = [0; 4];
        region.controller = None;
        region.order = None;
        region.order_house = None;
        region.garrison = 0;
        region.influence_token = None;
        region.mustering_points = 0;
    }
    for house in HouseId::ALL {
        if let Some(state) = game.state.house_mut(house) {
            state.power = 5;
            state.orders_placed.clear();
            state.orders_available = OrderKey::ALL.into_iter().collect();
        }
    }
    (game, data)
}

/// Python's `_first_house`: the Iron Throne holder.
fn first_house(game: &Game) -> HouseId {
    game.state
        .first_player()
        .expect("every game has an Iron Throne order")
}

/// Python's `_adjacent_land_pair`: the first land region in board order and its
/// first land neighbour.
fn adjacent_land_pair(data: &DataSet) -> (RegionId, RegionId) {
    for index in 0..data.map.region_count() as u16 {
        let region = RegionId(index);
        if !data.map.region(region).is_land() {
            continue;
        }
        if let Some(neighbour) = data.map.land_neighbours_of(region).first().copied() {
            return (region, neighbour);
        }
    }
    panic!("no adjacent land regions");
}

/// A house's card by its data id.
fn card(data: &DataSet, house: HouseId, id: &str) -> CardId {
    data.house_cards
        .card_index(house, id)
        .unwrap_or_else(|| panic!("{house} has no card {id}"))
}

/// House by name, for the tests that name a house in the Python original.
fn house(id: &str) -> HouseId {
    HouseId::from_name(id).unwrap_or_else(|| panic!("unknown house {id}"))
}

/// Region ids sorted the way Python sorts strings: by map id.
fn sorted_regions(data: &DataSet, mut ids: Vec<RegionId>) -> Vec<RegionId> {
    ids.sort_unstable_by_key(|id| data.map.id_of(*id));
    ids
}

/// The first sea region in Python's `sorted(sea_region_ids)` order.
fn first_sea(data: &DataSet) -> RegionId {
    (0..data.map.region_count() as u16)
        .map(RegionId)
        .filter(|id| data.map.region(*id).is_sea())
        .min_by_key(|id| data.map.id_of(*id))
        .expect("the board has seas")
}

/// Drop a game into the Wildling bidding step with one known card.
fn begin_wildling_attack(game: &mut Game, data: &DataSet, card_id: &str, strength: u8) {
    let index = data
        .wildlings
        .card_index(card_id)
        .unwrap_or_else(|| panic!("unknown wildling card {card_id}"));
    game.state.wildling_deck = DeckState::new([index]);
    game.state.wildling_discard.clear();
    game.state.wildling_threat = strength;
    game.state.phase = Phase::Westeros;
    game.resolve_wildlings()
        .expect("the wildlings must be able to attack");
    game.advance().expect("the bidding step must settle");
}

/// Python's `_westeros_choice_game`: round 2 with the three decks replaced.
fn westeros_choice_game(cards: [&[&str]; 3]) -> (Game, Arc<DataSet>) {
    let (mut game, data) = clean(0);
    game.state.round = 2;
    game.state.phase = Phase::Westeros;
    game.state.pending_westeros = None;
    game.state.order_restrictions.clear();
    for (deck, ids) in game.state.westeros_decks.iter_mut().zip(cards) {
        let resolved: Vec<u16> = ids
            .iter()
            .map(|id| {
                data.westeros
                    .card_index(id)
                    .unwrap_or_else(|| panic!("unknown Westeros card {id}"))
            })
            .collect();
        *deck = DeckState::new(resolved);
    }
    (game, data)
}

/// The card ids the engine offers for a holders' choice, as `CardId`s.
fn choice_cards(actions: &[Action]) -> Vec<CardId> {
    actions.iter().filter_map(|action| action.card).collect()
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Python: `test_neutral_forces_are_placed_at_setup`.
#[test]
fn neutral_forces_are_placed_at_setup() {
    let game = game(1);
    let kings_landing = region_of(&game.data, "kings_landing");
    let the_eyrie = region_of(&game.data, "the_eyrie");
    assert_eq!(game.state.region(kings_landing).neutral_force, 5);
    assert_eq!(game.state.region(the_eyrie).neutral_force, 6);
    // A Neutral Force area is not a home area and holds no garrison.
    assert_eq!(game.state.region(kings_landing).garrison, 0);
    assert_eq!(game.state.region(kings_landing).controller, None);
}

// ---------------------------------------------------------------------------
// Marches
// ---------------------------------------------------------------------------

/// Python: `test_a_march_may_leave_units_behind`.
///
/// Every subset of the eligible units is offered, so a march can split a force.
#[test]
fn a_march_may_leave_units_behind() {
    let (mut game, data) = blank(1);
    let winterfell = region_of(&data, "winterfell");
    let house = HouseId(2);
    {
        let region = game.state.region_mut(winterfell);
        region.controller = Some(house);
        region.owner = Some(house);
        region.units = [2, 1, 0, 0]; // two footmen, one knight
    }
    with_march_order(&mut game, winterfell, house);

    let actions = game.march_actions(house, winterfell);
    let subsets: Vec<[u8; 4]> = actions.iter().filter_map(|action| action.units).collect();
    // 3 non-empty subsets of (2 footmen, 1 knight) = 2 * 2 - 1.
    for expected in [
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [2, 1, 0, 0],
    ] {
        assert!(
            subsets.contains(&expected),
            "expected a march with {expected:?}, got {subsets:?}"
        );
    }
    assert!(
        !subsets.contains(&[0, 0, 0, 0]),
        "an empty march is not a move"
    );
}

/// Python: `test_a_march_that_exceeds_supply_is_not_offered`.
#[test]
fn a_march_that_exceeds_supply_is_not_offered() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let winterfell = region_of(&data, "winterfell");
    let white_harbor = region_of(&data, "white_harbor");
    // Supply level 0 allows armies of at most 2 and 2.
    if let Some(state) = game.state.house_mut(house) {
        state.supply_level = 0;
    }
    {
        let region = game.state.region_mut(winterfell);
        region.controller = Some(house);
        region.units = [3, 0, 0, 0];
    }
    {
        let region = game.state.region_mut(white_harbor);
        region.controller = Some(house);
        region.units = [1, 0, 0, 0];
    }
    with_march_order(&mut game, winterfell, house);

    // Moving one footman leaves 2 + 2, which is legal; moving two would make 1 + 3.
    let actions = game.march_actions(house, winterfell);
    let moves: Vec<[u8; 4]> = actions
        .iter()
        .filter(|action| action.target == Some(white_harbor))
        .filter_map(|action| action.units)
        .collect();
    assert!(
        moves.contains(&[1, 0, 0, 0]),
        "a legal move must be offered: {moves:?}"
    );
    assert!(
        !moves.contains(&[2, 0, 0, 0]),
        "a move that would break Supply must not be offered: {moves:?}"
    );
}

/// Python: `test_a_march_may_establish_control_with_a_power_token`.
#[test]
fn a_march_may_establish_control_with_a_power_token() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let winterfell = region_of(&data, "winterfell");
    let white_harbor = region_of(&data, "white_harbor");
    {
        // Not the house's own home area, so a Power token is what keeps it.
        let region = game.state.region_mut(winterfell);
        region.controller = Some(house);
        region.owner = None;
        region.units = [1, 0, 0, 0];
    }
    with_march_order(&mut game, winterfell, house);

    let actions = game.march_actions(house, winterfell);
    let establishing: Vec<&Action> = actions
        .iter()
        .filter(|action| action.target == Some(white_harbor) && action.establish)
        .collect();
    assert_eq!(
        establishing.len(),
        1,
        "vacating the area must offer establishing control"
    );

    // Without Power there is nothing to spend.
    if let Some(state) = game.state.house_mut(house) {
        state.power = 0;
    }
    let actions = game.march_actions(house, winterfell);
    assert!(
        !actions
            .iter()
            .any(|action| action.target == Some(white_harbor) && action.establish),
        "no Power means no token"
    );
}

/// Python: `test_routed_ships_can_still_transport`.
#[test]
fn routed_ships_can_still_transport() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let winterfell = region_of(&data, "winterfell");
    let narrow_sea = region_of(&data, "the_narrow_sea");
    {
        let region = game.state.region_mut(winterfell);
        region.controller = Some(house);
        region.owner = Some(house);
        region.units = [1, 0, 0, 0];
    }
    {
        // A *routed* ship still carries the army.
        let region = game.state.region_mut(narrow_sea);
        region.controller = Some(house);
        region.routed = [0, 0, 0, 1];
    }
    with_march_order(&mut game, winterfell, house);

    let actions = game.march_actions(house, winterfell);
    let transported = actions.iter().any(|action| {
        action
            .target
            .map(|target| data.map.region(target).is_land())
            == Some(true)
            && action.units != Some([0, 0, 0, 0])
    });
    assert!(
        transported,
        "a routed ship must still transport: {actions:?}"
    );
}

/// Python: `test_marching_into_a_port_respects_the_three_ship_cap`.
#[test]
fn marching_into_a_port_respects_the_three_ship_cap() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    // Enough Supply that merging the two fleets is legal: the test is about the
    // Port's three-Ship cap, not about Supply.
    if let Some(state) = game.state.house_mut(house) {
        state.supply_level = 4;
    }
    let winterfell_port = region_of(&data, "winterfell_port");
    // The sea the port opens onto, rather than a guessed neighbour.
    let bay = data
        .map
        .region(winterfell_port)
        .sea
        .expect("a port names its sea");
    {
        let region = game.state.region_mut(winterfell_port);
        region.controller = Some(house);
        region.units = [0, 0, 0, 2]; // two ships already there
    }
    {
        let region = game.state.region_mut(bay);
        region.controller = Some(house);
        region.units = [0, 0, 0, 3];
    }
    with_march_order(&mut game, bay, house);

    let actions = game.march_actions(house, bay);
    let into_port: Vec<[u8; 4]> = actions
        .iter()
        .filter(|action| action.target == Some(winterfell_port))
        .filter_map(|action| action.units)
        .collect();
    assert!(
        into_port.iter().all(|units| units[3] <= 1),
        "a Port holds three Ships, so one more fits: {into_port:?}"
    );
    assert!(
        into_port.contains(&[0, 0, 0, 1]),
        "the single ship that fits must be offered: {into_port:?}"
    );
}

// ---------------------------------------------------------------------------
// Consolidate Power
// ---------------------------------------------------------------------------

/// Python: `test_consolidate_on_a_sea_area_has_no_effect`.
#[test]
fn consolidate_on_a_sea_area_has_no_effect() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let sea = region_of(&data, "the_shivering_sea");
    {
        let region = game.state.region_mut(sea);
        region.controller = Some(house);
        region.order = Some(OrderKey::from_name("consolidate_a").unwrap());
        region.order_house = Some(house);
    }
    let before = game.state.house(house).unwrap().power;
    game.apply_consolidate(Action::resolve_consolidate(house, sea))
        .expect("the action is legal");
    assert_eq!(
        game.state.house(house).unwrap().power,
        before,
        "open water yields nothing"
    );
}

/// Python: `test_special_consolidate_may_muster_in_its_own_area`.
#[test]
fn special_consolidate_may_muster_in_its_own_area() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let winterfell = region_of(&data, "winterfell");
    {
        let region = game.state.region_mut(winterfell);
        region.controller = Some(house);
        region.owner = Some(house);
        region.units = [1, 0, 0, 0];
        region.order = Some(OrderKey::from_name("consolidate_star").unwrap());
        region.order_house = Some(house);
    }
    game.apply_consolidate_muster(Action::consolidate_muster(house, winterfell))
        .expect("the action is legal");
    // The area gets its printed mustering value and the phase returns there.
    assert_eq!(
        game.state.region(winterfell).mustering_points,
        game.data.map.region(winterfell).mustering
    );
    assert_eq!(game.state.phase, Phase::Mustering);
    assert_eq!(game.state.return_phase, Some(Phase::Action));
}

// ---------------------------------------------------------------------------
// Messenger Raven
// ---------------------------------------------------------------------------

/// Python: `test_the_messenger_raven_can_bury_the_wildling_card`.
#[test]
fn the_messenger_raven_can_bury_the_wildling_card() {
    let mut game = game(1);
    let house = HouseId(1);
    let before: Vec<u16> = game.state.wildling_deck.cards.iter().copied().collect();
    assert!(before.len() > 1, "the deck must have cards to rotate");

    game.apply_raven_deck(Action::raven_deck(house, true), true)
        .expect("burying the top card is legal");

    let after: Vec<u16> = game.state.wildling_deck.cards.iter().copied().collect();
    assert_eq!(after.len(), before.len());
    assert_eq!(
        after.last(),
        before.first(),
        "the top card goes to the bottom"
    );
    assert_eq!(&after[..after.len() - 1], &before[1..]);
    assert!(
        game.state.raven_used,
        "using the Raven spends it for the round"
    );
}

// ---------------------------------------------------------------------------
// End-of-game tiebreakers
// ---------------------------------------------------------------------------

/// Python: `test_tiebreakers_prefer_strongholds_over_plain_castles`.
#[test]
fn tiebreakers_prefer_strongholds_over_plain_castles() {
    let (mut game, data) = blank(1);
    let first = HouseId(0);
    let second = HouseId(1);
    // Same number of victory areas; one holds a Stronghold, the other a Castle.
    let stronghold = region_of(&data, "winterfell"); // mustering 2
    let castle = region_of(&data, "moat_cailin"); // mustering 1
    for region in [stronghold, castle] {
        game.state.region_mut(region).controller = None;
    }
    game.state.region_mut(stronghold).controller = Some(first);
    game.state.region_mut(castle).controller = Some(second);
    assert_eq!(
        game.state.castle_count(first),
        game.state.castle_count(second),
        "the test needs an equal castle count"
    );
    assert!(game.stronghold_count(first) > game.stronghold_count(second));
}

/// Python: `test_tiebreakers_use_supply_power_then_the_iron_throne`.
#[test]
fn tiebreakers_use_supply_power_then_the_iron_throne() {
    let (mut game, _data) = blank(1);
    let first = HouseId(0);
    let second = HouseId(1);
    // Nothing distinguishes them on the board, so Supply decides.
    for house in HouseId::ALL {
        if let Some(state) = game.state.house_mut(house) {
            state.supply_level = 1;
            state.power = 3;
        }
    }
    if let Some(state) = game.state.house_mut(first) {
        state.supply_level = 2;
    }
    assert!(game.stronghold_count(first) == game.stronghold_count(second));

    // Supply, then Power, then Iron Throne position: the ordering is a total
    // order, so a strict improvement on any key must win.
    let key = |game: &Game, house: HouseId| {
        (
            game.state.castle_count(house),
            game.stronghold_count(house),
            game.state.house(house).unwrap().supply_level,
            game.state.house(house).unwrap().power,
            -(game.state.track_position(house, Track::IronThrone) as i32),
        )
    };
    assert!(
        key(&game, first) > key(&game, second),
        "Supply decides first"
    );

    if let Some(state) = game.state.house_mut(first) {
        state.supply_level = 1;
        state.power = 9;
    }
    assert!(key(&game, first) > key(&game, second), "then Power");

    if let Some(state) = game.state.house_mut(first) {
        state.power = 3;
    }
    game.state.tracks[Track::IronThrone.index()] = SmallVec::from_slice(&[
        first,
        second,
        HouseId(2),
        HouseId(3),
        HouseId(4),
        HouseId(5),
    ]);
    assert!(
        key(&game, first) > key(&game, second),
        "then the Iron Throne"
    );
}

// ---------------------------------------------------------------------------
// Terrain helpers used above stay honest
// ---------------------------------------------------------------------------

/// Guards the two terrain facts the port tests rely on: a Port is not land, and
/// a Sea is where Ships live.
#[test]
fn terrain_of_the_regions_these_tests_use() {
    let data = data();
    for id in ["winterfell_port", "white_harbor_port"] {
        assert_eq!(
            data.map.region(region_of(&data, id)).terrain,
            Terrain::Port,
            "{id}"
        );
    }
    for id in ["the_shivering_sea", "the_narrow_sea"] {
        assert_eq!(
            data.map.region(region_of(&data, id)).terrain,
            Terrain::Sea,
            "{id}"
        );
    }
    let _ = UnitType::Ship;
    let _: Option<&State> = None;
    let _ = ActionKind::March;
}

// ---------------------------------------------------------------------------
// Data the rules depend on
// ---------------------------------------------------------------------------

/// Python: `test_stark_starts_with_one_supply`.
///
/// Stark begins one Supply level behind everyone else: a printed starting
/// position, not a rule the engine applies.
#[test]
fn stark_starts_with_one_supply() {
    let game = game(0);
    assert_eq!(game.state.house(HouseId(2)).unwrap().supply_level, 1);
    for house in HouseId::ALL
        .into_iter()
        .filter(|house| *house != HouseId(2))
    {
        assert_eq!(
            game.state.house(house).unwrap().supply_level,
            2,
            "{house} starts at supply 2"
        );
    }
}

/// Python: `test_every_house_card_ability_is_implemented`.
///
/// Every ability named in the card data must have an implementation, or a card
/// would silently do nothing in battle.
#[test]
fn every_house_card_ability_is_implemented() {
    let data = data();
    for house in HouseId::ALL {
        for card in data.house_cards.deck(house) {
            if card.ability.is_empty() || card.ability == "none" {
                continue;
            }
            assert!(
                crate::house_cards::IMPLEMENTED_ABILITIES.contains(&card.ability.as_str()),
                "{}: ability {:?} is declared in the data but not implemented",
                card.id,
                card.ability
            );
        }
    }
}

/// Python: `test_martell_deck_matches_the_reference_sheet`.
#[test]
fn martell_deck_matches_the_reference_sheet() {
    let data = data();
    let deck = data.house_cards.deck(HouseId(5));
    let card = |id: &str| {
        deck.iter()
            .find(|card| card.id == id)
            .unwrap_or_else(|| panic!("Martell has no card {id}"))
    };
    assert_eq!(card("the_red_viper").strength, 4);
    assert_eq!(card("the_red_viper").swords, 2);
    assert_eq!(card("the_red_viper").forts, 1);
    assert_eq!(card("areo_hotah").strength, 3);
    assert_eq!(card("obara_sand").strength, 2);
    assert_eq!(card("darkstar").strength, 2);
    assert_eq!(card("nymeria_sand").strength, 1);
    assert_eq!(card("arianne_martell").strength, 1);
    assert_eq!(card("doran_martell").strength, 0);
    assert!(deck.iter().all(|card| card.id != "ellaria_sand"));
    assert_eq!(deck.len(), 7);
}

// ---------------------------------------------------------------------------
// Ships, Neutral Forces and Ports
// ---------------------------------------------------------------------------

/// A land neighbour of a region, for tests that need somewhere to march from.
fn land_neighbour(data: &DataSet, region: RegionId) -> RegionId {
    data.map
        .land_neighbours_of(region)
        .into_iter()
        .find(|candidate| *candidate != region)
        .unwrap_or_else(|| panic!("{} has no land neighbour", data.map.id_of(region)))
}

/// Python: `test_ships_may_muster_into_a_friendly_sea_but_not_an_enemy_one`.
#[test]
fn ships_may_muster_into_a_friendly_sea_but_not_an_enemy_one() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let winterfell = region_of(&data, "winterfell");
    let port = data.map.port_of(winterfell).expect("Winterfell has a Port");
    let sea = data.map.region(port).sea.expect("a Port names its sea");
    {
        let region = game.state.region_mut(winterfell);
        region.controller = Some(house);
        region.owner = Some(house);
        region.mustering_points = 1; // enough for a Ship
    }
    let ship_targets = |game: &Game| -> Vec<RegionId> {
        game.mustering_actions(house)
            .iter()
            .filter(|action| action.unit == Some(UnitType::Ship))
            .filter_map(|action| action.target)
            .collect()
    };

    game.state.region_mut(sea).controller = Some(house);
    assert!(
        ship_targets(&game).contains(&sea),
        "a friendly sea may receive a new Ship"
    );

    // Only enemy Ships block mustering, which is what an enemy-held sea means.
    game.state.region_mut(sea).controller = Some(HouseId(3));
    assert!(
        !ship_targets(&game).contains(&sea),
        "an enemy-held sea refuses one"
    );
}

/// Python: `test_too_weak_an_army_cannot_attack_a_neutral_force`.
///
/// A Neutral Force may only be attacked with enough strength to destroy it, and
/// the check uses the *maximum possible* support, because support is not declared
/// yet when the legal marches are generated.
#[test]
fn too_weak_an_army_cannot_attack_a_neutral_force() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let kings_landing = region_of(&data, "kings_landing");
    let origin = land_neighbour(&data, kings_landing);
    assert_eq!(
        game.state.region(kings_landing).neutral_force,
        5,
        "the test assumes King's Landing's printed strength"
    );
    {
        let region = game.state.region_mut(origin);
        region.controller = Some(house);
        region.owner = None;
        region.units = [1, 0, 0, 0];
    }
    with_march_order(&mut game, origin, house);
    let offered = |game: &Game| {
        game.march_actions(house, origin)
            .iter()
            .any(|action| action.target == Some(kings_landing))
    };

    assert!(!offered(&game), "one Footman cannot break a force of five");

    // Three Knights are six strength, which is enough.
    game.state.region_mut(origin).units = [0, 3, 0, 0];
    assert!(offered(&game), "three Knights can");
}

/// Python: `test_marching_onto_a_neutral_force_takes_the_area`.
#[test]
fn marching_onto_a_neutral_force_takes_the_area() {
    let (mut game, data) = clean(0);
    let house = first_house(&game);
    let origin = region_of(&data, "kingswood");
    let target = region_of(&data, "kings_landing");
    game.state.region_mut(target).neutral_force = 5;
    {
        let region = game.state.region_mut(origin);
        region.controller = Some(house);
        region.units = [0, 3, 0, 0];
    }
    with_march_order(&mut game, origin, house);

    let actions = game.march_actions(house, origin);
    assert!(!actions.is_empty(), "three Knights can attack");
    game.apply_march(actions[0]).expect("the march is legal");

    let target_state = game.state.region(target);
    assert_eq!(target_state.neutral_force, 0);
    assert_eq!(target_state.controller, Some(house));
    assert_eq!(target_state.unit_count(UnitType::Knight), 3);
}

/// Python: `test_a_neutral_force_blocks_a_retreat`.
#[test]
fn a_neutral_force_blocks_a_retreat() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let kings_landing = region_of(&data, "kings_landing");
    let origin = land_neighbour(&data, kings_landing);

    let options = crate::combat::legal_retreats(&game.state, &data.map, origin, house, false, None);
    assert!(
        !options.contains(&kings_landing),
        "a Neutral Force token defends its area against retreaters too"
    );

    // With the token gone the area is open, because nothing else holds it.
    game.state.region_mut(kings_landing).neutral_force = 0;
    let options = crate::combat::legal_retreats(&game.state, &data.map, origin, house, false, None);
    assert!(options.contains(&kings_landing), "the area is now empty");
}

/// Python: `test_capturing_a_land_area_replaces_enemy_port_ships`.
#[test]
fn capturing_a_land_area_replaces_enemy_port_ships() {
    let (mut game, data) = blank(1);
    let house = HouseId(2);
    let winterfell = region_of(&data, "winterfell");
    let port = data.map.port_of(winterfell).expect("Winterfell has a Port");
    {
        let region = game.state.region_mut(port);
        region.controller = Some(HouseId(3));
        region.units = [0, 0, 0, 2];
    }
    if let Some(state) = game.state.house_mut(house) {
        state.supply_level = 4; // room for the captured fleet
    }

    game.capture_port(house, winterfell);

    let after = game.state.region(port);
    assert_eq!(after.controller, Some(house));
    assert_eq!(
        after.unit_count(UnitType::Ship),
        2,
        "two enemy Ships become two of the captor's"
    );
    assert_eq!(after.routed_count(UnitType::Ship), 0);
}

// ---------------------------------------------------------------------------
// Turn order
// ---------------------------------------------------------------------------

/// Python: `test_action_step_cycles_one_order_per_house`.
#[test]
fn action_step_cycles_one_order_per_house() {
    let (mut game, _data) = clean(0);
    game.state.phase = Phase::Action;
    game.state.action_step = ActionStep::Raid;
    game.state.action_house_last = None;

    let chosen: Vec<HouseId> = game.state.track_order(Track::IronThrone)[..2].to_vec();
    let regions: Vec<RegionId> = game.state.region_ids().collect();
    let raid = OrderKey::from_name("raid_a").expect("raid exists");
    for (index, house) in chosen.iter().enumerate() {
        for region_id in &regions[2 * index..2 * index + 2] {
            let region = game.state.region_mut(*region_id);
            region.order = Some(raid);
            region.order_house = Some(*house);
        }
    }

    let sequence: Vec<HouseId> = (0..4).filter_map(|_| game.next_action_house()).collect();
    assert_eq!(sequence.len(), 4);
    assert!(sequence.iter().all(|house| chosen.contains(house)));
    // One order each, alternating, instead of a house resolving both in a row.
    assert_ne!(sequence[0], sequence[1]);
    assert_ne!(sequence[1], sequence[2]);
    assert_eq!(sequence[0], sequence[2]);
    assert_eq!(sequence[1], sequence[3]);
}

// ---------------------------------------------------------------------------
// Messenger Raven
// ---------------------------------------------------------------------------

/// Python: `test_the_messenger_raven_can_swap_an_order`.
#[test]
fn the_messenger_raven_can_swap_an_order() {
    let (mut game, _data) = clean(0);
    let holder = game.state.raven_holder().expect("a raven holder");
    game.state.phase = Phase::Raven;
    game.state.active_house = Some(holder);
    game.state.raven_used = false;

    let region_id = game.state.region_ids().next().expect("a board");
    let raid = OrderKey::from_name("raid_a").unwrap();
    {
        let region = game.state.region_mut(region_id);
        region.controller = Some(holder);
        region.units = [1, 0, 0, 0];
        region.order = Some(raid);
        region.order_house = Some(holder);
    }
    if let Some(state) = game.state.house_mut(holder) {
        state.orders_available.retain(|key| *key != raid);
        state.orders_placed.push(raid);
    }

    let defend = OrderKey::from_name("defend_1a").unwrap();
    let swap = game
        .legal_actions()
        .into_iter()
        .find(|action| {
            action.kind == ActionKind::RavenSwap
                && action.region == Some(region_id)
                && action.order == Some(defend)
        })
        .expect("the raven may swap the order");
    game.apply_raven_swap(swap).expect("the swap is legal");

    assert_eq!(game.state.region(region_id).order, Some(defend));
    assert!(game.state.raven_used);
}

// ---------------------------------------------------------------------------
// Valyrian Steel Blade
// ---------------------------------------------------------------------------

/// Python: `test_using_the_valyrian_steel_blade_is_the_holders_choice`.
#[test]
fn using_the_valyrian_steel_blade_is_the_holders_choice() {
    let (mut game, data) = clean(0);
    let attacker = game.state.blade_holder().expect("a blade holder");
    let defender = HouseId::ALL
        .into_iter()
        .find(|house| *house != attacker)
        .expect("six houses");
    let (_, region_id) = adjacent_land_pair(&data);

    game.state.phase = Phase::Combat;
    game.state.active_house = Some(attacker);
    if let Some(state) = game.state.house_mut(attacker) {
        state.blade_used = false;
    }
    game.state.pending_battle = Some(PendingBattle {
        region: region_id,
        attacker,
        defender: Some(defender),
        neutral: false,
        origin: Some(region_id),
        attacker_order: None,
        attacker_units: [1, 0, 0, 0],
        defender_units: [1, 0, 0, 0],
        attacker_order_strength: 0,
        defender_order_strength: 0,
        // The Python regression parks `"chosen"` placeholders here; the stage is
        // already Blade, so no real card id is needed.
        attacker_card: None,
        defender_card: None,
        attacker_card_chosen: true,
        defender_card_chosen: true,
        doran_owner: None,
        doran_track: None,
        prepared: None,
        blade_done: false,
        attacker_blade: false,
        defender_blade: false,
        stage: BattleStage::Blade,
        support_queue: SmallVec::new(),
        support_index: 0,
        support_choices: SmallVec::new(),
        result: None,
        casualties_remaining: 0,
        retreat_options: SmallVec::new(),
        retreat_house: None,
        retreat_destination: None,
        defender_order: None,
        defender_order_house: None,
    });

    assert_eq!(game.next_combat_chooser(), Some(attacker));
    let actions = game.combat_actions(attacker);
    assert!(actions
        .iter()
        .any(|action| action.kind == ActionKind::UseBlade));
    assert!(actions.iter().any(|action| action.kind == ActionKind::Pass));

    game.apply_use_blade(Action::use_blade(attacker))
        .expect("the blade is the holder's choice");
    let battle = game.state.pending_battle.as_ref().expect("a battle");
    assert!(battle.attacker_blade);
    assert!(battle.blade_done);
}

// ---------------------------------------------------------------------------
// Wildling attacks
// ---------------------------------------------------------------------------

/// Python: `test_wildling_icons_advance_before_card_effects_resolve`.
#[test]
fn wildling_icons_advance_before_card_effects_resolve() {
    let (mut game, _data) = westeros_choice_game([
        &["wildlings_attack"],
        &["last_days_of_summer"],
        &["sea_of_storms"],
    ]);
    game.state.wildling_threat = 10;

    game.run_westeros_phase().expect("the phase must not stall");

    // The two icons push the track to 12 before the "Wildlings Attack" card
    // resolves, so the first attack happens at 12, not at the starting 10.
    assert_eq!(game.state.phase, Phase::WildlingAttack);
    assert_eq!(
        game.state
            .wildling_attack
            .as_ref()
            .map(|attack| attack.strength),
        Some(12)
    );
}

/// Python: `test_wildling_attack_bids_in_iron_throne_order`.
#[test]
fn wildling_attack_bids_in_iron_throne_order() {
    let (mut game, data) = clean(0);
    begin_wildling_attack(&mut game, &data, "silence", 6);
    let order: Vec<HouseId> = game.state.track_order(Track::IronThrone).to_vec();

    let mut seen: Vec<HouseId> = Vec::new();
    while game.state.phase == Phase::WildlingAttack {
        let active = game.state.active_house.expect("a bidder");
        seen.push(active);
        game.step(Action::bid(active, 1)).expect("a bid is legal");
    }

    assert_eq!(seen, order);
    // Six bids of 1 meet the strength of 6: the Night's Watch holds.
    assert_eq!(game.state.wildling_threat, 0);
    // Every house spent the Power it bid, win or lose.
    for house in &order {
        assert_eq!(game.state.house(*house).unwrap().power, 4);
    }
    assert_eq!(game.state.wildling_discard.len(), 1);
    assert_eq!(
        data.wildlings.card_id(game.state.wildling_discard[0]),
        "silence"
    );
}

/// Python: `test_wildlings_win_moves_the_track_back_two`.
#[test]
fn wildlings_win_moves_the_track_back_two() {
    let (mut game, data) = clean(0);
    begin_wildling_attack(&mut game, &data, "silence", 8);

    while game.state.phase == Phase::WildlingAttack {
        let active = game.state.active_house.expect("a bidder");
        game.step(Action::bid(active, 0)).expect("a bid is legal");
    }

    assert_eq!(game.state.wildling_threat, 6);
}

/// Python: `test_wildling_lowest_bidder_loses_units`.
#[test]
fn wildling_lowest_bidder_loses_units() {
    let (mut game, data) = clean(0);
    let house = first_house(&game);
    let kingswood = region_of(&data, "kingswood");
    {
        let region = game.state.region_mut(kingswood);
        region.controller = Some(house);
        region.units = [4, 0, 0, 0];
    }

    begin_wildling_attack(&mut game, &data, "mammoth_riders", 8);
    // Everyone bids zero. The wildlings win and the tied lowest bidders are
    // broken by Iron Throne position, so `house` is the lowest bidder.
    while game.state.phase == Phase::WildlingAttack {
        let active = game.state.active_house.expect("a bidder");
        game.step(Action::bid(active, 0)).expect("a bid is legal");
    }

    assert_eq!(
        game.state.region(kingswood).unit_count(UnitType::Footman),
        1
    );
}

/// Python: `test_preemptive_raid_starts_a_second_attack`.
#[test]
fn preemptive_raid_starts_a_second_attack() {
    let (mut game, data) = clean(0);
    begin_wildling_attack(&mut game, &data, "preemptive_raid", 4);

    let mut second_seen = false;
    let mut guard = 0;
    while game.state.phase == Phase::WildlingAttack && guard < 20 {
        if let Some(attack) = game.state.wildling_attack.as_ref() {
            if attack.strength == 6 && !attack.excluded.is_empty() {
                second_seen = true;
                break;
            }
        }
        let active = game.state.active_house.expect("a bidder");
        game.step(Action::bid(active, 1)).expect("a bid is legal");
        guard += 1;
    }

    assert!(second_seen, "the highest bidder faces a second attack");
}

/// Python: `test_westeros_cards_resolve_after_an_interrupting_wildling_attack`.
#[test]
fn westeros_cards_resolve_after_an_interrupting_wildling_attack() {
    let (mut game, data) = westeros_choice_game([
        &["wildlings_attack"],
        &["last_days_of_summer"],
        &["sea_of_storms"],
    ]);
    game.state.wildling_threat = 12;
    let silence = data.wildlings.card_index("silence").unwrap();
    game.state.wildling_deck = DeckState::new([silence]);
    game.state.wildling_discard.clear();

    game.run_westeros_phase().expect("the phase must not stall");
    // The two Wildling icons pushed the track to 12, so the attack interrupts
    // the Westeros phase before the cards resolve.
    assert_eq!(game.state.phase, Phase::WildlingAttack);

    let mut guard = 0;
    while game.state.phase == Phase::WildlingAttack && guard < 50 {
        let active = game.state.active_house.expect("a bidder");
        let amount = game
            .state
            .house(active)
            .map(|state| state.power)
            .unwrap_or(0);
        game.step(Action::bid(active, amount))
            .expect("a bid is legal");
        guard += 1;
    }

    // The queue drains instead of stalling, and the round is ready to continue.
    assert!(game.state.pending_westeros.is_none());
    assert!(!matches!(
        game.state.phase,
        Phase::Westeros | Phase::WildlingAttack
    ));
}

// ---------------------------------------------------------------------------
// Holders'-choice Westeros cards
// ---------------------------------------------------------------------------

/// Python: `test_throne_of_blades_holder_chooses_mustering`.
#[test]
fn throne_of_blades_holder_chooses_mustering() {
    let (mut game, data) = westeros_choice_game([
        &["throne_of_blades"],
        &["game_of_thrones"],
        &["sea_of_storms"],
    ]);
    let holder = first_house(&game);
    let fort = sorted_regions(&data, game.state.fortifications.to_vec())[0];
    {
        let region = game.state.region_mut(fort);
        region.controller = Some(holder);
        region.mustering_points = 0;
    }

    game.run_westeros_phase().expect("the choice appears");
    game.advance().expect("the holder is asked");
    assert_eq!(game.state.phase, Phase::WesterosChoice);
    assert_eq!(game.state.active_house, Some(holder));

    let offered = choice_cards(&game.legal_actions());
    for id in ["mustering", "supply", "none"] {
        let index = if id == "none" {
            NONE_OPTION
        } else {
            data.westeros.card_index(id).expect("known card")
        };
        assert!(
            offered.contains(&CardId(index as u8)),
            "missing option {id}: {offered:?}"
        );
    }
    assert_eq!(offered.len(), 3);

    let mustering = CardId(data.westeros.card_index("mustering").unwrap() as u8);
    game.step(Action::westeros_choice(holder, mustering))
        .expect("the choice is legal");

    assert_eq!(
        game.state.region(fort).mustering_points,
        data.map.region(fort).mustering
    );
    assert!(game.state.westeros_choice.is_none());
    assert!(matches!(
        game.state.phase,
        Phase::Mustering | Phase::Planning
    ));
}

/// Python: `test_dark_wings_holder_chooses_clash_of_kings_and_the_queue_resumes`.
#[test]
fn dark_wings_holder_chooses_clash_of_kings_and_the_queue_resumes() {
    let (mut game, data) =
        westeros_choice_game([&["supply"], &["dark_wings_dark_words"], &["sea_of_storms"]]);
    let holder = game.state.raven_holder().expect("a raven holder");

    game.run_westeros_phase().expect("the choice appears");
    game.advance().expect("the holder is asked");
    assert_eq!(game.state.phase, Phase::WesterosChoice);
    assert_eq!(game.state.active_house, Some(holder));

    let clash = CardId(data.westeros.card_index("clash_of_kings").unwrap() as u8);
    game.step(Action::westeros_choice(holder, clash))
        .expect("the choice is legal");
    assert_eq!(game.state.phase, Phase::Bidding);
    assert_eq!(
        game.state.bid_tracks.as_slice(),
        &[Track::IronThrone, Track::Fiefdoms, Track::KingsCourt]
    );

    let mut guard = 0;
    while game.state.phase == Phase::Bidding && guard < 100 {
        let active = game.state.active_house.expect("a bidder");
        game.step(Action::bid(active, 0)).expect("a bid is legal");
        guard += 1;
    }

    // The queued "Sea of Storms" still resolves after the bidding.
    let raid = OrderKey::from_name("raid_a").unwrap();
    assert!(game.state.has_order_restriction(raid));
    assert!(game.state.pending_westeros.is_none());
}

/// Python: `test_put_to_the_sword_holder_chooses_a_restriction`.
#[test]
fn put_to_the_sword_holder_chooses_a_restriction() {
    let (mut game, data) = clean(0);
    let holder = game.state.blade_holder().expect("a blade holder");
    let put_to_the_sword = data.westeros.card_index("put_to_the_sword").unwrap();
    let options: SmallVec<[u16; 3]> = ["rains_of_autumn", "storm_of_swords", "none"]
        .into_iter()
        .map(|id| {
            if id == "none" {
                NONE_OPTION
            } else {
                data.westeros.card_index(id).unwrap()
            }
        })
        .collect();

    game.state.phase = Phase::WesterosChoice;
    game.state.pending_westeros = Some(SmallVec::new());
    game.state.order_restrictions.clear();
    game.state.westeros_choice = Some(WesterosChoice {
        card: put_to_the_sword,
        holder,
        options,
    });

    game.advance().expect("the holder is asked");
    assert_eq!(game.state.active_house, Some(holder));
    let offered = choice_cards(&game.legal_actions());
    for id in ["rains_of_autumn", "storm_of_swords", "none"] {
        let index = if id == "none" {
            NONE_OPTION
        } else {
            data.westeros.card_index(id).unwrap()
        };
        assert!(offered.contains(&CardId(index as u8)), "missing {id}");
    }

    let rains = CardId(data.westeros.card_index("rains_of_autumn").unwrap() as u8);
    game.step(Action::westeros_choice(holder, rains))
        .expect("the choice is legal");

    let march_plus = OrderKey::from_name("march_plus").unwrap();
    assert!(game.state.has_order_restriction(march_plus));
    assert!(game.state.westeros_choice.is_none());
}

/// Python: `test_a_holders_choice_can_be_no_effect`.
#[test]
fn a_holders_choice_can_be_no_effect() {
    let (mut game, data) = clean(0);
    let holder = game.state.blade_holder().expect("a blade holder");
    let put_to_the_sword = data.westeros.card_index("put_to_the_sword").unwrap();
    let options: SmallVec<[u16; 3]> = ["rains_of_autumn", "storm_of_swords", "none"]
        .into_iter()
        .map(|id| {
            if id == "none" {
                NONE_OPTION
            } else {
                data.westeros.card_index(id).unwrap()
            }
        })
        .collect();

    game.state.phase = Phase::WesterosChoice;
    game.state.pending_westeros = Some(SmallVec::new());
    game.state.order_restrictions.clear();
    game.state.westeros_choice = Some(WesterosChoice {
        card: put_to_the_sword,
        holder,
        options,
    });

    game.step(Action::westeros_choice(holder, CardId(NONE_OPTION as u8)))
        .expect("no effect is a legal choice");

    assert!(game.state.order_restrictions.is_empty());
    assert!(game.state.westeros_choice.is_none());
}

// ---------------------------------------------------------------------------
// House card abilities in battle
// ---------------------------------------------------------------------------

/// A battle request with every modifier at its default.
fn battle(
    target: RegionId,
    attacker: HouseId,
    defender: HouseId,
    attacker_units: [u8; 4],
    defender_units: [u8; 4],
    attacker_card: Option<CardId>,
    defender_card: Option<CardId>,
) -> BattleRequest {
    let mut request =
        BattleRequest::new(target, attacker, defender, attacker_units, defender_units);
    request.attacker_card = attacker_card;
    request.defender_card = defender_card;
    request
}

/// Python: `test_stannis_gains_strength_when_the_opponent_holds_the_throne`.
#[test]
fn stannis_gains_strength_when_the_opponent_holds_the_throne() {
    let (mut game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (baratheon, lannister) = (house("baratheon"), house("lannister"));
    let stannis = card(&data, baratheon, "stannis_baratheon");
    game.state.tracks[Track::IronThrone.index()] = SmallVec::from_slice(&[
        lannister,
        baratheon,
        house("stark"),
        house("greyjoy"),
        house("tyrell"),
        house("martell"),
    ]);

    let request = battle(
        target,
        baratheon,
        lannister,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        Some(stannis),
        None,
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.attacker.card_strength, 5, "printed 4, +1");

    game.state.tracks[Track::IronThrone.index()] = SmallVec::from_slice(&[
        baratheon,
        lannister,
        house("stark"),
        house("greyjoy"),
        house("tyrell"),
        house("martell"),
    ]);
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.attacker.card_strength, 4);
}

/// Python: `test_davos_needs_stannis_in_the_discard_pile`.
#[test]
fn davos_needs_stannis_in_the_discard_pile() {
    let (mut game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (baratheon, stark) = (house("baratheon"), house("stark"));
    let stannis = card(&data, baratheon, "stannis_baratheon");
    game.state.house_mut(baratheon).unwrap().house_cards_used = SmallVec::from_slice(&[stannis]);

    let request = battle(
        target,
        baratheon,
        stark,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, baratheon, "davos_seaworth")),
        None,
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.attacker.card_strength, 3, "printed 2, +1");
    assert_eq!(result.attacker.swords, 1);
}

/// Python: `test_balon_zeroes_the_opponents_printed_strength`.
#[test]
fn balon_zeroes_the_opponents_printed_strength() {
    let (game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (greyjoy, stark) = (house("greyjoy"), house("stark"));

    let request = battle(
        target,
        greyjoy,
        stark,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, greyjoy, "balon_greyjoy")),
        Some(card(&data, stark, "eddard_stark")),
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.defender.card_strength, 0);
}

/// Python: `test_asha_gains_icons_when_not_supported`.
#[test]
fn asha_gains_icons_when_not_supported() {
    let (game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (greyjoy, stark) = (house("greyjoy"), house("stark"));

    let request = battle(
        target,
        greyjoy,
        stark,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, greyjoy, "asha_greyjoy")),
        None,
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.attacker.swords, 2);
    assert_eq!(result.attacker.forts, 1);
}

/// Python: `test_nymeria_switches_icon_by_role`.
#[test]
fn nymeria_switches_icon_by_role() {
    let (game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (martell, stark) = (house("martell"), house("stark"));
    let nymeria = card(&data, martell, "nymeria_sand");

    let attacking = battle(
        target,
        martell,
        stark,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        Some(nymeria),
        None,
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &attacking);
    assert_eq!(result.attacker.swords, 1);
    assert_eq!(result.attacker.forts, 0);

    let defending = battle(
        target,
        stark,
        martell,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        None,
        Some(nymeria),
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &defending);
    assert_eq!(result.defender.forts, 1);
    assert_eq!(result.defender.swords, 0);
}

/// Python: `test_theon_needs_a_fortification_when_defending`.
#[test]
fn theon_needs_a_fortification_when_defending() {
    let (game, data) = clean(0);
    let fort = sorted_regions(&data, game.state.fortifications.to_vec())[0];
    let (stark, greyjoy) = (house("stark"), house("greyjoy"));

    let request = battle(
        fort,
        stark,
        greyjoy,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        None,
        Some(card(&data, greyjoy, "theon_greyjoy")),
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert!(data.map.region(fort).is_fortification());
    assert_eq!(result.defender.card_strength, 3, "printed 2, +1");
    assert_eq!(result.defender.swords, 1);
}

/// Python: `test_kevan_doubles_attacking_footmen`.
#[test]
fn kevan_doubles_attacking_footmen() {
    let (game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (lannister, stark) = (house("lannister"), house("stark"));

    let request = battle(
        target,
        lannister,
        stark,
        [3, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, lannister, "kevan_lannister")),
        None,
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.attacker.units_strength, 6);
}

/// Python: `test_victarion_doubles_attacking_ships`.
#[test]
fn victarion_doubles_attacking_ships() {
    let (game, data) = clean(0);
    let sea = first_sea(&data);
    let (greyjoy, stark) = (house("greyjoy"), house("stark"));

    let request = battle(
        sea,
        greyjoy,
        stark,
        [0, 0, 0, 3],
        [0, 0, 0, 1],
        Some(card(&data, greyjoy, "victarion_greyjoy")),
        None,
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.attacker.units_strength, 6);
}

/// Python: `test_catelyn_doubles_a_defend_order`.
#[test]
fn catelyn_doubles_a_defend_order() {
    let (mut game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (lannister, stark) = (house("lannister"), house("stark"));
    let defend = OrderKey::from_name("defend_2").unwrap();
    {
        let region = game.state.region_mut(target);
        region.order = Some(defend);
        region.order_house = Some(stark);
    }

    let mut request = battle(
        target,
        lannister,
        stark,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        None,
        Some(card(&data, stark, "catelyn_stark")),
    );
    request.defender_order_strength = 2;
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.defender.order_strength, 4);
}

/// Python: `test_blackfish_ignores_casualties`.
#[test]
fn blackfish_ignores_casualties() {
    let (game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (lannister, stark) = (house("lannister"), house("stark"));

    let request = battle(
        target,
        lannister,
        stark,
        [3, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, lannister, "gregor_clegane")),
        Some(card(&data, stark, "blackfish")),
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.winner, lannister);
    assert_eq!(result.casualties, 0);
}

/// Python: `test_mace_destroys_an_opposing_footman`.
#[test]
fn mace_destroys_an_opposing_footman() {
    let (game, data) = clean(0);
    let (_, target) = adjacent_land_pair(&data);
    let (tyrell, stark) = (house("tyrell"), house("stark"));

    let request = battle(
        target,
        tyrell,
        stark,
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        Some(card(&data, tyrell, "mace_tyrell")),
        None,
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.defender.units_strength, 1);
    assert_eq!(result.context.defender_units[UnitType::Footman.index()], 1);
}

/// Python: `test_salladhor_reduces_non_baratheon_ships`.
#[test]
fn salladhor_reduces_non_baratheon_ships() {
    let (mut game, data) = clean(0);
    let sea = first_sea(&data);
    let neighbour = sorted_regions(&data, data.map.sea_neighbours_of(sea).to_vec())[0];
    let (baratheon, lannister) = (house("baratheon"), house("lannister"));
    {
        let region = game.state.region_mut(neighbour);
        region.controller = Some(baratheon);
        region.units = [0, 0, 0, 1];
        region.order = Some(OrderKey::from_name("support_a").unwrap());
        region.order_house = Some(baratheon);
    }

    let request = battle(
        sea,
        baratheon,
        lannister,
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        Some(card(&data, baratheon, "salladhor_saan")),
        None,
    );
    let result = resolve_battle(&game.state, &data.map, &data.house_cards, &request);
    assert_eq!(result.defender.units_strength, 0);
    assert_eq!(result.attacker.units_strength, 1);
}

/// Python: `test_tywin_gains_two_power_when_he_wins`.
#[test]
fn tywin_gains_two_power_when_he_wins() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (lannister, stark) = (house("lannister"), house("stark"));
    game.state.region_mut(origin).controller = Some(lannister);
    game.state.region_mut(origin).units = [0; 4];
    let before = game.state.house(lannister).unwrap().power;

    let mut request = battle(
        target,
        lannister,
        stark,
        [3, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, lannister, "tywin_lannister")),
        None,
    );
    request.attacker_origin = Some(origin);
    let result = resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(result.winner, lannister);
    assert_eq!(game.state.house(lannister).unwrap().power, before + 2);
}

/// Python: `test_roose_returns_the_discard_pile_on_a_loss`.
#[test]
fn roose_returns_the_discard_pile_on_a_loss() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (lannister, stark) = (house("lannister"), house("stark"));
    game.state.region_mut(origin).controller = Some(lannister);
    game.state.region_mut(origin).units = [0; 4];
    {
        let state = game.state.house_mut(stark).unwrap();
        state.house_cards_available = SmallVec::from_slice(&[
            card(&data, stark, "robb_stark"),
            card(&data, stark, "eddard_stark"),
        ]);
        state.house_cards_used = SmallVec::from_slice(&[
            card(&data, stark, "greatjon_umber"),
            card(&data, stark, "roose_bolton"),
        ]);
    }

    let mut request = battle(
        target,
        lannister,
        stark,
        [5, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, lannister, "gregor_clegane")),
        Some(card(&data, stark, "roose_bolton")),
    );
    request.attacker_origin = Some(origin);
    resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    let state = game.state.house(stark).unwrap();
    assert_eq!(state.house_cards_available.len(), 4);
    for id in [
        "robb_stark",
        "eddard_stark",
        "greatjon_umber",
        "roose_bolton",
    ] {
        assert!(state
            .house_cards_available
            .contains(&card(&data, stark, id)));
    }
    assert!(state.house_cards_used.is_empty());
}

/// Python: `test_arianne_blocks_the_conquest`.
#[test]
fn arianne_blocks_the_conquest() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (lannister, martell) = (house("lannister"), house("martell"));
    game.state.region_mut(origin).controller = Some(lannister);
    game.state.region_mut(origin).units = [0; 4];

    let mut request = battle(
        target,
        lannister,
        martell,
        [3, 0, 0, 0],
        [1, 0, 0, 0],
        None,
        Some(card(&data, martell, "arianne_martell")),
    );
    request.attacker_origin = Some(origin);
    let result = resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(result.winner, lannister);
    // The attacker may not occupy the area; it returns to where it marched from.
    assert_ne!(game.state.region(target).controller, Some(lannister));
    assert_eq!(game.state.region(origin).unit_count(UnitType::Footman), 3);
}

/// Python: `test_loras_replaces_the_march_order_in_the_conquered_area`.
#[test]
fn loras_replaces_the_march_order_in_the_conquered_area() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (tyrell, stark) = (house("tyrell"), house("stark"));
    game.state.region_mut(origin).controller = Some(tyrell);
    game.state.region_mut(origin).units = [0; 4];

    let mut request = battle(
        target,
        tyrell,
        stark,
        [3, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, tyrell, "loras_tyrell")),
        None,
    );
    request.attacker_origin = Some(origin);
    request.attacker_order = OrderKey::from_name("march");
    resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(
        game.state.region(target).order,
        OrderKey::from_name("march")
    );
    assert_eq!(game.state.region(target).order_house, Some(tyrell));
}

/// Python: `test_cersei_removes_an_opponent_order`.
#[test]
fn cersei_removes_an_opponent_order() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (lannister, stark) = (house("lannister"), house("stark"));
    game.state.region_mut(origin).controller = Some(lannister);
    game.state.region_mut(origin).units = [0; 4];
    let victim = sorted_regions(
        &data,
        game.state
            .region_ids()
            .filter(|region| *region != target)
            .collect(),
    )[0];
    {
        let region = game.state.region_mut(victim);
        region.order = OrderKey::from_name("march");
        region.order_house = Some(stark);
    }

    let mut request = battle(
        target,
        lannister,
        stark,
        [5, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, lannister, "cersei_lannister")),
        None,
    );
    request.attacker_origin = Some(origin);
    resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(game.state.region(victim).order, None);
}

/// Python: `test_queen_of_thorns_removes_an_adjacent_order`.
#[test]
fn queen_of_thorns_removes_an_adjacent_order() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (tyrell, stark) = (house("tyrell"), house("stark"));
    game.state.region_mut(origin).controller = Some(tyrell);
    game.state.region_mut(origin).units = [0; 4];
    let adjacent = sorted_regions(
        &data,
        data.map
            .neighbours_of(target)
            .iter()
            .copied()
            .filter(|region| *region != origin)
            .collect(),
    )[0];
    {
        let region = game.state.region_mut(adjacent);
        region.order = OrderKey::from_name("support_a");
        region.order_house = Some(stark);
    }

    let mut request = battle(
        target,
        tyrell,
        stark,
        [5, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, tyrell, "queen_of_thorns")),
        None,
    );
    request.attacker_origin = Some(origin);
    resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(game.state.region(adjacent).order, None);
}

/// Python: `test_doran_moves_the_opponent_to_the_bottom_of_a_track`.
#[test]
fn doran_moves_the_opponent_to_the_bottom_of_a_track() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (martell, stark) = (house("martell"), house("stark"));
    game.state.region_mut(origin).controller = Some(martell);
    game.state.region_mut(origin).units = [0; 4];
    game.state.tracks[Track::Fiefdoms.index()] = SmallVec::from_slice(&[
        stark,
        house("baratheon"),
        house("lannister"),
        house("greyjoy"),
        house("tyrell"),
        martell,
    ]);

    let mut request = battle(
        target,
        martell,
        stark,
        [5, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, martell, "doran_martell")),
        None,
    );
    request.attacker_origin = Some(origin);
    resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(game.state.track_order(Track::Fiefdoms).last(), Some(&stark));
}

/// Doran resolves before the blade step, and a used blade stays used.
#[test]
fn doran_moves_influence_before_blade_and_carries_the_used_face() {
    fn setup(used: bool) -> (Game, Arc<DataSet>, HouseId, HouseId) {
        let (mut game, data) = clean(0);
        let (origin, target) = adjacent_land_pair(&data);
        let (baratheon, martell) = (house("baratheon"), house("martell"));
        game.state.phase = Phase::Combat;
        game.state.active_house = Some(baratheon);
        game.state.region_mut(origin).controller = Some(baratheon);
        game.state.region_mut(origin).units = [1, 0, 0, 0];
        game.state.region_mut(target).controller = Some(martell);
        game.state.region_mut(target).units = [1, 0, 0, 0];
        let others = [
            house("stark"),
            house("lannister"),
            house("greyjoy"),
            house("tyrell"),
        ];
        game.state.tracks[Track::IronThrone.index()] = SmallVec::from_slice(&[
            others[0], martell, others[1], others[2], others[3], baratheon,
        ]);
        game.state.tracks[Track::Fiefdoms.index()] = SmallVec::from_slice(&[
            baratheon, martell, others[0], others[1], others[2], others[3],
        ]);
        game.state.tracks[Track::KingsCourt.index()] = SmallVec::from_slice(&[
            others[0], martell, others[1], others[2], others[3], baratheon,
        ]);
        game.state.house_mut(baratheon).unwrap().blade_used = used;
        game.state.house_mut(martell).unwrap().blade_used = false;
        game.state.pending_battle = Some(PendingBattle {
            region: target,
            attacker: baratheon,
            defender: Some(martell),
            neutral: false,
            origin: Some(origin),
            attacker_order: None,
            attacker_units: [1, 0, 0, 0],
            defender_units: [1, 0, 0, 0],
            attacker_order_strength: 0,
            defender_order_strength: 0,
            attacker_card: Some(card(&data, baratheon, "stannis_baratheon")),
            defender_card: Some(card(&data, martell, "doran_martell")),
            attacker_card_chosen: true,
            defender_card_chosen: true,
            doran_owner: None,
            doran_track: None,
            prepared: None,
            blade_done: false,
            attacker_blade: false,
            defender_blade: false,
            stage: BattleStage::Cards,
            support_queue: SmallVec::new(),
            support_index: 0,
            support_choices: SmallVec::new(),
            result: None,
            casualties_remaining: 0,
            retreat_options: SmallVec::new(),
            retreat_house: None,
            retreat_destination: None,
            defender_order: None,
            defender_order_house: None,
        });
        (game, data, baratheon, martell)
    }

    let (mut game, _data, baratheon, martell) = setup(false);
    assert_eq!(game.next_combat_chooser(), Some(martell));
    assert_eq!(game.state.track_order(Track::Fiefdoms)[0], martell);
    assert_eq!(
        game.state.pending_battle.as_ref().unwrap().stage,
        BattleStage::Blade
    );
    assert!(game
        .state
        .pending_battle
        .as_ref()
        .unwrap()
        .prepared
        .is_some());
    assert!(!game.state.house(baratheon).unwrap().blade_used);

    let (mut game, _data, _baratheon, martell) = setup(true);
    assert_eq!(game.next_combat_chooser(), None);
    assert_eq!(game.state.track_order(Track::Fiefdoms)[0], martell);
    assert!(game.state.house(martell).unwrap().blade_used);
    assert!(game.state.pending_battle.as_ref().unwrap().blade_done);

    let (mut game, _data, _baratheon, martell) = setup(false);
    game.doran_choice = true;
    assert_eq!(game.next_combat_chooser(), Some(martell));
    assert_eq!(
        game.state.pending_battle.as_ref().unwrap().stage,
        BattleStage::Doran
    );
    assert_eq!(
        game.state.track_order(Track::Fiefdoms)[0],
        house("baratheon")
    );
    game.choose_doran_track(Track::Fiefdoms).unwrap();
    assert_eq!(
        game.state.pending_battle.as_ref().unwrap().stage,
        BattleStage::Blade
    );
    assert_eq!(game.state.track_order(Track::Fiefdoms)[0], martell);
    assert_eq!(game.state.active_house, Some(martell));
}

/// Moving a King's Court position does not remove orders already on the board.
#[test]
fn doran_does_not_remove_already_placed_starred_orders() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (martell, stark, baratheon) = (house("martell"), house("stark"), house("baratheon"));
    game.state.region_mut(origin).controller = Some(martell);
    game.state.region_mut(origin).units = [0; 4];
    let order_region = game
        .state
        .region_ids()
        .find(|region| *region != origin && *region != target)
        .expect("a third region");
    let starred = OrderKey::from_name("march_plus").unwrap();
    game.state.region_mut(order_region).order = Some(starred);
    game.state.region_mut(order_region).order_house = Some(baratheon);
    game.state.tracks[Track::IronThrone.index()] = SmallVec::from_slice(&[
        baratheon,
        martell,
        stark,
        house("lannister"),
        house("greyjoy"),
        house("tyrell"),
    ]);
    game.state.tracks[Track::Fiefdoms.index()] = SmallVec::from_slice(&[
        baratheon,
        martell,
        stark,
        house("lannister"),
        house("greyjoy"),
        house("tyrell"),
    ]);
    game.state.tracks[Track::KingsCourt.index()] = SmallVec::from_slice(&[
        stark,
        martell,
        baratheon,
        house("lannister"),
        house("greyjoy"),
        house("tyrell"),
    ]);

    let mut request = battle(
        target,
        martell,
        stark,
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, martell, "doran_martell")),
        None,
    );
    request.attacker_origin = Some(origin);
    resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(game.state.region(order_region).order, Some(starred));
    assert_eq!(game.state.region(order_region).order_house, Some(baratheon));
}

/// Python: `test_tyrion_returns_the_opponents_card_and_forces_a_swap`.
#[test]
fn tyrion_returns_the_opponents_card_and_forces_a_swap() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (lannister, stark) = (house("lannister"), house("stark"));
    game.state.region_mut(origin).controller = Some(lannister);
    game.state.region_mut(origin).units = [0; 4];
    let (catelyn, eddard) = (
        card(&data, stark, "catelyn_stark"),
        card(&data, stark, "eddard_stark"),
    );
    {
        let state = game.state.house_mut(stark).unwrap();
        state.house_cards_available = SmallVec::from_slice(&[catelyn]);
        state.house_cards_used = SmallVec::from_slice(&[eddard]);
    }

    let mut request = battle(
        target,
        lannister,
        stark,
        [5, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, lannister, "tyrion_lannister")),
        Some(eddard),
    );
    request.attacker_origin = Some(origin);
    let result = resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(result.defender.card_strength, 0, "forced onto Catelyn");
    let state = game.state.house(stark).unwrap();
    assert!(state.house_cards_available.contains(&eddard));
    assert!(state.house_cards_used.contains(&catelyn));
}

/// Python: `test_tyrion_leaves_an_empty_handed_opponent_with_no_card`.
#[test]
fn tyrion_leaves_an_empty_handed_opponent_with_no_card() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (lannister, stark) = (house("lannister"), house("stark"));
    game.state.region_mut(origin).controller = Some(lannister);
    game.state.region_mut(origin).units = [0; 4];
    let eddard = card(&data, stark, "eddard_stark");
    {
        let state = game.state.house_mut(stark).unwrap();
        state.house_cards_available.clear();
        state.house_cards_used = SmallVec::from_slice(&[eddard]);
    }

    let mut request = battle(
        target,
        lannister,
        stark,
        [5, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, lannister, "tyrion_lannister")),
        Some(eddard),
    );
    request.attacker_origin = Some(origin);
    let result = resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    assert_eq!(result.defender.card_strength, 0);
    let state = game.state.house(stark).unwrap();
    assert_eq!(state.house_cards_available.as_slice(), &[eddard]);
    assert!(state.house_cards_used.is_empty());
}

/// Python: `test_aeron_pays_power_to_swap_itself_out`.
#[test]
fn aeron_pays_power_to_swap_itself_out() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (greyjoy, stark) = (house("greyjoy"), house("stark"));
    game.state.region_mut(origin).controller = Some(greyjoy);
    game.state.region_mut(origin).units = [0; 4];
    let (aeron, balon) = (
        card(&data, greyjoy, "aeron_damphair"),
        card(&data, greyjoy, "balon_greyjoy"),
    );
    {
        let state = game.state.house_mut(greyjoy).unwrap();
        state.power = 5;
        state.house_cards_available = SmallVec::from_slice(&[balon]);
        state.house_cards_used = SmallVec::from_slice(&[aeron]);
    }

    let mut request = battle(
        target,
        greyjoy,
        stark,
        [5, 0, 0, 0],
        [1, 0, 0, 0],
        Some(aeron),
        None,
    );
    request.attacker_origin = Some(origin);
    let result = resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    let state = game.state.house(greyjoy).unwrap();
    assert_eq!(state.power, 3);
    assert_eq!(result.attacker.card_strength, 2, "Balon Greyjoy");
    assert!(state.house_cards_used.contains(&balon));
}

/// Python: `test_patchface_discards_the_opponents_best_card`.
#[test]
fn patchface_discards_the_opponents_best_card() {
    let (mut game, data) = clean(0);
    let (origin, target) = adjacent_land_pair(&data);
    let (baratheon, stark) = (house("baratheon"), house("stark"));
    game.state.region_mut(origin).controller = Some(baratheon);
    game.state.region_mut(origin).units = [0; 4];
    let (greatjon, robb, eddard) = (
        card(&data, stark, "greatjon_umber"),
        card(&data, stark, "robb_stark"),
        card(&data, stark, "eddard_stark"),
    );
    {
        let state = game.state.house_mut(stark).unwrap();
        state.house_cards_available = SmallVec::from_slice(&[greatjon, robb]);
        state.house_cards_used = SmallVec::from_slice(&[eddard]);
    }

    let mut request = battle(
        target,
        baratheon,
        stark,
        [5, 0, 0, 0],
        [1, 0, 0, 0],
        Some(card(&data, baratheon, "patchface")),
        Some(eddard),
    );
    request.attacker_origin = Some(origin);
    resolve_and_apply(&mut game.state, &data.map, &data.house_cards, &request);

    let state = game.state.house(stark).unwrap();
    assert!(state.house_cards_used.contains(&robb));
    assert!(!state.house_cards_available.contains(&robb));
}
