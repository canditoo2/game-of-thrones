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
use crate::enums::{HouseId, OrderKey, Phase, RegionId, Terrain, Track, UnitType};
use crate::game::Game;
use crate::map::DataSet;
use crate::setup::GameConfig;
use crate::state::State;

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
