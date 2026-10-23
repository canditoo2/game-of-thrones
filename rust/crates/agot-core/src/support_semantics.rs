//! Cross-house support at the phase-machine level.
//!
//! Ported from `tests/test_cross_house_support.py`, which retires once this
//! lives here. The engine-level halves (how much support is worth, which areas
//! may give it) are already in `tests/combat_semantics.rs` under the same names;
//! what only the phase machine can answer is *who is asked*, *what they may
//! declare*, and what happens to a battle once the declarations are in.
//!
//! These live inside the crate because they need `Game::advance` and the
//! `pub(crate)` action apply steps, exactly like `rules_fixes.rs`.
//!
//! Three Python originals are pinned by name in the integration test instead:
//! `support_from_choices_adds_all_declared_areas`,
//! `siege_support_needs_a_fortified_target` and the persisted-order half of
//! `support_order_is_not_consumed_by_a_battle` (that one is here too, because
//! the phase machine advances the declaration index).

#![cfg(test)]

use std::sync::Arc;

use smallvec::SmallVec;

use crate::action::{Action, ActionKind, Side};
use crate::combat;
use crate::enums::{HouseId, OrderKey, Phase, RegionId, Track, UnitType};
use crate::game::Game;
use crate::map::DataSet;
use crate::setup::GameConfig;
use crate::state::{BattleStage, PendingBattle};

// Adjacent land regions used by the end-to-end tests, as in the Python file.
const TARGET: &str = "highgarden";
const ORIGIN: &str = "the_reach";
const SUPPORT_REGION: &str = "searoad_marches";
const OTHER_SUPPORT_REGION: &str = "oldtown";

const NEUTRAL_TARGET: &str = "kings_landing";
const NEUTRAL_ORIGIN: &str = "kingswood";
const NEUTRAL_SUPPORT_REGION: &str = "the_reach";

fn game(seed: u64) -> Game {
    let data = Arc::new(DataSet::load_default().expect("bundled data must load"));
    Game::new(data, GameConfig::with_seed(seed)).expect("game must start")
}

fn house(name: &str) -> HouseId {
    HouseId::from_name(name).unwrap_or_else(|| panic!("unknown house {name}"))
}

fn region_of(data: &DataSet, id: &str) -> RegionId {
    data.map
        .lookup(id)
        .unwrap_or_else(|| panic!("unknown region {id}"))
}

/// Python's `_clear_board`: the opening position with an empty board.
fn clear_board(game: &mut Game) {
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
        region.neutral_force = 0;
    }
    game.state.pending_battle = None;
    game.state.pending_march = None;
}

/// Python's `_place_support`.
fn place_support(
    game: &mut Game,
    data: &DataSet,
    owner: &str,
    region: &str,
    units: [u8; 4],
    order: &str,
) {
    let id = region_of(data, region);
    let key = OrderKey::from_name(order).unwrap_or_else(|| panic!("unknown order {order}"));
    let region_state = game.state.region_mut(id);
    region_state.controller = Some(house(owner));
    region_state.units = units;
    region_state.order = Some(key);
    region_state.order_house = Some(house(owner));
}

/// Python's `_setup_battle`: a battle parked at the Support stage.
fn setup_battle(
    attacker: &str,
    defender: &str,
    attacker_units: [u8; 4],
    defender_units: [u8; 4],
    supports: &[(&str, &str, [u8; 4], &str)],
) -> (Game, Arc<DataSet>) {
    let mut game = game(0);
    let data = Arc::clone(&game.data);
    clear_board(&mut game);
    let attacker = house(attacker);
    let defender = house(defender);
    let origin = region_of(&data, ORIGIN);
    let target = region_of(&data, TARGET);
    {
        let region = game.state.region_mut(origin);
        region.controller = Some(attacker);
        region.units = attacker_units;
    }
    {
        let region = game.state.region_mut(target);
        region.controller = Some(defender);
        region.units = defender_units;
    }
    for (owner, region, units, order) in supports {
        place_support(&mut game, &data, owner, region, *units, order);
    }
    game.state.phase = Phase::Combat;
    game.state.active_house = None;
    let queue: SmallVec<[RegionId; 8]> = combat::support_orders(&game.state, &data.map, target)
        .into_iter()
        .map(|(region, _)| region)
        .collect();
    game.state.pending_battle = Some(PendingBattle {
        region: target,
        attacker,
        defender: Some(defender),
        neutral: false,
        origin: Some(origin),
        attacker_order: None,
        attacker_units,
        defender_units,
        attacker_order_strength: 0,
        defender_order_strength: 0,
        attacker_card: None,
        defender_card: None,
        attacker_card_chosen: true,
        defender_card_chosen: true,
        doran_owner: None,
        doran_track: None,
        prepared: None,
        blade_done: true,
        attacker_blade: false,
        defender_blade: false,
        stage: BattleStage::Support,
        support_queue: queue,
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
    game.advance().expect("the battle must settle");
    (game, data)
}

/// Python's `_setup_neutral_battle`.
fn setup_neutral_battle(
    attacker_units: [u8; 4],
    supports: &[(&str, &str, [u8; 4], &str)],
) -> (Game, Arc<DataSet>) {
    let mut game = game(0);
    let data = Arc::clone(&game.data);
    clear_board(&mut game);
    let attacker = house("stark");
    let origin = region_of(&data, NEUTRAL_ORIGIN);
    let target = region_of(&data, NEUTRAL_TARGET);
    game.state.region_mut(origin).controller = Some(attacker);
    game.state.region_mut(origin).units = [0; 4];
    game.state.region_mut(target).neutral_force = 5;
    for (owner, region, units, order) in supports {
        place_support(&mut game, &data, owner, region, *units, order);
    }
    game.state.phase = Phase::Combat;
    game.state.active_house = None;
    let queue: SmallVec<[RegionId; 8]> = combat::support_orders(&game.state, &data.map, target)
        .into_iter()
        .map(|(region, _)| region)
        .collect();
    game.state.pending_battle = Some(PendingBattle {
        region: target,
        attacker,
        defender: None,
        neutral: true,
        origin: Some(origin),
        attacker_order: None,
        attacker_units,
        defender_units: [0; 4],
        attacker_order_strength: 0,
        defender_order_strength: 0,
        attacker_card: None,
        defender_card: None,
        attacker_card_chosen: true,
        defender_card_chosen: true,
        doran_owner: None,
        doran_track: None,
        prepared: None,
        blade_done: true,
        attacker_blade: false,
        defender_blade: false,
        stage: BattleStage::Support,
        support_queue: queue,
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
    game.advance().expect("the battle must settle");
    (game, data)
}

fn stage(game: &Game) -> Option<BattleStage> {
    game.state
        .pending_battle
        .as_ref()
        .map(|battle| battle.stage)
}

/// Python's `_finish`: step through any remaining casualties and retreats.
fn finish(game: &mut Game) {
    let mut guard = 0;
    while game.state.pending_battle.is_some() && guard < 20 {
        guard += 1;
        let actions = game.legal_actions();
        if actions.is_empty() {
            break;
        }
        game.step(actions[0]).expect("a legal action applies");
    }
}

fn sides(game: &Game) -> Vec<Side> {
    let mut sides: Vec<Side> = game
        .legal_actions()
        .iter()
        .filter_map(|action| action.side)
        .collect();
    sides.sort_unstable();
    sides.dedup();
    sides
}

// ---------------------------------------------------------------------------
// tests/test_cross_house_support.py (phase-machine half)
// ---------------------------------------------------------------------------

/// Python: `test_third_house_may_back_either_side_or_refuse`.
#[test]
fn third_house_may_back_either_side_or_refuse() {
    let (game, data) = setup_battle(
        "stark",
        "baratheon",
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        &[("tyrell", SUPPORT_REGION, [0, 2, 0, 0], "support_a")],
    );
    let _ = data;
    assert_eq!(stage(&game), Some(BattleStage::Support));
    assert_eq!(game.state.active_house, Some(house("tyrell")));
    assert_eq!(
        sides(&game),
        vec![Side::Attacker, Side::Defender, Side::Refuse]
    );
}

/// Python: `test_a_house_may_not_support_against_its_own_units`.
#[test]
fn a_house_may_not_support_against_its_own_units() {
    // The attacker itself may only back the attacker; likewise for the defender.
    let (game, _data) = setup_battle(
        "stark",
        "baratheon",
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        &[("stark", OTHER_SUPPORT_REGION, [0, 1, 0, 0], "support_a")],
    );
    assert_eq!(game.state.active_house, Some(house("stark")));
    assert_eq!(sides(&game), vec![Side::Attacker, Side::Refuse]);
}

/// Python: `test_third_house_support_flips_a_battle`.
#[test]
fn third_house_support_flips_a_battle() {
    // Without help the attacker (2) loses to the defender (4); two supporting
    // Knights add 4 and win it 6-4.
    let (mut game, data) = setup_battle(
        "stark",
        "baratheon",
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        &[("tyrell", SUPPORT_REGION, [0, 2, 0, 0], "support_a")],
    );
    let support = region_of(&data, SUPPORT_REGION);
    let tyrell = house("tyrell");
    game.step(Action::declare_support(tyrell, support, Side::Attacker))
        .expect("a declaration is legal");
    if let Some(result) = game
        .state
        .pending_battle
        .as_ref()
        .and_then(|b| b.result.as_ref())
    {
        assert_eq!(result.winner, house("stark"));
    }
    finish(&mut game);
    assert_eq!(
        game.state.region(region_of(&data, TARGET)).controller,
        Some(house("stark"))
    );
}

/// Python: `test_refused_support_leaves_the_attacker_short`.
#[test]
fn refused_support_leaves_the_attacker_short() {
    let (mut game, data) = setup_battle(
        "stark",
        "baratheon",
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        &[("tyrell", SUPPORT_REGION, [0, 2, 0, 0], "support_a")],
    );
    let support = region_of(&data, SUPPORT_REGION);
    game.step(Action::declare_support(
        house("tyrell"),
        support,
        Side::Refuse,
    ))
    .expect("a refusal is legal");
    finish(&mut game);
    // The attack failed; the defender keeps the area.
    assert_eq!(
        game.state.region(region_of(&data, TARGET)).controller,
        Some(house("baratheon"))
    );
}

/// Python: `test_support_order_is_not_consumed_by_a_battle`.
#[test]
fn support_order_is_not_consumed_by_a_battle() {
    let (mut game, data) = setup_battle(
        "stark",
        "baratheon",
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        &[("tyrell", SUPPORT_REGION, [0, 2, 0, 0], "support_a")],
    );
    let support = region_of(&data, SUPPORT_REGION);
    game.step(Action::declare_support(
        house("tyrell"),
        support,
        Side::Attacker,
    ))
    .expect("a declaration is legal");
    // While the battle is still resolving, the supporting order is untouched.
    assert_eq!(
        game.state.region(support).order,
        OrderKey::from_name("support_a")
    );
    assert_eq!(
        game.state.region(support).order_house,
        Some(house("tyrell"))
    );
}

/// Python: `test_declarations_follow_iron_throne_order`.
#[test]
fn declarations_follow_iron_throne_order() {
    let (mut game, data) = setup_battle(
        "stark",
        "baratheon",
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        &[
            ("tyrell", SUPPORT_REGION, [0, 1, 0, 0], "support_a"),
            ("martell", OTHER_SUPPORT_REGION, [0, 1, 0, 0], "support_a"),
        ],
    );
    let throne = game.state.track_order(Track::IronThrone);
    let expected = [house("tyrell"), house("martell")]
        .into_iter()
        .min_by_key(|house| {
            throne
                .iter()
                .position(|candidate| candidate == house)
                .unwrap()
        })
        .expect("one of the two");
    assert_eq!(game.state.active_house, Some(expected));

    // Declaring advances the queue to the other supporter.
    let _ = data;
    let first = game.legal_actions();
    assert_eq!(first[0].kind, ActionKind::DeclareSupport);
    game.step(first[0]).expect("the declaration is legal");
    assert_eq!(
        game.state.active_house,
        Some(if expected == house("tyrell") {
            house("martell")
        } else {
            house("tyrell")
        })
    );
    let second = game.legal_actions();
    game.step(second[0]).expect("the declaration is legal");
    assert!(game.state.pending_battle.is_none() || stage(&game) != Some(BattleStage::Support));
}

/// Python: `test_neutral_force_can_never_be_supported`.
#[test]
fn neutral_force_can_never_be_supported() {
    let (game, _data) = setup_neutral_battle(
        [1, 1, 0, 0],
        &[("tyrell", NEUTRAL_SUPPORT_REGION, [0, 2, 0, 0], "support_a")],
    );
    assert_eq!(stage(&game), Some(BattleStage::Support));
    assert_eq!(game.state.active_house, Some(house("tyrell")));
    // Only the attacker side, or refusing; the Neutral Force has no allies.
    assert_eq!(sides(&game), vec![Side::Attacker, Side::Refuse]);
}

/// Python: `test_third_house_support_can_destroy_a_neutral_force`.
#[test]
fn third_house_support_can_destroy_a_neutral_force() {
    let (mut game, data) = setup_neutral_battle(
        [1, 1, 0, 0], // 3
        &[("tyrell", NEUTRAL_SUPPORT_REGION, [0, 2, 0, 0], "support_a")],
    );
    let support = region_of(&data, NEUTRAL_SUPPORT_REGION);
    game.step(Action::declare_support(
        house("tyrell"),
        support,
        Side::Attacker,
    ))
    .expect("a declaration is legal");

    let target = region_of(&data, NEUTRAL_TARGET);
    assert_eq!(game.state.region(target).neutral_force, 0);
    assert_eq!(game.state.region(target).controller, Some(house("stark")));
}

/// Python: `test_neutral_attack_fails_without_support`.
#[test]
fn neutral_attack_fails_without_support() {
    let (mut game, data) = setup_neutral_battle(
        [1, 1, 0, 0], // 3 < 5
        &[("tyrell", NEUTRAL_SUPPORT_REGION, [0, 2, 0, 0], "support_a")],
    );
    let support = region_of(&data, NEUTRAL_SUPPORT_REGION);
    game.step(Action::declare_support(
        house("tyrell"),
        support,
        Side::Refuse,
    ))
    .expect("a refusal is legal");

    assert_eq!(
        game.state
            .region(region_of(&data, NEUTRAL_TARGET))
            .neutral_force,
        5
    );
    assert_eq!(
        game.state
            .region(region_of(&data, NEUTRAL_ORIGIN))
            .controller,
        Some(house("stark"))
    );
}

/// Python: `test_march_against_a_neutral_force_offers_support_stage`.
#[test]
fn march_against_a_neutral_force_offers_support_stage() {
    // A real march into King's Landing must pause for an adjacent Support order
    // rather than resolving immediately.
    let mut game = game(0);
    let data = Arc::clone(&game.data);
    clear_board(&mut game);
    let origin = region_of(&data, NEUTRAL_ORIGIN);
    let target = region_of(&data, NEUTRAL_TARGET);
    {
        let region = game.state.region_mut(origin);
        region.controller = Some(house("stark"));
        region.order = OrderKey::from_name("march");
        region.order_house = Some(house("stark"));
        region.units = [0, 3, 0, 0];
    }
    game.state.region_mut(target).neutral_force = 5;
    place_support(
        &mut game,
        &data,
        "tyrell",
        NEUTRAL_SUPPORT_REGION,
        [0, 2, 0, 0],
        "support_a",
    );

    let actions: Vec<Action> = game
        .march_actions(house("stark"), origin)
        .into_iter()
        .filter(|action| action.kind == ActionKind::March)
        .collect();
    assert!(!actions.is_empty(), "the march must be offered");
    game.apply_march(actions[0]).expect("the march is legal");
    assert_eq!(game.state.phase, Phase::Combat);
    assert_eq!(stage(&game), Some(BattleStage::Support));
}

/// Guards the regions this module relies on being adjacent.
#[test]
fn the_regions_are_adjacent_as_the_tests_assume() {
    let data = DataSet::load_default().expect("bundled data must load");
    let neighbours = |id: &str| -> Vec<&str> {
        data.map
            .neighbours_of(region_of(&data, id))
            .iter()
            .map(|region| data.map.id_of(*region))
            .collect()
    };
    for (region, neighbour) in [
        (TARGET, ORIGIN),
        (TARGET, SUPPORT_REGION),
        (TARGET, OTHER_SUPPORT_REGION),
        (NEUTRAL_TARGET, NEUTRAL_SUPPORT_REGION),
    ] {
        assert!(
            neighbours(region).contains(&neighbour),
            "{region} must neighbour {neighbour}"
        );
    }
    let _ = UnitType::Footman;
}
