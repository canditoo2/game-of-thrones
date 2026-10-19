//! Combat and House-card semantics, ported from the Python suite.
//!
//! The three Python files this file mirrors are
//! `tests/test_combat.py` (battle resolution, siege engines, the Fiefdoms
//! tiebreak, retreats), `tests/test_cross_house_support.py` (who may support
//! whom, how much support is worth, the Neutral-Force march) and
//! `tests/test_player_choice_combat.py` (the defeated player picks the casualty
//! and the retreat).
//!
//! The Python originals drive the `Game` phase machine. Here the positions are
//! built directly on a `State` (`region.controller`, `units`, `order`, tracks)
//! and the `combat` / `house_cards` functions are called the way the phase
//! machine calls them, which is what makes these tests independent of the
//! concurrently ported `action_phase.rs`.
//!
//! A few Python tests cannot be expressed at this level, and are named here so
//! the gap is explicit rather than silently dropped:
//!
//! * "a third house may back either side or refuse" and "a house may not support
//!   against its own units" are about the *action list* the phase machine
//!   offers (`Game._combat_actions`); `combat.rs` never restricts a declaration,
//!   it adds the side it is given. `support_from_choices_honours_the_declared_side`
//!   pins that boundary.
//! * "a Neutral Force can never be supported" is the same phase-machine rule
//!   (`battle["neutral"]` selects `("attacker", "refuse")`).
//! * the casualty/retreat *stages* are phase-machine state; the engine-level
//!   halves (casualty count, which units are legal picks, the retreat options)
//!   are all covered below.

use agot_core::action::Side;
use agot_core::combat::{
    apply_battle_result, apply_casualties, destroy_army, legal_retreats, order_modifier,
    potential_support_strength, resolve_and_apply, resolve_battle, retreat, retreat_keeps_supply,
    retreat_options, support_from_choices, support_orders, unit_strength, units_strength,
    BattleRequest,
};
use agot_core::enums::{CardId, HouseId, OrderKey, RegionId, Track, UnitType};
use agot_core::house_cards::{self, BattleContext};
use agot_core::map::{CardSpec, DataSet, GameMap, HouseCardData};
use agot_core::rng::Rng;
use agot_core::setup::{initial_state, GameConfig};
use agot_core::state::State;
use smallvec::{smallvec, SmallVec};

// ---------------------------------------------------------------------------
// Position builders
// ---------------------------------------------------------------------------

struct Board {
    data: DataSet,
}

impl Board {
    fn new() -> Board {
        Board {
            data: DataSet::load_default().expect("bundled data must load"),
        }
    }

    fn map(&self) -> &GameMap {
        &self.data.map
    }

    fn cards(&self) -> &HouseCardData {
        &self.data.house_cards
    }

    fn house(&self, name: &str) -> HouseId {
        HouseId::from_name(name).expect("known house")
    }

    fn region(&self, name: &str) -> RegionId {
        self.data.map.lookup(name).expect("known region")
    }

    fn card(&self, house: &str, card: &str) -> CardId {
        self.data
            .house_cards
            .card_index(self.house(house), card)
            .expect("known card")
    }

    /// The Python `_clean_game` / `_clear_board` position: the real opening
    /// position with every region emptied of units, orders and tokens.
    fn clean_state(&self) -> State {
        let config = GameConfig::with_seed(0);
        let mut rng = Rng::from_seed(config.seed);
        let mut state = initial_state(&self.data, &config, &mut rng);
        // Python's GameState always carries a log; the port only fills it when
        // the caller asked for one.
        state.log = Some(Vec::new());
        let ids: Vec<RegionId> = state.region_ids().collect();
        for id in ids {
            let region = state.region_mut(id);
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
        for house in HouseId::ALL {
            if let Some(house_state) = state.house_mut(house) {
                house_state.orders_placed.clear();
                house_state.orders_available = OrderKey::ALL.into_iter().collect();
            }
        }
        state
    }
}

/// The Python helper: the first land region in board order and its first land
/// neighbour.
fn adjacent_land_pair(map: &GameMap) -> (RegionId, RegionId) {
    for index in 0..map.region_count() as u16 {
        let region = RegionId(index);
        if !map.region(region).is_land() {
            continue;
        }
        if let Some(neighbour) = map
            .neighbours_of(region)
            .iter()
            .find(|other| map.region(**other).is_land())
        {
            return (region, *neighbour);
        }
    }
    panic!("no adjacent land regions");
}

/// `_place_support`: a house's Support order, with the units that carry it.
fn place_support(state: &mut State, house: HouseId, region: RegionId, units: [u8; 4], order: &str) {
    let region_state = state.region_mut(region);
    region_state.controller = Some(house);
    region_state.units = units;
    region_state.order = Some(OrderKey::from_name(order).expect("known order"));
    region_state.order_house = Some(house);
}

/// Python's `max(iterable, key=...)` keeps the *first* maximum.
fn max_swords(deck: &[CardSpec]) -> &CardSpec {
    deck.iter()
        .reduce(|best, card| {
            if card.swords > best.swords {
                card
            } else {
                best
            }
        })
        .expect("a deck has cards")
}

/// Python's `max(iterable, key=...)` keeps the *first* maximum.
fn max_forts(deck: &[CardSpec]) -> &CardSpec {
    deck.iter()
        .reduce(|best, card| if card.forts > best.forts { card } else { best })
        .expect("a deck has cards")
}

// ---------------------------------------------------------------------------
// tests/test_combat.py
// ---------------------------------------------------------------------------

#[test]
fn strength_counts_units_and_march_order() {
    let board = Board::new();
    let state = board.clean_state();
    let (_origin, target) = adjacent_land_pair(board.map());
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));

    let mut request = BattleRequest::new(target, stark, lannister, [3, 1, 0, 0], [2, 0, 0, 0]);
    request.attacker_order_strength = 1;
    request.defender_order_strength = 1;
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    // 3 Footmen (3) + 1 Knight (2) + the March order (1).
    assert_eq!(result.attacker.total(), 3 + 2 + 1);
    assert_eq!(result.defender.total(), 2 + 1);
    assert_eq!(result.winner, stark);
}

#[test]
fn merging_forces_would_be_a_bug_regression() {
    // The defender's units must never be credited to the attacker.
    let board = Board::new();
    let state = board.clean_state();
    let (_origin, target) = adjacent_land_pair(board.map());
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));

    let request = BattleRequest::new(target, stark, lannister, [1, 0, 0, 0], [0, 4, 0, 0]);
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert_eq!(result.attacker.total(), 1);
    assert_eq!(result.defender.total(), 8);
    assert_eq!(result.winner, lannister);
}

#[test]
fn ties_go_to_the_higher_fiefdoms_position() {
    let board = Board::new();
    let state = board.clean_state();
    let (_origin, target) = adjacent_land_pair(board.map());
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));

    let request = BattleRequest::new(target, stark, lannister, [2, 0, 0, 0], [2, 0, 0, 0]);
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert!(result.tiebreak);
    let stark_position = state.track_position(stark, Track::Fiefdoms);
    let lannister_position = state.track_position(lannister, Track::Fiefdoms);
    let expected = if stark_position < lannister_position {
        stark
    } else {
        lannister
    };
    assert_eq!(result.winner, expected);
    assert!(result
        .notes
        .iter()
        .any(|note| note.contains("decided on the Fiefdoms track")));
}

#[test]
fn casualties_are_swords_minus_forts() {
    let board = Board::new();
    let state = board.clean_state();
    let (_origin, target) = adjacent_land_pair(board.map());
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));

    let attacker_spec = max_swords(board.cards().deck(stark));
    let defender_spec = max_forts(board.cards().deck(lannister));
    let attacker_card = board
        .cards()
        .card_index(stark, &attacker_spec.id)
        .expect("card in the deck");
    let defender_card = board
        .cards()
        .card_index(lannister, &defender_spec.id)
        .expect("card in the deck");

    let mut request = BattleRequest::new(target, stark, lannister, [5, 0, 0, 0], [1, 0, 0, 0]);
    request.attacker_card = Some(attacker_card);
    request.defender_card = Some(defender_card);
    request.attacker_order_strength = 1;
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert_eq!(result.winner, stark);
    assert_eq!(
        result.casualties as i32,
        (attacker_spec.swords as i32 - defender_spec.forts as i32).max(0)
    );
    // Eddard Stark (2 Swords) against The Hound (2 Fortifications) is a wash.
    assert_eq!(result.attacker.swords, 2);
    assert_eq!(result.defender.forts, 2);
    assert_eq!(result.casualties, 0);
}

#[test]
fn losing_attacker_retreats_to_its_origin() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (origin, target) = adjacent_land_pair(board.map());
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));
    // The army has already marched out.
    state.region_mut(origin).units = [0; 4];

    let mut request = BattleRequest::new(target, stark, lannister, [1, 0, 0, 0], [0, 3, 0, 0]);
    request.attacker_origin = Some(origin);
    let result = resolve_and_apply(&mut state, board.map(), board.cards(), &request);

    assert_eq!(result.winner, lannister);
    assert_eq!(state.region(target).controller, Some(lannister));
    // The surviving attacker is routed in its origin region.
    assert_eq!(state.region(origin).routed_count(UnitType::Footman), 1);
    assert_eq!(state.region(origin).controller, Some(stark));
}

// ---------------------------------------------------------------------------
// tests/test_cross_house_support.py
// ---------------------------------------------------------------------------

#[test]
fn support_from_choices_adds_all_declared_areas() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (tyrell, martell) = (board.house("tyrell"), board.house("martell"));
    let (target, support, other) = (
        board.region("highgarden"),
        board.region("searoad_marches"),
        board.region("oldtown"),
    );

    place_support(&mut state, tyrell, support, [0, 2, 0, 0], "support_a");
    place_support(&mut state, martell, other, [1, 0, 0, 0], "support_plus");

    let totals = support_from_choices(
        &state,
        board.map(),
        target,
        &[(support, Side::Attacker), (other, Side::Defender)],
    );
    assert_eq!(totals.attacker_strength, 4); // 2 Knights
    assert_eq!(totals.attacker_units, [0, 2, 0, 0]);
    assert_eq!(totals.defender_strength, 2); // 1 Footman + the starred Support order
    assert_eq!(totals.defender_units, [1, 0, 0, 0]);
}

#[test]
fn siege_support_needs_a_fortified_target() {
    let board = Board::new();
    let mut state = board.clean_state();
    let tyrell = board.house("tyrell");
    let (target, support) = (board.region("highgarden"), board.region("searoad_marches"));
    let blackwater = board.region("blackwater");

    place_support(&mut state, tyrell, support, [0, 0, 1, 0], "support_a");

    // Against Highgarden (a Stronghold) the Siege Engine adds its 4.
    let against_fort =
        support_from_choices(&state, board.map(), target, &[(support, Side::Attacker)]);
    assert_eq!(against_fort.attacker_strength, 4);
    // Against a plain land area, and when backing the defender, it adds nothing.
    let against_open = support_from_choices(
        &state,
        board.map(),
        blackwater,
        &[(support, Side::Attacker)],
    );
    assert_eq!(against_open.attacker_strength, 0);
    let backing_defender =
        support_from_choices(&state, board.map(), target, &[(support, Side::Defender)]);
    assert_eq!(backing_defender.attacker_strength, 0);
    assert_eq!(backing_defender.defender_strength, 0);
}

#[test]
fn third_house_support_flips_a_battle() {
    // Without help the attacker (2) loses to the defender (4); two supporting
    // Knights add 4 and win it 6-4.
    let board = Board::new();
    let mut state = board.clean_state();
    let (stark, baratheon, tyrell) = (
        board.house("stark"),
        board.house("baratheon"),
        board.house("tyrell"),
    );
    let (origin, target, support) = (
        board.region("the_reach"),
        board.region("highgarden"),
        board.region("searoad_marches"),
    );
    state.region_mut(origin).controller = Some(stark);
    state.region_mut(target).controller = Some(baratheon);
    place_support(&mut state, tyrell, support, [0, 2, 0, 0], "support_a");

    let mut request = BattleRequest::new(target, stark, baratheon, [2, 0, 0, 0], [4, 0, 0, 0]);
    request.attacker_origin = Some(origin);
    request.support_choices = Some(smallvec![(support, Side::Attacker)]);
    let result = resolve_and_apply(&mut state, board.map(), board.cards(), &request);

    assert_eq!(result.attacker.total(), 6);
    assert_eq!(result.defender.total(), 4);
    assert_eq!(result.winner, stark);
    assert_eq!(state.region(target).controller, Some(stark));
}

#[test]
fn refused_support_leaves_the_attacker_short() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (stark, baratheon, tyrell) = (
        board.house("stark"),
        board.house("baratheon"),
        board.house("tyrell"),
    );
    let (origin, target, support) = (
        board.region("the_reach"),
        board.region("highgarden"),
        board.region("searoad_marches"),
    );
    state.region_mut(origin).controller = Some(stark);
    state.region_mut(target).controller = Some(baratheon);
    place_support(&mut state, tyrell, support, [0, 2, 0, 0], "support_a");

    let mut request = BattleRequest::new(target, stark, baratheon, [2, 0, 0, 0], [4, 0, 0, 0]);
    request.attacker_origin = Some(origin);
    request.support_choices = Some(smallvec![(support, Side::Refuse)]);
    let result = resolve_and_apply(&mut state, board.map(), board.cards(), &request);

    assert_eq!(result.winner, baratheon);
    // The attack failed; the defender keeps the area.
    assert_eq!(state.region(target).controller, Some(baratheon));
}

#[test]
fn support_order_is_not_consumed_by_a_battle() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (stark, baratheon, tyrell) = (
        board.house("stark"),
        board.house("baratheon"),
        board.house("tyrell"),
    );
    let (origin, target, support) = (
        board.region("the_reach"),
        board.region("highgarden"),
        board.region("searoad_marches"),
    );
    state.region_mut(origin).controller = Some(stark);
    state.region_mut(target).controller = Some(baratheon);
    place_support(&mut state, tyrell, support, [0, 2, 0, 0], "support_a");

    let mut request = BattleRequest::new(target, stark, baratheon, [2, 0, 0, 0], [4, 0, 0, 0]);
    request.attacker_origin = Some(origin);
    request.support_choices = Some(smallvec![(support, Side::Attacker)]);
    resolve_and_apply(&mut state, board.map(), board.cards(), &request);

    // The supporting order is untouched, wherever the battle went.
    assert_eq!(state.region(support).order, Some(OrderKey(6))); // support_a
    assert_eq!(state.region(support).order_house, Some(tyrell));
    assert_eq!(state.region(support).unit_count(UnitType::Knight), 2);
}

#[test]
fn declarations_follow_iron_throne_order() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (tyrell, martell) = (board.house("tyrell"), board.house("martell"));
    let (target, first, second) = (
        board.region("highgarden"),
        board.region("searoad_marches"),
        board.region("oldtown"),
    );
    place_support(&mut state, tyrell, first, [0, 1, 0, 0], "support_a");
    place_support(&mut state, martell, second, [0, 1, 0, 0], "support_a");

    let orders = support_orders(&state, board.map(), target);
    assert_eq!(orders.len(), 2);
    let throne = state.track_order(Track::IronThrone);
    let tyrell_position = throne.iter().position(|house| *house == tyrell).unwrap();
    let martell_position = throne.iter().position(|house| *house == martell).unwrap();
    // Python's `min(..., key=throne.index)` keeps the first of a tie.
    let expected = if tyrell_position <= martell_position {
        tyrell
    } else {
        martell
    };
    assert_eq!(orders[0].1, expected);
}

#[test]
fn support_from_choices_honours_the_declared_side_only() {
    // "A house may not support against its own units" is a phase-machine rule
    // (`Game._combat_actions`), not a `combat.py` one: the engine adds whatever
    // side a declaration names, and the caller is what keeps it legal.
    let board = Board::new();
    let mut state = board.clean_state();
    let tyrell = board.house("tyrell");
    let (target, support) = (board.region("highgarden"), board.region("searoad_marches"));
    place_support(&mut state, tyrell, support, [0, 2, 0, 0], "support_a");

    let totals = support_from_choices(&state, board.map(), target, &[(support, Side::Defender)]);
    assert_eq!(totals.attacker_strength, 0);
    assert_eq!(totals.defender_strength, 4);
    // An area whose order has since gone contributes nothing at all.
    state.region_mut(support).order = None;
    let gone = support_from_choices(&state, board.map(), target, &[(support, Side::Defender)]);
    assert_eq!(gone.defender_strength, 0);
}

#[test]
fn neutral_attack_strength_counts_declared_support() {
    // `Game._finalize_neutral_battle` adds the declared support to the moving
    // units; the Neutral Force can never be supported in return.
    let board = Board::new();
    let mut state = board.clean_state();
    let tyrell = board.house("tyrell");
    let kings_landing = board.region("kings_landing");
    let the_reach = board.region("the_reach");
    state.region_mut(kings_landing).neutral_force = 5;
    place_support(&mut state, tyrell, the_reach, [0, 2, 0, 0], "support_a");

    let moving = [1, 1, 0, 0]; // 3
    let is_fort = board.map().region(kings_landing).is_fortification();
    let support = support_from_choices(
        &state,
        board.map(),
        kings_landing,
        &[(the_reach, Side::Attacker)],
    )
    .attacker_strength;
    let with_support = units_strength(&moving, is_fort) as i32 + support as i32;
    assert_eq!(with_support, 7);
    assert!(with_support >= 5, "the token is destroyed");

    let refused = support_from_choices(
        &state,
        board.map(),
        kings_landing,
        &[(the_reach, Side::Refuse)],
    )
    .attacker_strength;
    let without_support = units_strength(&moving, is_fort) as i32 + refused as i32;
    assert_eq!(without_support, 3);
    assert!(without_support < 5, "the march is cancelled");
}

#[test]
fn potential_support_makes_a_neutral_march_look_winnable() {
    // `Game._neutral_attack_strength` uses the *upper bound* of support while
    // generating legal marches, so an undeclared Support order already counts.
    let board = Board::new();
    let mut state = board.clean_state();
    let (stark, tyrell) = (board.house("stark"), board.house("tyrell"));
    let (origin, target, support) = (
        board.region("kingswood"),
        board.region("kings_landing"),
        board.region("the_reach"),
    );
    state.region_mut(origin).controller = Some(stark);
    state.region_mut(origin).order = Some(OrderKey::from_name("march").unwrap());
    state.region_mut(origin).order_house = Some(stark);
    state.region_mut(target).neutral_force = 5;
    place_support(&mut state, tyrell, support, [0, 2, 0, 0], "support_a");

    let moving = [0, 3, 0, 0]; // 6
    let is_fort = board.map().region(target).is_fortification();
    let potential = potential_support_strength(&state, board.map(), target, true);
    assert_eq!(potential, 4);
    let order = order_modifier(&state, origin, stark, true);
    assert_eq!(order, 0, "a plain March order adds nothing");
    let strength = units_strength(&moving, is_fort) as i32 + order as i32 + potential as i32;
    assert_eq!(strength, 10);
    assert!(strength >= 5, "the march must be offered");
}

#[test]
fn ships_only_support_where_the_rules_allow_it() {
    // `_contributing_units`: Ships in an adjacent sea may support a land battle;
    // Ships in a Port may support the adjacent sea, but never a land area;
    // Footmen/Knights/Siege may never support a battle in a sea area.
    let board = Board::new();
    let mut state = board.clean_state();
    let tyrell = board.house("tyrell");
    let (highgarden, sea, port, oldtown) = (
        board.region("highgarden"),
        board.region("redwyne_straits"),
        board.region("oldtown_port"),
        board.region("oldtown"),
    );

    place_support(&mut state, tyrell, sea, [0, 0, 0, 2], "support_a");
    let land = support_from_choices(&state, board.map(), highgarden, &[(sea, Side::Attacker)]);
    assert_eq!(land.attacker_strength, 2);
    assert_eq!(land.attacker_units, [0, 0, 0, 2]);

    // A Port's ships never support a land battle.
    place_support(&mut state, tyrell, port, [0, 0, 0, 2], "support_a");
    let land = support_from_choices(&state, board.map(), highgarden, &[(port, Side::Attacker)]);
    assert_eq!(land.attacker_strength, 0);

    // ...but they do support the adjacent sea, unlike land units.
    place_support(&mut state, tyrell, oldtown, [2, 0, 0, 0], "support_a");
    let sea_battle = support_from_choices(&state, board.map(), sea, &[(oldtown, Side::Attacker)]);
    assert_eq!(sea_battle.attacker_strength, 0);
    let sea_battle = support_from_choices(&state, board.map(), sea, &[(port, Side::Attacker)]);
    assert_eq!(sea_battle.attacker_strength, 2);
}

#[test]
fn routed_ships_still_transport_a_retreat() {
    // `_ship_transport_targets`: a land force may retreat through a friendly
    // Ship chain, and routed Ships may still transport (rulebook page 23).
    let board = Board::new();
    let mut state = board.clean_state();
    let tyrell = board.house("tyrell");
    let (oldtown, sea, the_arbor) = (
        board.region("oldtown"),
        board.region("redwyne_straits"),
        board.region("the_arbor"),
    );
    state.region_mut(oldtown).controller = Some(tyrell);
    state.region_mut(oldtown).owner = Some(tyrell);

    // Without a fleet The Arbor is out of reach: it is not adjacent to Oldtown.
    let options = legal_retreats(&state, board.map(), oldtown, tyrell, false, None);
    assert!(!options.contains(&the_arbor));

    // A *routed* friendly Ship in the straits still transports.
    state.region_mut(sea).controller = Some(tyrell);
    state.region_mut(sea).routed = [0, 0, 0, 1];
    let options = legal_retreats(&state, board.map(), oldtown, tyrell, false, None);
    assert!(
        options.contains(&the_arbor),
        "routed ships may still transport"
    );

    // A Ship of another house does not.
    state.region_mut(sea).controller = Some(board.house("lannister"));
    let options = legal_retreats(&state, board.map(), oldtown, tyrell, false, None);
    assert!(!options.contains(&the_arbor));
}

// ---------------------------------------------------------------------------
// tests/test_player_choice_combat.py
// ---------------------------------------------------------------------------

/// Baratheon's Melisandre carries one Sword icon and no Fortification icons.
fn melisandre_attack(board: &Board, origin: RegionId, target: RegionId) -> BattleRequest {
    let (baratheon, stark) = (board.house("baratheon"), board.house("stark"));
    let mut request = BattleRequest::new(target, baratheon, stark, [4, 0, 0, 0], [1, 1, 0, 0]);
    request.attacker_card = Some(board.card("baratheon", "melisandre"));
    request.attacker_origin = Some(origin);
    request
}

#[test]
fn the_defeated_player_chooses_which_unit_dies() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (origin, target) = adjacent_land_pair(board.map());
    let (baratheon, stark) = (board.house("baratheon"), board.house("stark"));
    state.region_mut(origin).controller = Some(baratheon);
    state.region_mut(target).controller = Some(stark);

    let request = melisandre_attack(&board, origin, target);
    let mut result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert_eq!(result.winner, baratheon);
    assert_eq!(result.loser, stark);
    assert_eq!(result.casualties, 1);
    // Both units are legal picks, so the player may sacrifice either.
    assert_eq!(result.context.units_of(stark), &[1, 1, 0, 0]);
    assert!(result.context.remove_unit(stark, UnitType::Footman));
    assert_eq!(result.context.units_of(stark), &[0, 1, 0, 0]);
    // A unit that is not there cannot be picked.
    assert!(!result.context.remove_unit(stark, UnitType::Footman));
    assert!(!result.context.remove_unit(stark, UnitType::Ship));

    apply_battle_result(&mut state, board.map(), board.cards(), &result, 0, None);
    assert_eq!(state.region(target).controller, Some(baratheon));
    assert_eq!(state.region(target).units, [4, 0, 0, 0]);
}

#[test]
fn last_defender_destroyed_area_falls() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (origin, target) = adjacent_land_pair(board.map());
    let (baratheon, stark) = (board.house("baratheon"), board.house("stark"));
    state.region_mut(origin).controller = Some(baratheon);
    state.region_mut(target).controller = Some(stark);

    let mut request = melisandre_attack(&board, origin, target);
    request.defender_units = [1, 0, 0, 0];
    let mut result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert_eq!(result.casualties, 1);
    assert!(result.context.remove_unit(stark, UnitType::Footman));
    assert!(result
        .context
        .units_of(stark)
        .iter()
        .all(|count| *count == 0));

    apply_battle_result(&mut state, board.map(), board.cards(), &result, 0, None);
    assert_eq!(state.region(target).controller, Some(baratheon));
    assert_eq!(state.region(target).units, [4, 0, 0, 0]);
    assert_eq!(state.region(target).routed, [0; 4]);
}

#[test]
fn retreat_destination_is_a_player_choice() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (origin, target) = adjacent_land_pair(board.map());
    let (baratheon, stark) = (board.house("baratheon"), board.house("stark"));
    state.region_mut(origin).controller = Some(baratheon);
    state.region_mut(target).controller = Some(stark);

    let mut request = melisandre_attack(&board, origin, target);
    request.defender_units = [0, 2, 0, 0];
    let mut result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert_eq!(result.casualties, 1);
    assert!(result.context.remove_unit(stark, UnitType::Knight));
    let survivors = *result.context.units_of(stark);
    assert_eq!(survivors, [0, 1, 0, 0]);

    let options = retreat_options(
        &state,
        board.map(),
        target,
        stark,
        &survivors,
        result.context.attacker_origin,
    );
    assert!(!options.is_empty());
    let chosen = options[0];
    assert!(!options.contains(&origin), "never back into the attack");

    apply_battle_result(
        &mut state,
        board.map(),
        board.cards(),
        &result,
        0,
        Some(chosen),
    );
    assert_eq!(state.region(chosen).routed_count(UnitType::Knight), 1);
    assert_eq!(state.region(target).controller, Some(baratheon));
}

#[test]
fn blackfish_prevents_casualties_entirely() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (origin, target) = adjacent_land_pair(board.map());
    let (baratheon, stark) = (board.house("baratheon"), board.house("stark"));
    state.region_mut(origin).controller = Some(baratheon);
    state.region_mut(target).controller = Some(stark);

    let mut request = melisandre_attack(&board, origin, target);
    request.defender_units = [2, 0, 0, 0];
    request.defender_card = Some(board.card("stark", "blackfish"));
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    // The Blackfish takes no casualties, so the phase machine skips the
    // casualty step entirely.
    assert_eq!(result.casualties, 0);
    assert_eq!(result.loser, stark);
    assert_eq!(result.context.units_of(stark), &[2, 0, 0, 0]);
    assert!(result
        .notes
        .iter()
        .any(|note| note.contains("ignores casualties (The Blackfish)")));
}

#[test]
fn robb_stark_lets_the_winner_pick_the_retreat() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (origin, target) = adjacent_land_pair(board.map());
    let (stark, martell) = (board.house("stark"), board.house("martell"));
    state.region_mut(origin).controller = Some(stark);
    state.region_mut(target).controller = Some(martell);

    let mut request = BattleRequest::new(target, stark, martell, [4, 0, 0, 0], [2, 0, 0, 0]);
    request.attacker_card = Some(board.card("stark", "robb_stark"));
    request.attacker_origin = Some(origin);
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert_eq!(result.winner, stark);
    assert_eq!(result.casualties, 0, "Robb carries no Sword icons");
    assert!(house_cards::robb_chose_retreat(
        &result.context,
        board.cards(),
        result.winner
    ));

    // The winner picks, but only among the areas where the loser keeps Supply.
    let survivors = *result.context.units_of(result.loser);
    let options = retreat_options(
        &state,
        board.map(),
        target,
        martell,
        &survivors,
        result.context.attacker_origin,
    );
    assert!(!options.is_empty());
    let minimal: SmallVec<[RegionId; 8]> = options
        .iter()
        .copied()
        .filter(|option| retreat_keeps_supply(&state, board.map(), *option, martell, &survivors))
        .collect();
    let chosen = if minimal.is_empty() {
        options[0]
    } else {
        minimal[0]
    };

    apply_battle_result(
        &mut state,
        board.map(),
        board.cards(),
        &result,
        0,
        Some(chosen),
    );
    assert_eq!(state.region(target).controller, Some(stark));
    assert_eq!(state.region(chosen).routed_count(UnitType::Footman), 2);
}

#[test]
fn attacker_loss_does_not_offer_a_retreat_choice() {
    // A defeated attacker always retreats to where it marched from, so the
    // engine must not ask it to pick an area: a destination handed to
    // `apply_battle_result` is ignored.
    let board = Board::new();
    let mut state = board.clean_state();
    let (origin, target) = adjacent_land_pair(board.map());
    let (stark, baratheon) = (board.house("stark"), board.house("baratheon"));
    state.region_mut(origin).controller = Some(stark);
    state.region_mut(target).controller = Some(baratheon);

    let mut request = BattleRequest::new(target, stark, baratheon, [1, 0, 0, 0], [4, 0, 0, 0]);
    request.attacker_origin = Some(origin);
    let result = resolve_battle(&state, board.map(), board.cards(), &request);
    assert_eq!(result.winner, baratheon);

    let elsewhere = board.region("kings_landing");
    apply_battle_result(
        &mut state,
        board.map(),
        board.cards(),
        &result,
        0,
        Some(elsewhere),
    );

    assert_eq!(state.region(origin).routed_count(UnitType::Footman), 1);
    assert_eq!(state.region(origin).controller, Some(stark));
    assert_eq!(state.region(elsewhere).routed, [0; 4]);
    assert_eq!(state.region(target).controller, Some(baratheon));
    assert_eq!(state.region(target).units, [4, 0, 0, 0]);
}

// ---------------------------------------------------------------------------
// Region-level helpers the phase machine also uses
// ---------------------------------------------------------------------------

#[test]
fn unit_strength_follows_control_and_siege_rules() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));
    let winterfell = board.region("winterfell");
    state.region_mut(winterfell).controller = Some(stark);
    state.region_mut(winterfell).units = [1, 0, 1, 0];

    assert_eq!(unit_strength(&state, winterfell, stark, false), 1);
    assert_eq!(unit_strength(&state, winterfell, stark, true), 5);
    // Another house controls nothing here.
    assert_eq!(unit_strength(&state, winterfell, lannister, true), 0);
}

#[test]
fn order_modifier_only_pays_the_side_it_belongs_to() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));
    let winterfell = board.region("winterfell");
    state.region_mut(winterfell).order = Some(OrderKey::from_name("defend_2").unwrap());
    state.region_mut(winterfell).order_house = Some(stark);

    assert_eq!(order_modifier(&state, winterfell, stark, false), 2);
    assert_eq!(order_modifier(&state, winterfell, stark, true), 0);
    assert_eq!(order_modifier(&state, winterfell, lannister, false), 0);

    state.region_mut(winterfell).order = Some(OrderKey::from_name("march_minus").unwrap());
    assert_eq!(order_modifier(&state, winterfell, stark, true), -1);
    assert_eq!(order_modifier(&state, winterfell, stark, false), 0);
}

#[test]
fn apply_casualties_removes_cheapest_first_and_releases_the_area() {
    let board = Board::new();
    let mut state = board.clean_state();
    let stark = board.house("stark");
    let winterfell = board.region("winterfell");
    state.region_mut(winterfell).controller = Some(stark);
    state.region_mut(winterfell).owner = Some(stark);
    state.region_mut(winterfell).units = [1, 1, 1, 0];
    state.region_mut(winterfell).order = Some(OrderKey(0));
    state.region_mut(winterfell).order_house = Some(stark);

    let removed = apply_casualties(&mut state, winterfell, stark, 2);
    assert_eq!(removed.as_slice(), &[UnitType::Footman, UnitType::Siege]);
    assert_eq!(state.region(winterfell).units, [0, 1, 0, 0]);

    // A house that does not control the area loses nothing.
    assert!(apply_casualties(&mut state, winterfell, board.house("lannister"), 1).is_empty());

    // Emptying it drops control back to the printed owner and clears the order.
    apply_casualties(&mut state, winterfell, stark, 1);
    assert_eq!(state.region(winterfell).units, [0; 4]);
    assert_eq!(state.region(winterfell).controller, Some(stark));
    assert_eq!(state.region(winterfell).order, None);
    assert_eq!(state.region(winterfell).order_house, None);
}

#[test]
fn destroy_army_wipes_the_region_and_its_control() {
    let board = Board::new();
    let mut state = board.clean_state();
    let stark = board.house("stark");
    let lannister = board.house("lannister");
    let winterfell = board.region("winterfell");
    state.region_mut(winterfell).controller = Some(lannister);
    state.region_mut(winterfell).owner = Some(stark);
    state.region_mut(winterfell).influence_token = Some(lannister);
    state.region_mut(winterfell).units = [2, 0, 0, 0];
    state.region_mut(winterfell).routed = [0, 0, 0, 1];
    state.region_mut(winterfell).garrison = 2;
    state.region_mut(winterfell).order = Some(OrderKey(0));
    state.region_mut(winterfell).order_house = Some(lannister);

    destroy_army(&mut state, board.map(), winterfell);

    assert_eq!(state.region(winterfell).units, [0; 4]);
    assert_eq!(state.region(winterfell).routed, [0; 4]);
    assert_eq!(state.region(winterfell).garrison, 0);
    // The influence token outranks the printed owner.
    assert_eq!(state.region(winterfell).controller, Some(lannister));
    assert_eq!(state.region(winterfell).order, None);
}

#[test]
fn retreat_moves_the_army_out_and_routs_it() {
    let board = Board::new();
    let mut state = board.clean_state();
    let stark = board.house("stark");
    let (source, target) = adjacent_land_pair(board.map());
    state.region_mut(source).controller = Some(stark);
    state.region_mut(source).units = [1, 1, 0, 0];
    state.region_mut(source).owner = Some(stark);

    retreat(&mut state, board.map(), source, stark, target, false);

    assert_eq!(state.region(source).units, [0; 4]);
    assert_eq!(state.region(target).routed, [1, 1, 0, 0]);
    assert_eq!(state.region(target).controller, Some(stark));
}

#[test]
fn legal_retreats_skip_enemies_and_the_origin() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));
    let (origin, enemy) = adjacent_land_pair(board.map());
    // Winterfell's first land neighbour is Castle Black; Moat Cailin is the
    // next free one.
    let free = board.region("moat_cailin");
    state.region_mut(origin).controller = Some(stark);
    state.region_mut(enemy).controller = Some(lannister);
    state.region_mut(enemy).units = [1, 0, 0, 0];

    let all = legal_retreats(&state, board.map(), origin, stark, false, None);
    assert!(
        !all.contains(&enemy),
        "an area held by another house is never a retreat"
    );
    assert!(all.contains(&free), "an empty neutral area is");

    let excluding = legal_retreats(&state, board.map(), origin, stark, false, Some(free));
    assert!(
        !excluding.contains(&free),
        "the origin is excluded on request"
    );

    // Enemy presence blocks a retreat even when the area holds no units.
    state.region_mut(enemy).units = [0; 4];
    state.region_mut(enemy).influence_token = Some(lannister);
    let blocked = legal_retreats(&state, board.map(), origin, stark, false, None);
    assert!(!blocked.contains(&enemy), "an enemy Power token blocks it");

    // Every land destination stays on land.
    for option in &all {
        assert!(board.map().region(*option).is_land());
    }
}

#[test]
fn ports_only_accept_their_owners_ships() {
    // `legal_retreats`: a Port can hold at most three ships, and only its owner
    // may use it.
    let board = Board::new();
    let mut state = board.clean_state();
    let tyrell = board.house("tyrell");
    let port = board.region("oldtown_port");
    let sea = board.region("redwyne_straits");

    let options = legal_retreats(&state, board.map(), sea, tyrell, true, None);
    assert!(!options.contains(&port), "an unowned port is closed");
    state.region_mut(port).controller = Some(tyrell);
    let options = legal_retreats(&state, board.map(), sea, tyrell, true, None);
    assert!(options.contains(&port));
    state.region_mut(port).units = [0, 0, 0, 3];
    let options = legal_retreats(&state, board.map(), sea, tyrell, true, None);
    assert!(!options.contains(&port), "three ships already fill it");
    // A land force may never use a port.
    state.region_mut(port).units = [0; 4];
    let options = legal_retreats(&state, board.map(), sea, tyrell, false, None);
    assert!(!options.contains(&port));
}

#[test]
fn a_lone_siege_engine_cannot_retreat() {
    let board = Board::new();
    let mut state = board.clean_state();
    let stark = board.house("stark");
    let (origin, _target) = adjacent_land_pair(board.map());
    state.region_mut(origin).controller = Some(stark);

    let options = retreat_options(&state, board.map(), origin, stark, &[0, 0, 1, 0], None);
    assert!(options.is_empty());
    let options = retreat_options(&state, board.map(), origin, stark, &[0; 4], None);
    assert!(options.is_empty());
    // A Footman next to it can retreat.
    let options = retreat_options(&state, board.map(), origin, stark, &[1, 0, 1, 0], None);
    assert!(!options.is_empty());
}

// ---------------------------------------------------------------------------
// House-card abilities
// ---------------------------------------------------------------------------

#[test]
fn queen_of_thorns_defers_the_order_removal() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (tyrell, lannister) = (board.house("tyrell"), board.house("lannister"));
    let target = board.region("highgarden");
    let order_region = board.region("oldtown");
    let support = OrderKey::from_name("support_a").unwrap();

    state.region_mut(target).controller = Some(lannister);
    state.region_mut(order_region).controller = Some(lannister);
    state.region_mut(order_region).order = Some(support);
    state.region_mut(order_region).order_house = Some(lannister);
    state.house_mut(lannister).unwrap().take_order(support);

    let mut request = BattleRequest::new(target, tyrell, lannister, [1, 0, 0, 0], [1, 0, 0, 0]);
    request.attacker_card = Some(board.card("tyrell", "queen_of_thorns"));
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    // The pure resolution only records the removal.
    assert_eq!(result.context.orders_to_remove.as_slice(), &[order_region]);
    assert_eq!(state.region(order_region).order, Some(support));
    assert!(state.house(lannister).unwrap().has_order_placed(support));

    house_cards::commit_deferred(&mut state, board.cards(), &result.context);
    assert_eq!(state.region(order_region).order, None);
    assert_eq!(state.region(order_region).order_house, None);
    // Python takes the token out of the placed pool; it returns to the pool at
    // Cleanup, not here.
    assert!(!state.house(lannister).unwrap().has_order_placed(support));
}

#[test]
fn tyrion_swap_is_deferred_and_returns_the_card_to_hand() {
    // Tyrion forces the opponent to replace a card: the best card *left in the
    // opponent's hand* replaces the one they played, which only happens when the
    // played card was their strongest (`card_score(best) < card_score(old)`).
    let board = Board::new();
    let mut state = board.clean_state();
    let (lannister, stark) = (board.house("lannister"), board.house("stark"));
    let (_origin, target) = adjacent_land_pair(board.map());
    let eddard = board.card("stark", "eddard_stark");
    let robb = board.card("stark", "robb_stark");

    state.region_mut(target).controller = Some(stark);
    // The defender has already revealed its card: out of hand, into the discard.
    state.house_mut(stark).unwrap().play_card(eddard);

    let mut request = BattleRequest::new(target, lannister, stark, [5, 0, 0, 0], [1, 0, 0, 0]);
    request.attacker_card = Some(board.card("lannister", "tyrion_lannister"));
    request.defender_card = Some(eddard);
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    // The swap is recorded, not applied, and it downgrades the defender's side.
    assert_eq!(result.context.card_swaps.len(), 1);
    assert!(!state
        .house(stark)
        .unwrap()
        .house_cards_available
        .contains(&eddard));
    assert_eq!(result.defender.card_strength, 3); // Robb Stark
    assert_eq!(result.defender.swords, 0);
    assert!(result
        .notes
        .iter()
        .any(|note| note.contains("stark plays robb_stark instead of eddard_stark")));

    house_cards::commit_deferred(&mut state, board.cards(), &result.context);
    let stark_state = state.house(stark).unwrap();
    assert!(!stark_state.house_cards_available.contains(&robb));
    assert!(stark_state.house_cards_used.contains(&robb));
    // A `return_swap` hands the replaced card back.
    assert!(stark_state.house_cards_available.contains(&eddard));
    assert!(!stark_state.house_cards_used.contains(&eddard));
}

#[test]
fn aeron_damphair_pays_two_power_for_a_stronger_card() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (greyjoy, stark) = (board.house("greyjoy"), board.house("stark"));
    let (_origin, target) = adjacent_land_pair(board.map());
    let aeron = board.card("greyjoy", "aeron_damphair");
    let euron = board.card("greyjoy", "euron_crows_eye");

    state.region_mut(target).controller = Some(stark);
    state.house_mut(greyjoy).unwrap().play_card(aeron);

    let mut request = BattleRequest::new(target, greyjoy, stark, [1, 0, 0, 0], [1, 0, 0, 0]);
    request.attacker_card = Some(aeron);
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert_eq!(result.context.power_costs.as_slice(), &[(greyjoy, 2)]);
    assert_eq!(result.context.card_swaps.len(), 1);
    assert_eq!(result.attacker.card_strength, 4);
    assert_eq!(result.attacker.swords, 1);
    assert_eq!(state.house(greyjoy).unwrap().power, 5);

    house_cards::commit_deferred(&mut state, board.cards(), &result.context);
    let greyjoy_state = state.house(greyjoy).unwrap();
    assert_eq!(greyjoy_state.power, 3);
    assert!(!greyjoy_state.house_cards_available.contains(&euron));
    // A `discard_swap` does not hand the replaced card back.
    assert!(!greyjoy_state.house_cards_available.contains(&aeron));
    assert!(greyjoy_state.house_cards_used.contains(&euron));
}

#[test]
fn arianne_martell_blocks_the_conquest_and_returns_the_attacker() {
    let board = Board::new();
    let mut state = board.clean_state();
    let (stark, martell) = (board.house("stark"), board.house("martell"));
    let (origin, target) = adjacent_land_pair(board.map());
    state.region_mut(origin).controller = Some(stark);
    state.region_mut(target).controller = Some(martell);
    state.region_mut(target).owner = Some(martell);
    state.region_mut(target).garrison = 2;

    let mut request = BattleRequest::new(target, stark, martell, [5, 0, 0, 0], [1, 0, 0, 0]);
    request.defender_card = Some(board.card("martell", "arianne_martell"));
    request.attacker_origin = Some(origin);
    let result = resolve_battle(&state, board.map(), board.cards(), &request);
    assert_eq!(result.winner, stark);
    assert!(house_cards::arianne_holds(
        &result.context,
        board.cards(),
        result.loser
    ));

    apply_battle_result(&mut state, board.map(), board.cards(), &result, 0, None);

    // The attacker may not move in; it is back where it marched from, where
    // Supply enforcement trims it to the largest army Stark may support
    // (supply level 1 allows a 3-unit army), and the emptied area falls back to
    // its printed owner.
    assert_eq!(state.region(target).units, [0; 4]);
    assert_eq!(state.region(target).controller, Some(martell));
    assert_eq!(state.region(origin).unit_count(UnitType::Footman), 3);
    assert_eq!(state.region(origin).controller, Some(stark));
}

#[test]
fn mace_tyrell_destroys_a_footman_before_the_totals_are_compared() {
    let board = Board::new();
    let state = board.clean_state();
    let (tyrell, stark) = (board.house("tyrell"), board.house("stark"));
    let (_origin, target) = adjacent_land_pair(board.map());

    let mut request = BattleRequest::new(target, tyrell, stark, [1, 0, 0, 0], [1, 0, 0, 0]);
    request.attacker_card = Some(board.card("tyrell", "mace_tyrell"));
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    // The defender's Footman is gone from the effective pool *and* the totals.
    assert_eq!(result.context.units_of(stark), &[0, 0, 0, 0]);
    assert_eq!(result.defender.units_strength, 0);
    assert_eq!(result.defender.total(), 0);
    assert_eq!(result.winner, tyrell);
    assert!(result
        .notes
        .iter()
        .any(|note| note.contains("Mace Tyrell destroys a stark Footman")));
}

#[test]
fn the_blackfish_shields_the_side_that_played_it() {
    // `casualty_immunity` names the house that played the card, so an attacker
    // holding it is immune too - the immunity is never granted to the opponent.
    let board = Board::new();
    let state = board.clean_state();
    let (stark, lannister) = (board.house("stark"), board.house("lannister"));
    let (_origin, target) = adjacent_land_pair(board.map());

    let mut request = BattleRequest::new(target, stark, lannister, [1, 0, 0, 0], [4, 0, 0, 0]);
    request.attacker_card = Some(board.card("stark", "blackfish"));
    let result = resolve_battle(&state, board.map(), board.cards(), &request);

    assert_eq!(result.winner, lannister);
    assert_eq!(result.loser, stark);
    assert!(result
        .context
        .casualty_immunity
        .contains(&board.house("stark")));
    assert_eq!(result.casualties, 0);
}

#[test]
fn the_context_is_plain_data_the_phase_machine_can_edit() {
    // The structural change against Python: nothing in `BattleContext` aliases
    // the board, so a caller can hold it, mutate the pools and pass it on.
    let board = Board::new();
    let state = board.clean_state();
    let (baratheon, stark) = (board.house("baratheon"), board.house("stark"));
    let (_origin, target) = adjacent_land_pair(board.map());

    let mut request = BattleRequest::new(target, baratheon, stark, [4, 0, 0, 0], [1, 1, 0, 0]);
    request.attacker_card = Some(board.card("baratheon", "melisandre"));
    let mut result = resolve_battle(&state, board.map(), board.cards(), &request);

    let before: BattleContext = result.context.clone();
    result.context.remove_unit(stark, UnitType::Footman);
    result.context.units_of_mut(stark)[UnitType::Knight.index()] = 0;
    assert_eq!(before.units_of(stark), &[1, 1, 0, 0]);
    assert!(result
        .context
        .units_of(stark)
        .iter()
        .all(|count| *count == 0));
}
