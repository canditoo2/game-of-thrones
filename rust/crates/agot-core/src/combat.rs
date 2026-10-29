//! Battle resolution.
//!
//! Port of `src/agot/engine/combat.py`. The model follows the Second Edition
//! rulebook:
//!
//! * Combat Strength is the sum of each participating unit's printed Combat
//!   Strength, plus the March/Defend order modifier, plus support, plus the
//!   defending Garrison's strength, plus the revealed House card's strength,
//!   plus a possible Valyrian Steel Blade bonus.
//! * A Siege Engine contributes 4 Combat Strength only when attacking (or
//!   supporting an attack against) an area containing a Castle or Stronghold;
//!   otherwise it contributes 0.
//! * The higher total wins; ties go to the house higher on the Fiefdoms track.
//! * Only the loser takes casualties: one unit per Sword icon on the *winner's*
//!   House card, minus one per Fortification icon on the *loser's* card, floored
//!   at zero. The defeated player chooses which of their units die.
//! * The loser then retreats (or its remaining units are destroyed). A defeated
//!   attacker goes back where it marched from; a defeated defender picks a legal
//!   adjacent area. Routed units may never be chosen as casualties, and are
//!   destroyed if forced to retreat.
//!
//! A battle is stored with the two forces kept *separate* (`attacker_units` and
//! `defender_units`). They briefly share a region on the board, but merging them
//! into one pool would make it impossible to tell whose units are whose, which
//! is what casualty removal needs.
//!
//! [`resolve_battle`] is pure: it decides the winner, the number of casualties
//! and any deferred card effects. [`apply_battle_result`] moves the board
//! afterwards, so the phase machine can pause in between for the defeated
//! player's choices. [`resolve_and_apply`] is the convenience path that makes
//! those choices automatically (cheapest unit first, then the first legal
//! retreat) for callers that do not run the phase machine.
//!
//! Two structural changes against the Python original:
//!
//! * unit collections are `[u8; 4]` counts indexed by [`UnitType::index`], so an
//!   absent unit type is a zero and Python's `_prune` has no counterpart;
//! * the static board and the House-card tables are parameters (`map`, `cards`)
//!   instead of globals, and the card abilities live in [`crate::house_cards`].
//!
//! One deliberate simplification is still made here and called out in
//! `docs/roadmap.md`:
//!
//! * Support is resolved automatically in favour of each side's own Support
//!   orders when the caller passes no `support_choices`. The real game lets any
//!   house support either side; [`support_from_choices`] is the cross-house path.
//!
//! Two Python behaviours have no exact counterpart and are noted here:
//!
//! * `state.log.extend(result.notes)` becomes [`State::note`] calls, which add
//!   the `R{round} {phase}:` prefix; the log is never fingerprinted.
//! * `track_position` raises for a house that is not on a track, while
//!   [`State::track_position`] answers 0 (which then loses the Fiefdoms tie).
//!   Only a malformed state can reach that.

use smallvec::SmallVec;

use crate::action::Side;
use crate::enums::{
    CardId, HouseId, OrderKey, OrderSpec, OrderType, RegionId, Terrain, Track, UnitType,
    ORDER_SPECS,
};
use crate::house_cards::{self, BattleContext};
use crate::map::{GameMap, HouseCardData};
use crate::state::{Region, State};

/// Removal order for the *automatic* casualty path ([`resolve_and_apply`]) and
/// for Supply enforcement: cheapest combat value first, because a casualty
/// destroys a unit regardless of its strength.
const CASUALTY_ORDER: [UnitType; 4] = [
    UnitType::Footman,
    UnitType::Siege,
    UnitType::Knight,
    UnitType::Ship,
];

// ---------------------------------------------------------------------------
// Small helpers over the fixed unit arrays
// ---------------------------------------------------------------------------

#[inline]
fn has_units(units: &[u8; 4]) -> bool {
    units.iter().any(|count| *count > 0)
}

#[inline]
fn no_units(units: &[u8; 4]) -> bool {
    !has_units(units)
}

#[inline]
fn has_routed(region: &Region) -> bool {
    region.routed.iter().any(|count| *count > 0)
}

/// True when every unit in the collection is a Ship, mirroring Python's
/// `all(unit is UnitType.SHIP for unit in units)` over a *pruned* dict.
#[inline]
fn all_ships(units: &[u8; 4]) -> bool {
    let total: u32 = units.iter().map(|count| *count as u32).sum();
    total > 0 && units[UnitType::Ship.index()] as u32 == total
}

/// The spec of an order key, or `None` for a key outside the table.
///
/// Python reaches for `ORDERS_BY_KEY.get(...)` wherever a region's order may be
/// anything at all; this mirrors that lookup rather than indexing the table.
#[inline]
pub(crate) fn order_spec(key: OrderKey) -> Option<&'static OrderSpec> {
    ORDER_SPECS.get(key.0 as usize)
}

/// Sorts region ids the way Python's `sorted(...)` over the id strings does.
///
/// The port keeps regions in board order, so every place the Python sorts
/// region ids (which region loses its Order to the Queen of Thorns, which
/// retreat is offered first) has to compare the map's id strings explicitly.
pub(crate) fn sort_region_ids(map: &GameMap, ids: &mut [RegionId]) {
    ids.sort_by(|left, right| map.id_of(*left).cmp(map.id_of(*right)));
}

// ---------------------------------------------------------------------------
// Strength building blocks (also used by the heuristic agent)
// ---------------------------------------------------------------------------

/// Combat strength of an explicit collection of units.
///
/// `siege_active` must be true when the units are attacking (or supporting an
/// attack against) an area containing a Castle or Stronghold, which is the only
/// time Siege Engines contribute their printed strength (rulebook page 20).
pub fn units_strength(units: &[u8; 4], siege_active: bool) -> u16 {
    let mut total: u16 = 0;
    for unit in UnitType::ALL {
        let count = units[unit.index()] as u16;
        if count == 0 {
            continue;
        }
        if unit == UnitType::Siege && !siege_active {
            continue;
        }
        total += unit.combat_strength() as u16 * count;
    }
    total
}

/// Combat strength of the units `house` currently controls in a region.
pub fn unit_strength(state: &State, region: RegionId, house: HouseId, siege_active: bool) -> u16 {
    let region = state.region(region);
    if region.controller != Some(house) {
        return 0;
    }
    units_strength(&region.units, siege_active)
}

/// Which units in a supporting region can actually add strength.
///
/// Ships in an adjacent sea area may support a land battle; ships in a Port may
/// support the adjacent sea, but never a land area. Footmen/Knights/Siege may
/// never support a battle in a sea area. A supporting Siege Engine only
/// contributes when supporting the attacker against a Castle or Stronghold, and
/// it is the *embattled* area that must be fortified - not the area the Siege
/// Engine is supporting from.
fn contributing_units(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    battle_is_sea: bool,
    supporter_is_attacker: bool,
    target_is_fortification: bool,
) -> [u8; 4] {
    let terrain = map.region(region_id).terrain;
    let ships_ok = terrain == Terrain::Sea || (battle_is_sea && terrain == Terrain::Port);
    let units = &state.region(region_id).units;
    let mut contributing = [0u8; 4];

    for unit in UnitType::ALL {
        let count = units[unit.index()];
        if count == 0 {
            continue;
        }
        if unit == UnitType::Ship {
            if ships_ok {
                contributing[unit.index()] += count;
            }
        } else if !battle_is_sea {
            contributing[unit.index()] += count;
        }
    }
    if no_units(&contributing) {
        return contributing;
    }
    if !(supporter_is_attacker && target_is_fortification) {
        contributing[UnitType::Siege.index()] = 0;
    }
    contributing
}

/// Support a house receives from its own adjacent Support orders.
///
/// This is the legacy self-directed helper used when [`resolve_battle`] is
/// called without explicit `support_choices`. Cross-house support is resolved by
/// [`support_from_choices`].
pub fn support_strength(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    house: HouseId,
    supporting_attacker: bool,
) -> u16 {
    support_detail(state, map, region_id, house, supporting_attacker).0
}

/// Support strength *and* the units providing it.
///
/// The House-card abilities of Kevan Lannister, Victarion Greyjoy and Salladhor
/// Saan care about *which* units are supporting, so the aggregate is returned
/// alongside the strength.
pub fn support_detail(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    house: HouseId,
    supporting_attacker: bool,
) -> (u16, [u8; 4]) {
    let battle_is_sea = map.region(region_id).is_sea();
    let is_fortification = map.region(region_id).is_fortification();
    let mut total: i32 = 0;
    let mut units = [0u8; 4];

    for neighbour in map.neighbours_of(region_id) {
        let region = state.region(*neighbour);
        if region.order_house != Some(house) {
            continue;
        }
        let Some(order) = region.order else {
            continue;
        };
        let Some(spec) = order_spec(order) else {
            continue;
        };
        if spec.order_type != OrderType::Support {
            continue;
        }
        // An Order token alone does not support: there must still be a unit
        // (ready or routed) in the area. Routed units contribute 0 strength
        // but the Support order's printed bonus still applies (FFG ruling).
        if region.total(true) == 0 {
            continue;
        }
        let contributing = contributing_units(
            state,
            map,
            *neighbour,
            battle_is_sea,
            supporting_attacker,
            is_fortification,
        );
        for unit in UnitType::ALL {
            units[unit.index()] += contributing[unit.index()];
        }
        total += units_strength(&contributing, supporting_attacker && is_fortification) as i32;
        total += spec.strength as i32;
    }
    // Support orders never carry a negative bonus; the clamp only guards
    // malformed board data.
    (total.max(0) as u16, units)
}

/// Whether any adjacent Support order of `house` has at least one unit.
///
/// Public support totals are 0 when only routed units support with a plain
/// (non-starred) Support order, but the side is still "being supported" for
/// House-card text such as Salladhor Saan.
pub fn support_present(state: &State, map: &GameMap, region_id: RegionId, house: HouseId) -> bool {
    map.neighbours_of(region_id).iter().any(|neighbour| {
        let region = state.region(*neighbour);
        region.order_house == Some(house)
            && region.order.is_some_and(|order| {
                order_spec(order).is_some_and(|spec| spec.order_type == OrderType::Support)
            })
            && region.total(true) > 0
    })
}

/// Adjacent Support orders as `(region, owner)` in declaration order.
///
/// Support is declared in Iron Throne order (rulebook page 18). The embattled
/// area itself is never included: a Support order gives no defensive benefit to
/// an attack against its own area.
pub fn support_orders(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
) -> SmallVec<[(RegionId, HouseId); 8]> {
    let throne = state.track_order(Track::IronThrone);
    let mut found: SmallVec<[(RegionId, HouseId); 8]> = SmallVec::new();

    for neighbour in map.neighbours_of(region_id) {
        let region = state.region(*neighbour);
        let (Some(order), Some(owner)) = (region.order, region.order_house) else {
            continue;
        };
        let Some(spec) = order_spec(order) else {
            continue;
        };
        if spec.order_type != OrderType::Support {
            continue;
        }
        found.push((*neighbour, owner));
    }
    // A house missing from the track sorts last, exactly as Python does.
    found.sort_by_key(|(_, owner)| {
        throne
            .iter()
            .position(|candidate| candidate == owner)
            .unwrap_or(throne.len())
    });
    found
}

/// Upper bound on support: every adjacent Support order contributes fully.
///
/// Used to test whether a march against a Neutral Force *could* succeed once
/// support has been declared, since that is unknown when legal marches are
/// generated (rulebook page 26).
pub fn potential_support_strength(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    supporting_attacker: bool,
) -> u16 {
    support_strength_for(state, map, region_id, supporting_attacker, None)
}

/// Support strength that only counts orders owned by `house`.
///
/// A rival's Support order is not guaranteed to be declared for this attack, so
/// the legal-march filter must not treat it as available. The battle itself
/// only counts the support each house actually declares.
pub fn own_support_strength(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    house: HouseId,
    supporting_attacker: bool,
) -> u16 {
    support_strength_for(state, map, region_id, supporting_attacker, Some(house))
}

fn support_strength_for(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    supporting_attacker: bool,
    owner: Option<HouseId>,
) -> u16 {
    let battle_is_sea = map.region(region_id).is_sea();
    let is_fort = map.region(region_id).is_fortification();
    let mut total: i32 = 0;

    for (neighbour, order_owner) in support_orders(state, map, region_id) {
        if owner.is_some() && Some(order_owner) != owner {
            continue;
        }
        if state.region(neighbour).total(true) == 0 {
            continue;
        }
        let contributing = contributing_units(
            state,
            map,
            neighbour,
            battle_is_sea,
            supporting_attacker,
            is_fort,
        );
        // `support_orders` only returns regions that carry a Support order.
        let spec = state
            .region(neighbour)
            .order
            .expect("support order present")
            .spec();
        total += units_strength(&contributing, supporting_attacker && is_fort) as i32;
        total += spec.strength as i32;
    }
    total.max(0) as u16
}

/// The two sides' support, built from explicit `area -> side` declarations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SupportTotals {
    pub attacker_strength: u16,
    pub attacker_units: [u8; 4],
    pub defender_strength: u16,
    pub defender_units: [u8; 4],
    /// True when at least one declared Support order actually has a unit,
    /// even if routed units make its combat contribution 0.
    pub attacker_present: bool,
    pub defender_present: bool,
}

/// Build each side's support from explicit `area -> side` declarations.
///
/// A declaration of [`Side::Refuse`] (or an area whose order has since gone)
/// contributes nothing.
pub fn support_from_choices(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    choices: &[(RegionId, Side)],
) -> SupportTotals {
    let battle_is_sea = map.region(region_id).is_sea();
    let is_fort = map.region(region_id).is_fortification();
    let mut totals = SupportTotals::default();
    let mut attacker_total: i32 = 0;
    let mut defender_total: i32 = 0;

    for (support_region, side) in choices {
        let supporting_attacker = match side {
            Side::Attacker => true,
            Side::Defender => false,
            Side::Refuse => continue,
        };
        // Python's `state.regions.get(...)` yields nothing for an unknown area.
        if support_region.index() >= state.region_count as usize {
            continue;
        }
        let region = state.region(*support_region);
        let Some(order) = region.order else {
            continue;
        };
        let Some(spec) = order_spec(order) else {
            continue;
        };
        if spec.order_type != OrderType::Support {
            continue;
        }
        // A Support order with no unit left in its area supports nothing;
        // routed units still support at 0 plus the order bonus.
        if region.total(true) == 0 {
            continue;
        }
        let contributing = contributing_units(
            state,
            map,
            *support_region,
            battle_is_sea,
            supporting_attacker,
            is_fort,
        );
        let strength = units_strength(&contributing, supporting_attacker && is_fort) as i32
            + spec.strength as i32;
        if supporting_attacker {
            attacker_total += strength;
            totals.attacker_present = true;
            for unit in UnitType::ALL {
                totals.attacker_units[unit.index()] += contributing[unit.index()];
            }
        } else {
            defender_total += strength;
            totals.defender_present = true;
            for unit in UnitType::ALL {
                totals.defender_units[unit.index()] += contributing[unit.index()];
            }
        }
    }
    totals.attacker_strength = attacker_total.max(0) as u16;
    totals.defender_strength = defender_total.max(0) as u16;
    totals
}

/// March bonus for an attacker or Defend bonus for a defender.
pub fn order_modifier(state: &State, region: RegionId, house: HouseId, attacker: bool) -> i16 {
    let region = state.region(region);
    if region.order_house != Some(house) {
        return 0;
    }
    let Some(order) = region.order else {
        return 0;
    };
    let Some(spec) = order_spec(order) else {
        return 0;
    };
    if attacker && spec.order_type == OrderType::March {
        return spec.strength as i16;
    }
    if !attacker && spec.order_type == OrderType::Defend {
        return spec.strength as i16;
    }
    0
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// One participant in a battle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BattleSide {
    pub house: HouseId,
    pub units_strength: i16,
    pub order_strength: i16,
    pub support_strength: i16,
    pub garrison_strength: i16,
    pub card_strength: i16,
    pub swords: u8,
    pub forts: u8,
    pub blade: bool,
}

impl BattleSide {
    /// A side with every modifier at zero, as Python's dataclass defaults are.
    pub fn new(house: HouseId) -> BattleSide {
        BattleSide {
            house,
            units_strength: 0,
            order_strength: 0,
            support_strength: 0,
            garrison_strength: 0,
            card_strength: 0,
            swords: 0,
            forts: 0,
            blade: false,
        }
    }

    /// The Combat Strength the two sides are compared on.
    pub fn total(&self) -> i16 {
        self.units_strength
            + self.order_strength
            + self.support_strength
            + self.garrison_strength
            + self.card_strength
            + if self.blade { 1 } else { 0 }
    }
}

/// Assemble a [`BattleSide`] from explicit pieces.
///
/// Mirrors the Python `build_side`; the parameter list is the point, so the
/// clippy argument-count lint is silenced rather than worked around.
#[allow(clippy::too_many_arguments)]
pub fn build_side(
    cards: &HouseCardData,
    house: HouseId,
    units: &[u8; 4],
    order_strength: i16,
    support: u16,
    card: Option<CardId>,
    use_blade: bool,
    garrison_strength: u16,
    siege_active: bool,
) -> BattleSide {
    let mut side = BattleSide {
        house,
        units_strength: units_strength(units, siege_active) as i16,
        order_strength,
        support_strength: support as i16,
        garrison_strength: garrison_strength as i16,
        card_strength: 0,
        swords: 0,
        forts: 0,
        blade: false,
    };
    if let Some(spec) = house_cards::card_data(cards, house, card) {
        side.card_strength = spec.strength as i16;
        side.swords = spec.swords;
        side.forts = spec.forts;
    }
    side.blade = use_blade;
    side
}

/// Everything [`resolve_battle`] needs about one battle.
///
/// The phase machine fills this from `PendingBattle`; `attacker_order_strength`
/// and `defender_order_strength` are the pre-computed March/Defend modifiers
/// (the March order may already have left the board).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BattleRequest {
    pub region: RegionId,
    pub attacker: HouseId,
    pub defender: HouseId,
    pub attacker_units: [u8; 4],
    pub defender_units: [u8; 4],
    pub attacker_card: Option<CardId>,
    pub defender_card: Option<CardId>,
    pub attacker_order_strength: i16,
    pub defender_order_strength: i16,
    pub attacker_blade: bool,
    pub defender_blade: bool,
    pub attacker_order: Option<OrderKey>,
    pub attacker_origin: Option<RegionId>,
    /// Adjacent supporting areas and the side they declared, in the order the
    /// declarations were made. `None` falls back to the self-directed support
    /// of each side's own Support orders.
    pub support_choices: Option<SmallVec<[(RegionId, Side); 8]>>,
}

impl BattleRequest {
    /// A request with no cards, no orders and no declared support.
    pub fn new(
        region: RegionId,
        attacker: HouseId,
        defender: HouseId,
        attacker_units: [u8; 4],
        defender_units: [u8; 4],
    ) -> BattleRequest {
        BattleRequest {
            region,
            attacker,
            defender,
            attacker_units,
            defender_units,
            attacker_card: None,
            defender_card: None,
            attacker_order_strength: 0,
            defender_order_strength: 0,
            attacker_blade: false,
            defender_blade: false,
            attacker_order: None,
            attacker_origin: None,
            support_choices: None,
        }
    }
}

/// Outcome of a battle, ready to be applied to the state.
///
/// `context` carries the card effects and the effective unit pools: the phase
/// machine mutates `context.attacker_units` / `context.defender_units` while the
/// defeated player picks casualties, and [`apply_battle_result`] reads them back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BattleResult {
    pub region: RegionId,
    pub attacker: BattleSide,
    pub defender: BattleSide,
    pub winner: HouseId,
    pub loser: HouseId,
    pub casualties: u8,
    pub tiebreak: bool,
    pub notes: Vec<String>,
    pub context: BattleContext,
}

/// Compute the outcome of a battle without mutating `state`.
///
/// House-card abilities are applied here too, but only the pure ones: an ability
/// that moves an Order token, an Influence track or a card between hand and
/// discard is *recorded* on the returned [`BattleContext`] and committed by
/// [`resolve_and_apply`] (or by the phase machine's own `commit_deferred` call).
///
/// `support_choices` maps an adjacent supporting area to [`Side::Attacker`] or
/// [`Side::Defender`]. When it is `None` the engine falls back to the old
/// self-directed support, which keeps hand-written calls working.
pub fn resolve_battle(
    state: &State,
    map: &GameMap,
    cards: &HouseCardData,
    request: &BattleRequest,
) -> BattleResult {
    let context = prepare_battle(state, map, cards, request, None);
    finish_battle(state, request, context)
}

/// Build the card-effect context, before the Valyrian Steel Blade step.
///
/// The phase machine calls this as soon as both House cards are revealed, then
/// commits the immediate board changes before asking the blade holder whether
/// to use the blade. [`finish_battle`] later compares the already-modified
/// sides. Keeping the context also means card effects are not applied twice.
///
/// Support is built from the board before those deferred effects are
/// committed, but a Support order removed by the Queen of Thorns must not keep
/// contributing: the FAQ requires recalculating the Initial Combat Strength.
/// [`drop_support_from_removed_orders`] applies that correction, so the
/// deterministic path matches the interactive one, which chooses the Order
/// before this function runs. See `house_cards::queen_of_thorns`.
pub fn prepare_battle(
    state: &State,
    map: &GameMap,
    cards: &HouseCardData,
    request: &BattleRequest,
    doran_track: Option<Track>,
) -> BattleContext {
    let region_id = request.region;
    let is_fortification = map.region(region_id).is_fortification();
    let region = state.region(region_id);

    let (
        attacker_support,
        attacker_support_units,
        defender_support,
        defender_support_units,
        attacker_supported,
        defender_supported,
    ) = match &request.support_choices {
        None => {
            let (attacker_total, attacker_units) =
                support_detail(state, map, region_id, request.attacker, true);
            let (defender_total, defender_units) =
                support_detail(state, map, region_id, request.defender, false);
            (
                attacker_total,
                attacker_units,
                defender_total,
                defender_units,
                support_present(state, map, region_id, request.attacker),
                support_present(state, map, region_id, request.defender),
            )
        }
        Some(choices) => {
            let totals = support_from_choices(state, map, region_id, choices);
            (
                totals.attacker_strength,
                totals.attacker_units,
                totals.defender_strength,
                totals.defender_units,
                totals.attacker_present,
                totals.defender_present,
            )
        }
    };

    let attacker_side = build_side(
        cards,
        request.attacker,
        &request.attacker_units,
        request.attacker_order_strength,
        attacker_support,
        request.attacker_card,
        request.attacker_blade,
        0,
        is_fortification,
    );
    let defender_side = build_side(
        cards,
        request.defender,
        &request.defender_units,
        request.defender_order_strength,
        defender_support,
        request.defender_card,
        request.defender_blade,
        region.garrison as u16,
        // A defending Siege Engine never contributes its printed strength.
        false,
    );

    let mut ctx = BattleContext {
        region_id,
        attacker: request.attacker,
        defender: request.defender,
        attacker_units: request.attacker_units,
        defender_units: request.defender_units,
        attacker_card: request.attacker_card,
        defender_card: request.defender_card,
        attacker_supported,
        defender_supported,
        support_choices: request.support_choices.clone(),
        attacker_support_units,
        defender_support_units,
        attacker_order: request.attacker_order,
        attacker_origin: request.attacker_origin,
        attacker_side,
        defender_side,
        card_swaps: Vec::new(),
        power_costs: SmallVec::new(),
        orders_to_remove: SmallVec::new(),
        influence_move: None,
        doran_track,
        casualty_immunity: SmallVec::new(),
        notes: Vec::new(),
    };
    house_cards::apply_pre_battle(&mut ctx, state, map, cards);
    // Queen of Thorns removes an Order *after* support was locked in as part of
    // the Initial Combat Strength; the FAQ requires recalculating that strength,
    // so drop the support the removed area provided. The interactive path makes
    // the Order choice before this function runs, so its context has nothing to
    // drop and is unaffected.
    drop_support_from_removed_orders(&mut ctx, state, map, cards);
    ctx
}

/// Remove one supporting area's contribution after the Queen of Thorns.
///
/// The removed Order belongs to the opponent of the card's owner, but the area
/// may have declared for either side (or refused). The declaration list, when
/// present, is the authority; otherwise the legacy self-directed support of the
/// area's owner decides which side loses the strength. Kevan Lannister and
/// Victarion Greyjoy upgrade supporting Footmen/Ships, so their extra bonus is
/// removed from `units_strength` as well (the area no longer supports).
fn drop_support_from_removed_orders(
    ctx: &mut BattleContext,
    state: &State,
    map: &GameMap,
    cards: &HouseCardData,
) {
    if ctx.orders_to_remove.is_empty() {
        return;
    }
    let battle_is_sea = map.region(ctx.region_id).is_sea();
    let is_fort = map.region(ctx.region_id).is_fortification();
    let removed_orders = ctx.orders_to_remove.clone();
    for removed in removed_orders {
        let region = state.region(removed);
        let Some(order) = region.order else {
            continue;
        };
        let Some(spec) = order_spec(order) else {
            continue;
        };
        if spec.order_type != OrderType::Support || region.total(true) == 0 {
            continue;
        }
        let supporting_attacker = match &ctx.support_choices {
            Some(choices) => match choices
                .iter()
                .find(|(area, _)| *area == removed)
                .map(|(_, side)| *side)
            {
                Some(Side::Attacker) => true,
                Some(Side::Defender) => false,
                // Refused (or never declared): it contributed no support.
                _ => continue,
            },
            None => {
                if region.order_house == Some(ctx.attacker) {
                    true
                } else if region.order_house == Some(ctx.defender) {
                    false
                } else {
                    continue;
                }
            }
        };
        let contributing = contributing_units(
            state,
            map,
            removed,
            battle_is_sea,
            supporting_attacker,
            is_fort,
        );
        let strength = units_strength(&contributing, supporting_attacker && is_fort) as i16
            + i16::from(spec.strength);
        // The supported house's card may have already upgraded these very
        // support units; that extra lives in `units_strength`, not in the
        // Support order's printed bonus.
        let supporting_house = if supporting_attacker {
            ctx.attacker
        } else {
            ctx.defender
        };
        let upgraded = if supporting_attacker {
            match house_cards::ability_of(cards, supporting_house, ctx.card_of(supporting_house)) {
                "kevan_lannister" => contributing[UnitType::Footman.index()],
                "victarion_greyjoy" => contributing[UnitType::Ship.index()],
                _ => 0,
            }
        } else {
            0
        };
        let (side, support_units) = if supporting_attacker {
            (&mut ctx.attacker_side, &mut ctx.attacker_support_units)
        } else {
            (&mut ctx.defender_side, &mut ctx.defender_support_units)
        };
        side.support_strength = (side.support_strength - strength).max(0);
        side.units_strength = (side.units_strength - i16::from(upgraded)).max(0);
        for unit in UnitType::ALL {
            support_units[unit.index()] =
                support_units[unit.index()].saturating_sub(contributing[unit.index()]);
        }
    }
}

/// Compare the prepared sides and build the final battle result.
///
/// The immediate card effects were already applied by [`prepare_battle`]; this
/// only applies the blade choice made after them and decides the outcome.
pub fn finish_battle(
    state: &State,
    request: &BattleRequest,
    mut ctx: BattleContext,
) -> BattleResult {
    // The blade is selected after the card text. A prepared context was built
    // before the blade choice, so apply the two blade flags here.
    ctx.attacker_side.blade = request.attacker_blade;
    ctx.defender_side.blade = request.defender_blade;

    // The abilities may have changed either side's printed statistics (a swap,
    // a passive modifier), so the totals are read back from the context - the
    // same objects Python mutates in place.
    let attacker_side = ctx.attacker_side;
    let defender_side = ctx.defender_side;

    let mut notes: Vec<String> = ctx.notes.clone();
    let mut tiebreak = false;
    let mut winner = request.attacker;
    let mut loser = request.defender;
    let mut winner_side = attacker_side;
    let mut loser_side = defender_side;

    if attacker_side.total() > defender_side.total() {
        // The attacker already holds the default assignment.
    } else if defender_side.total() > attacker_side.total() {
        winner = request.defender;
        loser = request.attacker;
        winner_side = defender_side;
        loser_side = attacker_side;
    } else {
        tiebreak = true;
        let attacker_position = state.track_position(request.attacker, Track::Fiefdoms);
        let defender_position = state.track_position(request.defender, Track::Fiefdoms);
        if attacker_position >= defender_position {
            winner = request.defender;
            loser = request.attacker;
            winner_side = defender_side;
            loser_side = attacker_side;
        }
        notes.push("strength tied; decided on the Fiefdoms track".to_string());
    }

    let mut casualties = (winner_side.swords as i16 - loser_side.forts as i16).max(0) as u8;
    if ctx.casualty_immunity.contains(&loser) {
        casualties = 0;
        if winner_side.swords > 0 {
            notes.push(format!("{loser} ignores casualties (The Blackfish)"));
        }
    } else if casualties > 0 {
        notes.push(format!(
            "{loser} takes {casualties} casualt{}",
            if casualties == 1 { "y" } else { "ies" }
        ));
    } else if winner_side.swords > 0 {
        notes.push("all casualties cancelled by fortifications".to_string());
    }

    BattleResult {
        region: request.region,
        attacker: ctx.attacker_side,
        defender: ctx.defender_side,
        winner,
        loser,
        casualties,
        tiebreak,
        notes,
        context: ctx,
    }
}

/// Remove `count` units in place, cheapest first.
fn remove_units(units: &mut [u8; 4], count: u8) {
    let mut remaining = count;
    for unit in CASUALTY_ORDER {
        while remaining > 0 && units[unit.index()] > 0 {
            units[unit.index()] -= 1;
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
    }
}

/// Resolve a battle and mutate `state` accordingly.
///
/// This is the non-interactive convenience path: casualties are removed
/// cheapest-first and the defender's retreat is picked by the engine. The phase
/// machine instead calls [`resolve_battle`] and then [`apply_battle_result`] so
/// that the defeated player makes those choices.
pub fn resolve_and_apply(
    state: &mut State,
    map: &GameMap,
    cards: &HouseCardData,
    request: &BattleRequest,
) -> BattleResult {
    let mut result = resolve_battle(state, map, cards, request);
    // Card swaps, Order removals and Doran's influence move are deferred until
    // now so that resolve_battle stays free of state mutation.
    house_cards::commit_deferred(state, cards, &result.context);
    // FFG ruling: a losing force loses its participating Siege Engines
    // automatically, in addition to the normal casualties.
    let siege = result.context.remove_siege_engines(result.loser);
    if siege > 0 {
        state.note(format!(
            "{} Siege Engine(s) destroyed automatically in {}",
            result.loser,
            map.id_of(result.region)
        ));
    }
    apply_battle_result(state, map, cards, &result, result.casualties, None);
    result
}

/// Move the board to the post-battle position.
///
/// [`resolve_battle`] decides the winner, the number of casualties and any
/// deferred card effects; this function applies the consequences. The defeated
/// player's choices arrive here: `remaining_casualties` counts the units still
/// to be destroyed (0 once the player has chosen them, the effective pools in
/// `result.context` having been decremented) and `retreat_destination` is the
/// area the defender (or Robb Stark's controller) picked.
///
/// The Python original appends the "was destroyed" lines to `result.notes` and
/// finally does `state.log.extend(result.notes)`. `result` is immutable here, so
/// the same lines are pushed onto the state's log with [`State::note`] instead,
/// in the same order.
pub fn apply_battle_result(
    state: &mut State,
    map: &GameMap,
    cards: &HouseCardData,
    result: &BattleResult,
    remaining_casualties: u8,
    retreat_destination: Option<RegionId>,
) {
    let ctx = &result.context;
    let attacker = ctx.attacker;
    let defender = ctx.defender;
    let region_id = result.region;
    let mut attacker_units = ctx.attacker_units;
    let mut defender_units = ctx.defender_units;

    // Draw back a just-completed seven-card House deck before post-combat text
    // such as Patchface (FFG ruling). The phase machine also does this before
    // calling here; a populated hand makes the helper a no-op.
    recycle_house_cards(state, attacker);
    if defender != attacker {
        recycle_house_cards(state, defender);
    }

    // Cards and the convenience path normally strip the loser's Siege Engines
    // before this point. Keep the rule here too so a direct caller cannot
    // accidentally let one survive as a casualty.
    let auto_siege = if result.loser == attacker {
        let siege = attacker_units[UnitType::Siege.index()];
        attacker_units[UnitType::Siege.index()] = 0;
        siege
    } else {
        let siege = defender_units[UnitType::Siege.index()];
        defender_units[UnitType::Siege.index()] = 0;
        siege
    };

    let pre_routed = state.region(region_id).routed;

    let robb = house_cards::robb_chose_retreat(ctx, cards, result.winner);
    let arianne = house_cards::arianne_holds(ctx, cards, result.loser);

    let mut notes: Vec<String> = result.notes.clone();
    if auto_siege > 0 {
        notes.push(format!(
            "{} Siege Engine(s) destroyed automatically in {}",
            result.loser,
            map.id_of(region_id)
        ));
    }

    if result.loser == attacker {
        remove_units(&mut attacker_units, remaining_casualties);
        let surviving = attacker_units;
        {
            let region = state.region_mut(region_id);
            region.units = defender_units;
            region.controller = Some(defender);
            // The defender keeps its own routed units, order token and Power token.
            region.routed = pre_routed;
        }
        if has_units(&surviving) {
            retreat_attacker(
                state,
                map,
                region_id,
                attacker,
                &surviving,
                ctx.attacker_origin,
            );
        } else if result.casualties > 0 {
            notes.push(format!("{attacker} was destroyed"));
        }
    } else {
        remove_units(&mut defender_units, remaining_casualties);
        let surviving_defender = defender_units;
        {
            let region = state.region_mut(region_id);
            region.garrison = 0;
            region.order = None;
            region.order_house = None;
        }
        if arianne {
            // Arianne Martell: the attacker may not move in; it returns to the
            // area it marched from. The defender still retreats, but the
            // winner's other effects (including removing the Power token) apply.
            return_attacker(state, attacker, &attacker_units, ctx.attacker_origin);
            let region = state.region_mut(region_id);
            region.units = [0; 4];
            region.routed = [0; 4];
            region.influence_token = None;
        } else {
            let region = state.region_mut(region_id);
            region.units = attacker_units;
            region.controller = Some(attacker);
            // The defender's routed units are destroyed (routed units forced to
            // retreat are destroyed). The Garrison is permanently removed.
            region.routed = [0; 4];
            // Winning the attacker removes the defender's Order and Power tokens.
            region.influence_token = None;
        }
        if has_units(&surviving_defender) {
            retreat_defender(
                state,
                map,
                region_id,
                defender,
                &surviving_defender,
                ctx.attacker_origin,
                robb,
                retreat_destination,
            );
        } else if result.casualties > 0 {
            notes.push(format!("{defender} was destroyed"));
        }
    }

    let needs_release = {
        let region = state.region(region_id);
        // Mace Tyrell can remove the last unit while its owner still wins the
        // battle; the March/Consolidate Order then remains on the empty area
        // (FFG ruling). A surviving Order therefore blocks the release.
        region.is_empty() && !has_routed(region) && region.garrison == 0 && region.order.is_none()
    };
    if needs_release {
        release_region(state, region_id);
    }

    house_cards::apply_post_battle(ctx, state, map, cards, result);
    try_loras(ctx, state, cards, result);
    for note in notes {
        state.note(note);
    }
}

/// Legal retreat destinations for a defeated defender with `units` left.
///
/// Returns an empty list when the units cannot retreat at all (for example a
/// lone Siege Engine), which means they are destroyed instead.
pub fn retreat_options(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    house: HouseId,
    units: &[u8; 4],
    exclude_origin: Option<RegionId>,
) -> SmallVec<[RegionId; 8]> {
    if no_units(units) {
        return SmallVec::new();
    }
    let naval = all_ships(units);
    let (retreaters, _destroyed) = split_non_retreaters(units);
    if no_units(&retreaters) {
        return SmallVec::new();
    }
    legal_retreats(state, map, region_id, house, naval, exclude_origin)
}

/// Whether routing `units` into `destination` keeps the house legal.
///
/// Public alias of the internal Supply probe for the phase machine. Python
/// mutates `destination.routed` and puts it back; the port evaluates the same
/// Supply question on an untouched state, because `within_supply` only reads it.
/// The map is part of the signature for symmetry with the other retreat
/// helpers; Supply does not depend on the board topology.
pub fn retreat_keeps_supply(
    state: &State,
    _map: &GameMap,
    destination: RegionId,
    house: HouseId,
    units: &[u8; 4],
) -> bool {
    supply_after_routing(state, destination, house, units)
}

/// Send a blocked attacker (Arianne Martell) back where it marched from.
fn return_attacker(state: &mut State, house: HouseId, units: &[u8; 4], origin: Option<RegionId>) {
    if no_units(units) {
        return;
    }
    let Some(origin) = origin else {
        return;
    };
    {
        let target = state.region_mut(origin);
        for unit in UnitType::ALL {
            target.units[unit.index()] += units[unit.index()];
        }
        if target.controller.is_none() || target.controller == Some(house) {
            target.controller = Some(house);
        }
    }
    if !state.within_supply(house) {
        if state.supply_choice {
            if !state.supply_queue.contains(&house) {
                state.supply_queue.push(house);
            }
        } else {
            enforce_supply_at(state, origin, house);
        }
    }
}

/// Loras Tyrell re-places the March order in the conquered area.
fn try_loras(ctx: &BattleContext, state: &mut State, cards: &HouseCardData, result: &BattleResult) {
    if result.winner != ctx.attacker {
        return;
    }
    if house_cards::ability_of(cards, ctx.attacker, ctx.attacker_card) != "loras_tyrell" {
        return;
    }
    let Some(order) = ctx.attacker_order else {
        return;
    };
    {
        let region = state.region_mut(ctx.region_id);
        region.order = Some(order);
        region.order_house = Some(ctx.attacker);
    }
    state.note("Loras Tyrell re-places the March order to resolve again".to_string());
}

/// A losing attacker must retreat back to the area it marched from.
fn retreat_attacker(
    state: &mut State,
    map: &GameMap,
    region_id: RegionId,
    house: HouseId,
    units: &[u8; 4],
    origin: Option<RegionId>,
) {
    let Some(origin) = origin else {
        state.note(format!(
            "{house} has no retreat from {}; army destroyed",
            map.id_of(region_id)
        ));
        return;
    };
    let (retreaters, destroyed) = split_non_retreaters(units);
    if has_units(&destroyed) {
        state.note(format!(
            "{house} Siege Engine(s) destroyed retreating from {}",
            map.id_of(region_id)
        ));
    }
    if no_units(&retreaters) {
        return;
    }
    if !can_retreat_to(state, map, origin, house, false) {
        state.note(format!(
            "{house} cannot retreat to {}; army destroyed",
            map.id_of(origin)
        ));
        return;
    }
    move_routed(state, origin, house, &retreaters);
}

/// A losing defender retreats to a legal adjacent area, or is destroyed.
///
/// `destination` is the area the defender (or, under Robb Stark, the winner)
/// chose. When it is `None` the first legal area is used, which keeps the
/// non-interactive [`resolve_and_apply`] path behaving as before.
#[allow(clippy::too_many_arguments)]
fn retreat_defender(
    state: &mut State,
    map: &GameMap,
    region_id: RegionId,
    house: HouseId,
    units: &[u8; 4],
    attacker_origin: Option<RegionId>,
    prefer_min_loss: bool,
    destination: Option<RegionId>,
) {
    let naval = all_ships(units);
    let (retreaters, destroyed) = split_non_retreaters(units);
    if has_units(&destroyed) {
        state.note(format!(
            "{house} Siege Engine(s) destroyed retreating from {}",
            map.id_of(region_id)
        ));
    }
    if no_units(&retreaters) {
        return;
    }
    let options = legal_retreats(state, map, region_id, house, naval, attacker_origin);
    if options.is_empty() {
        state.note(format!(
            "{house} has no retreat from {}; army destroyed",
            map.id_of(region_id)
        ));
        return;
    }
    let chosen = match destination {
        Some(destination) if options.contains(&destination) => destination,
        _ => {
            let mut chosen = options[0];
            if prefer_min_loss {
                // Robb Stark chooses the retreat where the loser loses the
                // fewest units: prefer a destination that keeps the house
                // within Supply.
                chosen = options
                    .iter()
                    .copied()
                    .find(|option| retreat_keeps_supply(state, map, *option, house, &retreaters))
                    .unwrap_or(options[0]);
            }
            chosen
        }
    };
    move_routed(state, chosen, house, &retreaters);
}

/// Whether routing `units` into `destination` keeps the house legal.
///
/// Python adds the units to `destination.routed`, asks `within_supply` and puts
/// the region back. The port answers the same question by adding the units to
/// the destination's size while it walks the house's armies, so no state is
/// touched at all.
fn supply_after_routing(
    state: &State,
    destination: RegionId,
    house: HouseId,
    units: &[u8; 4],
) -> bool {
    let Some(house_state) = state.house(house) else {
        return true;
    };
    let limits = house_state.supply_limit();
    let extra: u32 = units.iter().map(|count| *count as u32).sum();
    let mut sizes: Vec<u32> = state
        .region_ids()
        .filter(|id| state.region(*id).controller == Some(house))
        .map(|id| {
            let mut size = state.region(id).total(true);
            if id == destination {
                size += extra;
            }
            size
        })
        .filter(|size| *size >= 2)
        .collect();
    sizes.sort_unstable_by(|left, right| right.cmp(left));
    for (index, size) in sizes.iter().enumerate() {
        if index >= limits.len() || *size > limits[index] as u32 {
            return false;
        }
    }
    true
}

/// Separate units that can retreat from those that are destroyed instead.
///
/// Python's `naval` parameter is unused there too: a Siege Engine is the only
/// unit that can never retreat.
fn split_non_retreaters(units: &[u8; 4]) -> ([u8; 4], [u8; 4]) {
    let mut retreaters = [0u8; 4];
    let mut destroyed = [0u8; 4];
    for unit in UnitType::ALL {
        let count = units[unit.index()];
        if count == 0 {
            continue;
        }
        if unit == UnitType::Siege {
            destroyed[unit.index()] += count;
        } else {
            retreaters[unit.index()] += count;
        }
    }
    (retreaters, destroyed)
}

/// Move routed units into `destination` and re-establish control if it is
/// unowned or already friendly.
fn move_routed(state: &mut State, destination: RegionId, house: HouseId, units: &[u8; 4]) {
    {
        let target = state.region_mut(destination);
        for unit in UnitType::ALL {
            target.routed[unit.index()] += units[unit.index()];
        }
        if target.controller.is_none() || target.controller == Some(house) {
            target.controller = Some(house);
        }
    }
    if !state.within_supply(house) {
        if state.supply_choice {
            if !state.supply_queue.contains(&house) {
                state.supply_queue.push(house);
            }
        } else {
            enforce_supply_at(state, destination, house);
        }
    }
}

/// Destroy routed/ready units in a region until Supply is satisfied.
fn enforce_supply_at(state: &mut State, region_id: RegionId, house: HouseId) {
    let mut guard = 0;
    while !state.within_supply(house) && guard < 100 {
        guard += 1;
        let region = state.region_mut(region_id);
        let mut removed = false;
        for unit in CASUALTY_ORDER {
            if region.routed_count(unit) > 0 {
                region.routed[unit.index()] -= 1;
                removed = true;
                break;
            }
            if region.unit_count(unit) > 0 {
                region.units[unit.index()] -= 1;
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Region-level helpers (also used for supply enforcement and cleaning up)
// ---------------------------------------------------------------------------

/// Remove `count` units controlled by `house` in a region, cheapest first.
pub fn apply_casualties(
    state: &mut State,
    region_id: RegionId,
    house: HouseId,
    count: u8,
) -> SmallVec<[UnitType; 8]> {
    if state.region(region_id).controller != Some(house) {
        return SmallVec::new();
    }
    let mut removed: SmallVec<[UnitType; 8]> = SmallVec::new();
    let mut remaining = count;
    {
        let region = state.region_mut(region_id);
        for unit in CASUALTY_ORDER {
            while remaining > 0 && region.unit_count(unit) > 0 {
                region.units[unit.index()] -= 1;
                remaining -= 1;
                removed.push(unit);
            }
            if remaining == 0 {
                break;
            }
        }
    }
    let needs_release = {
        let region = state.region(region_id);
        region.is_empty() && !has_routed(region) && region.garrison == 0
    };
    if needs_release {
        release_region(state, region_id);
    }
    removed
}

/// A Neutral Force token blocks retreats just like a defending army.
fn enemy_presence(state: &State, region_id: RegionId, house: HouseId) -> bool {
    let region = state.region(region_id);
    if region.controller.is_some() && region.controller != Some(house) {
        return true;
    }
    if region.controller.is_none() && region.total(false) > 0 {
        return true;
    }
    if region.neutral_force > 0 {
        return true;
    }
    if region.influence_token.is_some() && region.influence_token != Some(house) {
        return true;
    }
    false
}

/// Whether a retreating force may enter `destination`.
fn can_retreat_to(
    state: &State,
    map: &GameMap,
    destination: RegionId,
    house: HouseId,
    naval: bool,
) -> bool {
    let spec = map.region(destination);
    if naval {
        if spec.is_land() {
            return false;
        }
    } else if !spec.is_land() {
        return false;
    }
    if enemy_presence(state, destination, house) {
        return false;
    }
    true
}

/// Land areas reachable from `origin` through a friendly Ship chain.
///
/// Routed Ships may still transport (rulebook page 23).
fn ship_transport_targets(
    state: &State,
    map: &GameMap,
    house: HouseId,
    origin: RegionId,
) -> Vec<RegionId> {
    let naval: Vec<RegionId> = state
        .region_ids()
        .filter(|id| {
            let region = state.region(*id);
            region.controller == Some(house)
                && map.region(*id).is_sea()
                && region.unit_count(UnitType::Ship) + region.routed_count(UnitType::Ship) > 0
        })
        .collect();

    let mut seen: Vec<RegionId> = map
        .sea_neighbours_of(origin)
        .into_iter()
        .filter(|neighbour| naval.contains(neighbour))
        .collect();
    let mut frontier = seen.clone();
    while let Some(current) = frontier.pop() {
        for neighbour in map.sea_neighbours_of(current) {
            if naval.contains(&neighbour) && !seen.contains(&neighbour) {
                seen.push(neighbour);
                frontier.push(neighbour);
            }
        }
    }

    let mut targets: Vec<RegionId> = Vec::new();
    for sea in seen {
        for land in map.land_neighbours_of(sea) {
            if !targets.contains(&land) {
                targets.push(land);
            }
        }
    }
    targets
}

/// Adjacent regions a defeated force may retreat into.
pub fn legal_retreats(
    state: &State,
    map: &GameMap,
    region_id: RegionId,
    house: HouseId,
    naval: bool,
    exclude_origin: Option<RegionId>,
) -> SmallVec<[RegionId; 8]> {
    let mut options: Vec<RegionId> = Vec::new();

    for neighbour in map.neighbours_of(region_id) {
        if Some(*neighbour) == exclude_origin {
            continue;
        }
        let spec = map.region(*neighbour);
        if !can_retreat_to(state, map, *neighbour, house, naval) {
            continue;
        }
        if spec.is_port() {
            // A Port can hold at most three ships, and only its owner may use it.
            if !naval || state.region(*neighbour).controller != Some(house) {
                continue;
            }
            if state.region(*neighbour).total(true) >= 3 {
                continue;
            }
        }
        options.push(*neighbour);
    }

    if !naval {
        for neighbour in ship_transport_targets(state, map, house, region_id) {
            if Some(neighbour) == exclude_origin || options.contains(&neighbour) {
                continue;
            }
            if can_retreat_to(state, map, neighbour, house, false) {
                options.push(neighbour);
            }
        }
    }

    sort_region_ids(map, &mut options);
    options.into_iter().collect()
}

/// Move a controlled army out of `region_id` and rout it.
///
/// `map` and `naval` are part of the signature for symmetry with the Python
/// callers; splitting the retreaters only ever discards Siege Engines, so neither
/// is read.
pub fn retreat(
    state: &mut State,
    _map: &GameMap,
    region_id: RegionId,
    house: HouseId,
    target: RegionId,
    _naval: bool,
) {
    let units = state.region(region_id).units;
    state.region_mut(region_id).units = [0; 4];
    let (retreaters, _destroyed) = split_non_retreaters(&units);
    move_routed(state, target, house, &retreaters);
}

/// Wipe out everything in a region (used when no retreat is possible).
///
/// The map is part of the signature for symmetry with the Python callers; wiping
/// a region needs no board topology.
pub fn destroy_army(state: &mut State, _map: &GameMap, region_id: RegionId) {
    {
        let region = state.region_mut(region_id);
        region.units = [0; 4];
        region.routed = [0; 4];
        region.garrison = 0;
    }
    release_region(state, region_id);
}

/// Return six cards after a house plays the last card of its House deck.
///
/// The just-played card stays in the discard pile; the other six return to the
/// hand. Kept local to combat so the convenience `resolve_and_apply` path and
/// the phase machine share the same pre-Patchface timing.
fn recycle_house_cards(state: &mut State, house: HouseId) {
    let Some(house_state) = state.house_mut(house) else {
        return;
    };
    if !house_state.house_cards_available.is_empty() || house_state.house_cards_used.is_empty() {
        return;
    }
    let last_played = *house_state
        .house_cards_used
        .last()
        .expect("checked non-empty");
    house_state.house_cards_available = house_state
        .house_cards_used
        .iter()
        .copied()
        .filter(|card| *card != last_played)
        .collect();
    house_state.house_cards_used.clear();
    house_state.house_cards_used.push(last_played);
}

/// Drop control and any order token from a now-empty region.
fn release_region(state: &mut State, region_id: RegionId) {
    let region = state.region_mut(region_id);
    region.controller = region.influence_token.or(region.owner);
    region.order = None;
    region.order_house = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::DataSet;

    fn data() -> DataSet {
        DataSet::load_default().expect("bundled data must load")
    }

    #[test]
    fn unit_strength_ignores_inactive_siege_engines() {
        // 2 Footmen, 1 Knight, 1 Siege Engine, 1 Ship.
        let units = [2, 1, 1, 1];
        // 2 Footmen + Knight + Ship; the Siege Engine waits.
        assert_eq!(units_strength(&units, false), 5);
        // With a fortified target the Siege Engine adds its printed 4.
        assert_eq!(units_strength(&units, true), 9);
        assert_eq!(units_strength(&[0; 4], true), 0);
    }

    #[test]
    fn sides_total_their_modifiers_and_blade() {
        let mut side = BattleSide::new(HouseId(2));
        side.units_strength = 5;
        side.order_strength = 1;
        side.support_strength = 4;
        side.garrison_strength = 2;
        side.card_strength = 3;
        assert_eq!(side.total(), 15);
        side.blade = true;
        assert_eq!(side.total(), 16);
    }

    #[test]
    fn orders_are_read_from_the_region_and_its_holder() {
        let data = data();
        let mut state = State::blank();
        state.region_count = data.map.region_count() as u8;
        let winterfell = data.map.lookup("winterfell").unwrap();
        let house = HouseId(2);
        state.region_mut(winterfell).order = Some(OrderKey::from_name("march_plus").unwrap());
        state.region_mut(winterfell).order_house = Some(house);
        assert_eq!(order_modifier(&state, winterfell, house, true), 1);
        assert_eq!(order_modifier(&state, winterfell, house, false), 0);
        // The wrong house gets nothing.
        assert_eq!(order_modifier(&state, winterfell, HouseId(1), true), 0);
    }

    #[test]
    fn support_orders_follow_iron_throne_order() {
        let data = data();
        let mut state = State::blank();
        state.region_count = data.map.region_count() as u8;
        for track in Track::ALL {
            state.tracks[track.index()] = data.houses.initial_tracks[track.index()].clone();
        }
        let highgarden = data.map.lookup("highgarden").unwrap();
        // Two Support orders next to Highgarden: Martell and Tyrell.
        let (tyrell, martell) = (HouseId(4), HouseId(5));
        for (region, house) in [("oldtown", martell), ("searoad_marches", tyrell)] {
            let id = data.map.lookup(region).unwrap();
            state.region_mut(id).order = Some(OrderKey::from_name("support_a").unwrap());
            state.region_mut(id).order_house = Some(house);
        }
        let orders = support_orders(&state, &data.map, highgarden);
        assert_eq!(orders.len(), 2);
        // The Iron Throne order is baratheon, lannister, stark, martell,
        // greyjoy, tyrell, so Martell declares first.
        let throne = state.track_order(Track::IronThrone);
        let martell_position = throne.iter().position(|house| *house == martell).unwrap();
        let tyrell_position = throne.iter().position(|house| *house == tyrell).unwrap();
        assert!(martell_position < tyrell_position);
        assert_eq!(orders[0].1, martell);
        assert_eq!(orders[1].1, tyrell);
    }

    #[test]
    fn retreat_keeps_supply_probe_matches_within_supply() {
        let data = data();
        let mut state = State::blank();
        state.region_count = data.map.region_count() as u8;
        let house = HouseId(2);
        // Supply level 0: at most two armies of two.
        state.houses[house.index()] = Some(crate::state::House::new(7, 5, 0));
        let winterfell = data.map.lookup("winterfell").unwrap();
        let white_harbor = data.map.lookup("white_harbor").unwrap();
        state.region_mut(winterfell).controller = Some(house);
        state.region_mut(winterfell).units = [2, 0, 0, 0];
        state.region_mut(white_harbor).controller = Some(house);
        assert!(state.within_supply(house));
        // Routed units count towards Supply, so a second army of two fits...
        assert!(retreat_keeps_supply(
            &state,
            &data.map,
            white_harbor,
            house,
            &[2, 0, 0, 0]
        ));
        // ...a second army of three does not.
        assert!(!retreat_keeps_supply(
            &state,
            &data.map,
            white_harbor,
            house,
            &[3, 0, 0, 0]
        ));
        // A lone unit is not an army and never breaks Supply.
        assert!(retreat_keeps_supply(
            &state,
            &data.map,
            white_harbor,
            house,
            &[1, 0, 0, 0]
        ));
        // An area the house does not control does not count at all.
        let the_eyrie = data.map.lookup("the_eyrie").unwrap();
        assert!(retreat_keeps_supply(
            &state,
            &data.map,
            the_eyrie,
            house,
            &[4, 0, 0, 0]
        ));
    }

    #[test]
    fn casualty_removal_is_cheapest_first() {
        let mut units = [1u8, 1, 1, 1];
        remove_units(&mut units, 2);
        assert_eq!(units, [0, 1, 0, 1]);
        remove_units(&mut units, 5);
        assert_eq!(units, [0, 0, 0, 0]);
    }

    #[test]
    fn queen_of_thorns_removes_the_support_it_takes() {
        // FAQ v2.0: when the Queen removes a Support order, the supporting
        // area's strength is recalculated out of the Initial Combat Strength.
        let data = data();
        let map = &data.map;
        let cards = &data.house_cards;
        let (target, support) = (
            map.lookup("highgarden").unwrap(),
            map.lookup("searoad_marches").unwrap(),
        );
        assert!(map.neighbours_of(target).contains(&support));
        let (tyrell, stark) = (
            HouseId::from_name("tyrell").unwrap(),
            HouseId::from_name("stark").unwrap(),
        );
        let queen = cards.card_index(tyrell, "queen_of_thorns").unwrap();

        let build_state = || {
            let mut state = State::blank();
            state.region_count = map.region_count() as u8;
            {
                let region = state.region_mut(support);
                region.controller = Some(stark);
                region.units = [0, 2, 0, 0];
                region.order = Some(OrderKey::from_name("support_a").unwrap());
                region.order_house = Some(stark);
            }
            state
        };
        let mut request = BattleRequest::new(target, tyrell, stark, [2, 0, 0, 0], [1, 0, 0, 0]);
        request.attacker_card = Some(queen);

        // Explicit declaration: the defender loses the area's support.
        let state = build_state();
        let mut declared = request.clone();
        declared.support_choices = Some(SmallVec::from_slice(&[(support, Side::Defender)]));
        let result = resolve_battle(&state, map, cards, &declared);
        assert_eq!(result.context.orders_to_remove.as_slice(), &[support]);
        assert_eq!(result.defender.support_strength, 0);
        assert_eq!(result.context.defender_support_units, [0; 4]);
        assert_eq!(result.winner, tyrell);

        // Legacy self-directed support uses the Order's owner.
        let state = build_state();
        let result = resolve_battle(&state, map, cards, &request);
        assert_eq!(result.defender.support_strength, 0);
        assert_eq!(result.winner, tyrell);
    }
}
