//! Observation encoding: how a game state and its candidate actions become
//! tensors.
//!
//! Port of `src/agot/engine/observation.py`. The engine's action space is
//! *combinatorial*: the number of legal moves varies wildly (a Planning decision
//! can offer dozens of region/order pairs, a bid a handful of integers), so
//! there is no fixed move list to score. Like the Python original, this module
//! follows the pointer-network pattern: [`encode_state`] turns the whole
//! position into one fixed-length vector, [`encode_action`] turns a single
//! candidate into another one using only *relational* information (terrain, who
//! controls a region, unit counts, order type) rather than one-hot region ids,
//! and [`observe`] bundles the two.
//!
//! Everything is expressed over the canonical six-house order
//! ([`crate::enums::HOUSE_ORDER`]). Short-handed games leave the unused slots
//! zeroed, so the network input size never changes with the player count.
//!
//! # Bit-exactness
//!
//! This is a *numerical* port, not a rewrite: the Python reference computes
//! every fraction in `float64` and rounds to `float32` only on the store into
//! its `numpy` array (`head[3] = static.power / 3.0`, `min(1.0, n / 8.0)`, ...).
//! The Rust expressions below therefore keep the `f64` intermediate and cast
//! with `as f32` at exactly the point where Python rounds. A "cleaner" all-`f32`
//! formulation is a different function whose last bits differ, and a checkpoint
//! trained against one encoder would not score the other. `tests/encoder_parity.rs`
//! pins the behaviour against vectors recorded from the Python engine.

use std::sync::OnceLock;

use crate::action::{Action, Side, N_ACTION_KINDS, N_SIDES};
use crate::enums::{
    CardId, HouseId, RegionId, Track, UnitType, N_HOUSES, N_ORDERS, N_PHASES, N_STEPS,
    SUPPLY_LIMITS,
};
use crate::map::DataSet;
use crate::state::{Region, State, MAX_REGIONS};

// ---------------------------------------------------------------------------
// Canonical layout
// ---------------------------------------------------------------------------

/// The four unit kinds, in the order every per-unit block uses.
///
/// Matches `observation.UNIT_ORDER`; it is also the index order of
/// [`Region::units`], so a unit's slot is `unit.index()` everywhere.
pub const UNIT_ORDER: [UnitType; 4] = [
    UnitType::Footman,
    UnitType::Knight,
    UnitType::Siege,
    UnitType::Ship,
];

pub const N_UNITS: usize = UNIT_ORDER.len();

/// Sides a Support order can be declared for, plus the option to refuse
/// (rulebook pages 17-18). `Side::ALL` is declared in this order.
pub const SUPPORT_SIDES: [Side; N_SIDES] = Side::ALL;

/// House-card slots: every house brings exactly seven cards.
pub const CARD_SLOTS: usize = 7;

/// Soft normalisation constants. They only need to keep features in a sane
/// range.
pub const MAX_POWER: f64 = 20.0;
pub const MAX_THREAT: f64 = 12.0;

/// Feature count of one house's block.
pub const HOUSE_BLOCK: usize = 13;
/// Python's private alias for [`HOUSE_BLOCK`], kept so the layout reads the
/// same on both sides.
pub const _HOUSE_BLOCK: usize = HOUSE_BLOCK;

/// `round` + phase one-hot + step one-hot + threat / alive houses / region
/// count / mustering areas.
pub const GLOBAL_BLOCK: usize = 1 + N_PHASES + N_STEPS + 4;

/// terrain(3) + power/supply/mustering(3) + unit counts + flags(6)
/// + controller relation(4) + order one-hot + order flags(4).
pub const REGION_BLOCK: usize = 3 + 3 + N_UNITS + 6 + 4 + N_ORDERS + 4;

/// One House card's features.
pub const CARD_BLOCK: usize = 6;

/// An order one-hot plus the order's strength flag.
pub const ORDER_BLOCK: usize = N_ORDERS + 1;

/// An amount (normalised) plus a "given" flag.
pub const AMOUNT_BLOCK: usize = 2;

/// March detail: a normalised count per unit type, a "subset given" flag and an
/// "establish control" flag.
pub const MOVE_BLOCK: usize = N_UNITS + 2;

/// Length of the state feature vector.
pub const STATE_DIM: usize = N_HOUSES * HOUSE_BLOCK + GLOBAL_BLOCK;

/// Length of one action feature vector.
pub const ACTION_DIM: usize = N_ACTION_KINDS
    + N_HOUSES
    + 1 // is_active
    + REGION_BLOCK
    + REGION_BLOCK
    + CARD_BLOCK
    + ORDER_BLOCK
    + AMOUNT_BLOCK
    + N_UNITS
    + MOVE_BLOCK
    + N_SIDES;

// ---------------------------------------------------------------------------
// Static tables
// ---------------------------------------------------------------------------

/// Per-region constants and per-card features, derived once from the board
/// data.
///
/// Python caches these in module-level `lru_cache` functions
/// (`_static_region_tables`, `_card_table`). The tables only depend on the
/// board data, so [`static_tables`] builds them on the first encode call and
/// reuses them for the process lifetime — the crate ships one board, exactly as
/// Python's global `load_map()` does. Callers that need the tables for another
/// data set can build their own with [`StaticTables::from_data`].
#[derive(Debug)]
pub struct StaticTables {
    /// Six leading features per region: terrain one-hot plus the
    /// power/supply/mustering fractions.
    heads: Vec<[f32; 6]>,
    /// Printed home house per region (`None` for neutral areas).
    homes: Vec<Option<HouseId>>,
    /// Whether the region is a port or adjoins a sea (`host or sea` in Python).
    linked: Vec<bool>,
    /// One feature vector per House card, indexed by [`HouseId`] then deck slot.
    cards: Vec<Vec<[f32; CARD_BLOCK]>>,
}

impl StaticTables {
    /// Computes the tables from a loaded data set.
    pub fn from_data(data: &DataSet) -> StaticTables {
        let map = &data.map;
        let mut heads: Vec<[f32; 6]> = Vec::with_capacity(map.region_count());
        let mut homes: Vec<Option<HouseId>> = Vec::with_capacity(map.region_count());
        let mut linked: Vec<bool> = Vec::with_capacity(map.region_count());
        for index in 0..map.region_count() {
            let spec = map.region(RegionId(index as u16));
            let mut head = [0.0f32; 6];
            head[spec.terrain.index()] = 1.0;
            head[3] = (f64::from(spec.power) / 3.0) as f32;
            head[4] = (f64::from(spec.supply) / 3.0) as f32;
            head[5] = (f64::from(spec.mustering) / 2.0) as f32;
            heads.push(head);
            homes.push(spec.home);
            linked.push(spec.host.is_some() || spec.sea.is_some());
        }

        let mut cards: Vec<Vec<[f32; CARD_BLOCK]>> = Vec::with_capacity(N_HOUSES);
        for house in HouseId::ALL {
            let mut deck: Vec<[f32; CARD_BLOCK]> = Vec::new();
            for (slot, spec) in data.house_cards.deck(house).iter().enumerate() {
                let mut vector = [0.0f32; CARD_BLOCK];
                vector[0] = 1.0;
                vector[1] = (f64::from(spec.strength) / 5.0) as f32;
                vector[2] = (f64::from(spec.swords) / 3.0) as f32;
                vector[3] = (f64::from(spec.forts) / 3.0) as f32;
                vector[4] = ((slot + 1) as f64 / CARD_SLOTS as f64) as f32;
                vector[5] = 1.0;
                deck.push(vector);
            }
            cards.push(deck);
        }

        StaticTables {
            heads,
            homes,
            linked,
            cards,
        }
    }

    #[inline]
    fn head(&self, id: RegionId) -> &[f32; 6] {
        &self.heads[id.index()]
    }

    #[inline]
    fn home(&self, id: RegionId) -> Option<HouseId> {
        self.homes[id.index()]
    }

    #[inline]
    fn linked(&self, id: RegionId) -> bool {
        self.linked[id.index()]
    }

    /// Features of one House card, or the "unknown card" fallback.
    ///
    /// The Python reference keys its card table on `(house, card id)` and falls
    /// back to a vector that keeps only the "a card field is present" marker
    /// when the id is not in the house's deck. In Rust a card is a deck slot,
    /// so an out-of-range [`CardId`] is that same fallback — which is what a
    /// holders'-choice Westeros card (a card id from another deck) produces.
    fn card_features(&self, house: HouseId, card: CardId) -> [f32; CARD_BLOCK] {
        self.cards
            .get(house.index())
            .and_then(|deck| deck.get(card.0 as usize))
            .copied()
            .unwrap_or_else(|| {
                let mut vector = [0.0f32; CARD_BLOCK];
                vector[5] = 1.0;
                vector
            })
    }
}

/// The process-wide tables, built from the first data set an encode call sees.
fn static_tables(data: &DataSet) -> &'static StaticTables {
    static TABLES: OnceLock<StaticTables> = OnceLock::new();
    TABLES.get_or_init(|| StaticTables::from_data(data))
}

// ---------------------------------------------------------------------------
// State encoding
// ---------------------------------------------------------------------------

/// Encode a position as one fixed-length feature vector.
///
/// `actor` is the house the engine is asking to act; `None` falls back to
/// `state.active_house`, matching the Python default. It only affects a couple
/// of "*is this mine*" flags.
pub fn encode_state(state: &State, data: &DataSet, actor: Option<HouseId>) -> Vec<f32> {
    let mut out = vec![0.0f32; STATE_DIM];
    encode_state_into(state, data, actor, &mut out);
    out
}

/// Write a position's feature vector into `out` (length [`STATE_DIM`]).
///
/// `out` is fully overwritten, so it need not be zeroed first (Python allocates
/// a fresh `np.zeros` here). `data` is not read: the state vector only needs
/// the state and the static layout constants, and the parameter is kept so the
/// two encoders have the same shape.
pub fn encode_state_into(state: &State, data: &DataSet, actor: Option<HouseId>, out: &mut [f32]) {
    assert_eq!(out.len(), STATE_DIM, "state feature buffer");
    debug_assert!(
        state.region_count as usize <= data.map.region_count(),
        "state was built for a different board than the data set"
    );
    let actor = actor.or(state.active_house);

    // One pass over the board instead of one pass per house: accumulate each
    // house's army sizes, fleet size and castle count together.
    let mut totals = [HouseTotals::default(); N_HOUSES];
    let mut mustering_areas = 0u32;
    for id in state.region_ids() {
        let region = state.region(id);
        if region.mustering_points > 0 {
            mustering_areas += 1;
        }
        let Some(controller) = region.controller else {
            continue;
        };
        if state.house(controller).is_none() {
            continue;
        }
        if state.fortifications.contains(&id) {
            totals[controller.index()].castles += 1;
        }
        for unit in UNIT_ORDER {
            let count = u32::from(region.units[unit.index()]);
            if unit == UnitType::Ship {
                totals[controller.index()].ships += count;
            } else {
                totals[controller.index()].land += count;
            }
        }
    }

    out.fill(0.0);
    let mut offset = 0;
    for house_id in HouseId::ALL {
        if let Some(house) = state.house(house_id) {
            fill_house_block(
                state,
                house_id,
                house,
                actor,
                totals[house_id.index()],
                out,
                offset,
            );
        }
        // Houses absent from the game keep their zeroed block.
        offset += HOUSE_BLOCK;
    }

    // Global block.
    out[offset] = (f64::from(state.round) / f64::from(state.max_rounds.max(1))) as f32;
    offset += 1;
    out[offset + state.phase.index()] = 1.0;
    offset += N_PHASES;
    out[offset + state.action_step.index()] = 1.0;
    offset += N_STEPS;
    out[offset] = f64::min(1.0, f64::from(state.wildling_threat) / MAX_THREAT) as f32;
    let alive = HouseId::ALL
        .into_iter()
        .filter(|house| state.house(*house).map(|h| !h.eliminated).unwrap_or(false))
        .count();
    out[offset + 1] = (alive as f64 / N_HOUSES as f64) as f32;
    out[offset + 2] = (f64::from(state.region_count) / 64.0) as f32;
    out[offset + 3] = (f64::from(mustering_areas) / N_HOUSES as f64) as f32;
}

/// The per-house totals the board pass accumulates: ready land units, ships
/// and controlled castles/strongholds.
#[derive(Clone, Copy, Debug, Default)]
struct HouseTotals {
    land: u32,
    ships: u32,
    castles: u32,
}

/// Write one house's 13-slot block from the stats gathered in the board pass.
fn fill_house_block(
    state: &State,
    house_id: HouseId,
    house: &crate::state::House,
    actor: Option<HouseId>,
    totals: HouseTotals,
    out: &mut [f32],
    offset: usize,
) {
    out[offset] = 1.0;
    out[offset + 1] = if house.eliminated { 1.0 } else { 0.0 };
    out[offset + 2] = (f64::from(totals.castles) / f64::from(state.victory_castles.max(1))) as f32;
    out[offset + 3] = f64::min(1.0, f64::from(house.power) / MAX_POWER) as f32;
    out[offset + 4] =
        (f64::from(house.supply_level) / (SUPPLY_LIMITS.len() - 1).max(1) as f64) as f32;
    out[offset + 5] = track_feature(state, house_id, Track::IronThrone);
    out[offset + 6] = track_feature(state, house_id, Track::Fiefdoms);
    out[offset + 7] = track_feature(state, house_id, Track::KingsCourt);
    out[offset + 8] = f64::min(1.0, f64::from(totals.land) / 16.0) as f32;
    out[offset + 9] = f64::min(1.0, f64::from(totals.ships) / 12.0) as f32;
    out[offset + 10] = f64::min(1.0, house.orders_placed.len() as f64 / 12.0) as f32;
    out[offset + 11] = (house.house_cards_available.len() as f64 / CARD_SLOTS as f64) as f32;
    out[offset + 12] = if Some(house_id) == actor { 1.0 } else { 0.0 };
}

/// `1 - (position - 1) / (N_HOUSES - 1)`, computed in `f64` like the Python.
fn track_feature(state: &State, house: HouseId, track: Track) -> f32 {
    let position = state.track_position(house, track);
    (1.0 - (position as f64 - 1.0) / (N_HOUSES - 1) as f64) as f32
}

// ---------------------------------------------------------------------------
// Action encoding
// ---------------------------------------------------------------------------

/// Memo of region blocks, keyed by region, actor and reveal flag.
///
/// Port of Python's `RegionCache` (`dict[int, np.ndarray]` keyed on
/// `id(region)`). Python keys on the region *object* because a clone shares
/// region objects, so identity pins the contents; Rust regions are owned by
/// value and can never alias, so the same equivalence class is
/// `(RegionId, actor, orders_revealed)`. One cache is valid for one state only:
/// it assumes the regions do not change between calls, which is exactly the
/// Python contract.
#[derive(Clone, Debug)]
pub struct RegionCache {
    entries: Vec<Option<CachedBlock>>,
}

#[derive(Clone, Copy, Debug)]
struct CachedBlock {
    actor: Option<HouseId>,
    orders_revealed: bool,
    block: [f32; REGION_BLOCK],
}

impl RegionCache {
    pub fn new() -> RegionCache {
        RegionCache {
            entries: vec![None; MAX_REGIONS],
        }
    }

    /// Drops every memoised block, so the cache can serve another state.
    pub fn clear(&mut self) {
        self.entries.iter_mut().for_each(|entry| *entry = None);
    }

    /// The block for `id`, computing and memoising it on first use.
    fn block(
        &mut self,
        tables: &StaticTables,
        state: &State,
        id: RegionId,
        actor: Option<HouseId>,
    ) -> [f32; REGION_BLOCK] {
        let orders_revealed = state.orders_revealed();
        let index = id.index();
        if index >= self.entries.len() {
            self.entries.resize(index + 1, None);
        }
        if let Some(entry) = self.entries[index] {
            if entry.actor == actor && entry.orders_revealed == orders_revealed {
                return entry.block;
            }
        }
        let mut block = [0.0f32; REGION_BLOCK];
        fill_region_block(
            tables,
            state.region(id),
            id,
            actor,
            orders_revealed,
            &mut block,
        );
        self.entries[index] = Some(CachedBlock {
            actor,
            orders_revealed,
            block,
        });
        block
    }
}

impl Default for RegionCache {
    fn default() -> RegionCache {
        RegionCache::new()
    }
}

/// Static plus dynamic description of one region, from `actor`'s view.
///
/// Port of `_region_block`. Written as an in-place fill of one array: this runs
/// twice per candidate action, so it deliberately avoids temporaries.
fn fill_region_block(
    tables: &StaticTables,
    region: &Region,
    id: RegionId,
    actor: Option<HouseId>,
    orders_revealed: bool,
    out: &mut [f32; REGION_BLOCK],
) {
    out[..6].copy_from_slice(tables.head(id));

    let mut index = 6;
    for unit in UNIT_ORDER {
        let count = f64::from(region.units[unit.index()]) / 4.0;
        out[index + unit.index()] = if count > 1.5 { 1.5 } else { count as f32 };
    }
    index += N_UNITS;

    let ready: u32 = region.units.iter().map(|count| u32::from(*count)).sum();
    out[index] = f64::min(1.0, f64::from(ready) / 8.0) as f32;
    let routed: u32 = region.routed.iter().map(|count| u32::from(*count)).sum();
    out[index + 1] = f64::min(1.0, f64::from(routed) / 8.0) as f32;
    out[index + 2] = if region.influence_token.is_some() {
        1.0
    } else {
        0.0
    };
    out[index + 3] = if region.order.is_some() { 1.0 } else { 0.0 };
    out[index + 4] = (f64::from(region.garrison) / 4.0) as f32;
    out[index + 5] = (f64::from(region.neutral_force) / 6.0) as f32;
    index += 6;

    // The whole controller block is cleared first so no slot can leak a stale
    // value from a reused buffer, then the one flag that applies is set.
    out[index..index + 4].fill(0.0);
    match region.controller {
        None => out[index] = 1.0, // unowned
        Some(controller) if Some(controller) == actor => out[index + 2] = 1.0,
        Some(_) => out[index + 3] = 1.0,
    }
    index += 4;

    // A house may always see its own orders; the rest stay hidden until the
    // reveal (rulebook pages 11-12).
    let visible = actor.is_none() || orders_revealed || region.order_house == actor;
    let order = if visible { region.order } else { None };
    out[index..index + N_ORDERS].fill(0.0);
    if let Some(key) = order {
        out[index + key.index()] = 1.0;
    }
    index += N_ORDERS;

    out[index] = match order {
        Some(key) => (f64::from(key.strength()) / 2.0) as f32,
        None => 0.0, // hidden or absent order: no strength to show
    };
    out[index + 1] = if region.order_house == actor {
        1.0
    } else {
        0.0
    };
    out[index + 2] = if tables.home(id).is_some() && tables.home(id) == actor {
        1.0
    } else {
        0.0
    };
    out[index + 3] = if tables.linked(id) { 1.0 } else { 0.0 };
}

/// Encode one candidate action, allocating the result.
///
/// Thin wrapper over [`encode_action_into`]; batched callers fill preallocated
/// rows instead.
pub fn encode_action(
    state: &State,
    data: &DataSet,
    action: &Action,
    actor: Option<HouseId>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; ACTION_DIM];
    let mut cache = RegionCache::new();
    encode_action_into(state, data, action, actor, &mut out, &mut cache);
    out
}

/// Write one action's features into `out` (length [`ACTION_DIM`]).
///
/// `out` must already be zeroed: the writer sets only the slots that apply,
/// which is what lets a batched evaluator fill a whole action tensor without
/// allocating a temporary per candidate. `cache` is the optional memo of region
/// blocks; many candidates in one position share a region (every order for an
/// area, every march out of it), so the lookup is worth it.
pub fn encode_action_into(
    state: &State,
    data: &DataSet,
    action: &Action,
    actor: Option<HouseId>,
    out: &mut [f32],
    cache: &mut RegionCache,
) {
    assert_eq!(out.len(), ACTION_DIM, "action feature buffer");
    let tables = static_tables(data);
    let actor = actor.or(state.active_house);

    let mut offset = 0;
    out[offset + action.kind.index()] = 1.0;
    offset += N_ACTION_KINDS;

    if let Some(house) = action.house {
        out[offset + house.index()] = 1.0;
    }
    offset += N_HOUSES;

    if action.house == actor {
        out[offset] = 1.0;
    }
    offset += 1;

    // Region and target blocks. The row is already zeroed, so a field with no
    // region (or one outside the board) is simply skipped rather than copied
    // from an all-zero block.
    for field in [action.region, action.target] {
        if let Some(id) = field {
            if id.index() < state.region_count as usize {
                let block = cache.block(tables, state, id, actor);
                out[offset..offset + REGION_BLOCK].copy_from_slice(&block);
            }
        }
        offset += REGION_BLOCK;
    }

    if let (Some(house), Some(card)) = (action.house, action.card) {
        let block = tables.card_features(house, card);
        out[offset..offset + CARD_BLOCK].copy_from_slice(&block);
    }
    offset += CARD_BLOCK;

    if let Some(order) = action.order {
        out[offset + order.index()] = 1.0;
        out[offset + N_ORDERS] = (f64::from(order.strength()) / 2.0) as f32;
    }
    offset += ORDER_BLOCK;

    if let Some(amount) = action.amount {
        out[offset] = f64::min(1.0, f64::from(amount) / MAX_POWER) as f32;
        out[offset + 1] = 1.0;
    }
    offset += AMOUNT_BLOCK;

    if let Some(unit) = action.unit {
        out[offset + unit.index()] = 1.0;
    }
    offset += N_UNITS;

    if let Some(units) = action.units {
        for (index, count) in units.iter().enumerate() {
            out[offset + index] = f64::min(1.0, f64::from(*count) / 4.0) as f32;
        }
        out[offset + N_UNITS] = 1.0;
    }
    if action.establish {
        out[offset + N_UNITS + 1] = 1.0;
    }
    offset += MOVE_BLOCK;

    if let Some(side) = action.side {
        out[offset + side.index()] = 1.0;
    }
    offset += N_SIDES;

    debug_assert_eq!(offset, ACTION_DIM, "action feature layout");
}

// ---------------------------------------------------------------------------
// Bundled observation
// ---------------------------------------------------------------------------

/// Everything the policy/value network needs at one decision point.
#[derive(Clone, Debug)]
pub struct Observation {
    /// The house the decision belongs to (`None` when the state has no active
    /// house; see [`observe`]).
    pub actor: Option<HouseId>,
    pub actions: Vec<Action>,
    pub state_features: Vec<f32>,
    /// One row per action, `ACTION_DIM` wide, row-major.
    pub action_features: Vec<f32>,
}

impl Observation {
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Row `index` of the action matrix.
    pub fn action_row(&self, index: usize) -> &[f32] {
        &self.action_features[index * ACTION_DIM..(index + 1) * ACTION_DIM]
    }
}

/// Bundle a position with its candidate actions.
///
/// Python's `observe` takes the legal actions from the game itself and raises
/// on an empty list; here the caller passes them, and an empty list simply
/// yields an empty matrix (there is nothing for the network to score).
pub fn observe(
    state: &State,
    data: &DataSet,
    actions: &[Action],
    actor: Option<HouseId>,
) -> Observation {
    let actor = actor.or(state.active_house);
    let mut matrix = vec![0.0f32; actions.len() * ACTION_DIM];
    let mut cache = RegionCache::new();
    for (row, action) in actions.iter().enumerate() {
        encode_action_into(
            state,
            data,
            action,
            actor,
            &mut matrix[row * ACTION_DIM..(row + 1) * ACTION_DIM],
            &mut cache,
        );
    }
    Observation {
        actor,
        actions: actions.to_vec(),
        state_features: encode_state(state, data, actor),
        action_features: matrix,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionKind;
    use crate::enums::{ActionStep, OrderKey, Phase, ORDER_SPECS};
    use crate::rng::Rng;
    use crate::setup::{initial_state, GameConfig};

    fn data() -> DataSet {
        DataSet::load_default().expect("bundled data must load")
    }

    /// A fresh six-house opening position, left in the Planning Phase.
    fn planning_state(data: &DataSet) -> State {
        let config = GameConfig::with_seed(1);
        let mut rng = Rng::from_seed(config.seed);
        let mut state = initial_state(data, &config, &mut rng);
        state.phase = Phase::Planning;
        state.active_house = Some(HouseId(0));
        state
    }

    #[test]
    fn layout_constants_match_python() {
        // Recorded from `src/agot/engine/observation.py`; change either side
        // only together (a checkpoint depends on these numbers).
        assert_eq!(STATE_DIM, 97);
        assert_eq!(ACTION_DIM, 141);
        assert_eq!(HOUSE_BLOCK, 13);
        assert_eq!(GLOBAL_BLOCK, 19);
        assert_eq!(REGION_BLOCK, 39);
        assert_eq!(CARD_BLOCK, 6);
        assert_eq!(ORDER_BLOCK, 16);
        assert_eq!(AMOUNT_BLOCK, 2);
        assert_eq!(MOVE_BLOCK, 6);
        assert_eq!(N_UNITS, 4);
        assert_eq!(N_ORDERS, 15);
        assert_eq!(N_PHASES, 11);
        assert_eq!(N_STEPS, 3);
        assert_eq!(N_SIDES, 3);
        assert_eq!(CARD_SLOTS, 7);
    }

    #[test]
    fn order_keys_match_python_layout() {
        // `ORDER_KEYS = tuple(ORDERS_BY_KEY)` in Python; an OrderKey is its slot.
        let keys: Vec<&str> = ORDER_SPECS.iter().map(|spec| spec.key).collect();
        assert_eq!(
            keys,
            vec![
                "march",
                "march_minus",
                "march_plus",
                "defend_1a",
                "defend_1b",
                "defend_2",
                "support_a",
                "support_b",
                "support_plus",
                "raid_a",
                "raid_b",
                "raid_star",
                "consolidate_a",
                "consolidate_b",
                "consolidate_star",
            ]
        );
    }

    #[test]
    fn static_heads_are_the_map_fractions() {
        let data = data();
        let tables = StaticTables::from_data(&data);
        let winterfell = data.map.lookup("winterfell").expect("winterfell");
        let head = tables.head(winterfell);
        // land 1 / sea 0 / port 0, power 1/3, supply 1/3, mustering 2/2.
        assert_eq!(head[..3], [1.0, 0.0, 0.0]);
        assert_eq!(head[3], (1.0f64 / 3.0) as f32);
        assert_eq!(head[4], (1.0f64 / 3.0) as f32);
        assert_eq!(head[5], 1.0);
        assert_eq!(tables.home(winterfell), Some(HouseId(2)));
        // `linked` is `host or sea`, which only ports carry: a plain land
        // region and an open sea are both unlinked.
        assert!(!tables.linked(winterfell));

        let kings_landing = data.map.lookup("kings_landing").expect("kings landing");
        let head = tables.head(kings_landing);
        assert_eq!(head[3], (2.0f64 / 3.0) as f32);
        assert_eq!(head[5], 1.0);
        assert_eq!(tables.home(kings_landing), None);

        // A port knows its host and its sea, and neither is a home area.
        let port = data.map.lookup("winterfell_port").expect("winterfell port");
        let head = tables.head(port);
        assert_eq!(head[..3], [0.0, 0.0, 1.0]);
        assert_eq!(tables.home(port), None);
        assert!(tables.linked(port));
        let sea = data.map.lookup("shipbreaker_bay").expect("shipbreaker bay");
        assert!(!tables.linked(sea));
    }

    #[test]
    fn state_vector_marks_the_acting_house_and_the_global_slots() {
        let data = data();
        let state = planning_state(&data);
        let vector = encode_state(&state, &data, Some(HouseId(2)));
        assert_eq!(vector.len(), STATE_DIM);

        // Baratheon and Stark are present: their blocks start with the marker.
        assert_eq!(vector[0], 1.0);
        assert_eq!(vector[2 * HOUSE_BLOCK], 1.0);
        // The acting-house flag sits in slot 12 of the block.
        assert_eq!(vector[2 * HOUSE_BLOCK + 12], 1.0);
        assert_eq!(vector[12], 0.0);

        let global = N_HOUSES * HOUSE_BLOCK;
        assert_eq!(vector[global], (1.0f64 / 10.0) as f32);
        assert_eq!(vector[global + 1 + Phase::Planning.index()], 1.0);
        assert_eq!(
            vector[global + 1 + N_PHASES + ActionStep::Raid.index()],
            1.0
        );
        assert_eq!(
            vector[global + 1 + N_PHASES + N_STEPS],
            (2.0f64 / 12.0) as f32
        );
        assert_eq!(vector[global + 1 + N_PHASES + N_STEPS + 1], 1.0);
        assert_eq!(
            vector[global + 1 + N_PHASES + N_STEPS + 2],
            (58.0f64 / 64.0) as f32
        );
    }

    #[test]
    fn short_handed_games_leave_absent_blocks_zeroed() {
        let data = data();
        let config = GameConfig {
            houses: smallvec::smallvec![HouseId(2), HouseId(3)],
            seed: 4,
            ..GameConfig::default()
        };
        let mut rng = Rng::from_seed(config.seed);
        let state = initial_state(&data, &config, &mut rng);
        let vector = encode_state(&state, &data, None);
        // Baratheon is absent: its whole block stays zero.
        assert!(vector[..HOUSE_BLOCK].iter().all(|value| *value == 0.0));
        assert_eq!(vector[2 * HOUSE_BLOCK], 1.0);
        assert_eq!(vector[3 * HOUSE_BLOCK], 1.0);
    }

    #[test]
    fn action_encoding_sets_only_the_relevant_slots() {
        let data = data();
        let state = planning_state(&data);
        let winterfell = data.map.lookup("winterfell").expect("winterfell");
        let action = Action::place_order(HouseId(2), winterfell, OrderKey(2)); // march_plus
        let mut cache = RegionCache::new();
        let mut out = vec![0.0f32; ACTION_DIM];
        encode_action_into(
            &state,
            &data,
            &action,
            Some(HouseId(2)),
            &mut out,
            &mut cache,
        );

        assert_eq!(out[ActionKind::PlaceOrder.index()], 1.0);
        assert_eq!(out[N_ACTION_KINDS + HouseId(2).index()], 1.0);
        assert_eq!(out[N_ACTION_KINDS + N_HOUSES], 1.0); // is_active
        let order_offset = N_ACTION_KINDS + N_HOUSES + 1 + 2 * REGION_BLOCK + CARD_BLOCK;
        assert_eq!(out[order_offset + OrderKey(2).index()], 1.0);
        assert_eq!(out[order_offset + N_ORDERS], (1.0f64 / 2.0) as f32);
        // Slots for fields the action does not carry stay zero.
        assert_eq!(out[order_offset + N_ORDERS + 1], 0.0);
        assert_eq!(out[ACTION_DIM - 1], 0.0);
    }

    /// Python: `test_support_side_is_encoded`.
    #[test]
    fn support_side_is_encoded() {
        let data = data();
        let state = planning_state(&data);
        let winterfell = data.map.lookup("winterfell").expect("winterfell");

        let attacking = Action::declare_support(HouseId(2), winterfell, Side::Attacker);
        let vector = encode_action(&state, &data, &attacking, Some(HouseId(2)));
        assert_eq!(vector.len(), ACTION_DIM);
        assert_eq!(
            vector[ACTION_DIM - N_SIDES..],
            [1.0, 0.0, 0.0],
            "the attacker slot is the first side"
        );

        let refusing = Action::declare_support(HouseId(2), winterfell, Side::Refuse);
        let vector = encode_action(&state, &data, &refusing, Some(HouseId(2)));
        assert_eq!(vector[ACTION_DIM - N_SIDES..], [0.0, 0.0, 1.0]);
    }

    /// Python: `test_region_block_initialises_every_slot`.
    #[test]
    fn region_block_initialises_every_slot() {
        let data = data();
        let mut state = planning_state(&data);
        let golden_sound = data
            .map
            .lookup("the_golden_sound")
            .expect("the golden sound");
        {
            let region = state.region_mut(golden_sound);
            region.order = OrderKey::from_name("march");
            region.order_house = Some(HouseId(0));
        }
        state.phase = Phase::Planning; // so the order is hidden from Stark

        let block = block_of(&state, &data, golden_sound, Some(HouseId(2)));
        let controller = &block[16..20];
        assert_eq!(
            controller.iter().sum::<f32>(),
            1.0,
            "exactly one controller flag must be set"
        );
        assert_eq!(controller[1], 0.0, "the reserved slot stays zero");
        let order_start = 20;
        assert!(
            block[order_start..order_start + N_ORDERS]
                .iter()
                .all(|value| *value == 0.0),
            "a hidden order writes its whole one-hot as zero"
        );
        assert_eq!(
            block[order_start + N_ORDERS],
            0.0,
            "the order-strength slot is written even while the order is hidden"
        );
        assert!(block.iter().all(|value| value.is_finite()));
    }

    /// Python: `test_encoded_actions_stay_bounded_across_a_game`.
    #[test]
    fn encoded_actions_stay_bounded_across_a_game() {
        use std::sync::Arc;

        use crate::game::Game;

        let data = Arc::new(data());
        let config = GameConfig {
            seed: 3,
            max_rounds: 3,
            ..GameConfig::default()
        };
        let mut game = Game::new(Arc::clone(&data), config).expect("game must start");
        let mut steps = 0;
        while !game.is_terminal() && steps < 60 {
            let actions = game.legal_actions();
            if actions.is_empty() {
                break;
            }
            for action in &actions {
                let vector = encode_action(&game.state, &data, action, None);
                assert!(vector.iter().all(|value| value.is_finite()));
                let largest = vector
                    .iter()
                    .fold(0.0f32, |max, value| max.max(value.abs()));
                assert!(largest < 100.0, "runaway feature for {action:?}: {largest}");
            }
            let action = actions[0];
            game.step(action).expect("a legal action applies");
            steps += 1;
        }
    }

    #[test]
    fn region_blocks_hide_other_houses_orders_until_the_reveal() {
        let data = data();
        let mut state = planning_state(&data);
        let winterfell = data.map.lookup("winterfell").expect("winterfell");
        state.region_mut(winterfell).order = Some(OrderKey(5)); // defend_2
        state.region_mut(winterfell).order_house = Some(HouseId(2));

        // In Planning the order belongs to Stark, not to the actor.
        let block = block_of(&state, &data, winterfell, Some(HouseId(0)));
        let order_offset = 6 + N_UNITS + 6 + 4;
        assert!(block[order_offset..order_offset + N_ORDERS]
            .iter()
            .all(|value| *value == 0.0));
        assert_eq!(block[order_offset + N_ORDERS], 0.0);

        // Stark itself sees it, and so does everyone once orders are revealed.
        let own = block_of(&state, &data, winterfell, Some(HouseId(2)));
        assert_eq!(own[order_offset + 5], 1.0);
        assert_eq!(own[order_offset + N_ORDERS], (2.0f64 / 2.0) as f32);
        state.phase = Phase::Action;
        let revealed = block_of(&state, &data, winterfell, Some(HouseId(0)));
        assert_eq!(revealed[order_offset + 5], 1.0);
    }

    #[test]
    fn the_region_cache_is_transparent() {
        let data = data();
        let state = planning_state(&data);
        let actions: Vec<Action> = state
            .region_ids()
            .map(|id| Action::place_order(HouseId(0), id, OrderKey(0)))
            .collect();

        let mut shared = RegionCache::new();
        let mut memoised: Vec<Vec<f32>> = Vec::new();
        for action in &actions {
            let mut out = vec![0.0f32; ACTION_DIM];
            encode_action_into(
                &state,
                &data,
                action,
                Some(HouseId(0)),
                &mut out,
                &mut shared,
            );
            memoised.push(out);
        }
        for (action, expected) in actions.iter().zip(memoised.iter()) {
            let fresh = encode_action(&state, &data, action, Some(HouseId(0)));
            assert_eq!(
                fresh.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn unknown_cards_keep_only_the_presence_marker() {
        let data = data();
        let state = planning_state(&data);
        let action = Action::play_house_card(HouseId(2), Some(CardId(200)));
        let mut out = vec![0.0f32; ACTION_DIM];
        let mut cache = RegionCache::new();
        encode_action_into(
            &state,
            &data,
            &action,
            Some(HouseId(2)),
            &mut out,
            &mut cache,
        );
        let card_offset = N_ACTION_KINDS + N_HOUSES + 1 + 2 * REGION_BLOCK;
        assert_eq!(
            &out[card_offset..card_offset + CARD_BLOCK],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn observe_bundles_a_row_per_action() {
        let data = data();
        let state = planning_state(&data);
        let winterfell = data.map.lookup("winterfell").expect("winterfell");
        let actions = vec![
            Action::place_order(HouseId(0), winterfell, OrderKey(0)),
            Action::pass(HouseId(0)),
        ];
        let observation = observe(&state, &data, &actions, Some(HouseId(0)));
        assert_eq!(observation.len(), 2);
        assert_eq!(observation.state_features.len(), STATE_DIM);
        assert_eq!(observation.action_features.len(), 2 * ACTION_DIM);
        assert_eq!(observation.action_row(1)[ActionKind::Pass.index()], 1.0);
        assert_eq!(observation.actor, Some(HouseId(0)));
    }

    /// The region block of one region, straight through the cache.
    fn block_of(
        state: &State,
        data: &DataSet,
        id: RegionId,
        actor: Option<HouseId>,
    ) -> [f32; REGION_BLOCK] {
        let mut cache = RegionCache::new();
        cache.block(static_tables(data), state, id, actor)
    }
}
