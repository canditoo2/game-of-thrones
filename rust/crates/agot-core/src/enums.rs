//! Enumerations and small value types shared by the engine.
//!
//! Port of `src/agot/engine/enums.py`. Declaration order is load-bearing: the
//! observation encoder builds one-hot blocks from it, so it must stay identical
//! to the Python original (see `observation.py`, which derives `N_PHASES`,
//! `N_STEPS`, `N_ORDERS`, `N_UNITS` and the `HOUSE_ORDER` layout from it).
//!
//! String parsing lives in inherent `from_name` methods rather than a
//! `FromStr` impl: they are used by the JSON loaders and tests, where an
//! `Option` composes better than a `Result` with an error type that would never
//! carry information.

use std::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Canonical house order, mirroring `observation.HOUSE_ORDER`.
///
/// Every per-house block in an encoded vector, and every per-house value in a
/// search value vector, is indexed by this order. It is fixed so that a
/// checkpoint trained on one table can be loaded for another.
pub const HOUSE_ORDER: [&str; 6] = [
    "baratheon",
    "lannister",
    "stark",
    "greyjoy",
    "tyrell",
    "martell",
];

pub const N_HOUSES: usize = HOUSE_ORDER.len();

/// A house, as an index into [`HOUSE_ORDER`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct HouseId(pub u8);

impl HouseId {
    pub const ALL: [HouseId; N_HOUSES] = [
        HouseId(0),
        HouseId(1),
        HouseId(2),
        HouseId(3),
        HouseId(4),
        HouseId(5),
    ];

    pub fn from_name(name: &str) -> Option<HouseId> {
        HOUSE_ORDER
            .iter()
            .position(|candidate| *candidate == name)
            .map(|index| HouseId(index as u8))
    }

    pub fn name(self) -> &'static str {
        HOUSE_ORDER[self.0 as usize]
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for HouseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A board region, as an index into the loaded map's region list.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RegionId(pub u16);

impl RegionId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Index into one house's seven-card deck (see [`crate::map::HouseCardData`]).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CardId(pub u8);

// ---------------------------------------------------------------------------
// Terrain and units
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Terrain {
    Land = 0,
    Sea = 1,
    Port = 2,
}

impl Terrain {
    pub const ALL: [Terrain; 3] = [Terrain::Land, Terrain::Sea, Terrain::Port];

    pub fn as_str(self) -> &'static str {
        match self {
            Terrain::Land => "land",
            Terrain::Sea => "sea",
            Terrain::Port => "port",
        }
    }

    pub fn from_name(value: &str) -> Option<Terrain> {
        Terrain::ALL.into_iter().find(|t| t.as_str() == value)
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Terrain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The four unit kinds. Order matches `observation.UNIT_ORDER`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum UnitType {
    Footman = 0,
    Knight = 1,
    Siege = 2,
    Ship = 3,
}

impl UnitType {
    pub const ALL: [UnitType; 4] = [
        UnitType::Footman,
        UnitType::Knight,
        UnitType::Siege,
        UnitType::Ship,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            UnitType::Footman => "footman",
            UnitType::Knight => "knight",
            UnitType::Siege => "siege",
            UnitType::Ship => "ship",
        }
    }

    /// The plural key used in `houses.json`'s `starting_units`.
    pub fn as_json_plural(self) -> &'static str {
        match self {
            UnitType::Footman => "footmen",
            UnitType::Knight => "knights",
            UnitType::Siege => "siege",
            UnitType::Ship => "ships",
        }
    }

    pub fn from_name(value: &str) -> Option<UnitType> {
        UnitType::ALL.into_iter().find(|u| u.as_str() == value)
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Printed combat strength.
    ///
    /// A Siege Engine's printed 4 only applies when attacking (or supporting an
    /// attack against) an area containing a Castle or Stronghold; otherwise it
    /// contributes 0. See `combat.units_strength` in the Python engine.
    #[inline]
    pub fn combat_strength(self) -> u8 {
        match self {
            UnitType::Footman => 1,
            UnitType::Knight => 2,
            UnitType::Siege => 4,
            UnitType::Ship => 1,
        }
    }

    /// Whether the unit may occupy land regions (ships may not).
    #[inline]
    pub fn is_land(self) -> bool {
        !matches!(self, UnitType::Ship)
    }
}

impl fmt::Display for UnitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Strength printed on every House's Garrison token.
pub const GARRISON_STRENGTH: u8 = 2;

/// Mustering point cost to recruit one unit, indexed by [`UnitType::index`].
/// Upgrading a Footman to a Knight or Siege Engine costs 1 point instead.
pub const MUSTER_COST: [u8; 4] = [1, 2, 2, 1];

/// Power tokens supplied with each house.
///
/// The box holds 120 Power tokens, twenty per house, and the same "Exceeding
/// the Components Provided" rule applies: a house at twenty cannot gain more.
pub const POWER_LIMIT: u8 = 20;

/// Plastic figures supplied with each house, indexed by [`UnitType::index`].
///
/// This is the physical component cap from the rulebook ("Exceeding the
/// Components Provided", p. 4): a player may never have more figures of a type
/// on the board than the box contains. It is independent of Supply, which caps
/// army *sizes*; destroyed units return to the pool and can be mustered again.
pub const UNIT_LIMITS: [u8; 4] = [10, 5, 2, 6];

/// Units that may occupy land.
pub const LAND_UNITS: [UnitType; 3] = [UnitType::Footman, UnitType::Knight, UnitType::Siege];

// ---------------------------------------------------------------------------
// Orders
// ---------------------------------------------------------------------------

/// The five order kinds.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum OrderType {
    March = 0,
    Defend = 1,
    Support = 2,
    Raid = 3,
    ConsolidatePower = 4,
}

impl OrderType {
    pub const ALL: [OrderType; 5] = [
        OrderType::March,
        OrderType::Defend,
        OrderType::Support,
        OrderType::Raid,
        OrderType::ConsolidatePower,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            OrderType::March => "march",
            OrderType::Defend => "defend",
            OrderType::Support => "support",
            OrderType::Raid => "raid",
            OrderType::ConsolidatePower => "consolidate_power",
        }
    }

    pub fn from_name(value: &str) -> Option<OrderType> {
        OrderType::ALL.into_iter().find(|t| t.as_str() == value)
    }
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single order token type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OrderSpec {
    pub key: &'static str,
    pub order_type: OrderType,
    /// Combat modifier for march/defend orders, or the extra support strength
    /// of a starred support order.
    pub strength: i8,
    /// Starred tokens are limited by a house's King's Court position.
    pub starred: bool,
}

/// The fifteen order tokens every house owns, in the Python order.
///
/// The order is the encoder's `ORDER_KEYS` layout, so it must not change.
pub const ORDER_SPECS: [OrderSpec; 15] = [
    OrderSpec {
        key: "march",
        order_type: OrderType::March,
        strength: 0,
        starred: false,
    },
    OrderSpec {
        key: "march_minus",
        order_type: OrderType::March,
        strength: -1,
        starred: false,
    },
    OrderSpec {
        key: "march_plus",
        order_type: OrderType::March,
        strength: 1,
        starred: true,
    },
    OrderSpec {
        key: "defend_1a",
        order_type: OrderType::Defend,
        strength: 1,
        starred: false,
    },
    OrderSpec {
        key: "defend_1b",
        order_type: OrderType::Defend,
        strength: 1,
        starred: false,
    },
    OrderSpec {
        key: "defend_2",
        order_type: OrderType::Defend,
        strength: 2,
        starred: true,
    },
    OrderSpec {
        key: "support_a",
        order_type: OrderType::Support,
        strength: 0,
        starred: false,
    },
    OrderSpec {
        key: "support_b",
        order_type: OrderType::Support,
        strength: 0,
        starred: false,
    },
    OrderSpec {
        key: "support_plus",
        order_type: OrderType::Support,
        strength: 1,
        starred: true,
    },
    OrderSpec {
        key: "raid_a",
        order_type: OrderType::Raid,
        strength: 0,
        starred: false,
    },
    OrderSpec {
        key: "raid_b",
        order_type: OrderType::Raid,
        strength: 0,
        starred: false,
    },
    OrderSpec {
        key: "raid_star",
        order_type: OrderType::Raid,
        strength: 0,
        starred: true,
    },
    OrderSpec {
        key: "consolidate_a",
        order_type: OrderType::ConsolidatePower,
        strength: 0,
        starred: false,
    },
    OrderSpec {
        key: "consolidate_b",
        order_type: OrderType::ConsolidatePower,
        strength: 0,
        starred: false,
    },
    OrderSpec {
        key: "consolidate_star",
        order_type: OrderType::ConsolidatePower,
        strength: 0,
        starred: true,
    },
];

pub const N_ORDERS: usize = ORDER_SPECS.len();

/// Index into [`ORDER_SPECS`]: the Rust stand-in for Python's order-key string.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct OrderKey(pub u8);

impl OrderKey {
    pub const ALL: [OrderKey; N_ORDERS] = [
        OrderKey(0),
        OrderKey(1),
        OrderKey(2),
        OrderKey(3),
        OrderKey(4),
        OrderKey(5),
        OrderKey(6),
        OrderKey(7),
        OrderKey(8),
        OrderKey(9),
        OrderKey(10),
        OrderKey(11),
        OrderKey(12),
        OrderKey(13),
        OrderKey(14),
    ];

    #[inline]
    pub fn spec(self) -> &'static OrderSpec {
        &ORDER_SPECS[self.0 as usize]
    }

    #[inline]
    pub fn key(self) -> &'static str {
        self.spec().key
    }

    #[inline]
    pub fn order_type(self) -> OrderType {
        self.spec().order_type
    }

    #[inline]
    pub fn starred(self) -> bool {
        self.spec().starred
    }

    #[inline]
    pub fn strength(self) -> i8 {
        self.spec().strength
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_name(key: &str) -> Option<OrderKey> {
        ORDER_SPECS
            .iter()
            .position(|spec| spec.key == key)
            .map(|index| OrderKey(index as u8))
    }
}

impl fmt::Display for OrderKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.key())
    }
}

/// How many starred orders a house may place, by its 1-based position on the
/// King's Court track.
pub const STAR_ALLOWANCE_BY_COURT_POSITION: [u8; 6] = [3, 3, 2, 1, 0, 0];

/// Maximum army sizes a house may support, indexed by supply level. Each row
/// lists the largest army it may field, then the next largest, and so on.
pub const SUPPLY_LIMITS: [&[u8]; 7] = [
    &[2, 2],
    &[3, 2],
    &[3, 2, 2],
    &[3, 2, 2, 2],
    &[3, 3, 2, 2],
    &[4, 3, 2, 2],
    &[4, 3, 2, 2, 2],
];

// ---------------------------------------------------------------------------
// Phases and tracks
// ---------------------------------------------------------------------------

/// High-level round phases, in resolution order.
///
/// Declaration order defines the encoder's phase one-hot block.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Phase {
    Westeros = 0,
    WesterosChoice = 1,
    Bidding = 2,
    WildlingAttack = 3,
    Mustering = 4,
    Planning = 5,
    Raven = 6,
    Action = 7,
    Combat = 8,
    Cleanup = 9,
    GameOver = 10,
}

impl Phase {
    pub const ALL: [Phase; 11] = [
        Phase::Westeros,
        Phase::WesterosChoice,
        Phase::Bidding,
        Phase::WildlingAttack,
        Phase::Mustering,
        Phase::Planning,
        Phase::Raven,
        Phase::Action,
        Phase::Combat,
        Phase::Cleanup,
        Phase::GameOver,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Westeros => "westeros",
            Phase::WesterosChoice => "westeros_choice",
            Phase::Bidding => "bidding",
            Phase::WildlingAttack => "wildling_attack",
            Phase::Mustering => "mustering",
            Phase::Planning => "planning",
            Phase::Raven => "raven",
            Phase::Action => "action",
            Phase::Combat => "combat",
            Phase::Cleanup => "cleanup",
            Phase::GameOver => "game_over",
        }
    }

    pub fn from_name(value: &str) -> Option<Phase> {
        Phase::ALL.into_iter().find(|p| p.as_str() == value)
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub const N_PHASES: usize = Phase::ALL.len();

/// Sub-steps of the action phase, in resolution order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum ActionStep {
    Raid = 0,
    March = 1,
    ConsolidatePower = 2,
}

impl ActionStep {
    pub const ALL: [ActionStep; 3] = [
        ActionStep::Raid,
        ActionStep::March,
        ActionStep::ConsolidatePower,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ActionStep::Raid => "raid",
            ActionStep::March => "march",
            ActionStep::ConsolidatePower => "consolidate_power",
        }
    }

    pub fn from_name(value: &str) -> Option<ActionStep> {
        ActionStep::ALL.into_iter().find(|s| s.as_str() == value)
    }

    /// The order type resolved in this step.
    #[inline]
    pub fn order_type(self) -> OrderType {
        match self {
            ActionStep::Raid => OrderType::Raid,
            ActionStep::March => OrderType::March,
            ActionStep::ConsolidatePower => OrderType::ConsolidatePower,
        }
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

pub const N_STEPS: usize = ActionStep::ALL.len();

/// The three influence tracks.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Track {
    IronThrone = 0,
    Fiefdoms = 1,
    KingsCourt = 2,
}

impl Track {
    pub const ALL: [Track; 3] = [Track::IronThrone, Track::Fiefdoms, Track::KingsCourt];

    pub fn as_str(self) -> &'static str {
        match self {
            Track::IronThrone => "iron_throne",
            Track::Fiefdoms => "fiefdoms",
            Track::KingsCourt => "kings_court",
        }
    }

    pub fn from_name(value: &str) -> Option<Track> {
        Track::ALL.into_iter().find(|t| t.as_str() == value)
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Track {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_specs_match_python_layout() {
        // The encoder's ORDER_KEYS block is built from this order.
        assert_eq!(ORDER_SPECS[0].key, "march");
        assert_eq!(ORDER_SPECS[14].key, "consolidate_star");
        assert_eq!(N_ORDERS, 15);
        assert_eq!(OrderKey::from_name("march_plus"), Some(OrderKey(2)));
        assert!(OrderKey(2).starred());
        assert_eq!(OrderKey(1).strength(), -1);
        assert_eq!(OrderKey(5).order_type(), OrderType::Defend);
    }

    #[test]
    fn enum_orders_are_the_encoder_orders() {
        assert_eq!(Phase::Westeros.index(), 0);
        assert_eq!(Phase::GameOver.index(), 10);
        assert_eq!(ActionStep::Raid.index(), 0);
        assert_eq!(UnitType::Ship.index(), 3);
        assert_eq!(Track::KingsCourt.index(), 2);
        assert_eq!(HouseId::from_name("martell"), Some(HouseId(5)));
        assert_eq!(HouseId(0).name(), "baratheon");
    }

    #[test]
    fn star_allowance_and_supply_limits() {
        assert_eq!(STAR_ALLOWANCE_BY_COURT_POSITION[0], 3);
        assert_eq!(STAR_ALLOWANCE_BY_COURT_POSITION[5], 0);
        assert_eq!(SUPPLY_LIMITS[0], &[2, 2]);
        assert_eq!(SUPPLY_LIMITS[6], &[4, 3, 2, 2, 2]);
    }
}
