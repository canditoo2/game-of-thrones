//! The action space.
//!
//! Port of `src/agot/engine/actions.py`. Actions are small immutable records:
//! the action space is combinatorial, so the network scores a *variable* number
//! of candidate actions per decision point rather than emitting a fixed
//! distribution over a fixed move list.

use crate::enums::{CardId, HouseId, OrderKey, RegionId, UnitType};

/// Every kind of decision the engine can ask a player to make.
///
/// Declaration order matches the Python `ActionKind`, because the encoder's
/// one-hot block is built from it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum ActionKind {
    Pass = 0,
    PlaceOrder = 1,
    ResolveRaid = 2,
    March = 3,
    ResolveConsolidate = 4,
    ConsolidateMuster = 5,
    PlayHouseCard = 6,
    UseBlade = 7,
    RavenSwap = 8,
    RavenDeckTop = 9,
    RavenDeckBottom = 10,
    Retreat = 11,
    Disband = 12,
    /// The defeated player chooses which unit is destroyed for one casualty.
    ChooseCasualty = 13,
    /// A player declares which side an adjacent Support order helps.
    DeclareSupport = 14,
    Muster = 15,
    Upgrade = 16,
    Bid = 17,
    /// An Influence-token holder picks how a holders'-choice card resolves.
    WesterosChoice = 18,
}

impl ActionKind {
    pub const ALL: [ActionKind; 19] = [
        ActionKind::Pass,
        ActionKind::PlaceOrder,
        ActionKind::ResolveRaid,
        ActionKind::March,
        ActionKind::ResolveConsolidate,
        ActionKind::ConsolidateMuster,
        ActionKind::PlayHouseCard,
        ActionKind::UseBlade,
        ActionKind::RavenSwap,
        ActionKind::RavenDeckTop,
        ActionKind::RavenDeckBottom,
        ActionKind::Retreat,
        ActionKind::Disband,
        ActionKind::ChooseCasualty,
        ActionKind::DeclareSupport,
        ActionKind::Muster,
        ActionKind::Upgrade,
        ActionKind::Bid,
        ActionKind::WesterosChoice,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ActionKind::Pass => "pass",
            ActionKind::PlaceOrder => "place_order",
            ActionKind::ResolveRaid => "resolve_raid",
            ActionKind::March => "march",
            ActionKind::ResolveConsolidate => "resolve_consolidate",
            ActionKind::ConsolidateMuster => "consolidate_muster",
            ActionKind::PlayHouseCard => "play_house_card",
            ActionKind::UseBlade => "use_blade",
            ActionKind::RavenSwap => "raven_swap",
            ActionKind::RavenDeckTop => "raven_deck_top",
            ActionKind::RavenDeckBottom => "raven_deck_bottom",
            ActionKind::Retreat => "retreat",
            ActionKind::Disband => "disband",
            ActionKind::ChooseCasualty => "choose_casualty",
            ActionKind::DeclareSupport => "declare_support",
            ActionKind::Muster => "muster",
            ActionKind::Upgrade => "upgrade",
            ActionKind::Bid => "bid",
            ActionKind::WesterosChoice => "westeros_choice",
        }
    }

    pub fn from_name(value: &str) -> Option<ActionKind> {
        ActionKind::ALL.into_iter().find(|k| k.as_str() == value)
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

pub const N_ACTION_KINDS: usize = ActionKind::ALL.len();

/// Which side a Support order helps, in the Python `SUPPORT_SIDES` order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Side {
    Attacker = 0,
    Defender = 1,
    Refuse = 2,
}

impl Side {
    pub const ALL: [Side; 3] = [Side::Attacker, Side::Defender, Side::Refuse];

    pub fn as_str(self) -> &'static str {
        match self {
            Side::Attacker => "attacker",
            Side::Defender => "defender",
            Side::Refuse => "refuse",
        }
    }

    pub fn from_name(value: &str) -> Option<Side> {
        Side::ALL.into_iter().find(|s| s.as_str() == value)
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

pub const N_SIDES: usize = Side::ALL.len();

/// A single decision.
///
/// Only the fields relevant to `kind` are populated; the rest stay `None`.
/// Keeping one flat record (rather than a variant per kind) mirrors the Python
/// original: it makes actions cheap to copy, hashable for the PIMC root
/// aggregation, and trivial to encode as network input.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Action {
    pub kind: ActionKind,
    pub house: Option<HouseId>,
    pub region: Option<RegionId>,
    pub target: Option<RegionId>,
    pub order: Option<OrderKey>,
    pub card: Option<CardId>,
    pub unit: Option<UnitType>,
    pub side: Option<Side>,
    pub amount: Option<u8>,
    /// Units that move, as a count per entry of `UNIT_ORDER`. `None` means
    /// "every eligible unit".
    pub units: Option<[u8; 4]>,
    /// Whether a March uses a Power token to keep control of the area it
    /// vacates.
    pub establish: bool,
}

impl Action {
    /// A bare action of `kind` with every payload field empty.
    pub const fn new(kind: ActionKind) -> Action {
        Action {
            kind,
            house: None,
            region: None,
            target: None,
            order: None,
            card: None,
            unit: None,
            side: None,
            amount: None,
            units: None,
            establish: false,
        }
    }

    pub fn pass(house: HouseId) -> Action {
        Action {
            kind: ActionKind::Pass,
            house: Some(house),
            ..Action::new(ActionKind::Pass)
        }
    }

    pub fn place_order(house: HouseId, region: RegionId, order: OrderKey) -> Action {
        Action {
            kind: ActionKind::PlaceOrder,
            house: Some(house),
            region: Some(region),
            order: Some(order),
            ..Action::new(ActionKind::PlaceOrder)
        }
    }

    pub fn resolve_raid(house: HouseId, region: RegionId, target: RegionId) -> Action {
        Action {
            kind: ActionKind::ResolveRaid,
            house: Some(house),
            region: Some(region),
            target: Some(target),
            ..Action::new(ActionKind::ResolveRaid)
        }
    }

    pub fn march(
        house: HouseId,
        region: RegionId,
        target: RegionId,
        units: Option<[u8; 4]>,
        establish: bool,
    ) -> Action {
        Action {
            kind: ActionKind::March,
            house: Some(house),
            region: Some(region),
            target: Some(target),
            units,
            establish,
            ..Action::new(ActionKind::March)
        }
    }

    pub fn resolve_consolidate(house: HouseId, region: RegionId) -> Action {
        Action {
            kind: ActionKind::ResolveConsolidate,
            house: Some(house),
            region: Some(region),
            ..Action::new(ActionKind::ResolveConsolidate)
        }
    }

    pub fn consolidate_muster(house: HouseId, region: RegionId) -> Action {
        Action {
            kind: ActionKind::ConsolidateMuster,
            house: Some(house),
            region: Some(region),
            ..Action::new(ActionKind::ConsolidateMuster)
        }
    }

    pub fn play_house_card(house: HouseId, card: Option<CardId>) -> Action {
        Action {
            kind: ActionKind::PlayHouseCard,
            house: Some(house),
            card,
            ..Action::new(ActionKind::PlayHouseCard)
        }
    }

    pub fn use_blade(house: HouseId) -> Action {
        Action {
            kind: ActionKind::UseBlade,
            house: Some(house),
            ..Action::new(ActionKind::UseBlade)
        }
    }

    pub fn raven_swap(house: HouseId, region: RegionId, order: OrderKey) -> Action {
        Action {
            kind: ActionKind::RavenSwap,
            house: Some(house),
            region: Some(region),
            order: Some(order),
            ..Action::new(ActionKind::RavenSwap)
        }
    }

    pub fn raven_deck(house: HouseId, to_bottom: bool) -> Action {
        let kind = if to_bottom {
            ActionKind::RavenDeckBottom
        } else {
            ActionKind::RavenDeckTop
        };
        Action {
            kind,
            house: Some(house),
            ..Action::new(kind)
        }
    }

    pub fn retreat(house: HouseId, region: RegionId, target: RegionId) -> Action {
        Action {
            kind: ActionKind::Retreat,
            house: Some(house),
            region: Some(region),
            target: Some(target),
            ..Action::new(ActionKind::Retreat)
        }
    }

    pub fn disband(house: HouseId, region: RegionId, unit: UnitType) -> Action {
        Action {
            kind: ActionKind::Disband,
            house: Some(house),
            region: Some(region),
            unit: Some(unit),
            ..Action::new(ActionKind::Disband)
        }
    }

    /// Destroy one named unit for a battle casualty.
    pub fn choose_casualty(house: HouseId, region: RegionId, unit: UnitType) -> Action {
        Action {
            kind: ActionKind::ChooseCasualty,
            house: Some(house),
            region: Some(region),
            unit: Some(unit),
            ..Action::new(ActionKind::ChooseCasualty)
        }
    }

    /// Declare which side a Support order in `region` helps.
    pub fn declare_support(house: HouseId, region: RegionId, side: Side) -> Action {
        Action {
            kind: ActionKind::DeclareSupport,
            house: Some(house),
            region: Some(region),
            side: Some(side),
            ..Action::new(ActionKind::DeclareSupport)
        }
    }

    pub fn muster(
        house: HouseId,
        region: RegionId,
        unit: UnitType,
        target: Option<RegionId>,
    ) -> Action {
        Action {
            kind: ActionKind::Muster,
            house: Some(house),
            region: Some(region),
            unit: Some(unit),
            target,
            ..Action::new(ActionKind::Muster)
        }
    }

    pub fn upgrade(house: HouseId, region: RegionId, unit: UnitType) -> Action {
        Action {
            kind: ActionKind::Upgrade,
            house: Some(house),
            region: Some(region),
            unit: Some(unit),
            ..Action::new(ActionKind::Upgrade)
        }
    }

    pub fn bid(house: HouseId, amount: u8) -> Action {
        Action {
            kind: ActionKind::Bid,
            house: Some(house),
            amount: Some(amount),
            ..Action::new(ActionKind::Bid)
        }
    }

    /// The holder's pick for a holders'-choice Westeros card.
    pub fn westeros_choice(house: HouseId, card: CardId) -> Action {
        Action {
            kind: ActionKind::WesterosChoice,
            house: Some(house),
            card: Some(card),
            ..Action::new(ActionKind::WesterosChoice)
        }
    }

    /// Doran Martell's Influence-track choice, encoded in `amount`.
    ///
    /// The action kind is reused so the frozen action-feature layout does not
    /// change; the amount block distinguishes the three tracks.
    pub fn doran_choice(house: HouseId, track: u8) -> Action {
        Action {
            kind: ActionKind::WesterosChoice,
            house: Some(house),
            amount: Some(track),
            ..Action::new(ActionKind::WesterosChoice)
        }
    }

    /// The same action, with a region attached.
    ///
    /// Python's `Action.pass_(house, region=...)` names the area whose order is
    /// being declined; the flat record carries it on `region`.
    pub fn with_region(self, region: RegionId) -> Action {
        Action {
            region: Some(region),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_match_python_order() {
        assert_eq!(ActionKind::Pass.index(), 0);
        assert_eq!(ActionKind::WesterosChoice.index(), 18);
        assert_eq!(N_ACTION_KINDS, 19);
        assert_eq!(ActionKind::from_name("march"), Some(ActionKind::March));
    }

    #[test]
    fn constructors_fill_only_their_fields() {
        let action = Action::place_order(HouseId(2), RegionId(7), OrderKey(0));
        assert_eq!(action.kind, ActionKind::PlaceOrder);
        assert_eq!(action.house, Some(HouseId(2)));
        assert_eq!(action.region, Some(RegionId(7)));
        assert_eq!(action.order, Some(OrderKey(0)));
        assert_eq!(action.target, None);
        assert!(!action.establish);

        let bid = Action::bid(HouseId(1), 4);
        assert_eq!(bid.amount, Some(4));
        assert_eq!(bid.region, None);
    }

    #[test]
    fn actions_hash_by_value() {
        // PIMC root aggregation keys on the action; equal actions must collide.
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Action::bid(HouseId(1), 3));
        assert!(set.contains(&Action::bid(HouseId(1), 3)));
        assert!(!set.contains(&Action::bid(HouseId(1), 4)));
    }
}
