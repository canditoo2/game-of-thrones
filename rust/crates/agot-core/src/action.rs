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

/// Synthetic context tags for generic choice actions.
///
/// Generic House-card and Wildling sub-choices reuse a handful of action kinds, so
/// their encoder rows would otherwise be indistinguishable. The tag travels in
/// [`Action::order`] (an existing field, so `ACTION_DIM` is unchanged) and the
/// engine never reads it: `apply_choice` dispatches on the pending
/// [`crate::state::ChoiceTask`], not on the tag. Normal order tokens and normal
/// battle House cards keep `order: None`.
///
/// The values are the cross-agent contract: `agot.choice_context.CHOICE_TAGS`
/// in Python and the encoder's ORDER_BLOCK read this exact numbering, so append
/// new tags only in unused slots and never renumber. There are fifteen order
/// slots (`N_ORDERS`) and no more, so a new choice either reuses a tag or grows
/// the observation layer deliberately.
pub mod choice_context {
    /// Queen of Thorns: which adjacent enemy Order to remove.
    pub const QUEEN_ORDER: u8 = 0;
    /// Cersei Lannister: which of the loser's Orders to remove.
    pub const CERSEI_ORDER: u8 = 1;
    /// Renly Baratheon: which Footman is upgraded.
    pub const RENLY_FOOTMAN: u8 = 2;
    /// The Horde Descends: which fort takes the losses.
    pub const HORDE_FORT: u8 = 3;
    /// The Horde Descends / Wildling reward: which fort musters.
    pub const HORDE_MUSTER: u8 = 4;
    /// King Beyond the Wall: the other houses pick Fiefdoms or King's Court.
    pub const KING_BEYOND_OTHER: u8 = 5;
    /// King Beyond the Wall: the chosen house picks its highest track.
    pub const KING_BEYOND_HIGHEST: u8 = 6;
    /// Preemptive Raid: the A/B option and its follow-up track choice.
    pub const PREEMPTIVE: u8 = 7;
    /// Crow Killers: which Knight becomes a Footman.
    pub const CROW_KNIGHT: u8 = 8;
    /// Crow Killers: which Footman becomes a Knight.
    pub const CROW_UPGRADE: u8 = 9;
    /// A generic destroy-units task (not restricted to one fort).
    pub const GENERIC_DESTROY: u8 = 10;
    /// Tyrion Lannister: the controller declines or uses the card.
    pub const TYRION_USE: u8 = 11;
    /// Tyrion Lannister: the target's mandatory replacement card.
    pub const TYRION_CARD: u8 = 12;
    /// Aeron Damphair: the optional paid replacement card.
    pub const AERON_CARD: u8 = 13;
    /// Patchface / Massing at Milkwater / Mammoth Riders: card discard or retrieve.
    pub const CARD_DISCARD_OR_RETRIEVE: u8 = 14;

    /// Every tag, for tests and for the observation layer's documentation.
    pub const ALL: [u8; 15] = [
        QUEEN_ORDER,
        CERSEI_ORDER,
        RENLY_FOOTMAN,
        HORDE_FORT,
        HORDE_MUSTER,
        KING_BEYOND_OTHER,
        KING_BEYOND_HIGHEST,
        PREEMPTIVE,
        CROW_KNIGHT,
        CROW_UPGRADE,
        GENERIC_DESTROY,
        TYRION_USE,
        TYRION_CARD,
        AERON_CARD,
        CARD_DISCARD_OR_RETRIEVE,
    ];
}

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
    /// The house whose deck `card` was drawn from, when that is not `house`.
    ///
    /// A generic card choice can name a card from another house's deck:
    /// Patchface's controller discards a card from the opponent's hand, so the
    /// action belongs to the Patchface player but the card id has to be looked
    /// up in the target's deck. `None` means the card belongs to `house`.
    pub card_house: Option<HouseId>,
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
            card_house: None,
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

    /// A House-card action naming a card from another house's deck.
    ///
    /// Patchface (and any future "play a card from the opponent's hand" text)
    /// needs this: `house` is the player acting, `card_house` is the deck the
    /// card id belongs to. See [`Action::card_house`].
    pub fn play_house_card_owned(house: HouseId, card: CardId, card_house: HouseId) -> Action {
        Action {
            kind: ActionKind::PlayHouseCard,
            house: Some(house),
            card: Some(card),
            card_house: Some(card_house),
            ..Action::new(ActionKind::PlayHouseCard)
        }
    }

    /// The same action, with the deck of a cross-house card recorded.
    #[must_use]
    pub fn with_card_house(self, card_house: HouseId) -> Action {
        Action {
            card_house: Some(card_house),
            ..self
        }
    }

    /// Tag a generic choice with one of the [`choice_context`] constants.
    ///
    /// The tag lives in `order`, which the choice handlers ignore; it only
    /// enriches the encoder's ORDER_BLOCK so different choices are
    /// distinguishable. Passing a tag here never makes an order token real.
    #[must_use]
    pub fn with_context(self, tag: u8) -> Action {
        Action {
            order: Some(OrderKey(tag)),
            ..self
        }
    }

    /// The [`choice_context`] tag of a generic choice action, if it has one.
    #[inline]
    pub fn choice_context(self) -> Option<u8> {
        self.order.map(|key| key.0)
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

    /// A Supply disband aimed at a routed unit (`routed=true`) or a ready one.
    pub fn disband_state(house: HouseId, region: RegionId, unit: UnitType, routed: bool) -> Action {
        Action {
            kind: ActionKind::Disband,
            house: Some(house),
            region: Some(region),
            unit: Some(unit),
            amount: Some(u8::from(routed)),
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

    /// The Iron Throne holder's pick among tied Wildling bidders.
    ///
    /// Like Doran's choice this reuses `WesterosChoice` and stores the
    /// candidate index in `amount`, keeping the action-feature layout stable.
    pub fn wildling_tie_choice(house: HouseId, candidate: u8) -> Action {
        Action {
            kind: ActionKind::WesterosChoice,
            house: Some(house),
            amount: Some(candidate),
            ..Action::new(ActionKind::WesterosChoice)
        }
    }

    /// A generic card/text choice among regions or options.
    ///
    /// The action-feature layout is frozen, so these choices reuse
    /// `WesterosChoice`; the pending choice task supplies the meaning.
    pub fn choice_region(house: HouseId, region: RegionId) -> Action {
        Action {
            kind: ActionKind::WesterosChoice,
            house: Some(house),
            region: Some(region),
            target: Some(region),
            ..Action::new(ActionKind::WesterosChoice)
        }
    }

    pub fn choice_amount(house: HouseId, amount: u8) -> Action {
        Action {
            kind: ActionKind::WesterosChoice,
            house: Some(house),
            amount: Some(amount),
            ..Action::new(ActionKind::WesterosChoice)
        }
    }

    /// A unit choice for a pending House-card/Wildling effect.
    ///
    /// Legacy constructor: `amount` stays `None`, so the handler keeps the old
    /// ready-first behaviour. New action generation uses
    /// [`Action::choice_unit_state`] to name the state explicitly.
    pub fn choice_unit(house: HouseId, region: RegionId, unit: UnitType) -> Action {
        Action {
            kind: ActionKind::ChooseCasualty,
            house: Some(house),
            region: Some(region),
            unit: Some(unit),
            ..Action::new(ActionKind::ChooseCasualty)
        }
    }

    /// A unit choice that names a ready (`routed = false`) or routed unit.
    ///
    /// The state rides in `amount` (0 = ready, 1 = routed), following
    /// [`Action::disband_state`], so the frozen action layout does not change.
    pub fn choice_unit_state(
        house: HouseId,
        region: RegionId,
        unit: UnitType,
        routed: bool,
    ) -> Action {
        Action {
            kind: ActionKind::ChooseCasualty,
            house: Some(house),
            region: Some(region),
            unit: Some(unit),
            amount: Some(u8::from(routed)),
            ..Action::new(ActionKind::ChooseCasualty)
        }
    }

    /// A Footman-to-Knight upgrade for a pending Wildling/Renly effect.
    ///
    /// Legacy constructor: `amount` stays `None` (ready-first).
    pub fn choice_upgrade(house: HouseId, region: RegionId) -> Action {
        Action {
            kind: ActionKind::Upgrade,
            house: Some(house),
            region: Some(region),
            unit: Some(UnitType::Knight),
            ..Action::new(ActionKind::Upgrade)
        }
    }

    /// A Footman-to-Knight upgrade naming the *source* Footman's state.
    ///
    /// `amount` is 0 for a ready Footman and 1 for a routed one, like
    /// [`Action::choice_unit_state`].
    pub fn choice_upgrade_state(house: HouseId, region: RegionId, routed: bool) -> Action {
        Action {
            kind: ActionKind::Upgrade,
            house: Some(house),
            region: Some(region),
            unit: Some(UnitType::Knight),
            amount: Some(u8::from(routed)),
            ..Action::new(ActionKind::Upgrade)
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

    #[test]
    fn card_house_defaults_to_none_and_records_a_foreign_deck() {
        // Normal card and choice actions never name a foreign deck.
        assert_eq!(
            Action::play_house_card(HouseId(1), Some(CardId(2))).card_house,
            None
        );
        assert_eq!(
            Action::choice_unit(HouseId(1), RegionId(3), UnitType::Knight).card_house,
            None
        );
        // Patchface's action belongs to its player but names the target's deck.
        let patchface = Action::play_house_card_owned(HouseId(1), CardId(4), HouseId(2));
        assert_eq!(patchface.house, Some(HouseId(1)));
        assert_eq!(patchface.card, Some(CardId(4)));
        assert_eq!(patchface.card_house, Some(HouseId(2)));
        // The builder keeps the rest of the action intact.
        let built =
            Action::play_house_card(HouseId(1), Some(CardId(4))).with_card_house(HouseId(2));
        assert_eq!(built, patchface);
    }

    #[test]
    fn choice_context_tags_use_the_order_field() {
        use crate::enums::N_ORDERS;
        let action = Action::choice_amount(HouseId(1), 1).with_context(choice_context::PREEMPTIVE);
        assert_eq!(action.order, Some(OrderKey(choice_context::PREEMPTIVE)));
        assert_eq!(action.choice_context(), Some(choice_context::PREEMPTIVE));
        // The tag stays inside the real order vocabulary, so ORDER_BLOCK is a
        // valid one-hot row for it.
        for tag in choice_context::ALL {
            assert!((tag as usize) < N_ORDERS);
        }
        // A plain action carries no context.
        assert_eq!(Action::pass(HouseId(1)).choice_context(), None);
    }

    #[test]
    fn unit_state_constructors_encode_ready_and_routed() {
        let ready = Action::choice_unit_state(HouseId(1), RegionId(2), UnitType::Footman, false);
        assert_eq!(ready.amount, Some(0));
        assert_eq!(ready.kind, ActionKind::ChooseCasualty);
        let routed = Action::choice_unit_state(HouseId(1), RegionId(2), UnitType::Footman, true);
        assert_eq!(routed.amount, Some(1));
        assert_ne!(ready, routed);
        // The legacy constructors stay ready-first (`amount: None`).
        assert_eq!(
            Action::choice_unit(HouseId(1), RegionId(2), UnitType::Footman).amount,
            None
        );
        assert_eq!(Action::choice_upgrade(HouseId(1), RegionId(2)).amount, None);
        let routed_upgrade = Action::choice_upgrade_state(HouseId(1), RegionId(2), true);
        assert_eq!(routed_upgrade.amount, Some(1));
        assert_eq!(routed_upgrade.unit, Some(UnitType::Knight));
    }
}
