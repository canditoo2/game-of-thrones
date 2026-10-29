//! Mutable game state.
//!
//! Port of `src/agot/engine/state.py`, with one deliberate structural change:
//! the Python original keeps its regions in a dict that a clone *shares* until
//! a write copies one region (copy-on-write), plus bookkeeping (`_owned`) to
//! make that safe. Here the board is an owned array of small `Copy` regions, so
//! `State: Clone` is a plain structural copy and there is no sharing to get
//! wrong. That deletes a whole bug class rather than translating it.
//!
//! Two simplifications carried over from Python (they shape the state):
//!
//! 1. Order tokens are stored face up. The real game hides them until the
//!    reveal; `hidden.rs` resamples them for PIMC search.
//! 2. A region is controlled by at most one house at a time, which holds in the
//!    real game once battles have been resolved.

use smallvec::SmallVec;

use crate::action::Side;
use crate::combat::BattleResult;
use crate::enums::{
    ActionStep, CardId, HouseId, OrderKey, OrderType, Phase, RegionId, Track, UnitType, N_HOUSES,
    POWER_LIMIT, STAR_ALLOWANCE_BY_COURT_POSITION, SUPPLY_LIMITS,
};
use crate::house_cards::BattleContext;

/// Upper bound on regions a board may have (the base map has 58).
pub const MAX_REGIONS: usize = 64;

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/// What is currently in one region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Region {
    pub controller: Option<HouseId>,
    pub owner: Option<HouseId>,
    /// Ready unit counts, indexed by `UnitType`.
    pub units: [u8; 4],
    /// Routed unit counts, indexed by `UnitType`.
    pub routed: [u8; 4],
    pub neutral_force: u8,
    /// Printed strength of a Garrison token still present here (0 = none).
    /// Garrisons are not units: they add to a defender's Combat Strength only,
    /// never count towards Supply, and can never be assigned an order.
    pub garrison: u8,
    /// Mustering points still available to spend *in this area*. Unspent points
    /// are lost (they never pool).
    pub mustering_points: u8,
    pub influence_token: Option<HouseId>,
    pub order: Option<OrderKey>,
    pub order_house: Option<HouseId>,
}

impl Region {
    #[inline]
    pub fn unit_count(&self, unit: UnitType) -> u8 {
        self.units[unit.index()]
    }

    #[inline]
    pub fn routed_count(&self, unit: UnitType) -> u8 {
        self.routed[unit.index()]
    }

    #[inline]
    pub fn add_unit(&mut self, unit: UnitType, count: u8) {
        self.units[unit.index()] += count;
    }

    #[inline]
    pub fn has_units(&self) -> bool {
        self.units.iter().any(|count| *count > 0)
    }

    /// Total units, optionally including routed ones.
    #[inline]
    pub fn total(&self, include_routed: bool) -> u32 {
        let mut total: u32 = self.units.iter().map(|count| *count as u32).sum();
        if include_routed {
            total += self.routed.iter().map(|count| *count as u32).sum::<u32>();
        }
        total
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total(false) == 0
    }

    #[inline]
    pub fn contains_house(&self, house: HouseId) -> bool {
        self.controller == Some(house) && !self.is_empty()
    }

    /// Removes one routed unit, cheapest-first, returning what was removed.
    ///
    /// Mirrors `_remove_one_unit`'s routed branch: Footman, Siege, Knight,
    /// Ship.
    pub fn remove_one_routed(&mut self) -> Option<UnitType> {
        for unit in [
            UnitType::Footman,
            UnitType::Siege,
            UnitType::Knight,
            UnitType::Ship,
        ] {
            if self.routed_count(unit) > 0 {
                self.routed[unit.index()] -= 1;
                return Some(unit);
            }
        }
        None
    }

    /// Removes one ready unit, cheapest-first, returning what was removed.
    pub fn remove_one_ready(&mut self) -> Option<UnitType> {
        for unit in [
            UnitType::Footman,
            UnitType::Siege,
            UnitType::Knight,
            UnitType::Ship,
        ] {
            if self.unit_count(unit) > 0 {
                self.units[unit.index()] -= 1;
                return Some(unit);
            }
        }
        None
    }

    /// Restores control to the influence token's holder or the owner once the
    /// region is empty of units, matching the Python `_remove_one_unit` tail.
    pub fn settle_control(&mut self) {
        if self.is_empty() && self.routed.iter().all(|count| *count == 0) {
            self.controller = self.influence_token.or(self.owner);
        }
    }
}

// ---------------------------------------------------------------------------
// Houses
// ---------------------------------------------------------------------------

/// Per-house player state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct House {
    pub power: u8,
    pub supply_level: u8,
    pub orders_available: SmallVec<[OrderKey; 15]>,
    pub orders_placed: SmallVec<[OrderKey; 15]>,
    pub house_cards_available: SmallVec<[CardId; 7]>,
    pub house_cards_used: SmallVec<[CardId; 7]>,
    pub blade_used: bool,
    pub passed: bool,
    pub eliminated: bool,
}

impl House {
    /// A fresh house: every order token available and the full house deck.
    pub fn new(deck_size: usize, power: u8, supply_level: u8) -> House {
        let mut cards: SmallVec<[CardId; 7]> = SmallVec::new();
        for index in 0..deck_size {
            cards.push(CardId(index as u8));
        }
        House {
            power,
            supply_level,
            orders_available: OrderKey::ALL.into_iter().collect(),
            orders_placed: SmallVec::new(),
            house_cards_available: cards,
            house_cards_used: SmallVec::new(),
            blade_used: false,
            passed: false,
            eliminated: false,
        }
    }

    /// Starred orders this house has already placed.
    pub fn stars_placed(&self) -> u32 {
        self.orders_placed
            .iter()
            .filter(|key| key.starred())
            .count() as u32
    }

    /// Largest armies this house may support, largest first.
    pub fn supply_limit(&self) -> &'static [u8] {
        let level = (self.supply_level as usize).min(SUPPLY_LIMITS.len() - 1);
        SUPPLY_LIMITS[level]
    }

    pub fn has_order_placed(&self, order: OrderKey) -> bool {
        self.orders_placed.contains(&order)
    }

    /// Returns a placed order token to the pool.
    pub fn return_order(&mut self, order: OrderKey) -> bool {
        if let Some(position) = self.orders_placed.iter().position(|key| *key == order) {
            self.orders_placed.remove(position);
            self.orders_available.push(order);
            true
        } else {
            false
        }
    }

    /// Takes an order token out of the pool.
    pub fn take_order(&mut self, order: OrderKey) -> bool {
        if let Some(position) = self.orders_available.iter().position(|key| *key == order) {
            self.orders_available.remove(position);
            self.orders_placed.push(order);
            true
        } else {
            false
        }
    }

    pub fn card_available(&self, card: CardId) -> bool {
        self.house_cards_available.contains(&card)
    }

    pub fn play_card(&mut self, card: CardId) -> bool {
        if let Some(position) = self
            .house_cards_available
            .iter()
            .position(|candidate| *candidate == card)
        {
            self.house_cards_available.remove(position);
            self.house_cards_used.push(card);
            true
        } else {
            false
        }
    }

    pub fn retrieve_card(&mut self, card: CardId) -> bool {
        if let Some(position) = self
            .house_cards_used
            .iter()
            .position(|candidate| *candidate == card)
        {
            self.house_cards_used.remove(position);
            self.house_cards_available.push(card);
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Pending effects
// ---------------------------------------------------------------------------

/// A Planning-Phase ban on an order key or a whole order type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Restriction {
    Key(OrderKey),
    Type(OrderType),
}

impl Restriction {
    /// True when this restriction forbids `key`, matching Python's
    /// `_order_is_restricted` (`key in restrictions or order_type in restrictions`).
    #[inline]
    pub fn forbids(self, key: OrderKey) -> bool {
        match self {
            Restriction::Key(banned) => banned == key,
            Restriction::Type(banned) => banned == key.order_type(),
        }
    }
}

/// A Wildling attack in progress (only while [`Phase::WildlingAttack`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WildlingAttack {
    pub strength: u8,
    pub bids: SmallVec<[(HouseId, u8); N_HOUSES]>,
    /// Houses that sit this attack out (a "Preemptive Raid" second attack).
    pub excluded: SmallVec<[HouseId; N_HOUSES]>,
    pub card: Option<u16>,
    pub return_phase: Phase,
    pub second_attack: Option<SmallVec<[HouseId; N_HOUSES]>>,
    pub second_strength: u8,
}

impl Default for WildlingAttack {
    fn default() -> Self {
        WildlingAttack {
            strength: 0,
            bids: SmallVec::new(),
            excluded: SmallVec::new(),
            card: None,
            return_phase: Phase::Westeros,
            second_attack: None,
            second_strength: 6,
        }
    }
}

impl WildlingAttack {
    #[inline]
    pub fn bid_of(&self, house: HouseId) -> Option<u8> {
        self.bids
            .iter()
            .find(|(bidder, _)| *bidder == house)
            .map(|(_, amount)| *amount)
    }

    #[inline]
    pub fn set_bid(&mut self, house: HouseId, amount: u8) {
        if let Some(entry) = self.bids.iter_mut().find(|(bidder, _)| *bidder == house) {
            entry.1 = amount;
        } else {
            self.bids.push((house, amount));
        }
    }
}

/// The "no effect" option of a holders'-choice Westeros card.
///
/// The option list in `westeros.json` can contain the literal `"none"`, which
/// has no card row. Actions carry it as an out-of-range [`crate::enums::CardId`]
/// so the encoder still sets the "a card field is present" marker, matching the
/// Python action's string `"none"`.
pub const NONE_OPTION: u16 = u16::MAX;

/// Set while a holders'-choice Westeros card waits for its holder to pick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WesterosChoice {
    pub card: u16,
    pub holder: HouseId,
    pub options: SmallVec<[u16; 3]>,
}

/// A tied highest/lowest Wildling bid, waiting for the Iron Throne holder.
///
/// FFG's ruling is that the holder chooses between the tied bidders after the
/// Wildling card has been revealed but before its penalties or rewards resolve.
/// The choice is exposed as a [`crate::action::ActionKind::WesterosChoice`]
/// carrying the candidate index in `amount`, so the frozen action-feature
/// layout (and therefore existing checkpoints) does not change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WildlingTie {
    /// The tied bidders, in Iron Throne order.
    pub candidates: SmallVec<[HouseId; N_HOUSES]>,
    /// True when the Night's Watch won and the *highest* bidder is rewarded.
    pub highest: bool,
    /// The current Iron Throne holder, who makes the choice.
    pub chooser: HouseId,
}

/// The Iron Throne holder's ordering of tied Clash of Kings bids.
///
/// Bids are public once the track is resolved, so the holder decides the order
/// of the tied bidders (one choice per position). Groups are ordered by bid,
/// highest first, and `settled` accumulates the final track order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BidTie {
    pub track: Track,
    pub chooser: HouseId,
    /// Tied groups still to order, highest bid first.
    pub groups: SmallVec<[SmallVec<[HouseId; 6]>; 6]>,
    /// Houses already placed, in final order.
    pub settled: SmallVec<[HouseId; 6]>,
}

/// What to resume once the generic choice queue is empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChoiceResume {
    /// Continue the battle at the Cards stage (Doran/prepare follow).
    BattlePre,
    /// Finish the current Wildling attack after its card choices.
    Wildling,
    /// Post-battle card choices have been made; no extra resume is needed.
    PostBattle,
}

/// Which unit a generic unit-choice task is asking for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitChoice {
    /// Remove one unit from an allowed area.
    Destroy,
    /// Replace one Knight with a Footman.
    ReplaceKnight,
    /// Replace one Footman with a Knight.
    UpgradeFootman,
}

/// Which card a generic card-choice task is asking for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CardChoice {
    TyrionSwap { target: HouseId },
    AeronSwap,
    PatchfaceDiscard { target: HouseId },
    MassingDiscard,
    MammothRetrieve,
}

/// Which area a generic region-choice task is asking for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionChoice {
    QueenOrder { opponent: HouseId },
    CerseiOrder { opponent: HouseId },
    RenlyFootman,
    HordeFort { count: u8 },
    HordeMusterFort,
}

/// Which Influence track a generic track-choice task is asking for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackChoice {
    KingBeyondHighest,
    KingBeyondOther,
    /// Preemptive Raid option B: the house tied for its highest position on
    /// several tracks, so its controller picks which one to fall on.
    PreemptiveHighest,
}

/// Which either/or choice a generic option task is asking for.
///
/// The chosen option travels in [`crate::action::Action::amount`]: `0` always
/// means "decline" and `1` always means "use", so the two tasks below read the
/// same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionChoice {
    /// Preemptive Raid: destroy two units (`0`) or fall two positions (`1`).
    PreemptiveRaid,
    /// Tyrion Lannister's controller decides whether to use the card (`1`) or
    /// leave the battle as it is (`0`). Using it queues a **mandatory**
    /// [`CardChoice::TyrionSwap`] task for `target`.
    TyrionUse { target: HouseId },
    /// Robb Stark's controller may take over the loser's retreat (`1`) or let
    /// the loser choose (`0`).
    RobbRetreat,
}

/// The payload of one pending generic choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChoiceKind {
    /// Pick one unit. The action carries the unit's state in
    /// [`crate::action::Action::amount`] (0 = ready, 1 = routed), so a ready and
    /// a routed copy of the same unit type are separate decisions.
    Unit {
        purpose: UnitChoice,
        areas: SmallVec<[RegionId; 8]>,
        remaining: u8,
        allow_none: bool,
    },
    /// Pick one card from `cards`. The ids normally belong to the choosing
    /// house's deck; a Patchface task names the target's cards instead, and the
    /// emitted actions carry that deck in
    /// [`crate::action::Action::card_house`].
    Card {
        purpose: CardChoice,
        cards: SmallVec<[CardId; 7]>,
        allow_none: bool,
    },
    Region {
        purpose: RegionChoice,
        regions: SmallVec<[RegionId; 8]>,
        allow_none: bool,
    },
    Track {
        purpose: TrackChoice,
        tracks: SmallVec<[Track; 3]>,
    },
    Option {
        purpose: OptionChoice,
        options: SmallVec<[u8; 4]>,
    },
    /// A Wildling reward musters normally in one chosen Castle/Stronghold.
    Muster { region: RegionId, points: u8 },
}

/// One player decision that is not tied to a dedicated phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChoiceTask {
    pub house: HouseId,
    pub kind: ChoiceKind,
}

/// Set while a March Order is being resolved across several destinations.
///
/// The order token stays on the board until the march ends, so its bonus still
/// applies to a battle that happens after the non-combat moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingMarch {
    pub house: HouseId,
    pub origin: RegionId,
}

/// Which decision the Combat phase is currently asking for.
///
/// The Python original keeps this as a string on a dict; a Rust enum makes the
/// transition table checkable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BattleStage {
    /// Adjacent Support orders declare a side, one at a time.
    Support,
    /// A Neutral Force has no House card: go straight to resolution.
    NeutralResolve,
    /// Both sides pick a House card.
    Cards,
    /// Doran Martell's owner chooses which Influence track to move.
    Doran,
    /// The Valyrian Steel Blade holder decides whether to spend it.
    Blade,
    /// The loser destroys one unit per Sword icon.
    Casualties,
    /// The defeated defender picks a retreat destination.
    Retreat,
    /// Apply the outcome to the board.
    Finalize,
}

/// A battle awaiting choices (House cards, blade, casualties, retreat).
///
/// The two forces stay *separate* (`attacker_units` and `defender_units`): they
/// briefly share a region on the board, but merging them would make it
/// impossible to tell whose units are whose, which is what casualty removal
/// needs.
#[derive(Clone, Debug)]
pub struct PendingBattle {
    pub region: RegionId,
    pub attacker: HouseId,
    /// `None` while the defender is a Neutral Force token.
    pub defender: Option<HouseId>,
    pub neutral: bool,
    /// Where the attacker marched from; a losing attacker retreats there.
    pub origin: Option<RegionId>,
    /// The March order key, kept so Loras Tyrell can re-place it.
    pub attacker_order: Option<OrderKey>,
    pub attacker_units: [u8; 4],
    pub defender_units: [u8; 4],
    pub attacker_order_strength: i16,
    pub defender_order_strength: i16,
    pub attacker_card: Option<CardId>,
    pub defender_card: Option<CardId>,
    pub attacker_card_chosen: bool,
    pub defender_card_chosen: bool,
    /// Pre-battle card-text choices (Tyrion, Aeron, Queen of Thorns) have
    /// already been offered, so they must not be queued again.
    pub pre_choices_done: bool,
    /// Set while Doran Martell pauses for his owner's Influence-track choice.
    pub doran_owner: Option<HouseId>,
    pub doran_track: Option<Track>,
    /// Card text and immediate board changes, prepared after both cards are
    /// revealed and before the Valyrian Steel Blade step.
    pub prepared: Option<BattleContext>,
    pub blade_done: bool,
    pub attacker_blade: bool,
    pub defender_blade: bool,
    pub stage: BattleStage,
    /// Support orders are asked in Iron Throne order.
    pub support_queue: SmallVec<[RegionId; 8]>,
    pub support_index: usize,
    pub support_choices: SmallVec<[(RegionId, Side); 8]>,
    /// The resolved battle, parked while the players make their choices.
    pub result: Option<BattleResult>,
    pub casualties_remaining: u8,
    pub retreat_options: SmallVec<[RegionId; 8]>,
    /// Who picks the retreat: the loser, or the winner under Robb Stark.
    pub retreat_house: Option<HouseId>,
    pub retreat_destination: Option<RegionId>,
    /// The defender's Order token, returned to its pool if the attacker wins.
    pub defender_order: Option<OrderKey>,
    pub defender_order_house: Option<HouseId>,
}

impl PendingBattle {
    #[inline]
    pub fn support_choice(&self, region: RegionId) -> Option<Side> {
        self.support_choices
            .iter()
            .find(|(candidate, _)| *candidate == region)
            .map(|(_, side)| *side)
    }
}

/// One deck's draw pile and cursor.
///
/// Cards are stored inline: the Westeros decks hold ten cards and the Wildling
/// deck nine, so a clone of a `State` needs no allocation for them. That matters
/// because `State::clone` is the hottest operation in tree search.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DeckState {
    pub cards: SmallVec<[u16; 12]>,
    pub position: usize,
    pub drawn: Option<u16>,
}

impl DeckState {
    pub fn new(cards: impl IntoIterator<Item = u16>) -> DeckState {
        DeckState {
            cards: cards.into_iter().collect(),
            position: 0,
            drawn: None,
        }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.cards.len().saturating_sub(self.position)
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Complete mutable state of one game.
#[derive(Clone, Debug)]
pub struct State {
    pub max_rounds: u8,
    pub victory_castles: u8,

    pub round: u8,
    pub phase: Phase,
    pub action_step: ActionStep,
    pub active_house: Option<HouseId>,
    pub winner: Option<HouseId>,
    /// House that resolved the previous order in the current Action Phase step.
    pub action_house_last: Option<HouseId>,
    /// Set once the Messenger Raven has been used (or declined) in a round.
    pub raven_used: bool,
    /// Phase to return to once a Special Consolidate Power mustering finishes.
    pub return_phase: Option<Phase>,

    pub tracks: [SmallVec<[HouseId; N_HOUSES]>; 3],
    /// The board, inline: a `State` clone is then a plain memcpy with no
    /// allocation at all.
    pub regions: [Region; MAX_REGIONS],
    pub region_count: u8,
    pub houses: [Option<House>; N_HOUSES],

    pub wildling_threat: u8,
    pub wildling_attack: Option<WildlingAttack>,
    /// Set when the current attack's extreme bid is tied and the Iron Throne
    /// holder must choose the affected bidder.
    pub pending_wildling_tie: Option<WildlingTie>,
    /// The Iron Throne holder's pending ordering of tied Clash bids.
    pub pending_bid_tie: Option<BidTie>,
    pub westeros_decks: [DeckState; 3],
    /// Westeros card effects still to resolve this round.
    pub pending_westeros: Option<SmallVec<[u16; 3]>>,
    /// Desktop-play sequencing: the next deck to draw (0..3), and whether
    /// the table is paused between drawing and resolving one card.
    pub westeros_next_deck: u8,
    pub westeros_waiting: bool,
    pub westeros_choice: Option<WesterosChoice>,
    pub wildling_deck: DeckState,
    /// Unused by the engine (kept for parity with the Python state).
    pub wildling_deck_pos: u16,
    pub wildling_discard: SmallVec<[u16; 16]>,

    /// Order keys or order types that may not be placed this Planning Phase.
    pub order_restrictions: SmallVec<[Restriction; 4]>,
    /// Houses that still have to place order tokens this planning phase.
    pub pending_planning: SmallVec<[HouseId; N_HOUSES]>,
    /// Houses whose armies exceed their Supply limits and must disband.
    /// The first entry is the house currently choosing a unit.
    pub supply_queue: SmallVec<[HouseId; N_HOUSES]>,
    /// A completed interactive battle queues post-retreat Supply excess for the
    /// owner's disband choice; the low-level `combat::resolve_and_apply`
    /// convenience path keeps its automatic trim. Supply-card and Wildling
    /// excess always use [`State::supply_queue`].
    pub supply_choice: bool,
    /// Generic player choices from House cards and Wildling consequences.
    pub choice_queue: SmallVec<[ChoiceTask; 8]>,
    pub choice_resume: Option<ChoiceResume>,
    /// Interactive games expose all card/Wildling sub-choices; the low-level
    /// direct battle helpers retain their deterministic defaults for tests.
    pub choice_choices: bool,
    /// Pending bids for the current bidding round, keyed by house.
    pub pending_bids: SmallVec<[(HouseId, u8); N_HOUSES]>,
    pub pending_battle: Option<PendingBattle>,
    /// A battle encountered during a multi-destination March. It is promoted
    /// to `pending_battle` only after the March order is finished, so all
    /// movement is chosen before combat begins.
    pub deferred_battle: Option<PendingBattle>,
    pub pending_march: Option<PendingMarch>,
    /// Which influence tracks have still to be bid on, in resolution order.
    pub bid_tracks: SmallVec<[Track; 3]>,

    /// Region ids that count towards the victory condition.
    pub fortifications: SmallVec<[RegionId; 32]>,

    /// Human-readable trace. `None` during search: the log is per-node state
    /// that search never reads, and Python copied it at every clone.
    pub log: Option<Vec<String>>,
}

impl State {
    /// A state with no board contents; `setup.rs` fills it in.
    pub fn blank() -> State {
        State {
            max_rounds: 10,
            victory_castles: 7,
            round: 1,
            phase: Phase::Westeros,
            action_step: ActionStep::Raid,
            active_house: None,
            winner: None,
            action_house_last: None,
            raven_used: false,
            return_phase: None,
            tracks: std::array::from_fn(|_| SmallVec::new()),
            regions: [Region::default(); MAX_REGIONS],
            region_count: 0,
            houses: std::array::from_fn(|_| None),
            wildling_threat: 0,
            wildling_attack: None,
            pending_wildling_tie: None,
            pending_bid_tie: None,
            westeros_decks: std::array::from_fn(|_| DeckState::default()),
            pending_westeros: None,
            westeros_next_deck: 0,
            westeros_waiting: false,
            westeros_choice: None,
            wildling_deck: DeckState::default(),
            wildling_deck_pos: 0,
            wildling_discard: SmallVec::new(),
            order_restrictions: SmallVec::new(),
            pending_planning: SmallVec::new(),
            supply_queue: SmallVec::new(),
            supply_choice: false,
            choice_queue: SmallVec::new(),
            choice_resume: None,
            choice_choices: false,
            pending_bids: SmallVec::new(),
            pending_battle: None,
            deferred_battle: None,
            pending_march: None,
            bid_tracks: SmallVec::new(),
            fortifications: SmallVec::new(),
            log: None,
        }
    }

    // -- region access ---------------------------------------------------

    #[inline]
    pub fn region(&self, id: RegionId) -> &Region {
        &self.regions[id.index()]
    }

    #[inline]
    pub fn region_mut(&mut self, id: RegionId) -> &mut Region {
        &mut self.regions[id.index()]
    }

    /// Regions in board order, excluding the unused tail of the array.
    pub fn board(&self) -> &[Region] {
        &self.regions[..self.region_count as usize]
    }

    pub fn region_ids(&self) -> impl Iterator<Item = RegionId> + '_ {
        (0..self.region_count as u16).map(RegionId)
    }

    // -- house access ----------------------------------------------------

    #[inline]
    pub fn house(&self, house: HouseId) -> Option<&House> {
        self.houses[house.index()].as_ref()
    }

    #[inline]
    pub fn house_mut(&mut self, house: HouseId) -> Option<&mut House> {
        self.houses[house.index()].as_mut()
    }

    /// Houses taking part, in canonical order.
    pub fn present_houses(&self) -> SmallVec<[HouseId; N_HOUSES]> {
        HouseId::ALL
            .into_iter()
            .filter(|house| self.house(*house).is_some())
            .collect()
    }

    // -- queries ---------------------------------------------------------

    /// Whether the Order tokens on the board are face up.
    ///
    /// Orders are assigned face down and revealed simultaneously *before* the
    /// Messenger Raven step (rulebook: Assign Orders, Reveal Orders, Use
    /// Messenger Raven), so only the assignment step hides them.
    #[inline]
    pub fn orders_revealed(&self) -> bool {
        self.phase != Phase::Planning
    }

    /// 1-based position of a house on a track (1 is best).
    pub fn track_position(&self, house: HouseId, track: Track) -> usize {
        self.tracks[track.index()]
            .iter()
            .position(|candidate| *candidate == house)
            .map(|index| index + 1)
            .unwrap_or(0)
    }

    #[inline]
    pub fn track_order(&self, track: Track) -> &[HouseId] {
        &self.tracks[track.index()]
    }

    pub fn first_player(&self) -> Option<HouseId> {
        self.tracks[Track::IronThrone.index()].first().copied()
    }

    pub fn blade_holder(&self) -> Option<HouseId> {
        self.tracks[Track::Fiefdoms.index()].first().copied()
    }

    /// Carry the blade token's used side with a change of holder.
    ///
    /// `House::blade_used` records the token face for the current holder. When
    /// Doran Martell or a Clash of Kings bid moves the Fiefdoms leader, the
    /// token itself does not refresh: the new holder inherits the used face and
    /// cannot use the blade again that round.
    pub fn transfer_blade_used(&mut self, previous: Option<HouseId>) {
        let current = self.blade_holder();
        if current == previous {
            return;
        }
        let used = previous
            .and_then(|house| self.house(house))
            .map(|house| house.blade_used)
            .unwrap_or(false);
        if let Some(previous) = previous {
            if let Some(house) = self.house_mut(previous) {
                house.blade_used = false;
            }
        }
        if let Some(current) = current {
            if let Some(house) = self.house_mut(current) {
                house.blade_used = used;
            }
        }
    }

    pub fn raven_holder(&self) -> Option<HouseId> {
        self.tracks[Track::KingsCourt.index()].first().copied()
    }

    /// How many starred orders a house may place, by King's Court position.
    pub fn star_allowance(&self, house: HouseId) -> u8 {
        let position = self.track_position(house, Track::KingsCourt);
        if position == 0 {
            return 0;
        }
        let index = (position - 1).min(STAR_ALLOWANCE_BY_COURT_POSITION.len() - 1);
        STAR_ALLOWANCE_BY_COURT_POSITION[index]
    }

    /// Regions controlled by a house, in board order.
    pub fn controlled_regions(&self, house: HouseId) -> SmallVec<[RegionId; 32]> {
        self.region_ids()
            .filter(|id| self.region(*id).controller == Some(house))
            .collect()
    }

    /// Castles and strongholds controlled: the victory condition.
    pub fn castle_count(&self, house: HouseId) -> u32 {
        self.fortifications
            .iter()
            .filter(|id| self.region(**id).controller == Some(house))
            .count() as u32
    }

    /// Gain Power tokens, never past the twenty supplied with the house.
    pub fn gain_power(&mut self, house: HouseId, amount: u8) {
        let Some(state) = self.house_mut(house) else {
            return;
        };
        state.power = state.power.saturating_add(amount).min(POWER_LIMIT);
    }

    /// Regions that hold this house's units, in board order.
    ///
    /// Supply counts the armies on the board, so an area whose control flag was
    /// cleared while its ships stayed behind must still count: otherwise a stale
    /// flag would let a player muster past the limit.
    pub fn army_regions(&self, house: HouseId) -> SmallVec<[RegionId; 32]> {
        self.region_ids()
            .filter(|id| {
                let region = self.region(*id);
                region.controller == Some(house)
                    || (region.controller.is_none() && region.total(true) > 0)
            })
            .collect()
    }

    /// Sizes of a house's armies, largest first.
    ///
    /// A lone unit is not an army and is exempt from Supply, while routed units
    /// still count towards Supply.
    pub fn army_sizes(&self, house: HouseId) -> Vec<u32> {
        let mut sizes: Vec<u32> = self
            .army_regions(house)
            .iter()
            .map(|id| self.region(*id).total(true))
            .filter(|size| *size >= 2)
            .collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }

    pub fn within_supply(&self, house: HouseId) -> bool {
        let Some(state) = self.house(house) else {
            return true;
        };
        let limits = state.supply_limit();
        for (index, size) in self.army_sizes(house).into_iter().enumerate() {
            if index >= limits.len() || size > limits[index] as u32 {
                return false;
            }
        }
        true
    }

    pub fn has_mustering_points(&self, house: HouseId) -> bool {
        self.controlled_regions(house)
            .iter()
            .any(|id| self.region(*id).mustering_points > 0)
    }

    /// `(region, order)` pairs placed by a house.
    pub fn orders_on_board(&self, house: HouseId) -> SmallVec<[(RegionId, OrderKey); 16]> {
        self.region_ids()
            .filter_map(|id| {
                let region = self.region(id);
                match (region.order, region.order_house) {
                    (Some(order), Some(owner)) if owner == house => Some((id, order)),
                    _ => None,
                }
            })
            .collect()
    }

    pub fn has_order_type(&self, house: HouseId, order_type: OrderType) -> bool {
        self.region_ids().any(|id| {
            let region = self.region(id);
            region.order_house == Some(house)
                && region.order.map(|key| key.order_type()) == Some(order_type)
        })
    }

    /// Houses with at least one placed order of `order_type` on the board.
    pub fn houses_with_orders(&self, order_type: OrderType) -> SmallVec<[HouseId; N_HOUSES]> {
        let mut result: SmallVec<[HouseId; N_HOUSES]> = SmallVec::new();
        for id in self.region_ids() {
            let region = self.region(id);
            if region.order.map(|key| key.order_type()) == Some(order_type) {
                if let Some(owner) = region.order_house {
                    if !result.contains(&owner) {
                        result.push(owner);
                    }
                }
            }
        }
        result
    }

    /// `(region, order)` pairs of a house for one order type.
    pub fn orders_of_type(
        &self,
        house: HouseId,
        order_type: OrderType,
    ) -> SmallVec<[(RegionId, OrderKey); 16]> {
        self.region_ids()
            .filter_map(|id| {
                let region = self.region(id);
                if region.order_house != Some(house) {
                    return None;
                }
                match region.order {
                    Some(order) if order.order_type() == order_type => Some((id, order)),
                    _ => None,
                }
            })
            .collect()
    }

    pub fn has_order_restriction(&self, order: OrderKey) -> bool {
        self.order_restrictions
            .iter()
            .any(|restriction| restriction.forbids(order))
    }

    pub fn set_winner(&mut self, house: HouseId) {
        self.winner = Some(house);
        self.phase = Phase::GameOver;
        self.active_house = None;
    }

    /// Appends a trace line when the log is enabled.
    pub fn note(&mut self, message: String) {
        if let Some(log) = self.log.as_mut() {
            log.push(format!(
                "R{} {}: {}",
                self.round,
                self.phase.as_str(),
                message
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_counts_and_totals() {
        let mut region = Region::default();
        assert!(region.is_empty());
        region.add_unit(UnitType::Knight, 2);
        region.routed[UnitType::Footman.index()] = 1;
        assert_eq!(region.total(false), 2);
        assert_eq!(region.total(true), 3);
        assert!(!region.is_empty());
        assert!(region.has_units());
        region.controller = Some(HouseId(1));
        assert!(region.contains_house(HouseId(1)));
        assert_eq!(region.remove_one_ready(), Some(UnitType::Knight));
    }

    #[test]
    fn removing_the_last_unit_settles_control() {
        let mut region = Region {
            owner: Some(HouseId(2)),
            influence_token: Some(HouseId(3)),
            controller: Some(HouseId(1)),
            ..Region::default()
        };
        region.add_unit(UnitType::Footman, 1);
        assert_eq!(region.remove_one_ready(), Some(UnitType::Footman));
        region.settle_control();
        // The influence token outranks the printed owner.
        assert_eq!(region.controller, Some(HouseId(3)));
    }

    #[test]
    fn house_order_pool_round_trips() {
        let mut house = House::new(7, 5, 2);
        assert_eq!(house.orders_available.len(), 15);
        assert!(house.take_order(OrderKey(2)));
        assert!(house.has_order_placed(OrderKey(2)));
        assert_eq!(house.stars_placed(), 1);
        assert!(house.return_order(OrderKey(2)));
        assert_eq!(house.stars_placed(), 0);
        assert_eq!(house.orders_available.len(), 15);
        assert_eq!(house.house_cards_available.len(), 7);
        assert!(house.play_card(CardId(0)));
        assert!(house.card_available(CardId(1)));
        assert!(!house.card_available(CardId(0)));
        assert!(house.retrieve_card(CardId(0)));
    }

    #[test]
    fn restrictions_cover_keys_and_types() {
        let by_key = Restriction::Key(OrderKey(2)); // march_plus
        let by_type = Restriction::Type(OrderType::Raid);
        assert!(by_key.forbids(OrderKey(2)));
        assert!(!by_key.forbids(OrderKey(0)));
        assert!(by_type.forbids(OrderKey(9))); // raid_a
        assert!(by_type.forbids(OrderKey(11))); // raid_star
        assert!(!by_type.forbids(OrderKey(0)));
    }

    #[test]
    fn blank_state_has_no_board() {
        let state = State::blank();
        assert_eq!(state.region_count, 0);
        assert!(state.board().is_empty());
        assert_eq!(state.present_houses().len(), 0);
        assert!(state.orders_revealed());
    }

    #[test]
    fn supply_limits_follow_army_sizes() {
        let mut state = State::blank();
        state.houses[0] = Some(House::new(7, 5, 0)); // supply level 0 -> (2, 2)
        state.region_count = 2;
        state.regions[0].controller = Some(HouseId(0));
        state.regions[0].units[UnitType::Footman.index()] = 3;
        state.regions[1].controller = Some(HouseId(0));
        state.regions[1].units[UnitType::Footman.index()] = 1;
        assert_eq!(state.army_sizes(HouseId(0)), vec![3]);
        assert!(!state.within_supply(HouseId(0)));
        state.regions[0].units[UnitType::Footman.index()] = 2;
        assert!(state.within_supply(HouseId(0)));
    }
}
