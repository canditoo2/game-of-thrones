//! House-card special abilities.
//!
//! Port of `src/agot/engine/house_cards.py`. Every card in
//! `agot/data/house_cards.json` carries an `ability` id; this module turns those
//! ids into concrete effects during a battle. It is split out of
//! [`crate::combat`] so that the pure strength arithmetic stays readable.
//!
//! Three groups of effects exist:
//!
//! * **Passive modifiers** change a side's strength, swords or forts before the
//!   totals are compared (Stannis, Davos, Theon, Asha, Nymeria, Balon, Kevan,
//!   Victarion, Salladhor Saan, Catelyn).
//! * **Immediate effects** happen at reveal and touch the board (Mace destroys a
//!   Footman, the Queen of Thorns removes an adjacent Order, Doran moves an
//!   opponent down an Influence track, Tyrion and Aeron swap cards).
//! * **Post-combat effects** apply once the winner is known (Renly, Tywin,
//!   Cersei, Robb, Loras, Roose, Patchface, Arianne).
//!
//! The rulebook lets the card's owner make several sub-choices (which Footman,
//! which Order, which alternate card, which retreat area). Interactive play and
//! training queue those as generic [`crate::state::ChoiceTask`] decisions; the
//! direct low-level battle helpers keep deterministic fallbacks for the frozen
//! regression corpus. See `docs/roadmap.md`.
//!
//! # The deferred context
//!
//! Python's `BattleContext` holds a *reference to the mutable game state* and
//! writes through it (`ctx.state.mut(...)`) long after the battle was resolved.
//! That back-reference is what `GameState.clone` had to rebind by hand, and it
//! cannot exist here. The context therefore carries the deferred changes **as
//! data**, and the three board-touching entry points take `&mut State`
//! explicitly:
//!
//! | Python | Rust |
//! | --- | --- |
//! | `ctx.card_swaps` (dicts) | [`CardSwap`] records |
//! | `ctx.power_costs` | `Vec<(HouseId, u8)>` |
//! | `ctx.orders_to_remove` | `SmallVec<[RegionId; 2]>` |
//! | `ctx.influence_move` | `Option<(Track, HouseId)>` |
//! | `ctx.casualty_immunity` | `SmallVec<[HouseId; 2]>` |
//! | `ctx.state` | the `state` argument of every function |
//!
//! [`apply_pre_battle`] reads the state but never writes it: every pre-battle
//! effect that touches the board is *recorded* (swaps, power costs, Orders to
//! remove, Doran's move) or stays inside the context (Mace's casualty, the
//! passive modifiers), so the pure [`crate::combat::resolve_battle`] needs only
//! `&State` — exactly as in Python.
//!
//! Python also appends its messages to `ctx.notes`, which the phase machine never
//! reads and which the `BattleResult.notes` snapshot does not see; the port
//! writes them straight to the state's log so they are not lost.
//!
//! # What the port could not carry over unchanged
//!
//! * **`BattleResult.context` is not optional.** Python allows `None` and
//!   `resolve_and_apply` asserts it is set; here [`crate::combat::resolve_battle`]
//!   always builds one, so the field is plain data and the assert disappears.
//! * **Card lookups need the house.** Python's `card_data` indexes one global
//!   table by id string; a Rust [`CardId`] is an index into *one house's* deck,
//!   so [`card_data`] / [`ability_of`] / [`card_score`] take the house.
//! * **`max(..., key=card_score)` keeps the first maximum**, and so does
//!   [`best_card`]: it decides which card a swap or Patchface discards.
//! * **Region ids sort as strings.** Every `sorted(...)` over region ids in the
//!   Python (which Order the Queen of Thorns takes, which Order Cersei removes,
//!   which supporting Footman Renly upgrades) compares id strings, not board
//!   order; [`crate::combat::sort_region_ids`] preserves that.
//! * **Defensive lookups return `None` instead of raising.** A house or card
//!   that is not in play is skipped where Python would raise `KeyError`, and
//!   `State::track_position` yields 0 for an absent house where Python raises
//!   `ValueError`.

use smallvec::SmallVec;

use crate::action::Side;
use crate::combat::{sort_region_ids, BattleResult, BattleSide};
use crate::enums::{CardId, HouseId, OrderKey, OrderType, RegionId, Track, UnitType};
use crate::map::{CardSpec, GameMap, HouseCardData};
use crate::state::{CardChoice, ChoiceKind, ChoiceResume, ChoiceTask, RegionChoice, State};

// ---------------------------------------------------------------------------
// Ability lookup
// ---------------------------------------------------------------------------

/// Every ability id the engine actually implements.
///
/// A data-integrity test checks that the JSON never names an ability outside
/// this set.
pub const IMPLEMENTED_ABILITIES: [&str; 24] = [
    "stannis_baratheon",
    "davos_seaworth",
    "theon_greyjoy",
    "asha_greyjoy",
    "nymeria_sand",
    "balon_greyjoy",
    "kevan_lannister",
    "victarion_greyjoy",
    "salladhor_saan",
    "catelyn_stark",
    "blackfish",
    "mace_tyrell",
    "queen_of_thorns",
    "doran_martell",
    "tyrion_lannister",
    "aeron_damphair",
    "tywin_lannister",
    "renly_baratheon",
    "cersei_lannister",
    "robb_stark",
    "roose_bolton",
    "patchface",
    "loras_tyrell",
    "arianne_martell",
];

/// The printed statistics of a House card, or `None` for no card at all.
///
/// Python's `card_data` looks an id up in one global table; here a card is an
/// index into its own house's deck, so the house is part of the lookup.
pub fn card_data(cards: &HouseCardData, house: HouseId, card: Option<CardId>) -> Option<&CardSpec> {
    let card = card?;
    cards.deck(house).get(card.0 as usize)
}

/// The machine-readable ability id of a House card (`"none"` if any).
pub fn ability_of(cards: &HouseCardData, house: HouseId, card: Option<CardId>) -> &str {
    match card_data(cards, house, card) {
        Some(spec) if !spec.ability.is_empty() => &spec.ability,
        _ => "none",
    }
}

/// A crude value for choosing between cards (strength plus icons).
pub fn card_score(cards: &HouseCardData, house: HouseId, card: Option<CardId>) -> f64 {
    match card_data(cards, house, card) {
        Some(spec) => spec.strength as f64 + 0.8 * spec.swords as f64 + 0.5 * spec.forts as f64,
        None => 0.0,
    }
}

/// The best card of a hand by [`card_score`], ties going to the first card.
///
/// Python's `max(..., key=...)` keeps the first maximum, which decides *which*
/// card a swap or Patchface discards; `Iterator::max_by` would keep the last.
fn best_card(cards: &HouseCardData, house: HouseId, available: &[CardId]) -> Option<CardId> {
    let mut best: Option<(CardId, f64)> = None;
    for card in available {
        let score = card_score(cards, house, Some(*card));
        match best {
            Some((_, best_score)) if score <= best_score => {}
            _ => best = Some((*card, score)),
        }
    }
    best.map(|(card, _)| card)
}

/// True when a house has already played `card_id` (i.e. it is in the discard).
fn card_used(state: &State, cards: &HouseCardData, house: HouseId, card_id: &str) -> bool {
    let Some(house_state) = state.house(house) else {
        return false;
    };
    house_state.house_cards_used.iter().any(|card| {
        cards
            .deck(house)
            .get(card.0 as usize)
            .is_some_and(|spec| spec.id == card_id)
    })
}

// ---------------------------------------------------------------------------
// Battle context
// ---------------------------------------------------------------------------

/// How a House card reached the discard pile during a battle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapKind {
    /// Tyrion Lannister: the replaced card goes back to its owner's hand.
    ReturnSwap,
    /// Aeron Damphair: the replaced card is discarded.
    DiscardSwap,
}

/// One recorded card change, committed by [`commit_deferred`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardSwap {
    /// The house whose played card changes (Tyrion's target is the opponent).
    pub house: HouseId,
    pub old: Option<CardId>,
    pub new: Option<CardId>,
    pub kind: SwapKind,
}

/// Everything the abilities need about one battle.
///
/// `attacker_units` / `defender_units` are *effective* unit dictionaries: an
/// immediate ability such as Mace Tyrell's may already have removed a unit, and
/// the phase machine removes the defeated player's chosen casualties from them
/// before the battle is applied. `card_swaps`, `power_costs`, `orders_to_remove`
/// and `influence_move` are decided here (purely) and committed by
/// [`commit_deferred`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BattleContext {
    pub region_id: RegionId,
    pub attacker: HouseId,
    pub defender: HouseId,
    pub attacker_units: [u8; 4],
    pub defender_units: [u8; 4],
    pub attacker_card: Option<CardId>,
    pub defender_card: Option<CardId>,
    pub attacker_supported: bool,
    pub defender_supported: bool,
    /// The declared support for each adjacent area, when the caller supplied
    /// them (`None` is the legacy self-directed fallback). Internal to battle
    /// resolution: it is not serialized, so the frozen fixtures are unaffected.
    pub(crate) support_choices: Option<SmallVec<[(RegionId, Side); 8]>>,
    pub attacker_support_units: [u8; 4],
    pub defender_support_units: [u8; 4],
    pub attacker_order: Option<OrderKey>,
    pub attacker_origin: Option<RegionId>,
    pub attacker_side: BattleSide,
    pub defender_side: BattleSide,
    pub card_swaps: Vec<CardSwap>,
    pub power_costs: SmallVec<[(HouseId, u8); 2]>,
    pub orders_to_remove: SmallVec<[RegionId; 2]>,
    pub influence_move: Option<(Track, HouseId)>,
    /// Doran Martell's chosen track, when the desktop player picked one.
    pub doran_track: Option<Track>,
    /// Houses whose card (The Blackfish) ignores all casualties this battle.
    pub casualty_immunity: SmallVec<[HouseId; 2]>,
    pub notes: Vec<String>,
}

impl BattleContext {
    #[inline]
    pub fn side_of(&self, house: HouseId) -> &BattleSide {
        if house == self.attacker {
            &self.attacker_side
        } else {
            &self.defender_side
        }
    }

    #[inline]
    pub fn side_of_mut(&mut self, house: HouseId) -> &mut BattleSide {
        if house == self.attacker {
            &mut self.attacker_side
        } else {
            &mut self.defender_side
        }
    }

    #[inline]
    pub fn card_of(&self, house: HouseId) -> Option<CardId> {
        if house == self.attacker {
            self.attacker_card
        } else {
            self.defender_card
        }
    }

    /// Update the card id the later abilities read.
    ///
    /// A swap changes which card this side actually played, so every ability
    /// resolved after it (Mace, Balon, Renly, ...) must see the new id.
    #[inline]
    pub fn set_card_of(&mut self, house: HouseId, card: Option<CardId>) {
        if house == self.attacker {
            self.attacker_card = card;
        } else {
            self.defender_card = card;
        }
    }

    #[inline]
    pub fn units_of(&self, house: HouseId) -> &[u8; 4] {
        if house == self.attacker {
            &self.attacker_units
        } else {
            &self.defender_units
        }
    }

    #[inline]
    pub fn units_of_mut(&mut self, house: HouseId) -> &mut [u8; 4] {
        if house == self.attacker {
            &mut self.attacker_units
        } else {
            &mut self.defender_units
        }
    }

    #[inline]
    pub fn support_of(&self, house: HouseId) -> &[u8; 4] {
        if house == self.attacker {
            &self.attacker_support_units
        } else {
            &self.defender_support_units
        }
    }

    #[inline]
    pub fn is_attacker(&self, house: HouseId) -> bool {
        house == self.attacker
    }

    /// The other participant: Python's `ctx.defender if house == ctx.attacker
    /// else ctx.attacker`.
    #[inline]
    pub fn opponent_of(&self, house: HouseId) -> HouseId {
        if house == self.attacker {
            self.defender
        } else {
            self.attacker
        }
    }

    #[inline]
    pub fn supported(&self, house: HouseId) -> bool {
        if house == self.attacker {
            self.attacker_supported
        } else {
            self.defender_supported
        }
    }

    /// Removes one unit of `house` from its effective pool.
    ///
    /// The defeated player picks which unit dies (rulebook page 21); the phase
    /// machine calls this for every `ChooseCasualty` action and only then counts
    /// the casualty down. Returns false when the pool holds no such unit, which
    /// mirrors Python's `if pool.get(unit, 0) <= 0: return`.
    pub fn remove_unit(&mut self, house: HouseId, unit: UnitType) -> bool {
        let units = self.units_of_mut(house);
        if units[unit.index()] == 0 {
            return false;
        }
        units[unit.index()] -= 1;
        true
    }

    /// Remove every participating Siege Engine from one side.
    ///
    /// FFG's ruling: Siege Engines cannot be chosen as casualties; they are
    /// removed automatically, in addition to the normal casualties, before the
    /// loser picks any other unit.
    pub fn remove_siege_engines(&mut self, house: HouseId) -> u8 {
        let units = self.units_of_mut(house);
        let count = units[UnitType::Siege.index()];
        units[UnitType::Siege.index()] = 0;
        count
    }
}

#[inline]
fn count(units: &[u8; 4], unit: UnitType) -> u8 {
    units[unit.index()]
}

// ---------------------------------------------------------------------------
// Pre-battle abilities
// ---------------------------------------------------------------------------

/// Apply every ability that can change the battle before it is decided.
///
/// Reads `state` but never writes it: everything that would move the board is
/// recorded on `ctx` for [`commit_deferred`] to apply.
pub fn apply_pre_battle(
    ctx: &mut BattleContext,
    state: &State,
    map: &GameMap,
    cards: &HouseCardData,
) {
    let attacker = ctx.attacker;
    let defender = ctx.defender;
    // Card swaps first: they change the printed stats the other rules read.
    resolve_card_swaps(ctx, state, cards, attacker);
    resolve_card_swaps(ctx, state, cards, defender);
    // Immediate destruction of a Footman.
    mace(ctx, cards, attacker);
    mace(ctx, cards, defender);
    // Passive stat modifiers. Balon zeroes the opponent's card, so resolve it
    // before the opponent's own modifiers would matter (order is irrelevant to
    // its own side's stats).
    passive(ctx, state, map, cards, attacker);
    passive(ctx, state, map, cards, defender);
    // Salladhor Saan inspects support, so it runs once both sides are built.
    salladhor(ctx, cards, attacker);
    salladhor(ctx, cards, defender);
    // Board-touching immediate effects are recorded for the caller to commit.
    queen_of_thorns(ctx, state, map, cards, attacker);
    queen_of_thorns(ctx, state, map, cards, defender);
    doran(ctx, state, cards, attacker);
    doran(ctx, state, cards, defender);
}

/// Tyrion and Aeron may replace a card; auto-resolve the choice.
fn resolve_card_swaps(
    ctx: &mut BattleContext,
    state: &State,
    cards: &HouseCardData,
    house: HouseId,
) {
    if state.choice_choices {
        // Interactive play queues these choices in the Game layer.
        return;
    }
    let ability = ability_of(cards, house, ctx.card_of(house));
    if ability != "tyrion_lannister" && ability != "aeron_damphair" {
        return;
    }

    if ability == "tyrion_lannister" {
        // Tyrion belongs to Lannister and targets the *opponent's* card.
        let target = ctx.opponent_of(house);
        let target_available: Vec<CardId> = state
            .house(target)
            .map(|house_state| house_state.house_cards_available.to_vec())
            .unwrap_or_default();
        let old = ctx.card_of(target);
        if target_available.is_empty() {
            // No other card in hand: the opponent fights without a House card.
            apply_swap(ctx, cards, target, old, None, SwapKind::ReturnSwap);
            return;
        }
        if let Some(best_target) = best_card(cards, target, &target_available) {
            if card_score(cards, target, Some(best_target)) < card_score(cards, target, old) {
                apply_swap(
                    ctx,
                    cards,
                    target,
                    old,
                    Some(best_target),
                    SwapKind::ReturnSwap,
                );
            }
        }
        return;
    }

    // Aeron Damphair: pay two Power to trade this card for a stronger one.
    let Some(house_state) = state.house(house) else {
        return;
    };
    let available: Vec<CardId> = house_state.house_cards_available.to_vec();
    let current = ctx.card_of(house);
    if available.is_empty() {
        return;
    }
    if house_state.power < 2 {
        return;
    }
    let Some(best) = best_card(cards, house, &available) else {
        return;
    };
    if card_score(cards, house, Some(best)) <= card_score(cards, house, current) {
        return;
    }
    apply_swap(
        ctx,
        cards,
        house,
        current,
        Some(best),
        SwapKind::DiscardSwap,
    );
    ctx.power_costs.push((house, 2));
    ctx.notes
        .push(format!("{house} discards 2 Power to swap Aeron Damphair"));
}

/// Record a swap and update the side's printed stats immediately.
fn apply_swap(
    ctx: &mut BattleContext,
    cards: &HouseCardData,
    house: HouseId,
    old: Option<CardId>,
    new: Option<CardId>,
    kind: SwapKind,
) {
    let spec = card_data(cards, house, new);
    let (strength, swords, forts) = match spec {
        Some(spec) => (spec.strength as i16, spec.swords, spec.forts),
        None => (0, 0, 0),
    };
    let side = ctx.side_of_mut(house);
    side.card_strength = strength;
    side.swords = swords;
    side.forts = forts;
    // Later abilities (and the post-battle hooks) read the context card id, so
    // record the swap there too: the side is now fighting with `new`.
    ctx.set_card_of(house, new);
    ctx.card_swaps.push(CardSwap {
        house,
        old,
        new,
        kind,
    });
    ctx.notes.push(format!(
        "{house} plays {} instead of {}",
        card_name(cards, house, new).unwrap_or_else(|| "no House card".to_string()),
        card_name(cards, house, old).unwrap_or_else(|| "None".to_string()),
    ));
}

/// A card's printed id, for the log messages Python builds from ids.
fn card_name(cards: &HouseCardData, house: HouseId, card: Option<CardId>) -> Option<String> {
    card_data(cards, house, card).map(|spec| spec.id.clone())
}

/// Mace Tyrell destroys one of the opponent's participating Footmen.
fn mace(ctx: &mut BattleContext, cards: &HouseCardData, owner: HouseId) {
    if ability_of(cards, owner, ctx.card_of(owner)) != "mace_tyrell" {
        return;
    }
    let victim = ctx.opponent_of(owner);
    if count(ctx.units_of(victim), UnitType::Footman) == 0 {
        return;
    }
    ctx.units_of_mut(victim)[UnitType::Footman.index()] -= 1;
    ctx.side_of_mut(victim).units_strength -= 1;
    ctx.notes
        .push(format!("Mace Tyrell destroys a {victim} Footman"));
}

/// The passive stat modifiers, one card at a time.
fn passive(
    ctx: &mut BattleContext,
    state: &State,
    map: &GameMap,
    cards: &HouseCardData,
    house: HouseId,
) {
    let ability = ability_of(cards, house, ctx.card_of(house));
    if ability == "none" {
        return;
    }
    let opponent = ctx.opponent_of(house);
    let attacking = ctx.is_attacker(house);

    match ability {
        "stannis_baratheon" => {
            let mine = state.track_position(house, Track::IronThrone);
            let theirs = state.track_position(opponent, Track::IronThrone);
            // Position 1 is best, so a lower number means a higher position.
            if theirs < mine {
                ctx.side_of_mut(house).card_strength += 1;
            }
        }
        "davos_seaworth" => {
            if card_used(state, cards, house, "stannis_baratheon") {
                let side = ctx.side_of_mut(house);
                side.card_strength += 1;
                side.swords += 1;
            }
        }
        "theon_greyjoy" => {
            if !attacking && map.region(ctx.region_id).is_fortification() {
                let side = ctx.side_of_mut(house);
                side.card_strength += 1;
                side.swords += 1;
            }
        }
        "asha_greyjoy" => {
            if !ctx.supported(house) {
                let side = ctx.side_of_mut(house);
                side.swords += 2;
                side.forts += 1;
            }
        }
        "nymeria_sand" => {
            let side = ctx.side_of_mut(house);
            if attacking {
                side.swords += 1;
            } else {
                side.forts += 1;
            }
        }
        "balon_greyjoy" => {
            ctx.side_of_mut(opponent).card_strength = 0;
        }
        "kevan_lannister" => {
            if attacking {
                let footmen = count(ctx.units_of(house), UnitType::Footman)
                    + count(ctx.support_of(house), UnitType::Footman);
                ctx.side_of_mut(house).units_strength += footmen as i16;
            }
        }
        "victarion_greyjoy" => {
            if attacking {
                let ships = count(ctx.units_of(house), UnitType::Ship)
                    + count(ctx.support_of(house), UnitType::Ship);
                ctx.side_of_mut(house).units_strength += ships as i16;
            }
        }
        "catelyn_stark" => {
            let region = state.region(ctx.region_id);
            if !attacking
                && region.order_house == Some(house)
                && region.order.map(|order| order.order_type()) == Some(OrderType::Defend)
            {
                ctx.side_of_mut(house).order_strength *= 2;
            }
        }
        "blackfish" if !ctx.casualty_immunity.contains(&house) => {
            ctx.casualty_immunity.push(house);
        }
        _ => {}
    }
}

/// Salladhor Saan zeroes every non-Baratheon Ship in a supported battle.
///
/// Victarion may already have added +1 per attacking/supporting Ship to
/// `units_strength`; that bonus is ship strength too, so it must be removed
/// along with the printed ship strength. Order bonuses are left untouched.
fn salladhor(ctx: &mut BattleContext, cards: &HouseCardData, house: HouseId) {
    if ability_of(cards, house, ctx.card_of(house)) != "salladhor_saan" {
        return;
    }
    if !ctx.supported(house) {
        return;
    }
    for target in [ctx.attacker, ctx.defender] {
        if target.name() == "baratheon" {
            continue;
        }
        let ships = count(ctx.units_of(target), UnitType::Ship) as i16;
        let support_ships = count(ctx.support_of(target), UnitType::Ship) as i16;
        let victarion = ctx.is_attacker(target)
            && ability_of(cards, target, ctx.card_of(target)) == "victarion_greyjoy";
        let side = ctx.side_of_mut(target);
        // The printed Ship strength is in `units_strength`; supporting Ships
        // also appear in `support_strength`. Victarion's extra +1 per Ship is
        // in `units_strength` regardless of whether the Ship is supporting.
        side.units_strength -= ships;
        side.support_strength -= support_ships;
        if victarion {
            side.units_strength -= ships + support_ships;
        }
    }
    ctx.notes
        .push("Salladhor Saan reduces non-Baratheon Ships to 0".to_string());
}

/// The Queen of Thorns removes an adjacent opponent Order.
///
/// # Support timing
///
/// Support is locked in at step 2 of combat ("Calculate Initial Combat
/// Strength"), while House-card text resolves at step 3. The official FAQ
/// nevertheless requires the initial strength to be recalculated when this card
/// removes a Support order: the opponent "does not add the strength of the
/// supporting units in the area from where the Support order was removed"
/// (AGoT 2E FAQ v2.0, Queen of Thorns). [`crate::combat::prepare_battle`]
/// applies that correction after recording the removal, so both the
/// deterministic and the interactive paths score the battle without that
/// area's support. Interactive play chooses the area before
/// [`crate::combat::prepare_battle`] runs, so it already starts from the
/// post-removal board and needs no extra adjustment.
fn queen_of_thorns(
    ctx: &mut BattleContext,
    state: &State,
    map: &GameMap,
    cards: &HouseCardData,
    house: HouseId,
) {
    if state.choice_choices {
        // Interactive play queues the adjacent-Order choice in the Game layer.
        return;
    }
    if ability_of(cards, house, ctx.card_of(house)) != "queen_of_thorns" {
        return;
    }
    let opponent = ctx.opponent_of(house);
    let mut neighbours: Vec<RegionId> = map.neighbours_of(ctx.region_id).to_vec();
    sort_region_ids(map, &mut neighbours);
    for neighbour in neighbours {
        let region = state.region(neighbour);
        if region.order.is_none() || region.order_house != Some(opponent) {
            continue;
        }
        if Some(neighbour) == ctx.attacker_origin {
            continue;
        }
        ctx.orders_to_remove.push(neighbour);
        ctx.notes.push(format!(
            "Queen of Thorns removes the {opponent} Order in {}",
            map.id_of(neighbour)
        ));
        return;
    }
}

/// Doran Martell moves the opponent to the bottom of an Influence track.
fn doran(ctx: &mut BattleContext, state: &State, cards: &HouseCardData, house: HouseId) {
    if ability_of(cards, house, ctx.card_of(house)) != "doran_martell" {
        return;
    }
    let opponent = ctx.opponent_of(house);
    let track = ctx.doran_track.unwrap_or_else(|| {
        // Python's `min` keeps the first minimum, so the track order decides
        // ties.
        let mut best: Option<(Track, usize)> = None;
        for track in Track::ALL {
            let position = state.track_position(opponent, track);
            match best {
                Some((_, best_position)) if position >= best_position => {}
                _ => best = Some((track, position)),
            }
        }
        best.map(|(track, _)| track).unwrap_or(Track::IronThrone)
    });
    ctx.influence_move = Some((track, opponent));
    ctx.notes.push(format!(
        "Doran Martell sends {opponent} to the bottom of {}",
        track.as_str()
    ));
}

// ---------------------------------------------------------------------------
// Committing deferred effects
// ---------------------------------------------------------------------------

/// Apply the board changes the pure resolution only recorded.
///
/// The card table is part of the signature so that the three card entry points
/// read alike; committing a swap, a Power cost, an Order removal or an Influence
/// move works on the recorded ids and needs no lookup.
pub fn commit_deferred(state: &mut State, _cards: &HouseCardData, ctx: &BattleContext) {
    for entry in &ctx.card_swaps {
        commit_swap(state, entry);
    }
    for (house, cost) in &ctx.power_costs {
        if let Some(house_state) = state.house_mut(*house) {
            house_state.power = house_state.power.saturating_sub(*cost);
        }
    }
    for region_id in &ctx.orders_to_remove {
        let (order, holder) = {
            let region = state.region_mut(*region_id);
            let order = region.order;
            let holder = region.order_house;
            region.order = None;
            region.order_house = None;
            (order, holder)
        };
        if let (Some(order), Some(holder)) = (order, holder) {
            if let Some(house_state) = state.house_mut(holder) {
                if let Some(position) = house_state
                    .orders_placed
                    .iter()
                    .position(|key| *key == order)
                {
                    house_state.orders_placed.remove(position);
                }
            }
        }
    }
    if let Some((track, house)) = ctx.influence_move {
        let previous_blade = if track == Track::Fiefdoms {
            state.blade_holder()
        } else {
            None
        };
        let order = &mut state.tracks[track.index()];
        if let Some(position) = order.iter().position(|candidate| *candidate == house) {
            order.remove(position);
            order.push(house);
        }
        if track == Track::Fiefdoms {
            state.transfer_blade_used(previous_blade);
        }
    }
}

fn commit_swap(state: &mut State, entry: &CardSwap) {
    let Some(house_state) = state.house_mut(entry.house) else {
        return;
    };
    if entry.kind == SwapKind::ReturnSwap {
        if let Some(old) = entry.old {
            if let Some(position) = house_state
                .house_cards_used
                .iter()
                .position(|card| *card == old)
            {
                house_state.house_cards_used.remove(position);
            }
            if !house_state.house_cards_available.contains(&old) {
                house_state.house_cards_available.push(old);
            }
        }
    }
    if let Some(new) = entry.new {
        if let Some(position) = house_state
            .house_cards_available
            .iter()
            .position(|card| *card == new)
        {
            house_state.house_cards_available.remove(position);
        }
        house_state.house_cards_used.push(new);
    }
}

// ---------------------------------------------------------------------------
// Post-battle abilities
// ---------------------------------------------------------------------------

/// Apply winner/loser abilities once the battle has been placed.
///
/// Unlike Python's version this takes `&mut State` and logs its messages there
/// instead of appending them to `ctx.notes` (see the module docs).
pub fn apply_post_battle(
    ctx: &BattleContext,
    state: &mut State,
    map: &GameMap,
    cards: &HouseCardData,
    result: &BattleResult,
) {
    let winner = result.winner;
    let loser = result.loser;
    let winner_ability = ability_of(cards, winner, ctx.card_of(winner));
    let loser_ability = ability_of(cards, loser, ctx.card_of(loser));

    match winner_ability {
        "tywin_lannister" => {
            if let Some(house_state) = state.house_mut(winner) {
                house_state.power += 2;
            }
            state.note("Tywin Lannister gains 2 Power".to_string());
        }
        "renly_baratheon" if !state.choice_choices => renly_upgrade(ctx, state, map, winner),
        "cersei_lannister" if !state.choice_choices => remove_any_order(ctx, state, map, loser),
        _ => {}
    }

    if loser_ability == "roose_bolton" {
        if let Some(house_state) = state.house_mut(loser) {
            let used: SmallVec<[CardId; 7]> = house_state.house_cards_used.clone();
            house_state.house_cards_available.extend(used);
            house_state.house_cards_used.clear();
        }
        state.note("Roose Bolton returns the discard pile to hand".to_string());
    }

    if state.choice_choices {
        queue_post_battle_choices(ctx, state, map, cards, result);
    } else {
        // Patchface acts after combat whichever side won.
        for house in [ctx.attacker, ctx.defender] {
            if ability_of(cards, house, ctx.card_of(house)) == "patchface" {
                patchface(ctx, state, cards, house);
            }
        }
    }
}

/// Queue the interactive post-battle card choices.
fn queue_post_battle_choices(
    ctx: &BattleContext,
    state: &mut State,
    map: &GameMap,
    cards: &HouseCardData,
    result: &BattleResult,
) {
    let winner = result.winner;
    let loser = result.loser;
    let mut queued = false;

    if ability_of(cards, winner, ctx.card_of(winner)) == "renly_baratheon" {
        let winner_side = if winner == ctx.attacker {
            Side::Attacker
        } else {
            Side::Defender
        };
        // With explicit declarations only the areas that really supported the
        // winner may be upgraded: a Support order that refused (or helped the
        // opponent) is not a supporting Footman. `None` keeps the legacy
        // self-directed helper, where every adjacent Support order supports.
        let declared_for_winner = |region: RegionId| match &ctx.support_choices {
            None => true,
            Some(choices) => choices
                .iter()
                .any(|(area, side)| *area == region && *side == winner_side),
        };
        let mut regions: SmallVec<[RegionId; 8]> = SmallVec::new();
        if state.region(ctx.region_id).unit_count(UnitType::Footman) > 0 {
            regions.push(ctx.region_id);
        }
        let mut neighbours: Vec<RegionId> = map.neighbours_of(ctx.region_id).to_vec();
        sort_region_ids(map, &mut neighbours);
        for neighbour in neighbours {
            let region = state.region(neighbour);
            if region.unit_count(UnitType::Footman) == 0 {
                continue;
            }
            let eligible = match &ctx.support_choices {
                // No declarations: the legacy helper treats every adjacent
                // Support order of the winner as supporting it.
                None => {
                    region.order_house == Some(winner)
                        && region
                            .order
                            .is_some_and(|order| order.order_type() == OrderType::Support)
                }
                // With declarations the declaration itself says whether the
                // area supported the winner. The Support order may already be
                // gone (Queen of Thorns) without undoing the support it gave.
                Some(_) => region.controller == Some(winner) && declared_for_winner(neighbour),
            };
            if eligible {
                regions.push(neighbour);
            }
        }
        if !regions.is_empty() {
            state.choice_queue.push(ChoiceTask {
                house: winner,
                kind: ChoiceKind::Region {
                    purpose: RegionChoice::RenlyFootman,
                    regions,
                    allow_none: true,
                },
            });
            queued = true;
        }
    }

    if ability_of(cards, winner, ctx.card_of(winner)) == "cersei_lannister" {
        let mut regions: Vec<RegionId> = state
            .region_ids()
            .filter(|region| state.region(*region).order_house == Some(loser))
            .collect();
        sort_region_ids(map, &mut regions);
        if !regions.is_empty() {
            state.choice_queue.push(ChoiceTask {
                house: winner,
                kind: ChoiceKind::Region {
                    purpose: RegionChoice::CerseiOrder { opponent: loser },
                    regions: SmallVec::from_vec(regions),
                    allow_none: true,
                },
            });
            queued = true;
        }
    }

    for house in [ctx.attacker, ctx.defender] {
        if ability_of(cards, house, ctx.card_of(house)) == "patchface" {
            let target = if house == ctx.attacker {
                ctx.defender
            } else {
                ctx.attacker
            };
            let hand = state
                .house(target)
                .map(|state| state.house_cards_available.clone())
                .unwrap_or_default();
            if !hand.is_empty() {
                state.choice_queue.push(ChoiceTask {
                    house,
                    kind: ChoiceKind::Card {
                        purpose: CardChoice::PatchfaceDiscard { target },
                        cards: hand,
                        allow_none: true,
                    },
                });
                queued = true;
            }
        }
    }

    if queued {
        state.choice_resume = Some(ChoiceResume::PostBattle);
    }
}

/// Renly Baratheon upgrades a Footman to a Knight.
fn renly_upgrade(ctx: &BattleContext, state: &mut State, map: &GameMap, house: HouseId) {
    let region_id = ctx.region_id;
    if state.region(region_id).unit_count(UnitType::Footman) > 0 {
        let region = state.region_mut(region_id);
        region.units[UnitType::Footman.index()] -= 1;
        region.units[UnitType::Knight.index()] += 1;
        state.note("Renly Baratheon upgrades a Footman to a Knight".to_string());
        return;
    }
    // Otherwise a supporting Baratheon Footman may be upgraded.
    let mut neighbours: Vec<RegionId> = map.neighbours_of(region_id).to_vec();
    sort_region_ids(map, &mut neighbours);
    for neighbour in neighbours {
        let region = state.region(neighbour);
        if region.order_house != Some(house) {
            continue;
        }
        let Some(order) = region.order else {
            continue;
        };
        if order.order_type() != OrderType::Support {
            continue;
        }
        if region.unit_count(UnitType::Footman) == 0 {
            continue;
        }
        let region = state.region_mut(neighbour);
        region.units[UnitType::Footman.index()] -= 1;
        region.units[UnitType::Knight.index()] += 1;
        state.note("Renly Baratheon upgrades a supporting Footman".to_string());
        return;
    }
}

/// Cersei Lannister removes one of the loser's Orders.
fn remove_any_order(_ctx: &BattleContext, state: &mut State, map: &GameMap, house: HouseId) {
    let mut ids: Vec<RegionId> = state.region_ids().collect();
    sort_region_ids(map, &mut ids);
    for region_id in ids {
        let region = state.region(region_id);
        if region.order.is_none() || region.order_house != Some(house) {
            continue;
        }
        let order = region.order.expect("checked above");
        {
            let region = state.region_mut(region_id);
            region.order = None;
            region.order_house = None;
        }
        if let Some(house_state) = state.house_mut(house) {
            if let Some(position) = house_state
                .orders_placed
                .iter()
                .position(|key| *key == order)
            {
                house_state.orders_placed.remove(position);
            }
        }
        state.note(format!(
            "Cersei Lannister removes the {house} Order in {}",
            map.id_of(region_id)
        ));
        return;
    }
}

/// Patchface discards the opponent's strongest remaining card.
fn patchface(ctx: &BattleContext, state: &mut State, cards: &HouseCardData, house: HouseId) {
    let opponent = ctx.opponent_of(house);
    let Some(house_state) = state.house_mut(opponent) else {
        return;
    };
    let available: Vec<CardId> = house_state.house_cards_available.to_vec();
    let Some(card) = best_card(cards, opponent, &available) else {
        return;
    };
    if let Some(position) = house_state
        .house_cards_available
        .iter()
        .position(|candidate| *candidate == card)
    {
        house_state.house_cards_available.remove(position);
    }
    house_state.house_cards_used.push(card);
    let name = card_name(cards, opponent, Some(card)).unwrap_or_default();
    state.note(format!("Patchface discards {opponent}'s {name}"));
}

/// True when Robb Stark won and gets to pick the retreat of the loser.
pub fn robb_chose_retreat(ctx: &BattleContext, cards: &HouseCardData, winner: HouseId) -> bool {
    ability_of(cards, winner, ctx.card_of(winner)) == "robb_stark"
}

/// True when Arianne Martell defended and lost, blocking the conquest.
pub fn arianne_holds(ctx: &BattleContext, cards: &HouseCardData, loser: HouseId) -> bool {
    loser == ctx.defender && ability_of(cards, loser, ctx.card_of(loser)) == "arianne_martell"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combat::{prepare_battle, resolve_battle, BattleRequest};
    use crate::map::DataSet;
    use crate::state::House;
    use smallvec::smallvec;

    /// A blank board sized like the bundled data, with no houses in play.
    fn blank_state(data: &DataSet) -> State {
        let mut state = State::blank();
        state.region_count = data.map.region_count() as u8;
        state
    }

    /// The Python helper: the first land region in board order and its first
    /// land neighbour.
    fn first_land_pair(map: &GameMap) -> (RegionId, RegionId) {
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

    #[test]
    fn ability_lookup_uses_the_house_deck() {
        let data = DataSet::load_default().unwrap();
        let cards = &data.house_cards;
        let stark = HouseId::from_name("stark").unwrap();
        let blackfish = cards.card_index(stark, "blackfish").unwrap();
        assert_eq!(ability_of(cards, stark, Some(blackfish)), "blackfish");
        assert_eq!(ability_of(cards, stark, None), "none");
        // A card id from another house's deck is not a card of this house.
        assert!(card_data(cards, stark, Some(CardId(200))).is_none());
    }

    #[test]
    fn every_ability_in_the_data_is_implemented() {
        // Mirrors `test_every_house_card_ability_is_implemented`: a card with no
        // special ability declares "none", which is not an implemented id.
        let data = DataSet::load_default().unwrap();
        for house in HouseId::ALL {
            for spec in data.house_cards.deck(house) {
                if spec.ability == "none" {
                    continue;
                }
                assert!(
                    IMPLEMENTED_ABILITIES.contains(&spec.ability.as_str()),
                    "{}: unknown ability {:?}",
                    spec.id,
                    spec.ability
                );
            }
        }
    }

    #[test]
    fn best_card_keeps_the_first_of_equal_scores() {
        let data = DataSet::load_default().unwrap();
        let cards = &data.house_cards;
        let tyrell = HouseId::from_name("tyrell").unwrap();
        // Margaery Tyrell and Alester Florent print the same 1/0/1 stats.
        let margaery = cards.card_index(tyrell, "margaery_tyrell").unwrap();
        let alester = cards.card_index(tyrell, "alester_florent").unwrap();
        assert_eq!(
            card_score(cards, tyrell, Some(margaery)),
            card_score(cards, tyrell, Some(alester))
        );
        assert_eq!(
            best_card(cards, tyrell, &[alester, margaery]),
            Some(alester),
            "ties must keep the first card, as Python's max does"
        );
        assert_eq!(
            best_card(cards, tyrell, &[margaery, alester]),
            Some(margaery)
        );
        assert_eq!(best_card(cards, tyrell, &[]), None);
    }

    #[test]
    fn apply_swap_updates_the_context_card_id_for_later_abilities() {
        let data = DataSet::load_default().unwrap();
        let cards = &data.house_cards;
        let (target, _) = first_land_pair(&data.map);
        let stark = HouseId::from_name("stark").unwrap();
        let lannister = HouseId::from_name("lannister").unwrap();
        let robb = cards.card_index(stark, "robb_stark").unwrap();
        let jaime = cards.card_index(lannister, "jaime_lannister").unwrap();
        let request = BattleRequest::new(target, stark, lannister, [1, 0, 0, 0], [1, 0, 0, 0]);
        let mut ctx = prepare_battle(&blank_state(&data), &data.map, cards, &request, None);

        apply_swap(
            &mut ctx,
            cards,
            stark,
            None,
            Some(robb),
            SwapKind::DiscardSwap,
        );
        assert_eq!(ctx.card_of(stark), Some(robb));
        assert_eq!(ctx.attacker_card, Some(robb));
        assert_eq!(ctx.defender_card, None);
        let spec = card_data(cards, stark, Some(robb)).unwrap();
        assert_eq!(ctx.side_of(stark).card_strength, spec.strength as i16);
        assert_eq!(ctx.side_of(stark).swords, spec.swords);
        assert_eq!(ctx.side_of(stark).forts, spec.forts);

        // The defender's field is updated independently.
        apply_swap(
            &mut ctx,
            cards,
            lannister,
            None,
            Some(jaime),
            SwapKind::ReturnSwap,
        );
        assert_eq!(ctx.defender_card, Some(jaime));
        assert_eq!(ctx.attacker_card, Some(robb));
        assert_eq!(ctx.card_of(lannister), Some(jaime));
    }

    #[test]
    fn aeron_swap_is_visible_to_the_abilities_after_it() {
        let data = DataSet::load_default().unwrap();
        let cards = &data.house_cards;
        let (target, _) = first_land_pair(&data.map);
        let greyjoy = HouseId::from_name("greyjoy").unwrap();
        let stark = HouseId::from_name("stark").unwrap();
        let aeron = cards.card_index(greyjoy, "aeron_damphair").unwrap();
        let euron = cards.card_index(greyjoy, "euron_crows_eye").unwrap();

        let mut state = blank_state(&data);
        state.houses[greyjoy.index()] = Some(House::new(7, 5, 3));
        state.house_mut(greyjoy).unwrap().play_card(aeron);

        let mut request = BattleRequest::new(target, greyjoy, stark, [1, 0, 0, 0], [1, 0, 0, 0]);
        request.attacker_card = Some(aeron);
        let result = resolve_battle(&state, &data.map, cards, &request);

        // The side now fights with Euron, and every later ability reads him.
        assert_eq!(result.context.attacker_card, Some(euron));
        assert_eq!(result.attacker.card_strength, 4);
        assert_eq!(result.attacker.swords, 1);
        assert_eq!(
            ability_of(cards, greyjoy, result.context.card_of(greyjoy)),
            "none"
        );
    }

    #[test]
    fn renly_only_upgrades_a_footman_that_actually_supported_him() {
        let data = DataSet::load_default().unwrap();
        let map = &data.map;
        let cards = &data.house_cards;
        let baratheon = HouseId::from_name("baratheon").unwrap();
        let stark = HouseId::from_name("stark").unwrap();
        let target = map.lookup("highgarden").unwrap();
        let support = map.lookup("searoad_marches").unwrap();
        assert!(map.neighbours_of(target).contains(&support));

        let mut state = blank_state(&data);
        state.choice_choices = true;
        state.region_mut(target).controller = Some(stark);
        {
            let region = state.region_mut(support);
            region.controller = Some(baratheon);
            region.units = [1, 0, 0, 0];
            region.order = Some(OrderKey::from_name("support_a").unwrap());
            region.order_house = Some(baratheon);
        }
        let renly = cards.card_index(baratheon, "renly_baratheon").unwrap();
        let request = |side: Option<Side>| {
            let mut request = BattleRequest::new(target, baratheon, stark, [2, 0, 0, 0], [0; 4]);
            request.attacker_card = Some(renly);
            request.support_choices = side.map(|side| smallvec![(support, side)]);
            request
        };

        // The Support order refused: Renly did not win with that Footman's
        // help, so the area must not be offered.
        let result = resolve_battle(&state, map, cards, &request(Some(Side::Refuse)));
        assert_eq!(result.winner, baratheon);
        apply_post_battle(&result.context, &mut state, map, cards, &result);
        assert!(
            state.choice_queue.is_empty(),
            "a refused Support order is not an eligible Renly area"
        );

        // Declared for the winner: the supporting Footman is eligible.
        let result = resolve_battle(&state, map, cards, &request(Some(Side::Attacker)));
        apply_post_battle(&result.context, &mut state, map, cards, &result);
        match &state
            .choice_queue
            .first()
            .expect("Renly must offer the declared supporter")
            .kind
        {
            ChoiceKind::Region {
                purpose: RegionChoice::RenlyFootman,
                regions,
                allow_none,
            } => {
                assert!(*allow_none);
                assert_eq!(regions.as_slice(), &[support]);
            }
            other => panic!("expected a Renly Footman choice, got {other:?}"),
        }

        // The legacy self-directed helper (no declarations) keeps offering the
        // winner's own adjacent Support order.
        state.choice_queue.clear();
        let result = resolve_battle(&state, map, cards, &request(None));
        apply_post_battle(&result.context, &mut state, map, cards, &result);
        match &state
            .choice_queue
            .first()
            .expect("the legacy helper still offers the Support order")
            .kind
        {
            ChoiceKind::Region {
                purpose: RegionChoice::RenlyFootman,
                regions,
                ..
            } => assert_eq!(regions.as_slice(), &[support]),
            other => panic!("expected a Renly Footman choice, got {other:?}"),
        }
    }
}
