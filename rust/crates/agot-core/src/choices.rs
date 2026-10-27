//! Generic player choices that are not tied to a dedicated phase.
//!
//! House-card text and Wildling consequences used to auto-resolve their
//! sub-choices. They now enqueue [`ChoiceTask`]s; the engine pauses on the task
//! owner, exposes the legal actions in [`Game::choice_actions`], and resumes
//! the surrounding battle/Wildling sequence when the queue is empty.
//!
//! The action-feature layout is frozen, so the tasks reuse existing action
//! kinds: `PlayHouseCard` for card choices, `WesterosChoice` for regions,
//! tracks and either/or options, `ChooseCasualty`/`Upgrade` for units and
//! `Muster` for the Horde's mustering reward. Two existing fields carry the
//! extra information the reused kinds would otherwise lose:
//!
//! * `amount` names the unit state of a generic unit choice (0 = ready,
//!   1 = routed), the same convention as Supply disbanding.
//! * `order` carries a [`crate::action::choice_context`] tag so the encoder's
//!   ORDER_BLOCK can tell, say, a Queen of Thorns pick from a Horde pick. No
//!   handler reads it.
//!
//! `card_house` names the deck of a card that does not belong to the choosing
//! house (Patchface discards from the opponent's hand).

use smallvec::SmallVec;

use crate::action::{choice_context, Action};
use crate::combat;
use crate::enums::{CardId, HouseId, RegionId, Track, UnitType};
use crate::game::{EngineError, Game};
use crate::map::GameMap;
use crate::state::{
    CardChoice, ChoiceKind, ChoiceResume, ChoiceTask, OptionChoice, RegionChoice, TrackChoice,
    UnitChoice,
};

impl Game {
    /// Resume the sequence that paused for generic choices.
    pub(crate) fn resume_choice_queue(&mut self) -> Result<(), EngineError> {
        match self.state.choice_resume.take() {
            Some(ChoiceResume::BattlePre) => {
                // The Cards stage re-runs the Doran check and then prepares
                // the battle without queueing the same choices again.
                if let Some(battle) = self.state.pending_battle.as_mut() {
                    battle.stage = crate::state::BattleStage::Cards;
                }
            }
            Some(ChoiceResume::Wildling) => {
                self.complete_wildling_after_choices()?;
            }
            Some(ChoiceResume::PostBattle) | None => {}
        }
        Ok(())
    }

    /// The actions offered for the first pending generic choice.
    pub(crate) fn choice_actions(&self, task: &ChoiceTask) -> Vec<Action> {
        let house = task.house;
        match &task.kind {
            ChoiceKind::Unit {
                purpose,
                areas,
                remaining,
                allow_none,
            } => {
                let context = unit_choice_context(*purpose, areas, &self.data.map);
                let mut actions = Vec::new();
                if *remaining > 0 {
                    for region in areas {
                        let state = self.state.region(*region);
                        if *purpose == UnitChoice::UpgradeFootman {
                            // The action names the Knight it becomes; `amount`
                            // says whether the source Footman is ready.
                            if state.unit_count(UnitType::Footman) > 0 {
                                actions.push(
                                    Action::choice_upgrade_state(house, *region, false)
                                        .with_context(context),
                                );
                            }
                            if state.routed_count(UnitType::Footman) > 0 {
                                actions.push(
                                    Action::choice_upgrade_state(house, *region, true)
                                        .with_context(context),
                                );
                            }
                            continue;
                        }
                        for unit in UnitType::ALL {
                            if *purpose == UnitChoice::ReplaceKnight && unit != UnitType::Knight {
                                continue;
                            }
                            // Ready and routed copies are separate decisions:
                            // routing is a real difference (a routed unit fights
                            // at 0 and cannot be the one a "destroy a ready
                            // unit" text takes).
                            if state.unit_count(unit) > 0 {
                                actions.push(
                                    Action::choice_unit_state(house, *region, unit, false)
                                        .with_context(context),
                                );
                            }
                            if state.routed_count(unit) > 0 {
                                actions.push(
                                    Action::choice_unit_state(house, *region, unit, true)
                                        .with_context(context),
                                );
                            }
                        }
                    }
                }
                if *allow_none || actions.is_empty() {
                    actions.push(Action::pass(house));
                }
                actions
            }
            ChoiceKind::Card {
                purpose,
                cards,
                allow_none,
            } => {
                let owner = card_choice_owner(purpose);
                let context = card_choice_context(purpose);
                let mut actions: Vec<Action> = cards
                    .iter()
                    .map(|card| {
                        let mut action = Action::play_house_card(house, Some(*card));
                        if let Some(owner) = owner {
                            if owner != house {
                                action = action.with_card_house(owner);
                            }
                        }
                        action.with_context(context)
                    })
                    .collect();
                if *allow_none || actions.is_empty() {
                    actions.push(Action::pass(house));
                }
                actions
            }
            ChoiceKind::Region {
                purpose,
                regions,
                allow_none,
            } => {
                let context = region_choice_context(purpose);
                let mut actions: Vec<Action> = regions
                    .iter()
                    .map(|region| Action::choice_region(house, *region).with_context(context))
                    .collect();
                if *allow_none || actions.is_empty() {
                    actions.push(Action::pass(house));
                }
                actions
            }
            ChoiceKind::Track { purpose, tracks } => {
                let context = track_choice_context(*purpose);
                tracks
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        Action::choice_amount(house, index as u8).with_context(context)
                    })
                    .collect()
            }
            ChoiceKind::Option { purpose, options } => {
                let context = option_choice_context(*purpose);
                options
                    .iter()
                    .map(|option| {
                        let action = Action::choice_amount(house, *option);
                        match context {
                            Some(tag) => action.with_context(tag),
                            None => action,
                        }
                    })
                    .collect()
            }
            ChoiceKind::Muster { region, points } => {
                let mut actions = Vec::new();
                if *points > 0 {
                    for unit in [UnitType::Footman, UnitType::Knight, UnitType::Siege] {
                        let cost = crate::enums::MUSTER_COST[unit.index()];
                        if cost <= *points && self.muster_fits(house, *region, unit, 1, None) {
                            actions.push(
                                Action::muster(house, *region, unit, None)
                                    .with_context(choice_context::HORDE_MUSTER),
                            );
                        }
                    }
                    if self.state.region(*region).unit_count(UnitType::Footman) > 0 {
                        for unit in [UnitType::Knight, UnitType::Siege] {
                            if self.muster_fits(house, *region, unit, 1, None) {
                                actions.push(
                                    Action::upgrade(house, *region, unit)
                                        .with_context(choice_context::HORDE_MUSTER),
                                );
                            }
                        }
                    }
                    for destination in self.ship_muster_targets(house, *region) {
                        if self.muster_fits(house, destination, UnitType::Ship, 1, None) {
                            actions.push(
                                Action::muster(house, *region, UnitType::Ship, Some(destination))
                                    .with_context(choice_context::HORDE_MUSTER),
                            );
                        }
                    }
                }
                actions.push(Action::pass(house));
                actions
            }
        }
    }

    /// Apply one generic choice. Returns an error only for a malformed action;
    /// an action outside the current task is rejected so a stale GUI click
    /// cannot silently mutate the board.
    pub(crate) fn apply_choice(&mut self, action: Action) -> Result<(), EngineError> {
        let Some(task) = self.state.choice_queue.first().cloned() else {
            return Ok(());
        };
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("choice without a house".into()))?;
        if house != task.house {
            return Err(EngineError::IllegalAction(
                "choice belongs to another house".into(),
            ));
        }

        // Remove the task first: handlers may insert follow-up tasks at the
        // front of the queue (Horde fort, Preemptive Raid, ...).
        self.state.choice_queue.remove(0);
        let mut task = task;
        let keep = match &mut task.kind {
            ChoiceKind::Unit {
                purpose,
                areas,
                remaining,
                allow_none,
            } => self.apply_unit_choice(house, action, *purpose, areas, remaining, *allow_none)?,
            ChoiceKind::Card {
                purpose,
                cards,
                allow_none,
            } => {
                if action.kind == crate::action::ActionKind::Pass {
                    if *allow_none {
                        false
                    } else if cards.is_empty() {
                        // Tyrion with no alternative card: the forced result is
                        // simply "no House card", not a declined choice.
                        self.apply_card_choice(house, purpose, None)?;
                        false
                    } else {
                        return Err(EngineError::IllegalAction(
                            "this House-card choice is mandatory".into(),
                        ));
                    }
                } else {
                    let card = action.card;
                    if let Some(card) = card {
                        if !cards.contains(&card) {
                            return Err(EngineError::IllegalAction(
                                "card is not available for this choice".into(),
                            ));
                        }
                    }
                    self.apply_card_choice(house, purpose, card)?;
                    false
                }
            }
            ChoiceKind::Region {
                purpose,
                regions,
                allow_none,
            } => {
                if action.kind == crate::action::ActionKind::Pass {
                    if !*allow_none {
                        return Err(EngineError::IllegalAction(
                            "this area choice is mandatory".into(),
                        ));
                    }
                    false
                } else {
                    let region = action.region.or(action.target).ok_or_else(|| {
                        EngineError::IllegalAction("choice without an area".into())
                    })?;
                    if !regions.contains(&region) {
                        return Err(EngineError::IllegalAction(
                            "area is not available for this choice".into(),
                        ));
                    }
                    self.apply_region_choice(house, purpose, region)?;
                    false
                }
            }
            ChoiceKind::Track { purpose, tracks } => {
                let index = action.amount.ok_or_else(|| {
                    EngineError::IllegalAction("track choice without an index".into())
                })? as usize;
                let track = *tracks.get(index).ok_or_else(|| {
                    EngineError::IllegalAction("track choice index out of range".into())
                })?;
                match purpose {
                    TrackChoice::KingBeyondHighest => self.move_influence(track, house, true),
                    TrackChoice::KingBeyondOther => self.move_influence(track, house, false),
                    TrackChoice::PreemptiveHighest => {
                        self.apply_preemptive_track_drop(house, track)
                    }
                }
                false
            }
            ChoiceKind::Option { purpose, options } => {
                let amount = action.amount.ok_or_else(|| {
                    EngineError::IllegalAction("option choice without a value".into())
                })?;
                if !options.contains(&amount) {
                    return Err(EngineError::IllegalAction(
                        "option is not offered by this choice".into(),
                    ));
                }
                match purpose {
                    OptionChoice::PreemptiveRaid => self.apply_preemptive_choice(house, amount)?,
                    OptionChoice::TyrionUse { target } => {
                        self.apply_tyrion_use(house, *target, amount)?
                    }
                    OptionChoice::RobbRetreat => self.apply_robb_retreat(house, amount)?,
                }
                false
            }
            ChoiceKind::Muster { region, points } => {
                self.apply_wildling_muster(house, action, *region, points)?
            }
        };
        if keep {
            self.state.choice_queue.insert(0, task);
        }
        Ok(())
    }

    fn apply_unit_choice(
        &mut self,
        house: HouseId,
        action: Action,
        purpose: UnitChoice,
        areas: &mut SmallVec<[RegionId; 8]>,
        remaining: &mut u8,
        allow_none: bool,
    ) -> Result<bool, EngineError> {
        if action.kind == crate::action::ActionKind::Pass {
            let has_units = areas.iter().any(|region| {
                let state = self.state.region(*region);
                match purpose {
                    UnitChoice::Destroy => state.total(true) > 0,
                    UnitChoice::ReplaceKnight => {
                        state.unit_count(UnitType::Knight) + state.routed_count(UnitType::Knight)
                            > 0
                    }
                    UnitChoice::UpgradeFootman => {
                        state.unit_count(UnitType::Footman) + state.routed_count(UnitType::Footman)
                            > 0
                    }
                }
            });
            return if allow_none || !has_units || *remaining == 0 {
                Ok(false)
            } else {
                Err(EngineError::IllegalAction(
                    "this unit choice is mandatory".into(),
                ))
            };
        }
        let region = action
            .region
            .ok_or_else(|| EngineError::IllegalAction("unit choice without an area".into()))?;
        if !areas.contains(&region) {
            return Err(EngineError::IllegalAction(
                "area is not available for this choice".into(),
            ));
        }
        let mut unit = action
            .unit
            .ok_or_else(|| EngineError::IllegalAction("unit choice without a unit".into()))?;
        if purpose == UnitChoice::UpgradeFootman {
            // The action names the Knight it becomes; the source is a Footman.
            unit = UnitType::Footman;
        }
        let region_state = self.state.region_mut(region);
        if region_state.controller != Some(house) {
            return Err(EngineError::IllegalAction(
                "the choosing house no longer controls that area".into(),
            ));
        }
        let ready = region_state.unit_count(unit) > 0;
        let has_routed = region_state.routed_count(unit) > 0;
        // `amount` names the state (0 = ready, 1 = routed). An action without
        // it keeps the historical ready-first behaviour, which is also what a
        // hand-written or Python-created action does.
        let routed = match action.amount {
            Some(0) => false,
            Some(1) => true,
            _ => !ready && has_routed,
        };
        if routed {
            if !has_routed {
                return Err(EngineError::IllegalAction(
                    "that routed unit no longer exists in the area".into(),
                ));
            }
        } else if !ready {
            return Err(EngineError::IllegalAction(
                "that unit no longer exists in the area".into(),
            ));
        }
        match purpose {
            UnitChoice::Destroy => {
                if routed {
                    region_state.routed[unit.index()] -= 1;
                } else {
                    region_state.units[unit.index()] -= 1;
                }
                region_state.settle_control();
                let state_text = if routed { "routed " } else { "" };
                self.state.note(format!(
                    "{house} destroys a {state_text}{unit} in {region:?}"
                ));
            }
            UnitChoice::ReplaceKnight => {
                if unit != UnitType::Knight {
                    return Err(EngineError::IllegalAction(
                        "only Knights may be replaced".into(),
                    ));
                }
                if routed {
                    region_state.routed[UnitType::Knight.index()] -= 1;
                    region_state.routed[UnitType::Footman.index()] += 1;
                } else {
                    region_state.units[UnitType::Knight.index()] -= 1;
                    region_state.units[UnitType::Footman.index()] += 1;
                }
                self.state.note(format!(
                    "{house} replaces a Knight with a Footman in {region:?}"
                ));
            }
            UnitChoice::UpgradeFootman => {
                if unit != UnitType::Footman {
                    return Err(EngineError::IllegalAction(
                        "only Footmen may be upgraded".into(),
                    ));
                }
                if routed {
                    region_state.routed[UnitType::Footman.index()] -= 1;
                    region_state.routed[UnitType::Knight.index()] += 1;
                } else {
                    region_state.units[UnitType::Footman.index()] -= 1;
                    region_state.units[UnitType::Knight.index()] += 1;
                }
                self.state.note(format!(
                    "{house} upgrades a Footman to a Knight in {region:?}"
                ));
            }
        }
        *remaining = remaining.saturating_sub(1);
        Ok(*remaining > 0)
    }

    fn apply_card_choice(
        &mut self,
        house: HouseId,
        purpose: &CardChoice,
        card: Option<CardId>,
    ) -> Result<(), EngineError> {
        match purpose {
            CardChoice::TyrionSwap { target } => {
                let old = self.battle_card(*target);
                let target_state = self
                    .state
                    .house_mut(*target)
                    .ok_or(EngineError::MissingHouse(*target))?;
                if let Some(old) = old {
                    if let Some(position) = target_state
                        .house_cards_used
                        .iter()
                        .position(|candidate| *candidate == old)
                    {
                        target_state.house_cards_used.remove(position);
                    }
                    if !target_state.house_cards_available.contains(&old) {
                        target_state.house_cards_available.push(old);
                    }
                }
                if let Some(card) = card {
                    if let Some(position) = target_state
                        .house_cards_available
                        .iter()
                        .position(|candidate| *candidate == card)
                    {
                        target_state.house_cards_available.remove(position);
                    } else {
                        return Err(EngineError::IllegalAction(
                            "Tyrion choice is not in the target's hand".into(),
                        ));
                    }
                    target_state.house_cards_used.push(card);
                }
                self.set_battle_card(*target, card);
                self.state.note(format!(
                    "{target} plays {} after Tyrion's choice",
                    card.map(|card| format!("{card:?}"))
                        .unwrap_or("no card".into())
                ));
            }
            CardChoice::AeronSwap => {
                let Some(card) = card else {
                    return Ok(());
                };
                let power = self
                    .state
                    .house(house)
                    .map(|state| state.power)
                    .unwrap_or(0);
                if power < 2 {
                    return Err(EngineError::IllegalAction("Aeron needs 2 Power".into()));
                }
                {
                    let state = self.state.house_mut(house).expect("checked");
                    if let Some(position) = state
                        .house_cards_available
                        .iter()
                        .position(|candidate| *candidate == card)
                    {
                        state.house_cards_available.remove(position);
                    } else {
                        return Err(EngineError::IllegalAction(
                            "Aeron choice is not in hand".into(),
                        ));
                    }
                    state.house_cards_used.push(card);
                    state.power = state.power.saturating_sub(2);
                }
                self.set_battle_card(house, Some(card));
                self.state
                    .note("Aeron Damphair pays 2 Power for a different card".into());
            }
            CardChoice::PatchfaceDiscard { target } => {
                let Some(card) = card else {
                    return Ok(());
                };
                let state = self
                    .state
                    .house_mut(*target)
                    .ok_or(EngineError::MissingHouse(*target))?;
                if let Some(position) = state
                    .house_cards_available
                    .iter()
                    .position(|candidate| *candidate == card)
                {
                    state.house_cards_available.remove(position);
                } else {
                    return Err(EngineError::IllegalAction(
                        "Patchface choice is not in the target's hand".into(),
                    ));
                }
                state.house_cards_used.push(card);
                self.state
                    .note(format!("Patchface discards {target}'s {card:?}"));
            }
            CardChoice::MassingDiscard => {
                let Some(card) = card else {
                    return Ok(());
                };
                let state = self.state.house_mut(house).expect("choice house exists");
                if let Some(position) = state
                    .house_cards_available
                    .iter()
                    .position(|candidate| *candidate == card)
                {
                    state.house_cards_available.remove(position);
                } else {
                    return Err(EngineError::IllegalAction(
                        "Massing choice is not in hand".into(),
                    ));
                }
                state.house_cards_used.push(card);
                self.state
                    .note(format!("{house} discards {card:?} to Massing at Milkwater"));
            }
            CardChoice::MammothRetrieve => {
                let Some(card) = card else {
                    return Ok(());
                };
                let state = self.state.house_mut(house).expect("choice house exists");
                if let Some(position) = state
                    .house_cards_used
                    .iter()
                    .position(|candidate| *candidate == card)
                {
                    state.house_cards_used.remove(position);
                    state.house_cards_available.push(card);
                } else {
                    return Err(EngineError::IllegalAction(
                        "Mammoth choice is not in the discard".into(),
                    ));
                }
                self.state
                    .note(format!("{house} retrieves {card:?} from the discard"));
            }
        }
        Ok(())
    }

    fn apply_region_choice(
        &mut self,
        house: HouseId,
        purpose: &RegionChoice,
        region: RegionId,
    ) -> Result<(), EngineError> {
        match purpose {
            RegionChoice::QueenOrder { opponent } => {
                self.remove_order_at(region, *opponent)?;
                self.state.note(format!(
                    "Queen of Thorns removes {opponent}'s Order in {region:?}"
                ));
            }
            RegionChoice::CerseiOrder { opponent } => {
                self.remove_order_at(region, *opponent)?;
                self.state.note(format!(
                    "Cersei Lannister removes {opponent}'s Order in {region:?}"
                ));
            }
            RegionChoice::RenlyFootman => {
                let state = self.state.region_mut(region);
                if state.controller != Some(house) {
                    return Err(EngineError::IllegalAction(
                        "Renly's area is no longer controlled".into(),
                    ));
                }
                if state.unit_count(UnitType::Footman) == 0 {
                    return Err(EngineError::IllegalAction(
                        "Renly's area no longer has a Footman".into(),
                    ));
                }
                state.units[UnitType::Footman.index()] -= 1;
                state.units[UnitType::Knight.index()] += 1;
                self.state
                    .note("Renly Baratheon upgrades a Footman to a Knight".into());
            }
            RegionChoice::HordeFort { count } => {
                let units = self.state.region(region).total(true) as u8;
                let tasks = if units >= *count {
                    vec![ChoiceTask {
                        house,
                        kind: ChoiceKind::Unit {
                            purpose: UnitChoice::Destroy,
                            areas: SmallVec::from_slice(&[region]),
                            remaining: *count,
                            allow_none: false,
                        },
                    }]
                } else {
                    // Cannot take the full penalty at the chosen fort: the
                    // card falls back to destroying units anywhere.
                    self.destroy_tasks(house, *count)
                };
                self.push_choice_tasks(tasks);
                self.state
                    .note(format!("The Horde Descends destroys units in {region:?}"));
            }
            RegionChoice::HordeMusterFort => {
                let points = self.data.map.region(region).mustering;
                if points > 0 {
                    self.state.choice_queue.insert(
                        0,
                        ChoiceTask {
                            house,
                            kind: ChoiceKind::Muster { region, points },
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn apply_wildling_muster(
        &mut self,
        house: HouseId,
        action: Action,
        region: RegionId,
        points: &mut u8,
    ) -> Result<bool, EngineError> {
        if action.kind == crate::action::ActionKind::Pass {
            return Ok(false);
        }
        if self.state.region(region).controller != Some(house) {
            return Err(EngineError::IllegalAction(
                "the mustering area is no longer controlled".into(),
            ));
        }
        match action.kind {
            crate::action::ActionKind::Muster => {
                let unit = action
                    .unit
                    .ok_or_else(|| EngineError::IllegalAction("muster without a unit".into()))?;
                let cost = crate::enums::MUSTER_COST[unit.index()];
                let destination = if unit == UnitType::Ship {
                    action.target.ok_or_else(|| {
                        EngineError::IllegalAction("ship mustering without a destination".into())
                    })?
                } else {
                    region
                };
                if unit == UnitType::Ship
                    && !self
                        .ship_muster_targets(house, region)
                        .contains(&destination)
                {
                    return Err(EngineError::IllegalAction(
                        "that sea is not a legal ship-mustering destination".into(),
                    ));
                }
                if cost > *points || !self.muster_fits(house, destination, unit, 1, None) {
                    return Err(EngineError::IllegalAction(
                        "that unit cannot be mustered here".into(),
                    ));
                }
                let region_state = self.state.region_mut(destination);
                // A Ship mustered into an empty sea or port takes control of it,
                // exactly like normal mustering: leaving an uncontrolled area
                // holding the house's only Ship would break transport and
                // Support checks. An enemy-controlled area can never be a
                // destination (`ship_muster_targets` filters it), so this only
                // ever fills a neutral or friendly area.
                if unit == UnitType::Ship
                    && region_state.is_empty()
                    && region_state.routed.iter().all(|count| *count == 0)
                    && region_state
                        .controller
                        .map(|controller| controller == house)
                        .unwrap_or(true)
                {
                    region_state.controller = Some(house);
                }
                region_state.units[unit.index()] += 1;
                *points -= cost;
                self.state
                    .note(format!("{house} musters a {unit} in {destination:?}"));
            }
            crate::action::ActionKind::Upgrade => {
                let target = action.unit.unwrap_or(UnitType::Knight);
                if target != UnitType::Knight && target != UnitType::Siege {
                    return Err(EngineError::IllegalAction(
                        "invalid mustering upgrade".into(),
                    ));
                }
                if *points < 1
                    || self.state.region(region).unit_count(UnitType::Footman) == 0
                    || !self.muster_fits(house, region, target, 1, None)
                {
                    return Err(EngineError::IllegalAction(
                        "that upgrade cannot be mustered here".into(),
                    ));
                }
                let region_state = self.state.region_mut(region);
                region_state.units[UnitType::Footman.index()] -= 1;
                region_state.units[target.index()] += 1;
                *points -= 1;
                self.state.note(format!(
                    "{house} upgrades a Footman to a {target} in {region:?}"
                ));
            }
            _ => {
                return Err(EngineError::IllegalAction(
                    "invalid Wildling mustering action".into(),
                ));
            }
        }
        Ok(*points > 0)
    }

    fn apply_preemptive_choice(&mut self, house: HouseId, option: u8) -> Result<(), EngineError> {
        if option == 0 {
            self.push_choice_tasks(self.destroy_tasks(house, 2));
            self.state
                .note(format!("{house} chooses to destroy 2 units"));
            return Ok(());
        }
        // Option B: the house falls exactly two positions on its *highest*
        // Influence track. A house can sit equally high on two or three tracks
        // (the printed starting positions already produce some ties, and Clash
        // of Kings bids can create more), and the card does not say which one;
        // the house decides.
        let mut best = usize::MAX;
        let mut highest: SmallVec<[Track; 3]> = SmallVec::new();
        for track in Track::ALL {
            let position = self.state.track_position(house, track);
            if position == 0 {
                continue; // the house is not on this track
            }
            if position < best {
                best = position;
                highest.clear();
                highest.push(track);
            } else if position == best {
                highest.push(track);
            }
        }
        if highest.is_empty() {
            return Ok(());
        }
        if highest.len() > 1 {
            self.state.choice_queue.insert(
                0,
                ChoiceTask {
                    house,
                    kind: ChoiceKind::Track {
                        purpose: TrackChoice::PreemptiveHighest,
                        tracks: highest,
                    },
                },
            );
            self.state
                .note(format!("{house} chooses which highest track to fall on"));
            return Ok(());
        }
        self.apply_preemptive_track_drop(house, highest[0]);
        Ok(())
    }

    /// Move a house down exactly two positions on one Influence track.
    ///
    /// Shared by Preemptive Raid's single-highest shortcut and its tied-track
    /// follow-up. A change of Fiefdoms leader carries the Valyrian Steel
    /// Blade's used face to the new holder, like every other track move.
    fn apply_preemptive_track_drop(&mut self, house: HouseId, track: Track) {
        let previous_blade = if track == Track::Fiefdoms {
            self.state.blade_holder()
        } else {
            None
        };
        let order = &mut self.state.tracks[track.index()];
        if let Some(position) = order.iter().position(|candidate| *candidate == house) {
            order.remove(position);
            let destination = (position + 2).min(order.len());
            order.insert(destination, house);
        }
        if track == Track::Fiefdoms {
            self.state.transfer_blade_used(previous_blade);
        }
        self.state
            .note(format!("{house} drops two positions on {}", track.as_str()));
    }

    /// Tyrion Lannister's controller decides whether to use the card.
    ///
    /// The card text makes the opponent's replacement mandatory: using it
    /// queues a `Card` task with `allow_none: false` for the target, so the
    /// target can only pick one of its own remaining cards (or, with an empty
    /// hand, fights without a House card).
    fn apply_tyrion_use(
        &mut self,
        house: HouseId,
        target: HouseId,
        option: u8,
    ) -> Result<(), EngineError> {
        if option == 0 {
            self.state
                .note(format!("{house} declines to use Tyrion Lannister"));
            return Ok(());
        }
        let cards = self
            .state
            .house(target)
            .map(|state| state.house_cards_available.clone())
            .unwrap_or_default();
        self.state.choice_queue.insert(
            0,
            ChoiceTask {
                house: target,
                kind: ChoiceKind::Card {
                    purpose: CardChoice::TyrionSwap { target },
                    cards,
                    allow_none: false,
                },
            },
        );
        self.state.note(format!(
            "{house} uses Tyrion Lannister: {target} must play another card"
        ));
        Ok(())
    }

    /// Robb Stark's controller takes over (or waives) the loser's retreat.
    fn apply_robb_retreat(&mut self, house: HouseId, option: u8) -> Result<(), EngineError> {
        let Some((loser, options)) = self.state.pending_battle.as_ref().and_then(|battle| {
            battle
                .result
                .as_ref()
                .map(|result| (result.loser, battle.retreat_options.clone()))
        }) else {
            return Ok(());
        };
        if option == 0 {
            // Declined: the defeated player chooses among every legal area.
            if let Some(battle) = self.state.pending_battle.as_mut() {
                battle.retreat_house = Some(loser);
            }
            self.state
                .note(format!("{house} declines Robb Stark's retreat choice"));
            return Ok(());
        }
        let retreaters: [u8; 4] = self
            .state
            .pending_battle
            .as_ref()
            .and_then(|battle| battle.result.as_ref())
            .map(|result| *result.context.units_of(loser))
            .unwrap_or([0; 4]);
        // The winner may only pick an area where the loser loses the fewest
        // units, i.e. one that keeps the routed army within Supply.
        let mut minimal: SmallVec<[RegionId; 8]> = options
            .iter()
            .copied()
            .filter(|option| {
                combat::retreat_keeps_supply(
                    &self.state,
                    &self.data.map,
                    *option,
                    loser,
                    &retreaters,
                )
            })
            .collect();
        if minimal.is_empty() {
            minimal = options;
        }
        if let Some(battle) = self.state.pending_battle.as_mut() {
            battle.retreat_options = minimal;
            battle.retreat_house = Some(house);
        }
        self.state
            .note(format!("{house} chooses the retreat for {loser}"));
        Ok(())
    }

    fn destroy_tasks(&self, house: HouseId, count: u8) -> Vec<ChoiceTask> {
        let areas: SmallVec<[RegionId; 8]> =
            self.state.controlled_regions(house).into_iter().collect();
        if count == 0 || areas.is_empty() {
            return Vec::new();
        }
        vec![ChoiceTask {
            house,
            kind: ChoiceKind::Unit {
                purpose: UnitChoice::Destroy,
                areas,
                remaining: count,
                allow_none: false,
            },
        }]
    }

    pub(crate) fn push_choice_tasks(&mut self, tasks: Vec<ChoiceTask>) {
        for task in tasks.into_iter().rev() {
            self.state.choice_queue.insert(0, task);
        }
    }

    pub(crate) fn battle_card(&self, house: HouseId) -> Option<CardId> {
        let battle = self.state.pending_battle.as_ref()?;
        if house == battle.attacker {
            battle.attacker_card
        } else {
            battle.defender_card
        }
    }

    pub(crate) fn set_battle_card(&mut self, house: HouseId, card: Option<CardId>) {
        let Some(battle) = self.state.pending_battle.as_mut() else {
            return;
        };
        if house == battle.attacker {
            battle.attacker_card = card;
        } else {
            battle.defender_card = card;
        }
    }

    fn remove_order_at(&mut self, region: RegionId, owner: HouseId) -> Result<(), EngineError> {
        let region_state = self.state.region_mut(region);
        if region_state.order_house != Some(owner) {
            return Err(EngineError::IllegalAction(
                "the chosen Order no longer belongs to that house".into(),
            ));
        }
        let order = region_state.order.take();
        region_state.order_house = None;
        if let (Some(order), Some(state)) = (order, self.state.house_mut(owner)) {
            state.return_order(order);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Choice context tags
// ---------------------------------------------------------------------------
//
// The tags ride in `Action::order`; see `crate::action::choice_context`. They
// exist only so the encoder can tell otherwise identical generic choices apart.
// `apply_choice` never reads them.

/// Which [`choice_context`] a unit task belongs to.
fn unit_choice_context(purpose: UnitChoice, areas: &[RegionId], map: &GameMap) -> u8 {
    match purpose {
        UnitChoice::ReplaceKnight => choice_context::CROW_KNIGHT,
        UnitChoice::UpgradeFootman => choice_context::CROW_UPGRADE,
        // The Horde's unit task is restricted to the single fort its region
        // task picked; a generic destroy spans every controlled area. A house
        // down to one fort can look like both, so the fortification check keeps
        // the common case exact.
        UnitChoice::Destroy => match areas {
            [only] if map.region(*only).is_fortification() => choice_context::HORDE_FORT,
            _ => choice_context::GENERIC_DESTROY,
        },
    }
}

/// The house whose deck a card task's cards come from, when it is not the
/// choosing house.
fn card_choice_owner(purpose: &CardChoice) -> Option<HouseId> {
    match purpose {
        CardChoice::PatchfaceDiscard { target } => Some(*target),
        CardChoice::TyrionSwap { target } => Some(*target),
        CardChoice::AeronSwap | CardChoice::MassingDiscard | CardChoice::MammothRetrieve => None,
    }
}

fn card_choice_context(purpose: &CardChoice) -> u8 {
    match purpose {
        CardChoice::TyrionSwap { .. } => choice_context::TYRION_CARD,
        CardChoice::AeronSwap => choice_context::AERON_CARD,
        CardChoice::PatchfaceDiscard { .. }
        | CardChoice::MassingDiscard
        | CardChoice::MammothRetrieve => choice_context::CARD_DISCARD_OR_RETRIEVE,
    }
}

fn region_choice_context(purpose: &RegionChoice) -> u8 {
    match purpose {
        RegionChoice::QueenOrder { .. } => choice_context::QUEEN_ORDER,
        RegionChoice::CerseiOrder { .. } => choice_context::CERSEI_ORDER,
        RegionChoice::RenlyFootman => choice_context::RENLY_FOOTMAN,
        RegionChoice::HordeFort { .. } => choice_context::HORDE_FORT,
        RegionChoice::HordeMusterFort => choice_context::HORDE_MUSTER,
    }
}

fn track_choice_context(purpose: TrackChoice) -> u8 {
    match purpose {
        TrackChoice::KingBeyondHighest => choice_context::KING_BEYOND_HIGHEST,
        TrackChoice::KingBeyondOther => choice_context::KING_BEYOND_OTHER,
        TrackChoice::PreemptiveHighest => choice_context::PREEMPTIVE,
    }
}

fn option_choice_context(purpose: OptionChoice) -> Option<u8> {
    match purpose {
        OptionChoice::PreemptiveRaid => Some(choice_context::PREEMPTIVE),
        OptionChoice::TyrionUse { .. } => Some(choice_context::TYRION_USE),
        // Robb Stark's retreat offer is always within the retreat stage, which
        // the state already encodes; the 0..14 tag block has no slot for it.
        OptionChoice::RobbRetreat => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionKind;
    use crate::combat::BattleRequest;
    use crate::enums::{OrderKey, Phase};
    use crate::map::DataSet;
    use crate::setup::GameConfig;
    use crate::state::{BattleStage, PendingBattle};
    use std::sync::Arc;

    fn game(seed: u64) -> Game {
        let data = Arc::new(DataSet::load_default().expect("bundled data must load"));
        Game::new(data, GameConfig::with_seed(seed)).expect("game must start")
    }

    fn region_of(data: &DataSet, id: &str) -> RegionId {
        data.map
            .lookup(id)
            .unwrap_or_else(|| panic!("unknown region {id}"))
    }

    fn house_of(id: &str) -> HouseId {
        HouseId::from_name(id).unwrap_or_else(|| panic!("unknown house {id}"))
    }

    fn card(data: &DataSet, house: HouseId, id: &str) -> CardId {
        data.house_cards
            .card_index(house, id)
            .unwrap_or_else(|| panic!("{house} has no card {id}"))
    }

    fn adjacent_land_pair(data: &DataSet) -> (RegionId, RegionId) {
        for index in 0..data.map.region_count() as u16 {
            let region = RegionId(index);
            if !data.map.region(region).is_land() {
                continue;
            }
            if let Some(neighbour) = data.map.land_neighbours_of(region).first().copied() {
                return (region, neighbour);
            }
        }
        panic!("no adjacent land regions");
    }

    /// A battle paused at the House-card stage, both cards already revealed.
    fn card_stage_battle(
        attacker: HouseId,
        defender: HouseId,
        region: RegionId,
        origin: Option<RegionId>,
        attacker_card: CardId,
        defender_card: CardId,
    ) -> PendingBattle {
        PendingBattle {
            region,
            attacker,
            defender: Some(defender),
            neutral: false,
            origin,
            attacker_order: None,
            attacker_units: [4, 0, 0, 0],
            defender_units: [1, 0, 0, 0],
            attacker_order_strength: 0,
            defender_order_strength: 0,
            attacker_card: Some(attacker_card),
            defender_card: Some(defender_card),
            attacker_card_chosen: true,
            defender_card_chosen: true,
            pre_choices_done: false,
            doran_owner: None,
            doran_track: None,
            prepared: None,
            blade_done: true,
            attacker_blade: false,
            defender_blade: false,
            stage: BattleStage::Cards,
            support_queue: SmallVec::new(),
            support_index: 0,
            support_choices: SmallVec::new(),
            result: None,
            casualties_remaining: 0,
            retreat_options: SmallVec::new(),
            retreat_house: None,
            retreat_destination: None,
            defender_order: None,
            defender_order_house: None,
        }
    }

    /// A game paused on Tyrion Lannister's pre-battle choice.
    fn tyrion_game() -> (Game, Arc<DataSet>, HouseId, HouseId, CardId, CardId) {
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let (origin, target) = adjacent_land_pair(&data);
        let lannister = house_of("lannister");
        let stark = house_of("stark");
        let tyrion = card(&data, lannister, "tyrion_lannister");
        let eddard = card(&data, stark, "eddard_stark");
        let greatjon = card(&data, stark, "greatjon_umber");
        let robb = card(&data, stark, "robb_stark");
        {
            let state = game.state.house_mut(stark).expect("stark plays");
            state.house_cards_available = SmallVec::from_slice(&[greatjon, robb]);
            state.house_cards_used = SmallVec::from_slice(&[eddard]);
            // High Supply keeps a lost battle from interrupting the test with
            // a Supply-disband queue.
            state.supply_level = 5;
        }
        if let Some(state) = game.state.house_mut(lannister) {
            state.supply_level = 5;
        }
        game.state.choice_choices = true;
        game.choice_choices = true;
        game.state.phase = Phase::Combat;
        game.state.pending_battle = Some(card_stage_battle(
            lannister,
            stark,
            target,
            Some(origin),
            tyrion,
            eddard,
        ));
        game.advance().expect("the pre-battle choice appears");
        (game, data, lannister, stark, eddard, greatjon)
    }

    #[test]
    fn ready_and_routed_unit_choices_are_separate_actions() {
        let mut game = game(1);
        game.state.choice_choices = true;
        game.choice_choices = true;
        let house = game.state.first_player().expect("a first player");
        let data = Arc::clone(&game.data);
        let winterfell = region_of(&data, "winterfell");
        let white_harbor = region_of(&data, "white_harbor");
        for region in [winterfell, white_harbor] {
            let state = game.state.region_mut(region);
            state.controller = Some(house);
            state.units = [0; 4];
            state.routed = [0; 4];
        }
        {
            let state = game.state.region_mut(winterfell);
            state.units[UnitType::Footman.index()] = 1;
            state.routed[UnitType::Footman.index()] = 1;
        }
        game.state.choice_queue.push(ChoiceTask {
            house,
            kind: ChoiceKind::Unit {
                purpose: UnitChoice::Destroy,
                // Two areas keep this a generic destroy rather than the Horde's
                // single-fort task.
                areas: SmallVec::from_slice(&[winterfell, white_harbor]),
                remaining: 2,
                allow_none: false,
            },
        });
        game.advance().expect("the destroy choice appears");

        let actions = game.legal_actions();
        let ready = actions
            .iter()
            .copied()
            .find(|action| {
                action.region == Some(winterfell)
                    && action.unit == Some(UnitType::Footman)
                    && action.amount == Some(0)
            })
            .expect("the ready Footman is offered");
        let routed = actions
            .iter()
            .copied()
            .find(|action| {
                action.region == Some(winterfell)
                    && action.unit == Some(UnitType::Footman)
                    && action.amount == Some(1)
            })
            .expect("the routed Footman is a separate action");
        assert_ne!(ready, routed);
        assert_eq!(ready.order, Some(OrderKey(choice_context::GENERIC_DESTROY)));

        game.step(ready).expect("the ready Footman is destroyed");
        assert_eq!(
            game.state.region(winterfell).unit_count(UnitType::Footman),
            0
        );
        assert_eq!(
            game.state
                .region(winterfell)
                .routed_count(UnitType::Footman),
            1,
            "the routed copy must survive a ready-unit choice"
        );

        // The second casualty can still take the routed copy.
        let second = game
            .legal_actions()
            .into_iter()
            .find(|action| {
                action.region == Some(winterfell)
                    && action.unit == Some(UnitType::Footman)
                    && action.amount == Some(1)
            })
            .expect("the routed Footman is still offered");
        game.step(second).expect("the routed Footman is destroyed");
        assert_eq!(
            game.state
                .region(winterfell)
                .routed_count(UnitType::Footman),
            0
        );
    }

    #[test]
    fn a_legacy_unit_choice_without_amount_still_takes_the_ready_unit() {
        let mut game = game(1);
        let house = game.state.first_player().expect("a first player");
        let data = Arc::clone(&game.data);
        let winterfell = region_of(&data, "winterfell");
        {
            let state = game.state.region_mut(winterfell);
            state.controller = Some(house);
            state.units = [0; 4];
            state.routed = [0; 4];
            state.units[UnitType::Knight.index()] = 1;
            state.routed[UnitType::Knight.index()] = 1;
        }
        game.state.choice_queue.push(ChoiceTask {
            house,
            kind: ChoiceKind::Unit {
                purpose: UnitChoice::Destroy,
                areas: SmallVec::from_slice(&[winterfell]),
                remaining: 1,
                allow_none: false,
            },
        });
        game.advance().expect("the choice appears");
        // `Action::choice_unit` keeps `amount: None`, the historical encoding.
        let legacy = Action::choice_unit(house, winterfell, UnitType::Knight);
        game.step(legacy).expect("the legacy action applies");
        assert_eq!(
            game.state.region(winterfell).unit_count(UnitType::Knight),
            0
        );
        assert_eq!(
            game.state.region(winterfell).routed_count(UnitType::Knight),
            1
        );
    }

    #[test]
    fn patchface_card_actions_name_the_targets_deck() {
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let patchface_house = house_of("baratheon");
        let target = house_of("stark");
        let greatjon = card(&data, target, "greatjon_umber");
        game.state.choice_queue.push(ChoiceTask {
            house: patchface_house,
            kind: ChoiceKind::Card {
                purpose: CardChoice::PatchfaceDiscard { target },
                cards: SmallVec::from_slice(&[greatjon]),
                allow_none: true,
            },
        });

        let actions = game.legal_actions();
        let discard = actions
            .iter()
            .copied()
            .find(|action| action.card == Some(greatjon))
            .expect("the target's card is offered");
        assert_eq!(discard.kind, ActionKind::PlayHouseCard);
        assert_eq!(
            discard.house,
            Some(patchface_house),
            "the decision belongs to the Patchface player"
        );
        assert_eq!(
            discard.card_house,
            Some(target),
            "the card id must be looked up in the target's deck"
        );
        assert_eq!(
            discard.order,
            Some(OrderKey(choice_context::CARD_DISCARD_OR_RETRIEVE))
        );
        // Declining is still possible and names no deck.
        let pass = actions
            .iter()
            .copied()
            .find(|action| action.kind == ActionKind::Pass)
            .expect("Patchface may decline");
        assert_eq!(pass.card_house, None);
    }

    #[test]
    fn tyrion_decline_leaves_the_battle_alone() {
        let (mut game, _data, lannister, stark, eddard, greatjon) = tyrion_game();
        let task = game
            .state
            .choice_queue
            .first()
            .cloned()
            .expect("the controller is asked first");
        assert_eq!(task.house, lannister);
        match task.kind {
            ChoiceKind::Option {
                purpose: OptionChoice::TyrionUse { target },
                ..
            } => assert_eq!(target, stark),
            other => panic!("unexpected Tyrion task: {other:?}"),
        }
        let actions = game.legal_actions();
        assert!(
            actions
                .iter()
                .all(|action| action.kind != ActionKind::PlayHouseCard),
            "the target's cards must not be offered before the card is used"
        );
        let decline = actions
            .iter()
            .copied()
            .find(|action| action.amount == Some(0))
            .expect("declining is offered");

        game.step(decline).expect("declining is legal");
        let stark_state = game.state.house(stark).expect("stark plays");
        assert!(
            stark_state.house_cards_used.contains(&eddard),
            "the revealed card stays in the discard pile"
        );
        assert!(
            stark_state.house_cards_available.contains(&greatjon),
            "no replacement card is forced"
        );
        assert!(game.state.choice_queue.is_empty());
    }

    #[test]
    fn tyrion_use_forces_the_targets_card() {
        let (mut game, _data, _lannister, stark, eddard, greatjon) = tyrion_game();
        let use_card = game
            .legal_actions()
            .into_iter()
            .find(|action| action.amount == Some(1))
            .expect("using Tyrion is offered");
        assert_eq!(use_card.order, Some(OrderKey(choice_context::TYRION_USE)));
        game.step(use_card).expect("using Tyrion is legal");

        let task = game
            .state
            .choice_queue
            .first()
            .cloned()
            .expect("the target's card task is queued");
        assert_eq!(task.house, stark);
        match task.kind {
            ChoiceKind::Card {
                purpose: CardChoice::TyrionSwap { target },
                cards,
                allow_none,
            } => {
                assert_eq!(target, stark);
                assert!(!allow_none, "the target cannot decline the swap");
                assert!(cards.contains(&greatjon));
            }
            other => panic!("unexpected follow-up task: {other:?}"),
        }
        let actions = game.legal_actions();
        assert!(
            actions
                .iter()
                .all(|action| action.kind == ActionKind::PlayHouseCard),
            "the target must choose a card, not decline"
        );
        let replacement = actions
            .iter()
            .copied()
            .find(|action| action.card == Some(greatjon))
            .expect("the replacement card is offered");
        assert_eq!(
            replacement.card_house, None,
            "Tyrion's target chooses its own card"
        );
        assert_eq!(
            replacement.order,
            Some(OrderKey(choice_context::TYRION_CARD))
        );

        game.step(replacement).expect("the replacement applies");
        let stark_state = game.state.house(stark).expect("stark plays");
        assert!(stark_state.house_cards_used.contains(&greatjon));
        assert!(
            stark_state.house_cards_available.contains(&eddard),
            "the replaced card returns to hand"
        );
    }

    #[test]
    fn tyrion_against_an_empty_hand_forces_no_house_card() {
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let (origin, target) = adjacent_land_pair(&data);
        let lannister = house_of("lannister");
        let stark = house_of("stark");
        let tyrion = card(&data, lannister, "tyrion_lannister");
        let eddard = card(&data, stark, "eddard_stark");
        {
            let state = game.state.house_mut(stark).expect("stark plays");
            // No alternative card: using Tyrion can only strip the card away.
            state.house_cards_available.clear();
            state.house_cards_used = SmallVec::from_slice(&[eddard]);
            state.supply_level = 5;
        }
        if let Some(state) = game.state.house_mut(lannister) {
            state.supply_level = 5;
        }
        game.state.choice_choices = true;
        game.choice_choices = true;
        game.state.phase = Phase::Combat;
        game.state.pending_battle = Some(card_stage_battle(
            lannister,
            stark,
            target,
            Some(origin),
            tyrion,
            eddard,
        ));
        game.advance().expect("the pre-battle choice appears");
        let use_card = game
            .legal_actions()
            .into_iter()
            .find(|action| action.amount == Some(1))
            .expect("using Tyrion is offered");
        game.step(use_card).expect("using Tyrion is legal");

        // The task is mandatory, but an empty hand leaves only the forced
        // "no House card" pass, which is not a decline.
        let actions = game.legal_actions();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, ActionKind::Pass);
        game.step(actions[0])
            .expect("the forced no-card result applies");
        let stark_state = game.state.house(stark).expect("stark plays");
        assert!(
            stark_state.house_cards_available.contains(&eddard),
            "the replaced card returns to hand"
        );
        assert!(stark_state.house_cards_used.is_empty());
    }

    #[test]
    fn a_tied_highest_track_is_a_real_choice() {
        let mut game = game(1);
        game.state.choice_choices = true;
        game.choice_choices = true;
        let houses = game.state.present_houses();
        let house = houses[0];
        // Identical tracks put the house at the same (highest) position on all
        // three, which is exactly the tie the card has to break.
        let mut order: SmallVec<[HouseId; 6]> = SmallVec::new();
        order.push(house);
        for candidate in houses.iter().copied() {
            if candidate != house {
                order.push(candidate);
            }
        }
        for track in Track::ALL {
            game.state.tracks[track.index()] = order.clone();
        }
        game.state.choice_queue.push(ChoiceTask {
            house,
            kind: ChoiceKind::Option {
                purpose: OptionChoice::PreemptiveRaid,
                options: SmallVec::from_slice(&[0, 1]),
            },
        });
        game.advance().expect("the A/B option appears");
        let option_b = game
            .legal_actions()
            .into_iter()
            .find(|action| action.amount == Some(1))
            .expect("option B is offered");
        game.step(option_b).expect("option B applies");

        let task = game
            .state
            .choice_queue
            .first()
            .cloned()
            .expect("the initial tracks tie, so the house must choose");
        let tracks = match task.kind {
            ChoiceKind::Track {
                purpose: TrackChoice::PreemptiveHighest,
                tracks,
            } => tracks,
            other => panic!("unexpected Preemptive task: {other:?}"),
        };
        assert!(tracks.len() > 1, "a tie must offer more than one track");

        let index = tracks
            .iter()
            .position(|track| *track == Track::KingsCourt)
            .expect("King's Court is one of the tied tracks");
        let choice = game
            .legal_actions()
            .into_iter()
            .find(|action| action.amount == Some(index as u8))
            .expect("the tied track is offered");
        assert_eq!(choice.order, Some(OrderKey(choice_context::PREEMPTIVE)));
        let before = game.state.track_position(house, Track::KingsCourt);
        game.step(choice).expect("the track choice applies");
        let after = game.state.track_position(house, Track::KingsCourt);
        assert_eq!(
            after,
            (before + 2).min(game.state.track_order(Track::KingsCourt).len()),
            "the house falls exactly two positions on the chosen track"
        );
    }

    #[test]
    fn a_single_highest_track_is_applied_without_another_question() {
        let mut game = game(1);
        game.state.choice_choices = true;
        game.choice_choices = true;
        let houses = game.state.present_houses();
        assert!(houses.len() >= 3, "the base setup has six houses");
        let house = houses[0];
        let other = houses[1];
        let third = houses[2];
        // A unique highest position on the Iron Throne, deeper elsewhere.
        let mut throne: SmallVec<[HouseId; 6]> = SmallVec::new();
        throne.push(house);
        for candidate in houses.iter().copied() {
            if candidate != house {
                throne.push(candidate);
            }
        }
        let mut fiefdoms: SmallVec<[HouseId; 6]> = SmallVec::new();
        fiefdoms.push(other);
        fiefdoms.push(house);
        for candidate in houses.iter().copied() {
            if candidate != other && candidate != house {
                fiefdoms.push(candidate);
            }
        }
        let mut court: SmallVec<[HouseId; 6]> = SmallVec::new();
        court.push(other);
        court.push(third);
        court.push(house);
        for candidate in houses.iter().copied() {
            if candidate != other && candidate != third && candidate != house {
                court.push(candidate);
            }
        }
        game.state.tracks[Track::IronThrone.index()] = throne;
        game.state.tracks[Track::Fiefdoms.index()] = fiefdoms;
        game.state.tracks[Track::KingsCourt.index()] = court;

        game.state.choice_queue.push(ChoiceTask {
            house,
            kind: ChoiceKind::Option {
                purpose: OptionChoice::PreemptiveRaid,
                options: SmallVec::from_slice(&[0, 1]),
            },
        });
        game.advance().expect("the A/B option appears");
        let option_b = game
            .legal_actions()
            .into_iter()
            .find(|action| action.amount == Some(1))
            .expect("option B is offered");
        let before = game.state.track_position(house, Track::IronThrone);
        game.step(option_b).expect("option B applies");

        assert!(
            game.state.choice_queue.is_empty(),
            "a unique highest track needs no follow-up choice"
        );
        assert_eq!(
            game.state.track_position(house, Track::IronThrone),
            (before + 2).min(game.state.track_order(Track::IronThrone).len())
        );
        assert_eq!(
            game.state.track_position(house, Track::Fiefdoms),
            2,
            "the other tracks are untouched"
        );
    }

    #[test]
    fn a_preemptive_drop_on_fiefdoms_carries_the_used_blade() {
        let mut game = game(1);
        let houses = game.state.present_houses();
        let house = houses[0];
        let other = houses[1];
        let mut fiefdoms: SmallVec<[HouseId; 6]> = SmallVec::new();
        fiefdoms.push(house);
        fiefdoms.push(other);
        for candidate in houses.iter().copied() {
            if candidate != house && candidate != other {
                fiefdoms.push(candidate);
            }
        }
        game.state.tracks[Track::Fiefdoms.index()] = fiefdoms;
        game.state.house_mut(house).expect("house plays").blade_used = true;

        game.apply_preemptive_track_drop(house, Track::Fiefdoms);

        assert_eq!(game.state.blade_holder(), Some(other));
        assert!(
            game.state.house(other).expect("house plays").blade_used,
            "the new Fiefdoms leader inherits the used blade"
        );
        assert!(
            !game.state.house(house).expect("house plays").blade_used,
            "the old holder's flag is cleared"
        );
    }

    #[test]
    fn the_horde_ship_takes_control_of_an_empty_port() {
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let (port, host) = *data.map.port_hosts.first().expect("the board has a port");
        let house = game.state.first_player().expect("a first player");
        {
            let host_state = game.state.region_mut(host);
            host_state.controller = Some(house);
            host_state.units = [0; 4];
            host_state.routed = [0; 4];
        }
        {
            let port_state = game.state.region_mut(port);
            port_state.controller = None;
            port_state.units = [0; 4];
            port_state.routed = [0; 4];
        }
        game.state
            .house_mut(house)
            .expect("house plays")
            .supply_level = 5;

        let mut points = 1u8;
        let action = Action::muster(house, host, UnitType::Ship, Some(port));
        game.apply_wildling_muster(house, action, host, &mut points)
            .expect("the reward may muster a Ship");
        assert_eq!(game.state.region(port).unit_count(UnitType::Ship), 1);
        assert_eq!(
            game.state.region(port).controller,
            Some(house),
            "an empty sea or port never keeps a Ship without control"
        );
    }

    /// A land area with an attacker origin and a second retreat neighbour.
    fn target_with_two_neighbours(data: &DataSet) -> (RegionId, RegionId, RegionId) {
        for index in 0..data.map.region_count() as u16 {
            let target = RegionId(index);
            if !data.map.region(target).is_land() {
                continue;
            }
            let neighbours = data.map.land_neighbours_of(target);
            if neighbours.len() >= 2 {
                return (neighbours[0], target, neighbours[1]);
            }
        }
        panic!("no land area with two land neighbours");
    }

    /// A finished battle where Robb Stark won and the defender must retreat.
    fn robb_retreat_game() -> (Game, Arc<DataSet>, HouseId, HouseId, RegionId) {
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let regions: Vec<RegionId> = game.state.region_ids().collect();
        for region in regions {
            let state = game.state.region_mut(region);
            state.units = [0; 4];
            state.routed = [0; 4];
            state.controller = None;
            state.order = None;
            state.order_house = None;
            state.garrison = 0;
            state.neutral_force = 0;
        }
        let (origin, target, retreat) = target_with_two_neighbours(&data);
        let stark = house_of("stark");
        let baratheon = house_of("baratheon");
        let robb = card(&data, stark, "robb_stark");
        {
            let state = game.state.region_mut(origin);
            state.controller = Some(stark);
            state.units[UnitType::Footman.index()] = 6;
        }
        {
            let state = game.state.region_mut(target);
            state.controller = Some(baratheon);
            state.units[UnitType::Footman.index()] = 1;
        }
        for house in [stark, baratheon] {
            let state = game.state.house_mut(house).expect("house plays");
            state.supply_level = 5;
            state.house_cards_available.clear();
            state.house_cards_used.clear();
            state.blade_used = false;
        }

        let request = BattleRequest {
            region: target,
            attacker: stark,
            defender: baratheon,
            attacker_units: [6, 0, 0, 0],
            defender_units: [1, 0, 0, 0],
            attacker_card: Some(robb),
            defender_card: None,
            attacker_order_strength: 0,
            defender_order_strength: 0,
            attacker_blade: false,
            defender_blade: false,
            attacker_order: None,
            attacker_origin: Some(origin),
            support_choices: Some(SmallVec::new()),
        };
        let result = combat::resolve_battle(&game.state, &data.map, &data.house_cards, &request);
        assert_eq!(result.winner, stark, "Robb Stark must win the test battle");
        assert_eq!(result.loser, baratheon);

        game.state.choice_choices = true;
        game.choice_choices = true;
        game.state.phase = Phase::Combat;
        game.state.pending_battle = Some(PendingBattle {
            region: target,
            attacker: stark,
            defender: Some(baratheon),
            neutral: false,
            origin: Some(origin),
            attacker_order: None,
            attacker_units: [6, 0, 0, 0],
            defender_units: [1, 0, 0, 0],
            attacker_order_strength: 0,
            defender_order_strength: 0,
            attacker_card: Some(robb),
            defender_card: None,
            attacker_card_chosen: true,
            defender_card_chosen: true,
            pre_choices_done: true,
            doran_owner: None,
            doran_track: None,
            prepared: Some(result.context.clone()),
            blade_done: true,
            attacker_blade: false,
            defender_blade: false,
            stage: BattleStage::Finalize,
            support_queue: SmallVec::new(),
            support_index: 0,
            support_choices: SmallVec::new(),
            result: Some(result),
            casualties_remaining: 0,
            retreat_options: SmallVec::new(),
            retreat_house: None,
            retreat_destination: None,
            defender_order: None,
            defender_order_house: None,
        });
        game.prepare_retreat();
        assert_eq!(
            game.state
                .pending_battle
                .as_ref()
                .map(|battle| battle.stage),
            Some(BattleStage::Retreat),
            "the loser has a legal retreat"
        );
        (game, data, stark, baratheon, retreat)
    }

    #[test]
    fn robb_stark_can_decline_the_retreat_choice() {
        let (mut game, _data, stark, baratheon, retreat) = robb_retreat_game();
        let task = game
            .state
            .choice_queue
            .first()
            .cloned()
            .expect("Robb's controller is asked");
        assert_eq!(task.house, stark);
        match task.kind {
            ChoiceKind::Option {
                purpose: OptionChoice::RobbRetreat,
                ..
            } => {}
            other => panic!("unexpected Robb task: {other:?}"),
        }
        assert_eq!(
            game.state
                .pending_battle
                .as_ref()
                .and_then(|battle| battle.retreat_house),
            None,
            "nobody owns the retreat until the card is used"
        );

        game.advance().expect("the option appears");
        let decline = game
            .legal_actions()
            .into_iter()
            .find(|action| action.amount == Some(0))
            .expect("declining is offered");
        game.step(decline).expect("declining is legal");
        assert_eq!(
            game.state
                .pending_battle
                .as_ref()
                .and_then(|battle| battle.retreat_house),
            Some(baratheon),
            "declining hands the retreat back to the loser"
        );
        let actions = game.legal_actions();
        assert!(!actions.is_empty());
        assert!(actions
            .iter()
            .all(|action| action.kind == ActionKind::Retreat));
        assert!(
            actions.iter().any(|action| action.target == Some(retreat)),
            "the empty neighbour is a legal retreat"
        );
        assert_eq!(game.current_player(), Some(baratheon));
    }

    #[test]
    fn robb_stark_can_take_over_the_retreat() {
        let (mut game, _data, stark, baratheon, retreat) = robb_retreat_game();
        game.advance().expect("the option appears");
        let use_card = game
            .legal_actions()
            .into_iter()
            .find(|action| action.amount == Some(1))
            .expect("using Robb is offered");
        game.step(use_card).expect("using Robb is legal");
        assert_eq!(
            game.state
                .pending_battle
                .as_ref()
                .and_then(|battle| battle.retreat_house),
            Some(stark),
            "the winner chooses the loser's retreat"
        );
        let actions = game.legal_actions();
        assert!(!actions.is_empty());
        assert!(actions
            .iter()
            .all(|action| action.kind == ActionKind::Retreat));
        assert!(actions.iter().all(|action| action.house == Some(stark)));
        assert!(
            actions.iter().any(|action| action.target == Some(retreat)),
            "the minimal-loss filter cannot drop the empty neighbour"
        );
        assert_eq!(game.current_player(), Some(stark));
        // The loser is still the one who retreats.
        assert_eq!(
            game.state
                .pending_battle
                .as_ref()
                .and_then(|battle| battle.result.as_ref())
                .map(|result| result.loser),
            Some(baratheon)
        );
    }
}
