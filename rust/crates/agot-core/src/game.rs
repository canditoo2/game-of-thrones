//! The game object: legal actions, phase progression and victory.
//!
//! Port of `src/agot/engine/game.py`, built up phase by phase. This first slice
//! covers the opening position and the whole Planning Phase; `advance` refuses
//! any phase that has not been ported yet rather than guessing, so a caller can
//! never silently play by different rules.

use std::fmt;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::action::{Action, ActionKind};
use crate::enums::{ActionStep, HouseId, OrderKey, OrderType, Phase, RegionId, Track, N_HOUSES};
use crate::map::DataSet;
use crate::rng::Rng;
use crate::setup::{initial_state, GameConfig};
use crate::state::State;

/// Everything that can go wrong driving a game.
#[derive(Debug)]
pub enum EngineError {
    /// The action does not fit the position it was applied to.
    IllegalAction(String),
    /// The engine reached a phase whose rules have not been ported yet.
    UnportedPhase(Phase),
    /// The phase machine ran the guard out.
    PhaseDidNotSettle,
    /// The action kind has not been ported yet.
    UnportedAction(ActionKind),
    /// A house named by the position is missing from it.
    MissingHouse(HouseId),
    /// `step` was called on a finished game.
    GameOver,
    /// The configuration is not usable.
    BadConfig(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::IllegalAction(message) => write!(f, "illegal action: {message}"),
            EngineError::UnportedPhase(phase) => {
                write!(f, "phase {phase} has not been ported yet")
            }
            EngineError::PhaseDidNotSettle => {
                write!(f, "phase progression did not settle")
            }
            EngineError::UnportedAction(kind) => {
                write!(f, "action kind {} has not been ported yet", kind.as_str())
            }
            EngineError::MissingHouse(house) => write!(f, "house {house} is not in the game"),
            EngineError::GameOver => write!(f, "game is already over"),
            EngineError::BadConfig(message) => write!(f, "bad config: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// A single game of A Game of Thrones: The Board Game (2nd Edition).
pub struct Game {
    /// Static tables. Shared by `Arc` so a cloned game stays cheap.
    pub data: Arc<DataSet>,
    pub state: State,
    pub config: GameConfig,
    /// Desktop-play rule option: queue the one battle of a March order until
    /// all of its destinations have been chosen. Self-play keeps the original
    /// immediate-battle behaviour so the frozen reference fixtures remain valid.
    pub deferred_battles: bool,
    pub rng: Rng,
}

impl Clone for Game {
    /// Structural copy, the analogue of Python's `Game.clone`.
    ///
    /// One deliberate difference: the clone's generator is derived from a
    /// *copy* of the parent's stream, so cloning does not advance the source.
    /// Python seeds the child with `self.rng.random()` and therefore does move
    /// the parent's stream; the differential harness is action-scripted, so no
    /// stream equality is required, and a clone that leaves its source alone is
    /// the safer invariant for search.
    fn clone(&self) -> Self {
        let mut source = self.rng.clone();
        Game {
            data: Arc::clone(&self.data),
            state: self.state.clone(),
            config: self.config.clone(),
            deferred_battles: self.deferred_battles,
            rng: source.fork(),
        }
    }
}

impl fmt::Debug for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Game")
            .field("round", &self.state.round)
            .field("phase", &self.state.phase)
            .field("active_house", &self.state.active_house)
            .finish_non_exhaustive()
    }
}

impl Game {
    // -- construction ----------------------------------------------------

    pub fn new(data: Arc<DataSet>, config: GameConfig) -> Result<Game, EngineError> {
        config.validate(&data).map_err(EngineError::BadConfig)?;
        let mut rng = Rng::from_seed(config.seed);
        let state = initial_state(&data, &config, &mut rng);
        let mut game = Game {
            data,
            state,
            config,
            deferred_battles: false,
            rng,
        };
        game.advance()?;
        Ok(game)
    }

    // -- basic queries ---------------------------------------------------

    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.state.phase == Phase::GameOver
    }

    /// The house that must act, or `None` when the engine auto-advances.
    #[inline]
    pub fn current_player(&self) -> Option<HouseId> {
        self.state.active_house
    }

    #[inline]
    pub fn winner(&self) -> Option<HouseId> {
        self.state.winner
    }

    // -- legal actions ---------------------------------------------------

    /// All actions the current player may take.
    pub fn legal_actions(&self) -> Vec<Action> {
        let Some(house) = self.state.active_house else {
            return Vec::new();
        };
        match self.state.phase {
            Phase::Planning => self.planning_actions(house),
            Phase::Raven => self.raven_actions(house),
            Phase::Action => self.action_actions(house),
            Phase::Combat => self.combat_actions(house),
            Phase::Mustering => self.mustering_actions(house),
            Phase::Bidding => self.bidding_actions(house),
            Phase::WesterosChoice => self.westeros_choice_actions(house),
            Phase::WildlingAttack => self.wildling_bid_actions(house),
            _ => Vec::new(),
        }
    }

    // -- stepping --------------------------------------------------------

    /// Applies one action, then lets the engine auto-advance.
    pub fn step(&mut self, action: Action) -> Result<(), EngineError> {
        if self.is_terminal() {
            return Err(EngineError::GameOver);
        }
        match action.kind {
            ActionKind::PlaceOrder => self.apply_place_order(action)?,
            ActionKind::RavenSwap => self.apply_raven_swap(action)?,
            ActionKind::RavenDeckTop => self.apply_raven_deck(action, false)?,
            ActionKind::RavenDeckBottom => self.apply_raven_deck(action, true)?,
            ActionKind::ResolveRaid => self.apply_raid(action)?,
            ActionKind::March => self.apply_march(action)?,
            ActionKind::ResolveConsolidate => self.apply_consolidate(action)?,
            ActionKind::ConsolidateMuster => self.apply_consolidate_muster(action)?,
            ActionKind::DeclareSupport => self.apply_declare_support(action)?,
            ActionKind::PlayHouseCard => self.apply_house_card(action)?,
            ActionKind::UseBlade => self.apply_use_blade(action)?,
            ActionKind::ChooseCasualty => self.apply_choose_casualty(action)?,
            ActionKind::Retreat => self.apply_retreat_action(action)?,
            ActionKind::Muster => self.apply_muster(action)?,
            ActionKind::Upgrade => self.apply_upgrade(action)?,
            ActionKind::Bid => self.apply_bid(action)?,
            ActionKind::WesterosChoice => self.apply_westeros_choice(action)?,
            ActionKind::Pass => self.apply_pass(action)?,
            // `Disband` is never generated by the engine: Supply enforcement and
            // casualties are resolved internally, as in the Python original.
            other => return Err(EngineError::UnportedAction(other)),
        }
        self.advance()
    }

    // ==================================================================
    // Phase progression
    // ==================================================================

    /// Resolves every phase that needs no player input.
    ///
    /// `pub(crate)` so the rule-fix regressions can mirror Python's `_advance`
    /// when a test sets a phase up directly instead of playing into it.
    pub(crate) fn advance(&mut self) -> Result<(), EngineError> {
        for _ in 0..500 {
            if self.state.phase == Phase::GameOver {
                self.state.active_house = None;
                return Ok(());
            }

            self.check_victory();
            if self.state.phase == Phase::GameOver {
                return Ok(());
            }

            match self.state.phase {
                Phase::Westeros => self.run_westeros_phase()?,
                Phase::WesterosChoice => match self.state.westeros_choice.as_ref() {
                    Some(choice) => {
                        let holder = choice.holder;
                        self.state.active_house = Some(holder);
                        return Ok(());
                    }
                    None => self.state.phase = Phase::Westeros,
                },
                Phase::Bidding => match self.next_bidder() {
                    Some(house) => {
                        self.state.active_house = Some(house);
                        return Ok(());
                    }
                    None => self.resolve_bids(),
                },
                Phase::WildlingAttack => match self.next_wildling_bidder() {
                    Some(house) => {
                        self.state.active_house = Some(house);
                        return Ok(());
                    }
                    None => self.resolve_wildling_attack()?,
                },
                Phase::Mustering => match self.next_musterer() {
                    Some(house) => {
                        self.state.active_house = Some(house);
                        return Ok(());
                    }
                    None => {
                        if self.state.return_phase == Some(Phase::Action) {
                            // A Special Consolidate Power Order mustered in its
                            // own area; resume the Action Phase where it stopped.
                            self.state.return_phase = None;
                            self.state.phase = Phase::Action;
                        } else {
                            self.state.phase = Phase::Planning;
                        }
                    }
                },
                Phase::Planning => {
                    match self.next_planner() {
                        Some(house) => {
                            self.state.active_house = Some(house);
                            return Ok(());
                        }
                        // Assigning orders is done; the Messenger Raven holder
                        // acts before orders are revealed.
                        None => self.state.phase = Phase::Raven,
                    }
                }
                Phase::Raven => match self.next_raven_actor() {
                    Some(house) => {
                        self.state.active_house = Some(house);
                        return Ok(());
                    }
                    None => {
                        self.state.phase = Phase::Action;
                        self.state.action_step = ActionStep::Raid;
                        self.state.action_house_last = None;
                    }
                },
                Phase::Action => {
                    if let Some(pending) = self.state.pending_march {
                        // Mid-march: keep the same house active until it has no
                        // further moves; only then does play pass on.
                        if !self.march_actions(pending.house, pending.origin).is_empty() {
                            self.state.active_house = Some(pending.house);
                            return Ok(());
                        }
                        self.finish_march();
                        continue;
                    }
                    match self.next_action_house() {
                        Some(house) => {
                            self.state.active_house = Some(house);
                            return Ok(());
                        }
                        None => self.next_action_step(),
                    }
                }
                Phase::Combat => match self.next_combat_chooser() {
                    Some(house) => {
                        self.state.active_house = Some(house);
                        return Ok(());
                    }
                    None => self.resolve_pending_battle()?,
                },
                Phase::Cleanup => self.run_cleanup_phase()?,
                Phase::GameOver => unreachable!("handled at the top of the loop"),
            }
        }
        Err(EngineError::PhaseDidNotSettle)
    }

    /// The game ends immediately when a house controls enough castles.
    fn check_victory(&mut self) {
        if self.state.winner.is_some() {
            return;
        }
        let mut counts = [0u32; N_HOUSES];
        for region_id in self.state.fortifications.iter().copied() {
            if let Some(controller) = self.state.region(region_id).controller {
                counts[controller.index()] += 1;
            }
        }
        let target = self.state.victory_castles as u32;
        let order = self.state.tracks[Track::IronThrone.index()].clone();
        for house in order {
            if self
                .state
                .house(house)
                .map(|h| h.eliminated)
                .unwrap_or(true)
            {
                continue;
            }
            if counts[house.index()] >= target {
                self.declare_winner(house, "controls enough castles");
                return;
            }
        }
    }

    fn declare_winner(&mut self, house: HouseId, reason: &str) {
        self.state.winner = Some(house);
        self.state.phase = Phase::GameOver;
        self.state.active_house = None;
        self.state.note(format!("{house} wins: {reason}"));
    }

    // -- Planning Phase --------------------------------------------------

    /// Areas that still must receive an order token this Planning Phase.
    ///
    /// Every area containing units (including routed units and Ships in a Port)
    /// must be assigned exactly one order. Areas containing only a Garrison are
    /// exempt.
    pub fn planning_required(&self, house: HouseId) -> SmallVec<[RegionId; 32]> {
        self.state
            .region_ids()
            .filter(|region_id| {
                let region = self.state.region(*region_id);
                region.controller == Some(house) && region.order.is_none() && region.total(true) > 0
            })
            .collect()
    }

    fn next_planner(&mut self) -> Option<HouseId> {
        while let Some(house) = self.state.pending_planning.first().copied() {
            let eliminated = self
                .state
                .house(house)
                .map(|state| state.eliminated)
                .unwrap_or(true);
            if eliminated || self.planning_required(house).is_empty() {
                self.state.pending_planning.remove(0);
                continue;
            }
            return Some(house);
        }
        None
    }

    fn planning_actions(&self, house: HouseId) -> Vec<Action> {
        let required = self.planning_required(house);
        if required.is_empty() {
            return Vec::new();
        }
        let Some(house_state) = self.state.house(house) else {
            return Vec::new();
        };
        let stars_left = self
            .state
            .star_allowance(house)
            .saturating_sub(house_state.stars_placed() as u8);
        // Filter the order pool once: starred and restriction checks depend
        // only on the token, not on the region it might go in.
        let mut placeable: SmallVec<[(crate::enums::OrderKey, bool); 15]> = SmallVec::new();
        for order in house_state.orders_available.iter().copied() {
            if order.starred() && stars_left == 0 {
                continue;
            }
            if self.state.has_order_restriction(order) {
                continue;
            }
            placeable.push((order, order.order_type() == OrderType::March));
        }

        let mut actions: Vec<Action> = Vec::new();
        for region_id in &required {
            let has_units = self.state.region(*region_id).has_units();
            for (order, is_march) in &placeable {
                if *is_march && !has_units {
                    continue;
                }
                actions.push(Action::place_order(house, *region_id, *order));
            }
        }
        // Placement is mandatory, but if the pool and restrictions leave no
        // legal order the player must still be able to finish.
        if actions.is_empty() {
            actions.push(Action::pass(house));
        }
        actions
    }

    fn apply_place_order(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("place_order without a house".into()))?;
        let region_id = action
            .region
            .ok_or_else(|| EngineError::IllegalAction("place_order without a region".into()))?;
        let order = action
            .order
            .ok_or_else(|| EngineError::IllegalAction("place_order without an order".into()))?;

        // Return any order already in the region to its owner's pool.
        let existing = self.state.region(region_id);
        if let (Some(previous), Some(owner)) = (existing.order, existing.order_house) {
            if let Some(owner_state) = self.state.house_mut(owner) {
                owner_state.return_order(previous);
            }
        }

        let house_state = self
            .state
            .house_mut(house)
            .ok_or(EngineError::MissingHouse(house))?;
        if !house_state.take_order(order) {
            return Err(EngineError::IllegalAction(format!(
                "{house} has no {order} token available"
            )));
        }

        let region = self.state.region_mut(region_id);
        region.order = Some(order);
        region.order_house = Some(house);
        Ok(())
    }

    fn apply_pass(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("pass without a house".into()))?;
        match self.state.phase {
            Phase::Planning => {
                if let Some(position) = self
                    .state
                    .pending_planning
                    .iter()
                    .position(|candidate| *candidate == house)
                {
                    self.state.pending_planning.remove(position);
                }
                Ok(())
            }
            Phase::Raven => {
                self.state.raven_used = true;
                Ok(())
            }
            Phase::Action => {
                if self.state.pending_march.is_some() {
                    // Finishing an unfinished multi-destination march.
                    self.finish_march();
                    return Ok(());
                }
                match action.region {
                    None => {
                        // Forfeit every order of the step's type.
                        let order_type = self.state.action_step.order_type();
                        for (region, _) in self.state.orders_of_type(house, order_type) {
                            self.decline_order(house, region);
                        }
                    }
                    Some(region) => self.decline_order(house, region),
                }
                Ok(())
            }
            Phase::Mustering => {
                for region in self.state.controlled_regions(house) {
                    self.state.region_mut(region).mustering_points = 0;
                }
                Ok(())
            }
            Phase::Combat => {
                if let Some(battle) = self.state.pending_battle.as_mut() {
                    // Declining the Valyrian Steel Blade.
                    battle.blade_done = true;
                }
                Ok(())
            }
            other => Err(EngineError::UnportedPhase(other)),
        }
    }

    /// Resolve one order with no effect, returning its token to the pool.
    fn decline_order(&mut self, house: HouseId, region: RegionId) {
        let existing = self.state.region(region);
        let Some(order) = existing.order else {
            return;
        };
        if existing.order_house != Some(house) {
            return;
        }
        {
            let region_state = self.state.region_mut(region);
            region_state.order = None;
            region_state.order_house = None;
        }
        if let Some(house_state) = self.state.house_mut(house) {
            house_state.return_order(order);
        }
    }

    // ==================================================================
    // Cleanup phase
    // ==================================================================

    pub(crate) fn run_cleanup_phase(&mut self) -> Result<(), EngineError> {
        // Routed units recover.
        for region in self
            .state
            .region_ids()
            .collect::<SmallVec<[RegionId; 32]>>()
        {
            if self
                .state
                .region(region)
                .routed
                .iter()
                .all(|count| *count == 0)
            {
                continue;
            }
            let region_state = self.state.region_mut(region);
            for index in 0..4 {
                region_state.units[index] += region_state.routed[index];
                region_state.routed[index] = 0;
            }
        }

        self.clear_orders();

        for house in self.state.present_houses() {
            if let Some(state) = self.state.house_mut(house) {
                state.blade_used = false;
                state.orders_placed.clear();
                state.orders_available = OrderKey::ALL.into_iter().collect();
            }
        }

        self.state.raven_used = false;
        self.state.action_house_last = None;
        self.state.pending_march = None;
        self.state.deferred_battle = None;
        self.state.return_phase = None;

        for region in self
            .state
            .region_ids()
            .collect::<SmallVec<[RegionId; 32]>>()
        {
            let (empty, mustering, token, owner) = {
                let state = self.state.region(region);
                (
                    state.is_empty() && state.routed.iter().all(|count| *count == 0),
                    state.mustering_points,
                    state.influence_token,
                    state.owner,
                )
            };
            if empty {
                self.state.region_mut(region).controller = token.or(owner);
            }
            // Unspent mustering points are lost, and order restrictions last
            // only for the round in which their card was drawn.
            if mustering > 0 {
                self.state.region_mut(region).mustering_points = 0;
            }
        }
        self.state.order_restrictions.clear();

        // Victory check: immediate at seven castles, otherwise after the last
        // round with the FAQ v2.0 tiebreakers.
        self.check_victory();
        if self.state.phase == Phase::GameOver {
            return Ok(());
        }
        if self.state.round >= self.state.max_rounds {
            let best = self.final_ranking_best();
            self.declare_winner(best, "won the end-of-game tiebreakers");
            return Ok(());
        }

        self.state.round += 1;
        self.state.pending_planning = self.state.tracks[Track::IronThrone.index()].clone();
        self.state.bid_tracks.clear();
        self.state.pending_bids.clear();
        self.state.pending_westeros = None;
        self.state.wildling_attack = None;
        self.state.westeros_choice = None;
        for deck in self.state.westeros_decks.iter_mut() {
            deck.drawn = None;
        }
        let round = self.state.round;
        self.state.note(format!("round {round} begins"));
        self.state.phase = Phase::Westeros;
        Ok(())
    }

    /// FAQ v2.0 end-of-game tiebreaker ordering (highest wins).
    ///
    /// Victory-track areas, then Strongholds, then Supply, then available Power,
    /// then Iron Throne position.
    fn final_ranking_key(&self, house: HouseId) -> (u32, u32, u8, u8, i32) {
        (
            self.state.castle_count(house),
            self.stronghold_count(house),
            self.state
                .house(house)
                .map(|state| state.supply_level)
                .unwrap_or(0),
            self.state
                .house(house)
                .map(|state| state.power)
                .unwrap_or(0),
            -(self.state.track_position(house, Track::IronThrone) as i32),
        )
    }

    fn final_ranking_best(&self) -> HouseId {
        let track = self.state.track_order(Track::IronThrone).to_vec();
        let mut best = track[0];
        let mut best_key = self.final_ranking_key(best);
        for house in track.into_iter().skip(1) {
            let key = self.final_ranking_key(house);
            if key > best_key {
                best = house;
                best_key = key;
            }
        }
        best
    }

    /// Controlled Strongholds, the second end-of-game tiebreaker.
    pub fn stronghold_count(&self, house: HouseId) -> u32 {
        self.state
            .controlled_regions(house)
            .iter()
            .filter(|region| self.data.map.region(**region).is_stronghold())
            .count() as u32
    }

    /// Compact description of the state, used by the CLI and logging.
    pub fn summary(&self) -> serde_json::Value {
        let mut castles = serde_json::Map::new();
        let mut power = serde_json::Map::new();
        for house in self.state.present_houses() {
            castles.insert(
                house.name().to_string(),
                serde_json::json!(self.state.castle_count(house)),
            );
            power.insert(
                house.name().to_string(),
                serde_json::json!(self
                    .state
                    .house(house)
                    .map(|state| state.power)
                    .unwrap_or(0)),
            );
        }
        serde_json::json!({
            "round": self.state.round,
            "phase": self.state.phase.as_str(),
            "active_house": self.state.active_house.map(|house| house.name()),
            "winner": self.state.winner.map(|house| house.name()),
            "castles": castles,
            "power": power,
        })
    }

    // -- Messenger Raven step --------------------------------------------

    /// The Messenger Raven holder acts once per round, before orders are
    /// revealed.
    fn next_raven_actor(&self) -> Option<HouseId> {
        if self.state.raven_used {
            return None;
        }
        let holder = self.state.raven_holder()?;
        if self
            .state
            .house(holder)
            .map(|state| state.eliminated)
            .unwrap_or(true)
        {
            return None;
        }
        Some(holder)
    }

    /// The Messenger Raven may replace one of its own orders or inspect the top
    /// Wildling card. Only one of the two abilities may be used each round.
    fn raven_actions(&self, house: HouseId) -> Vec<Action> {
        let mut actions: Vec<Action> = Vec::new();
        let allowance = self.state.star_allowance(house) as i32;
        let stars = self
            .state
            .house(house)
            .map(|state| state.stars_placed())
            .unwrap_or(0) as i32;
        let available: SmallVec<[crate::enums::OrderKey; 15]> = self
            .state
            .house(house)
            .map(|state| state.orders_available.clone())
            .unwrap_or_default();

        for region_id in self.state.controlled_regions(house) {
            let region = self.state.region(region_id);
            if region.order.is_none() || region.order_house != Some(house) {
                continue;
            }
            let old_order = region.order.expect("checked above");
            let old_starred = if old_order.starred() { 1 } else { 0 };
            for order in available.iter().copied() {
                let new_stars = stars - old_starred + if order.starred() { 1 } else { 0 };
                if new_stars > allowance {
                    continue;
                }
                if self.state.has_order_restriction(order) {
                    continue;
                }
                if order.order_type() == OrderType::March && !region.has_units() {
                    continue;
                }
                actions.push(Action::raven_swap(house, region_id, order));
            }
        }
        actions.push(Action::raven_deck(house, false));
        actions.push(Action::raven_deck(house, true));
        actions.push(Action::pass(house));
        actions
    }

    pub(crate) fn apply_raven_swap(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("raven_swap without a house".into()))?;
        let region_id = action
            .region
            .ok_or_else(|| EngineError::IllegalAction("raven_swap without a region".into()))?;
        let order = action
            .order
            .ok_or_else(|| EngineError::IllegalAction("raven_swap without an order".into()))?;

        let old_order = self.state.region(region_id).order;
        if let Some(old) = old_order {
            if let Some(house_state) = self.state.house_mut(house) {
                house_state.return_order(old);
            }
        }
        let house_state = self
            .state
            .house_mut(house)
            .ok_or(EngineError::MissingHouse(house))?;
        if !house_state.take_order(order) {
            return Err(EngineError::IllegalAction(format!(
                "{house} has no {order} token available"
            )));
        }
        let region = self.state.region_mut(region_id);
        region.order = Some(order);
        region.order_house = Some(house);
        self.state.raven_used = true;
        let old_text = old_order.map(|key| key.key()).unwrap_or("none");
        let region_text = self.data.map.id_of(region_id).to_string();
        self.state.note(format!(
            "{house} swaps {old_text} for {order} in {region_text} with the Messenger Raven"
        ));
        Ok(())
    }

    pub(crate) fn apply_raven_deck(
        &mut self,
        action: Action,
        to_bottom: bool,
    ) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("raven_deck without a house".into()))?;
        if to_bottom && !self.state.wildling_deck.cards.is_empty() {
            let card = self.state.wildling_deck.cards.remove(0);
            self.state.wildling_deck.cards.push(card);
        }
        self.state.raven_used = true;
        let direction = if to_bottom { "bottom" } else { "top" };
        self.state.note(format!(
            "{house} inspects the top Wildling card and returns it to the {direction}"
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Arc<DataSet> {
        Arc::new(DataSet::load_default().expect("bundled data must load"))
    }

    #[test]
    fn a_fresh_game_asks_the_first_planner() {
        let game = Game::new(data(), GameConfig::with_seed(1)).unwrap();
        assert_eq!(game.state.phase, Phase::Planning);
        assert_eq!(game.state.round, 1);
        // Iron Throne order decides who plans first.
        let first = game.state.tracks[Track::IronThrone.index()][0];
        assert_eq!(game.current_player(), Some(first));
        assert!(!game.is_terminal());
    }

    #[test]
    fn planning_actions_cover_every_required_area() {
        let game = Game::new(data(), GameConfig::with_seed(1)).unwrap();
        let house = game.current_player().unwrap();
        let required = game.planning_required(house);
        let actions = game.legal_actions();
        assert!(!required.is_empty());
        assert!(!actions.is_empty());

        // Every action is a place_order for a required area, or the fallback pass.
        for action in &actions {
            assert_eq!(action.house, Some(house));
            match action.kind {
                ActionKind::PlaceOrder => {
                    let region = action.region.unwrap();
                    assert!(required.contains(&region));
                    assert!(action.order.is_some());
                }
                ActionKind::Pass => {}
                other => panic!("unexpected kind {other:?}"),
            }
        }

        // March orders need units; empty areas cannot take one.
        let starks = HouseId(2);
        if game.current_player() == Some(starks) {
            for action in &actions {
                if let (ActionKind::PlaceOrder, Some(order), Some(region)) =
                    (action.kind, action.order, action.region)
                {
                    if order.order_type() == OrderType::March {
                        assert!(game.state.region(region).has_units());
                    }
                }
            }
        }
    }

    #[test]
    fn placing_orders_drains_the_planning_queue() {
        let mut game = Game::new(data(), GameConfig::with_seed(1)).unwrap();
        let mut guard = 0;
        while game.state.phase == Phase::Planning {
            guard += 1;
            assert!(guard < 500, "planning phase must terminate");
            let actions = game.legal_actions();
            assert!(!actions.is_empty(), "a planner always has a legal action");
            let action = actions[0];
            if let Err(error) = game.step(action) {
                // The final placement hands control to the Messenger Raven
                // step, which is the next slice to port.
                assert!(
                    matches!(error, EngineError::UnportedPhase(Phase::Raven)),
                    "unexpected error: {error}"
                );
                break;
            }
        }
        assert_eq!(game.state.phase, Phase::Raven);
        assert!(game.state.pending_planning.is_empty());
        // Every area with units that a house controls now carries an order.
        for house in game.state.present_houses() {
            assert!(game.planning_required(house).is_empty());
        }
    }

    #[test]
    fn step_refuses_unported_phases_instead_of_guessing() {
        let mut game = Game::new(data(), GameConfig::with_seed(1)).unwrap();
        // Drain planning, then one more step lands on the Raven phase.
        while game.state.phase == Phase::Planning {
            let action = game.legal_actions()[0];
            if let Err(error) = game.step(action) {
                assert!(
                    matches!(error, EngineError::UnportedPhase(Phase::Raven)),
                    "expected the port boundary, got {error}"
                );
                break;
            }
        }
        assert_eq!(game.state.phase, Phase::Raven);
    }

    #[test]
    fn cloned_games_do_not_share_state() {
        let game = Game::new(data(), GameConfig::with_seed(3)).unwrap();
        let house = game.current_player().unwrap();
        let mut child = game.clone();
        assert_eq!(child.state.phase, Phase::Planning);

        let action = child.legal_actions()[0];
        child.step(action).unwrap();
        // The source is untouched by the clone's move.
        assert_eq!(game.current_player(), Some(house));
        assert_eq!(
            game.state.region(action.region.unwrap()).order,
            None,
            "the clone's order must not leak into the source"
        );
    }

    /// The other direction of the clone invariant.
    ///
    /// Python's copy-on-write bookkeeping once let the *source* write into an
    /// existing clone; an owned state makes that impossible, and this pins the
    /// direction that bug lived in (the retired `test_clone_isolation.py`).
    #[test]
    fn stepping_the_source_leaves_a_clone_untouched() {
        use crate::fingerprint::state_fingerprint;

        let mut game = Game::new(data(), GameConfig::with_seed(4)).unwrap();
        let clone = game.clone();
        let before = state_fingerprint(&clone.state, &clone.data);
        let action = game.legal_actions()[0];
        let region = action.region;
        game.step(action).unwrap();

        assert_eq!(
            state_fingerprint(&clone.state, &clone.data),
            before,
            "stepping the source must never reach into a clone"
        );
        if let Some(region) = region {
            assert_eq!(
                clone.state.region(region).order,
                None,
                "the source's order must not leak into the clone"
            );
        }
        assert!(!clone.legal_actions().is_empty());
    }
}
