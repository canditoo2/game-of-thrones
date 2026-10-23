//! The Action Phase and the Combat Phase.
//!
//! Port of the corresponding sections of `src/agot/engine/game.py`: order
//! resolution (Raid, March, Consolidate Power), multi-destination marches,
//! battles and the choices inside them.
//!
//! Two things are easy to get wrong here and are handled deliberately:
//!
//! * **Action order is part of the fingerprint.** Python sorts march targets,
//!   ship mustering targets and retreat options as *strings* and breaks ties in
//!   `max(...)` on `(size, region_id)` tuples, so the port sorts by the map's
//!   region id rather than by index — an index sort would silently reorder the
//!   legal-action list and diverge from the reference.
//! * **The two forces of a battle stay separate** until the outcome is applied,
//!   because casualty removal has to know whose units are whose.

use smallvec::SmallVec;

use crate::action::{Action, Side};
use crate::combat::{self, BattleRequest};
use crate::enums::{ActionStep, HouseId, OrderType, Phase, RegionId, Terrain, Track, UnitType};
use crate::game::{EngineError, Game};
use crate::house_cards;
use crate::state::{BattleStage, PendingBattle, PendingMarch};

/// A house's supply limits plus the current size of every area it controls.
///
/// March and mustering legality test Supply after the move, for every candidate
/// amount; computing the sizes once per decision keeps that from rescanning the
/// board per candidate.
pub(crate) struct SupplyProfile {
    limits: SmallVec<[u8; 8]>,
    controlled: SmallVec<[(RegionId, u32); 32]>,
}

impl SupplyProfile {
    /// `(region, unit count)` for every area the house controls.
    pub(crate) fn controlled_sizes(&self) -> &[(RegionId, u32)] {
        &self.controlled
    }

    pub(crate) fn limit_count(&self) -> usize {
        self.limits.len()
    }

    pub(crate) fn limit(&self, index: usize) -> u8 {
        self.limits[index]
    }
}

impl Game {
    // ==================================================================
    // Action Phase: whose turn, and what may they do
    // ==================================================================

    /// The next house to resolve one order, cycling by Iron Throne order.
    ///
    /// Each house resolves a single order before the turn passes on; a house
    /// with no orders of the current type is skipped. The step ends once no
    /// house can act.
    pub(crate) fn next_action_house(&mut self) -> Option<HouseId> {
        let order_type = self.state.action_step.order_type();
        let order: SmallVec<[HouseId; 6]> = self.state.tracks[Track::IronThrone.index()]
            .iter()
            .copied()
            .filter(|house| {
                self.state
                    .house(*house)
                    .map(|state| !state.eliminated)
                    .unwrap_or(false)
            })
            .collect();
        if order.is_empty() {
            return None;
        }
        let has_orders = self.state.houses_with_orders(order_type);
        let last = self.state.action_house_last;
        let start = match last {
            Some(house) => order
                .iter()
                .position(|candidate| *candidate == house)
                .map(|index| index + 1)
                .unwrap_or(0),
            None => 0,
        };
        for offset in 0..order.len() {
            let house = order[(start + offset) % order.len()];
            if has_orders.contains(&house) {
                self.state.action_house_last = Some(house);
                return Some(house);
            }
        }
        self.state.action_house_last = None;
        None
    }

    pub(crate) fn action_actions(&self, house: HouseId) -> Vec<Action> {
        if let Some(pending) = self.state.pending_march {
            if pending.house != house {
                return Vec::new();
            }
            let origin = pending.origin;
            // Further moves from the same area, or finish the march: units may
            // move to several adjacent areas under one order.
            let mut actions = self.march_actions(house, origin);
            actions.push(Action::pass(house).with_region(origin));
            return actions;
        }
        let order_type = self.state.action_step.order_type();
        let matches = self.state.orders_of_type(house, order_type);
        if matches.is_empty() {
            return Vec::new();
        }
        let mut actions: Vec<Action> = Vec::new();
        for (region_id, _) in matches {
            actions.extend(self.order_actions(house, region_id, order_type));
            // A player may always resolve an order with no effect; only that one
            // token is removed.
            actions.push(Action::pass(house).with_region(region_id));
        }
        actions
    }

    fn order_actions(
        &self,
        house: HouseId,
        region: RegionId,
        order_type: OrderType,
    ) -> Vec<Action> {
        match order_type {
            OrderType::Raid => self.raid_actions(house, region),
            OrderType::March => self.march_actions(house, region),
            OrderType::ConsolidatePower => {
                let mut actions = vec![Action::resolve_consolidate(house, region)];
                let spec = self
                    .state
                    .region(region)
                    .order
                    .map(|order| order.spec())
                    .expect("a consolidate action implies a placed order");
                // A Special Consolidate Power Order in a Castle or Stronghold
                // may muster there instead of collecting Power.
                if spec.starred && self.data.map.region(region).is_fortification() {
                    actions.push(Action::consolidate_muster(house, region));
                }
                actions
            }
            _ => Vec::new(),
        }
    }

    fn raid_actions(&self, house: HouseId, region: RegionId) -> Vec<Action> {
        let spec = self
            .state
            .region(region)
            .order
            .map(|order| order.spec())
            .expect("a raid action implies a placed order");
        let mut allowed = [
            OrderType::Support,
            OrderType::Raid,
            OrderType::ConsolidatePower,
            OrderType::March,
        ];
        // A starred Raid may also target a Defend order; the extra slot is only
        // read when the order is starred.
        let allowed_len = if spec.starred {
            allowed[3] = OrderType::Defend;
            4
        } else {
            3
        };
        let source_terrain = self.data.map.region(region).terrain;
        let mut actions = Vec::new();
        for neighbour in self.data.map.neighbours_of(region).iter().copied() {
            let target = self.state.region(neighbour);
            let Some(target_order) = target.order else {
                continue;
            };
            if target.order_house == Some(house) {
                continue;
            }
            let target_terrain = self.data.map.region(neighbour).terrain;
            // A Raid on land never reaches a sea area; a Raid in a Port only
            // reaches the adjacent sea; land areas may not raid a Port.
            if source_terrain == Terrain::Land && target_terrain != Terrain::Land {
                continue;
            }
            if source_terrain == Terrain::Port && target_terrain != Terrain::Sea {
                continue;
            }
            if allowed[..allowed_len].contains(&target_order.order_type()) {
                actions.push(Action::resolve_raid(house, region, neighbour));
            }
        }
        actions
    }

    // ==================================================================
    // Marches
    // ==================================================================

    pub(crate) fn supply_profile(&self, house: HouseId) -> SupplyProfile {
        let limits = self
            .state
            .house(house)
            .map(|state| state.supply_limit().iter().copied().collect())
            .unwrap_or_default();
        let controlled = self
            .state
            .controlled_regions(house)
            .into_iter()
            .map(|region| (region, self.state.region(region).total(true)))
            .collect();
        SupplyProfile { limits, controlled }
    }

    pub(crate) fn march_actions(&self, house: HouseId, region: RegionId) -> Vec<Action> {
        if self.state.region(region).is_empty() {
            return Vec::new();
        }
        let profile = self.supply_profile(house);
        let mut actions = Vec::new();
        let mut targets = self.march_targets(house, region);
        // Python sorts the target set as strings; the order is part of the
        // legal-action list, so sort by region id, not by index.
        let map = &self.data.map;
        targets.sort_unstable_by_key(|target| map.id_of(*target));
        let deferred_battle = self.deferred_battles && self.state.deferred_battle.is_some();
        for target in targets {
            if deferred_battle && self.battle_target(house, target) {
                // A March order can contain at most one battle; the first
                // battle is resolved after the order's movement is finished.
                continue;
            }
            let terrain = self.data.map.region(target).terrain;
            if terrain == Terrain::Sea || terrain == Terrain::Port {
                actions.extend(self.naval_march_actions(house, region, target, &profile));
            } else {
                actions.extend(self.land_march_actions(house, region, target, &profile));
            }
        }
        actions
    }

    fn naval_march_actions(
        &self,
        house: HouseId,
        origin: RegionId,
        target: RegionId,
        profile: &SupplyProfile,
    ) -> Vec<Action> {
        let ships = self.state.region(origin).unit_count(UnitType::Ship);
        if ships == 0 {
            return Vec::new();
        }
        let mut limit = ships;
        if self.data.map.region(target).terrain == Terrain::Port {
            // A Port holds at most three Ships.
            let occupied = self.state.region(target).total(false);
            limit = limit.min(3u8.saturating_sub(occupied as u8));
        }
        let mut actions = Vec::new();
        for count in 1..=limit {
            if !self.march_supply_ok(origin, target, count as u32, profile) {
                continue;
            }
            actions.push(Action::march(
                house,
                origin,
                target,
                Some([0, 0, 0, count]),
                false,
            ));
        }
        actions
    }

    fn land_march_actions(
        &self,
        house: HouseId,
        origin: RegionId,
        target: RegionId,
        profile: &SupplyProfile,
    ) -> Vec<Action> {
        let region = self.state.region(origin);
        let counts = [
            region.unit_count(UnitType::Footman),
            region.unit_count(UnitType::Knight),
            region.unit_count(UnitType::Siege),
        ];
        if counts.iter().sum::<u8>() == 0 {
            return Vec::new();
        }
        let mut actions = Vec::new();
        // Every subset of the eligible units, in the Python product() order.
        for footmen in 0..=counts[0] {
            for knights in 0..=counts[1] {
                for siege in 0..=counts[2] {
                    let moving = footmen as u32 + knights as u32 + siege as u32;
                    if moving == 0 {
                        continue;
                    }
                    if !self.march_supply_ok(origin, target, moving, profile) {
                        continue;
                    }
                    let units = [footmen, knights, siege, 0];
                    if !self.beats_neutral(house, origin, target, units) {
                        // A Neutral Force may only be attacked with enough
                        // strength to destroy it.
                        continue;
                    }
                    actions.push(Action::march(house, origin, target, Some(units), false));
                    if self.can_establish_control(house, origin, moving) {
                        actions.push(Action::march(house, origin, target, Some(units), true));
                    }
                }
            }
        }
        actions
    }

    /// Upper bound of a march's Combat Strength against a Neutral Force.
    ///
    /// The attacker may receive support from any adjacent Support order, but
    /// only once it is declared, so legal-march generation uses the maximum
    /// possible support. A Neutral Force may not use a House card or the
    /// Valyrian Steel Blade, and ties go to the attacker.
    fn neutral_attack_strength(
        &self,
        house: HouseId,
        origin: RegionId,
        target: RegionId,
        units: [u8; 4],
    ) -> u32 {
        let is_fort = self.data.map.region(target).is_fortification();
        combat::units_strength(&units, is_fort) as u32
            + combat::order_modifier(&self.state, origin, house, true).max(0) as u32
            + combat::potential_support_strength(&self.state, &self.data.map, target, true) as u32
    }

    fn beats_neutral(
        &self,
        house: HouseId,
        origin: RegionId,
        target: RegionId,
        units: [u8; 4],
    ) -> bool {
        let neutral = self.state.region(target).neutral_force;
        if neutral == 0 {
            return true;
        }
        self.neutral_attack_strength(house, origin, target, units) >= neutral as u32
    }

    /// Would this move leave the house within its Supply limits?
    fn march_supply_ok(
        &self,
        origin: RegionId,
        target: RegionId,
        moving: u32,
        profile: &SupplyProfile,
    ) -> bool {
        let mut sizes: SmallVec<[u32; 8]> = SmallVec::new();
        let mut target_seen = false;
        for (region, base) in &profile.controlled {
            let mut size = *base;
            if *region == origin {
                size = size.saturating_sub(moving);
            }
            if *region == target {
                size += moving;
                target_seen = true;
            }
            if size >= 2 {
                sizes.push(size);
            }
        }
        if !target_seen && moving >= 2 {
            // Taking an area with a new army of our own.
            sizes.push(moving);
        }
        sizes.sort_unstable_by(|left, right| right.cmp(left));
        sizes.iter().enumerate().all(|(index, size)| {
            index < profile.limits.len() && *size <= profile.limits[index] as u32
        })
    }

    /// Can this march leave a Power token in the area it vacates?
    fn can_establish_control(&self, house: HouseId, origin: RegionId, moving: u32) -> bool {
        let region = self.state.region(origin);
        if self.data.map.region(origin).terrain != Terrain::Land {
            return false;
        }
        if region.owner == Some(house) {
            return false; // the printed Home-area shield is enough
        }
        if region.influence_token.is_some() {
            return false; // only one Power token per area
        }
        if self
            .state
            .house(house)
            .map(|state| state.power)
            .unwrap_or(0)
            == 0
        {
            return false;
        }
        region.total(true) - moving == 0
    }

    fn battle_target(&self, house: HouseId, target: RegionId) -> bool {
        let region = self.state.region(target);
        region.neutral_force > 0
            || (region.controller.is_some()
                && region.controller != Some(house)
                && (!region.is_empty() || region.garrison > 0))
    }

    fn march_targets(&self, house: HouseId, origin: RegionId) -> SmallVec<[RegionId; 16]> {
        let region = self.state.region(origin);
        let origin_terrain = self.data.map.region(origin).terrain;
        let has_land = [UnitType::Footman, UnitType::Knight, UnitType::Siege]
            .iter()
            .any(|unit| region.unit_count(*unit) > 0);
        let has_ships = region.unit_count(UnitType::Ship) > 0;
        let mut targets: SmallVec<[RegionId; 16]> = SmallVec::new();

        if has_ships {
            for neighbour in self.data.map.neighbours_of(origin).iter().copied() {
                let terrain = self.data.map.region(neighbour).terrain;
                let is_sea = terrain == Terrain::Sea;
                let is_own_port = terrain == Terrain::Port
                    && self.state.region(neighbour).controller == Some(house);
                if is_sea || is_own_port {
                    targets.push(neighbour);
                }
            }
        }
        if has_land && origin_terrain == Terrain::Land {
            for neighbour in self.data.map.land_neighbours_of(origin) {
                targets.push(neighbour);
            }
            for neighbour in self.transport_targets(house, origin) {
                targets.push(neighbour);
            }
        }
        targets.retain(|target| *target != origin);
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    /// Land regions reachable by moving across a chain of owned ships.
    ///
    /// Routed Ships may still transport.
    fn transport_targets(&self, house: HouseId, origin: RegionId) -> SmallVec<[RegionId; 16]> {
        let naval = self.naval_ship_regions(house);
        let mut frontier: SmallVec<[RegionId; 16]> = self
            .data
            .map
            .sea_neighbours_of(origin)
            .into_iter()
            .filter(|region| naval.contains(region))
            .collect();
        let mut seen = frontier.clone();
        while let Some(current) = frontier.pop() {
            for neighbour in self.data.map.sea_neighbours_of(current) {
                if naval.contains(&neighbour) && !seen.contains(&neighbour) {
                    seen.push(neighbour);
                    frontier.push(neighbour);
                }
            }
        }
        let mut targets: SmallVec<[RegionId; 16]> = SmallVec::new();
        for sea in seen {
            for land in self.data.map.land_neighbours_of(sea) {
                if !targets.contains(&land) {
                    targets.push(land);
                }
            }
        }
        targets
    }

    /// Sea areas where a house has at least one (ready or routed) Ship.
    pub(crate) fn naval_ship_regions(&self, house: HouseId) -> SmallVec<[RegionId; 16]> {
        self.state
            .region_ids()
            .filter(|region| {
                let state = self.state.region(*region);
                state.controller == Some(house)
                    && self.data.map.region(*region).is_sea()
                    && state.unit_count(UnitType::Ship) + state.routed_count(UnitType::Ship) > 0
            })
            .collect()
    }

    // ==================================================================
    // Applying action-phase orders
    // ==================================================================

    pub(crate) fn apply_raid(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("raid without a house".into()))?;
        let region = action
            .region
            .ok_or_else(|| EngineError::IllegalAction("raid without a region".into()))?;
        let target_id = action
            .target
            .ok_or_else(|| EngineError::IllegalAction("raid without a target".into()))?;

        let removed = self.state.region(target_id).order;
        let owner = self.state.region(target_id).order_house;
        let pillaging = removed
            .map(|order| order.order_type() == OrderType::ConsolidatePower)
            .unwrap_or(false);
        {
            let target = self.state.region_mut(target_id);
            target.order = None;
            target.order_house = None;
        }
        {
            let source = self.state.region_mut(region);
            source.order = None;
            source.order_house = None;
        }
        if let (Some(order), Some(owner)) = (removed, owner) {
            if let Some(owner_state) = self.state.house_mut(owner) {
                owner_state.return_order(order);
            }
            if pillaging {
                if let Some(gainer) = self.state.house_mut(house) {
                    gainer.power += 1;
                }
                if let Some(victim) = self.state.house_mut(owner) {
                    victim.power = victim.power.saturating_sub(1);
                }
                let target_text = self.data.map.id_of(target_id).to_string();
                self.state.note(format!(
                    "{house} pillages {target_text}: +1 power, {owner} loses 1"
                ));
            } else {
                let target_text = self.data.map.id_of(target_id).to_string();
                self.state
                    .note(format!("{house} raids {order} from {target_text}"));
            }
        }
        Ok(())
    }

    pub(crate) fn apply_march(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("march without a house".into()))?;
        let origin = action
            .region
            .ok_or_else(|| EngineError::IllegalAction("march without a region".into()))?;
        let target_id = action
            .target
            .ok_or_else(|| EngineError::IllegalAction("march without a target".into()))?;
        let target_terrain = self.data.map.region(target_id).terrain;

        // The March order stays on the board until the march ends, so its bonus
        // still applies to a battle after the non-combat moves. That is what
        // lets one March Order reach several destinations.
        let attacker_order_strength = combat::order_modifier(&self.state, origin, house, true);
        let attacker_order_key = self.state.region(origin).order;

        let moving = self.moving_units(origin, target_terrain, action.units);
        let moving_total: u32 = moving.iter().map(|count| *count as u32).sum();
        let vacates = self.state.region(origin).total(true) - moving_total == 0;
        {
            let source = self.state.region_mut(origin);
            for (index, count) in moving.iter().enumerate() {
                source.units[index] = source.units[index].saturating_sub(*count);
            }
        }

        // The decision to establish control is made before any battle begins.
        if action.establish
            && vacates
            && self.data.map.region(origin).is_land()
            && self.state.region(origin).owner != Some(house)
        {
            self.state.region_mut(origin).influence_token = Some(house);
            if let Some(state) = self.state.house_mut(house) {
                state.power = state.power.saturating_sub(1);
            }
            let origin_text = self.data.map.id_of(origin).to_string();
            self.state
                .note(format!("{house} establishes control of {origin_text}"));
        }

        let target = self.state.region(target_id);
        let enemy_here = target.controller.is_some()
            && target.controller != Some(house)
            && (!target.is_empty() || target.garrison > 0);
        let neutral = target.neutral_force > 0;

        if neutral {
            // Attacking a Neutral Force is still a battle for *support*: the
            // attacker may receive support, the token itself never does.
            let queue: SmallVec<[RegionId; 8]> =
                combat::support_orders(&self.state, &self.data.map, target_id)
                    .into_iter()
                    .map(|(region, _)| region)
                    .collect();
            let battle = PendingBattle {
                region: target_id,
                attacker: house,
                defender: None,
                neutral: true,
                origin: Some(origin),
                attacker_order: attacker_order_key,
                attacker_units: moving,
                defender_units: [0; 4],
                attacker_order_strength,
                defender_order_strength: 0,
                attacker_card: None,
                defender_card: None,
                attacker_card_chosen: false,
                defender_card_chosen: false,
                blade_done: false,
                attacker_blade: false,
                defender_blade: false,
                stage: BattleStage::Support,
                support_queue: queue,
                support_index: 0,
                support_choices: SmallVec::new(),
                result: None,
                casualties_remaining: 0,
                retreat_options: SmallVec::new(),
                retreat_house: None,
                retreat_destination: None,
                defender_order: None,
                defender_order_house: None,
            };
            let target_text = self.data.map.id_of(target_id).to_string();
            if self.deferred_battles {
                self.state.deferred_battle = Some(battle);
                self.state.pending_march = Some(PendingMarch { house, origin });
                self.state.note(format!(
                    "{house} marches into {target_text}; the Neutral Force battle is queued"
                ));
                self.settle_source_control(origin);
                return Ok(());
            }
            self.state.pending_battle = Some(battle);
            self.consume_order(house, origin);
            self.state.pending_march = None;
            self.state.phase = Phase::Combat;
            self.state.note(format!(
                "{house} marches into {target_text} and attacks the Neutral Force"
            ));
            if self
                .state
                .pending_battle
                .as_ref()
                .map(|battle| battle.support_queue.is_empty())
                .unwrap_or(false)
            {
                // No Support order can help: resolve straight away.
                self.finalize_neutral_battle()?;
            }
            self.settle_source_control(origin);
            return Ok(());
        }

        if enemy_here {
            let defender = target.controller.expect("checked above");
            let defender_units = target.units;
            let defender_order_strength =
                combat::order_modifier(&self.state, target_id, defender, false);
            let queue: SmallVec<[RegionId; 8]> =
                combat::support_orders(&self.state, &self.data.map, target_id)
                    .into_iter()
                    .map(|(region, _)| region)
                    .collect();
            let battle = PendingBattle {
                region: target_id,
                attacker: house,
                defender: Some(defender),
                neutral: false,
                origin: Some(origin),
                attacker_order: attacker_order_key,
                attacker_units: moving,
                defender_units,
                attacker_order_strength,
                defender_order_strength,
                attacker_card: None,
                defender_card: None,
                attacker_card_chosen: false,
                defender_card_chosen: false,
                blade_done: false,
                attacker_blade: false,
                defender_blade: false,
                stage: BattleStage::Support,
                support_queue: queue,
                support_index: 0,
                support_choices: SmallVec::new(),
                result: None,
                casualties_remaining: 0,
                retreat_options: SmallVec::new(),
                retreat_house: None,
                retreat_destination: None,
                defender_order: None,
                defender_order_house: None,
            };
            let target_text = self.data.map.id_of(target_id).to_string();
            if self.deferred_battles {
                self.state.deferred_battle = Some(battle);
                self.state.pending_march = Some(PendingMarch { house, origin });
                self.state.note(format!(
                    "{house} marches into {target_text}; the battle is queued"
                ));
                self.settle_source_control(origin);
                return Ok(());
            }
            self.state.pending_battle = Some(battle);
            self.consume_order(house, origin);
            self.state.pending_march = None;
            self.state.phase = Phase::Combat;
            self.state.note(format!(
                "{house} marches into {target_text} and attacks {defender}"
            ));
            self.settle_source_control(origin);
            return Ok(());
        }

        // Marching onto a lone enemy Power token discards it; it is not a battle
        // and the Power token does not defend.
        {
            let target = self.state.region_mut(target_id);
            if target.influence_token.is_some() && target.influence_token != Some(house) {
                target.influence_token = None;
            }
            for (index, count) in moving.iter().enumerate() {
                target.units[index] += *count;
            }
            target.controller = Some(house);
        }
        if self.data.map.region(target_id).is_land() {
            self.capture_port(house, target_id);
        }
        let origin_text = self.data.map.id_of(origin).to_string();
        let target_text = self.data.map.id_of(target_id).to_string();
        self.state
            .note(format!("{house} marches {origin_text} -> {target_text}"));
        // The order is only spent once there is nothing left to move.
        self.state.pending_march = Some(PendingMarch { house, origin });
        if self.march_actions(house, origin).is_empty() {
            self.finish_march();
        }
        self.settle_source_control(origin);
        Ok(())
    }

    /// An emptied area falls back to its Power token or printed owner.
    fn settle_source_control(&mut self, origin: RegionId) {
        let source = self.state.region(origin);
        if source.is_empty() && source.routed.iter().all(|count| *count == 0) {
            let next = source.influence_token.or(source.owner);
            self.state.region_mut(origin).controller = next;
        }
    }

    pub(crate) fn finish_march(&mut self) {
        let Some(pending) = self.state.pending_march else {
            return;
        };
        self.consume_order(pending.house, pending.origin);
        self.state.pending_march = None;
        if let Some(battle) = self.state.deferred_battle.take() {
            // All destinations have been chosen: the queued battle now starts.
            self.state.pending_battle = Some(battle);
            self.state.phase = Phase::Combat;
            self.state.active_house = None;
        }
    }

    /// Units that leave a marching area.
    ///
    /// `units` is a count per `(Footman, Knight, Siege, Ship)`; `None` means
    /// every eligible unit, which keeps hand-written calls working.
    fn moving_units(
        &self,
        origin: RegionId,
        target_terrain: Terrain,
        units: Option<[u8; 4]>,
    ) -> [u8; 4] {
        let eligible: [bool; 4] =
            if target_terrain == Terrain::Sea || target_terrain == Terrain::Port {
                [false, false, false, true]
            } else {
                [true, true, true, false]
            };
        let source = self.state.region(origin);
        match units {
            None => {
                let mut moving = [0u8; 4];
                for index in 0..4 {
                    if eligible[index] {
                        moving[index] = source.units[index];
                    }
                }
                moving
            }
            Some(counts) => {
                let mut moving = [0u8; 4];
                for index in 0..4 {
                    if counts[index] > 0 && eligible[index] {
                        moving[index] = counts[index];
                    }
                }
                moving
            }
        }
    }

    /// Take the Port of a captured land area.
    ///
    /// The new land controller may replace the enemy Ships in the Port with an
    /// equal number of his own, limited by the Port's three-Ship capacity and by
    /// Supply.
    pub(crate) fn capture_port(&mut self, house: HouseId, land_region: RegionId) {
        let Some(port_id) = self.data.map.port_of(land_region) else {
            return;
        };
        let port = self.state.region(port_id);
        let enemy_ships = port.unit_count(UnitType::Ship) + port.routed_count(UnitType::Ship);
        {
            let port = self.state.region_mut(port_id);
            port.units[UnitType::Ship.index()] = 0;
            port.routed[UnitType::Ship.index()] = 0;
            port.controller = Some(house);
        }
        let mut placed = 0u8;
        for _ in 0..enemy_ships {
            if placed >= 3 || !self.muster_fits(house, port_id, UnitType::Ship, 1, None) {
                break;
            }
            self.state.region_mut(port_id).units[UnitType::Ship.index()] += 1;
            placed += 1;
        }
        if enemy_ships > 0 {
            let port_text = self.data.map.id_of(port_id).to_string();
            self.state.note(format!(
                "{house} captures {port_text}, replacing {placed}/{enemy_ships} Ships"
            ));
        }
    }

    /// Clear an area's order token and return it to its owner's pool.
    pub(crate) fn consume_order(
        &mut self,
        house: HouseId,
        region: RegionId,
    ) -> Option<crate::enums::OrderKey> {
        let order_key = self.state.region(region).order;
        {
            let region_state = self.state.region_mut(region);
            region_state.order = None;
            region_state.order_house = None;
        }
        let order_key = order_key?;
        if let Some(house_state) = self.state.house_mut(house) {
            house_state.return_order(order_key);
        }
        Some(order_key)
    }

    pub(crate) fn apply_consolidate(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("consolidate without a house".into()))?;
        let region = action
            .region
            .ok_or_else(|| EngineError::IllegalAction("consolidate without a region".into()))?;
        let static_region = self.data.map.region(region).clone();
        self.consume_order(house, region);
        // A Consolidate Power Order in a Sea area has no effect.
        if static_region.is_sea() {
            let region_text = self.data.map.id_of(region).to_string();
            self.state.note(format!(
                "{house}'s consolidate in {region_text} on open water: no effect"
            ));
            return Ok(());
        }
        // In a Port it is removed without effect while enemy Ships hold the sea.
        if static_region.is_port() {
            if let Some(sea_id) = static_region.sea {
                let sea = self.state.region(sea_id);
                let enemy_ships = sea.unit_count(UnitType::Ship) + sea.routed_count(UnitType::Ship);
                if sea.controller.is_some() && sea.controller != Some(house) && enemy_ships > 0 {
                    let region_text = self.data.map.id_of(region).to_string();
                    self.state.note(format!(
                        "{house}'s consolidate in {region_text} cancelled by enemy ships"
                    ));
                    return Ok(());
                }
            }
        }
        let gained = 1 + static_region.power;
        if let Some(state) = self.state.house_mut(house) {
            state.power += gained;
        }
        let region_text = self.data.map.id_of(region).to_string();
        self.state.note(format!(
            "{house} consolidates in {region_text} for {gained} power"
        ));
        Ok(())
    }

    pub(crate) fn apply_consolidate_muster(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action.house.ok_or_else(|| {
            EngineError::IllegalAction("consolidate_muster without a house".into())
        })?;
        let region = action.region.ok_or_else(|| {
            EngineError::IllegalAction("consolidate_muster without a region".into())
        })?;
        let mustering = self.data.map.region(region).mustering;
        self.consume_order(house, region);
        self.state.region_mut(region).mustering_points += mustering;
        self.state.return_phase = Some(Phase::Action);
        self.state.phase = Phase::Mustering;
        let region_text = self.data.map.id_of(region).to_string();
        self.state.note(format!(
            "{house} musters in {region_text} from a Special Consolidate Power Order"
        ));
        Ok(())
    }

    pub(crate) fn next_action_step(&mut self) {
        self.state.action_house_last = None;
        self.state.pending_march = None;
        self.state.deferred_battle = None;
        match self.state.action_step {
            ActionStep::Raid => self.state.action_step = ActionStep::March,
            ActionStep::March => self.state.action_step = ActionStep::ConsolidatePower,
            ActionStep::ConsolidatePower => {
                self.clear_orders();
                self.state.phase = Phase::Cleanup;
            }
        }
    }

    /// Return every order token on the board to its owner's pool.
    pub(crate) fn clear_orders(&mut self) {
        for region_id in self
            .state
            .region_ids()
            .collect::<SmallVec<[RegionId; 32]>>()
        {
            let region = self.state.region(region_id);
            let Some(order) = region.order else {
                continue;
            };
            let owner = region.order_house;
            if let Some(owner) = owner {
                if let Some(house_state) = self.state.house_mut(owner) {
                    house_state.return_order(order);
                }
            }
            let region = self.state.region_mut(region_id);
            region.order = None;
            region.order_house = None;
        }
    }

    // ==================================================================
    // Combat Phase
    // ==================================================================

    pub(crate) fn next_combat_chooser(&mut self) -> Option<HouseId> {
        // The battle is read as a snapshot and written back in short bursts: a
        // `&mut` into `pending_battle` cannot be held while the rest of the
        // state is queried.
        let snapshot = self.state.pending_battle.as_ref()?;
        match snapshot.stage {
            BattleStage::Support => {
                if let Some(region) = snapshot.support_queue.get(snapshot.support_index).copied() {
                    return self.state.region(region).order_house;
                }
                // All support declared: a Neutral Force has no defending card,
                // so it goes straight to resolution.
                let next = if snapshot.neutral {
                    BattleStage::NeutralResolve
                } else {
                    BattleStage::Cards
                };
                if let Some(battle) = self.state.pending_battle.as_mut() {
                    battle.stage = next;
                }
                self.next_combat_chooser()
            }
            BattleStage::Cards => {
                if !snapshot.attacker_card_chosen {
                    return Some(snapshot.attacker);
                }
                if !snapshot.defender_card_chosen {
                    return snapshot.defender;
                }
                if let Some(battle) = self.state.pending_battle.as_mut() {
                    battle.stage = BattleStage::Blade;
                }
                self.next_combat_chooser()
            }
            BattleStage::Blade => {
                let attacker = snapshot.attacker;
                let defender = snapshot.defender;
                let blade_done = snapshot.blade_done;
                if !blade_done {
                    if let Some(holder) = self.state.blade_holder() {
                        let involved = Some(holder) == Some(attacker) || Some(holder) == defender;
                        let used = self
                            .state
                            .house(holder)
                            .map(|state| state.blade_used)
                            .unwrap_or(true);
                        if involved && !used {
                            return Some(holder);
                        }
                    }
                    if let Some(battle) = self.state.pending_battle.as_mut() {
                        battle.blade_done = true;
                    }
                }
                None
            }
            BattleStage::Casualties => {
                if snapshot.casualties_remaining > 0 {
                    snapshot.result.as_ref().map(|result| result.loser)
                } else {
                    None
                }
            }
            BattleStage::Retreat => snapshot.retreat_house,
            BattleStage::NeutralResolve | BattleStage::Finalize => None,
        }
    }

    pub(crate) fn combat_actions(&self, house: HouseId) -> Vec<Action> {
        let Some(battle) = self.state.pending_battle.as_ref() else {
            return Vec::new();
        };
        match battle.stage {
            BattleStage::Support => {
                let index = battle.support_index;
                if index >= battle.support_queue.len() {
                    return Vec::new();
                }
                let region = battle.support_queue[index];
                if self.state.region(region).order_house != Some(house) {
                    return Vec::new();
                }
                let sides: &[Side] = if battle.neutral || Some(house) == Some(battle.attacker) {
                    // A Neutral Force never receives support, and a player never
                    // supports an opponent against their own units.
                    &[Side::Attacker, Side::Refuse]
                } else if Some(house) == battle.defender {
                    &[Side::Defender, Side::Refuse]
                } else {
                    &[Side::Attacker, Side::Defender, Side::Refuse]
                };
                sides
                    .iter()
                    .map(|side| Action::declare_support(house, region, *side))
                    .collect()
            }
            BattleStage::Cards => {
                let available = self
                    .state
                    .house(house)
                    .map(|state| state.house_cards_available.clone())
                    .unwrap_or_default();
                if available.is_empty() {
                    // Every card has been played; the house fights with no card.
                    return vec![Action::play_house_card(house, None)];
                }
                available
                    .into_iter()
                    .map(|card| Action::play_house_card(house, Some(card)))
                    .collect()
            }
            BattleStage::Blade => vec![Action::use_blade(house), Action::pass(house)],
            BattleStage::Casualties => {
                let Some(result) = battle.result.as_ref() else {
                    return Vec::new();
                };
                let pool = result.context.units_of(result.loser);
                [
                    UnitType::Footman,
                    UnitType::Knight,
                    UnitType::Siege,
                    UnitType::Ship,
                ]
                .into_iter()
                .filter(|unit| pool[unit.index()] > 0)
                .map(|unit| Action::choose_casualty(house, result.region, unit))
                .collect()
            }
            BattleStage::Retreat => battle
                .retreat_options
                .iter()
                .map(|destination| Action::retreat(house, battle.region, *destination))
                .collect(),
            _ => Vec::new(),
        }
    }

    pub(crate) fn apply_declare_support(&mut self, action: Action) -> Result<(), EngineError> {
        let region = action.region.unwrap_or(RegionId(0));
        let house = action.house.unwrap_or(HouseId(0));
        // Read the queue position before taking a mutable borrow of the battle.
        let (queued, index) = match self.state.pending_battle.as_ref() {
            Some(battle) => (
                battle.support_queue.get(battle.support_index).copied(),
                battle.support_index,
            ),
            None => return Ok(()),
        };
        if queued != Some(region) {
            return Ok(());
        }
        if self.state.region(region).order_house != Some(house) {
            return Ok(());
        }
        let side = action.side.unwrap_or(Side::Refuse);
        if let Some(battle) = self.state.pending_battle.as_mut() {
            if side != Side::Refuse {
                if let Some(entry) = battle
                    .support_choices
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == region)
                {
                    entry.1 = side;
                } else {
                    battle.support_choices.push((region, side));
                }
            }
            battle.support_index = index + 1;
        }
        let verb = match side {
            Side::Attacker => "supports the attacker",
            Side::Defender => "supports the defender",
            Side::Refuse => "refuses to support",
        };
        let region_text = self.data.map.id_of(region).to_string();
        self.state
            .note(format!("{house} {verb} from {region_text}"));
        Ok(())
    }

    pub(crate) fn apply_use_blade(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action.house.unwrap_or(HouseId(0));
        let Some(battle) = self.state.pending_battle.as_mut() else {
            return Ok(());
        };
        if Some(house) == Some(battle.attacker) {
            battle.attacker_blade = true;
        } else {
            battle.defender_blade = true;
        }
        battle.blade_done = true;
        self.state
            .note(format!("{house} uses the Valyrian Steel Blade"));
        Ok(())
    }

    pub(crate) fn apply_house_card(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action.house.unwrap_or(HouseId(0));
        let card = action.card;
        if let Some(card) = card {
            if let Some(state) = self.state.house_mut(house) {
                state.play_card(card);
            }
        }
        let Some(battle) = self.state.pending_battle.as_mut() else {
            return Ok(());
        };
        if Some(house) == Some(battle.attacker) {
            battle.attacker_card = card;
            battle.attacker_card_chosen = true;
        } else {
            battle.defender_card = card;
            battle.defender_card_chosen = true;
        }
        Ok(())
    }

    /// The defeated player destroys one unit of their choice.
    pub(crate) fn apply_choose_casualty(&mut self, action: Action) -> Result<(), EngineError> {
        let Some(unit) = action.unit else {
            return Ok(());
        };
        let Some(battle) = self.state.pending_battle.as_mut() else {
            return Ok(());
        };
        let Some(result) = battle.result.as_mut() else {
            return Ok(());
        };
        if result.context.remove_unit(result.loser, unit) {
            battle.casualties_remaining = battle.casualties_remaining.saturating_sub(1);
            let region_text = self.data.map.id_of(result.region).to_string();
            let loser = result.loser;
            self.state
                .note(format!("{loser} destroys a {unit} in {region_text}"));
        }
        Ok(())
    }

    /// The retreating player (or Robb Stark's controller) picks an area.
    pub(crate) fn apply_retreat_action(&mut self, action: Action) -> Result<(), EngineError> {
        let destination = action.target;
        let Some(battle) = self.state.pending_battle.as_mut() else {
            return Ok(());
        };
        if let Some(destination) = destination {
            if battle.retreat_options.contains(&destination) {
                battle.retreat_destination = Some(destination);
            }
        }
        // The choice is made: hand control back to the engine so it applies the
        // outcome rather than re-asking the same question.
        battle.retreat_house = None;
        Ok(())
    }

    /// Decide the battle, then pause for the defeated player's choices.
    pub(crate) fn begin_battle_resolution(&mut self) -> Result<(), EngineError> {
        let Some(battle) = self.state.pending_battle.as_ref() else {
            return Ok(());
        };
        let request = BattleRequest {
            region: battle.region,
            attacker: battle.attacker,
            defender: battle.defender.unwrap_or(battle.attacker),
            attacker_units: battle.attacker_units,
            defender_units: battle.defender_units,
            attacker_card: battle.attacker_card,
            defender_card: battle.defender_card,
            attacker_order_strength: battle.attacker_order_strength,
            defender_order_strength: battle.defender_order_strength,
            attacker_blade: battle.attacker_blade,
            defender_blade: battle.defender_blade,
            attacker_order: battle.attacker_order,
            attacker_origin: battle.origin,
            // Always `Some`, even when empty: Python passes the battle's
            // `support_choices` dict (which starts empty) rather than `None`, so
            // an empty set means "nobody supported" — not "fall back to the
            // self-directed support the convenience path uses". Passing `None`
            // here silently grants support from any adjacent Support order.
            support_choices: Some(battle.support_choices.clone()),
        };
        let result = combat::resolve_battle(
            &self.state,
            &self.data.map,
            &self.data.house_cards,
            &request,
        );
        house_cards::commit_deferred(&mut self.state, &self.data.house_cards, &result.context);
        // Remember the defender's Order token: if the attacker wins, the battle
        // clears it from the board and it returns to its owner's pool.
        let defender_order = self.state.region(result.region).order;
        let defender_order_house = self.state.region(result.region).order_house;
        let loser_units = result.context.units_of(result.loser);
        let remaining = result
            .casualties
            .min(loser_units.iter().map(|count| *count as u32).sum::<u32>() as u8);

        let battle = self
            .state
            .pending_battle
            .as_mut()
            .expect("checked at the top of this function");
        battle.defender_order = defender_order;
        battle.defender_order_house = defender_order_house;
        battle.casualties_remaining = remaining;
        battle.retreat_options = SmallVec::new();
        battle.retreat_house = None;
        battle.retreat_destination = None;
        battle.result = Some(result);
        if remaining > 0 {
            battle.stage = BattleStage::Casualties;
        } else {
            self.prepare_retreat();
        }
        Ok(())
    }

    /// Work out whether anyone chooses the retreat, then set the stage.
    pub(crate) fn prepare_retreat(&mut self) {
        let Some(battle) = self.state.pending_battle.as_ref() else {
            return;
        };
        let Some(result) = battle.result.as_ref() else {
            return;
        };
        let surviving = result.context.units_of(result.loser);
        if surviving.iter().all(|count| *count == 0) {
            self.set_battle_stage(BattleStage::Finalize);
            return;
        }
        if Some(result.loser) == Some(result.context.attacker) {
            // A defeated attacker always retreats to where it marched from.
            self.set_battle_stage(BattleStage::Finalize);
            return;
        }
        let mut options = combat::retreat_options(
            &self.state,
            &self.data.map,
            result.region,
            result.loser,
            surviving,
            result.context.attacker_origin,
        );
        // Robb Stark lets the winner choose, but only among the legal areas
        // where the loser loses the fewest units.
        let robb =
            house_cards::robb_chose_retreat(&result.context, &self.data.house_cards, result.winner);
        if robb && !options.is_empty() {
            let minimal: SmallVec<[RegionId; 8]> = options
                .iter()
                .copied()
                .filter(|option| {
                    combat::retreat_keeps_supply(
                        &self.state,
                        &self.data.map,
                        *option,
                        result.loser,
                        surviving,
                    )
                })
                .collect();
            if !minimal.is_empty() {
                options = minimal;
            }
        }
        if options.is_empty() {
            self.set_battle_stage(BattleStage::Finalize);
            return;
        }
        let retreat_house = if robb { result.winner } else { result.loser };
        if let Some(battle) = self.state.pending_battle.as_mut() {
            battle.retreat_options = options;
            battle.retreat_house = Some(retreat_house);
            battle.stage = BattleStage::Retreat;
        }
    }

    fn set_battle_stage(&mut self, stage: BattleStage) {
        if let Some(battle) = self.state.pending_battle.as_mut() {
            battle.stage = stage;
        }
    }

    /// Apply the battle to the board and leave the Combat phase.
    pub(crate) fn finalize_battle(&mut self) -> Result<(), EngineError> {
        let Some(battle) = self.state.pending_battle.take() else {
            self.state.phase = Phase::Action;
            return Ok(());
        };
        let Some(result) = battle.result.clone() else {
            self.state.phase = Phase::Action;
            return Ok(());
        };
        let attacker = battle.attacker;
        let defender = battle.defender;
        let region_id = battle.region;
        self.state.phase = Phase::Action;

        combat::apply_battle_result(
            &mut self.state,
            &self.data.map,
            &self.data.house_cards,
            &result,
            0,
            battle.retreat_destination,
        );
        // If the attacker won, the defender's Order token was removed by the
        // battle; return it to its owner's pool. If the defender won, its Order
        // and Power tokens are unaffected.
        if Some(result.winner) == Some(attacker) {
            if let (Some(order), Some(owner)) = (battle.defender_order, battle.defender_order_house)
            {
                if let Some(state) = self.state.house_mut(owner) {
                    state.return_order(order);
                }
            }
        }
        if battle.attacker_blade {
            if let Some(state) = self.state.house_mut(attacker) {
                state.blade_used = true;
            }
        }
        if let Some(defender) = defender {
            if battle.defender_blade {
                if let Some(state) = self.state.house_mut(defender) {
                    state.blade_used = true;
                }
            }
        }
        if Some(result.winner) == Some(attacker) && self.data.map.region(region_id).is_land() {
            // Capturing the area also takes its Port.
            self.capture_port(attacker, region_id);
        }
        let region_text = self.data.map.id_of(region_id).to_string();
        self.state.note(format!(
            "battle in {region_text}: {} defeats {} ({} vs {})",
            result.winner,
            result.loser,
            result.attacker.total(),
            result.defender.total()
        ));
        self.recycle_house_cards_for(attacker);
        if let Some(defender) = defender {
            self.recycle_house_cards_for(defender);
        }
        Ok(())
    }

    pub(crate) fn resolve_pending_battle(&mut self) -> Result<(), EngineError> {
        let Some(battle) = self.state.pending_battle.as_ref() else {
            self.state.phase = Phase::Action;
            return Ok(());
        };
        match battle.stage {
            BattleStage::Cards | BattleStage::Blade => self.begin_battle_resolution(),
            BattleStage::NeutralResolve => self.finalize_neutral_battle(),
            BattleStage::Casualties => {
                self.prepare_retreat();
                Ok(())
            }
            _ => self.finalize_battle(),
        }
    }

    /// Return six House cards after a house plays its last (seventh).
    ///
    /// The rulebook leaves the just-played card in the discard pile and returns
    /// the other six to hand.
    fn recycle_house_cards_for(&mut self, house: HouseId) {
        let Some(state) = self.state.house_mut(house) else {
            return;
        };
        if !state.house_cards_available.is_empty() || state.house_cards_used.is_empty() {
            return;
        }
        let last_played = *state.house_cards_used.last().expect("checked non-empty");
        state.house_cards_available = state
            .house_cards_used
            .iter()
            .copied()
            .filter(|card| *card != last_played)
            .collect();
        state.house_cards_used.clear();
        state.house_cards_used.push(last_played);
    }

    /// Resolve a march against a Neutral Force token.
    ///
    /// The attacker must equal or exceed the token's printed strength; there is
    /// no House card, no Valyrian Steel Blade and no casualty step. On success
    /// the token is removed and the area (with its fortification) is taken.
    pub(crate) fn finalize_neutral_battle(&mut self) -> Result<(), EngineError> {
        let Some(battle) = self.state.pending_battle.take() else {
            self.state.phase = Phase::Action;
            return Ok(());
        };
        let house = battle.attacker;
        let origin = battle.origin;
        let target_id = battle.region;
        let moving = battle.attacker_units;
        let is_fort = self.data.map.region(target_id).is_fortification();
        let support = if battle.support_choices.is_empty() {
            0
        } else {
            combat::support_from_choices(
                &self.state,
                &self.data.map,
                target_id,
                &battle.support_choices,
            )
            .attacker_strength as i16
        };
        let strength = combat::units_strength(&moving, is_fort) as i16
            + battle.attacker_order_strength
            + support;
        self.state.phase = Phase::Action;

        let neutral = self.state.region(target_id).neutral_force;
        if strength < neutral as i16 {
            // Support may have been refused; the attack then fails and the army
            // returns to where it marched from.
            if let Some(origin) = origin {
                let source = self.state.region_mut(origin);
                for (index, count) in moving.iter().enumerate() {
                    source.units[index] += *count;
                }
                if source.controller != Some(house) {
                    source.controller = Some(house);
                }
            }
            let target_text = self.data.map.id_of(target_id).to_string();
            self.state.note(format!(
                "{house} cannot destroy the Neutral Force in {target_text} ({strength} < {neutral}); march cancelled"
            ));
            return Ok(());
        }
        {
            let target = self.state.region_mut(target_id);
            target.neutral_force = 0;
            target.influence_token = None;
            for (index, count) in moving.iter().enumerate() {
                target.units[index] += *count;
            }
            target.controller = Some(house);
        }
        if self.data.map.region(target_id).is_land() {
            self.capture_port(house, target_id);
        }
        let target_text = self.data.map.id_of(target_id).to_string();
        self.state.note(format!(
            "{house} destroys the Neutral Force ({neutral}) in {target_text} with strength {strength}"
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::DataSet;
    use crate::setup::GameConfig;
    use std::sync::Arc;

    fn game(seed: u64) -> Game {
        let data = Arc::new(DataSet::load_default().expect("bundled data must load"));
        Game::new(data, GameConfig::with_seed(seed)).expect("game must start")
    }

    #[test]
    fn a_raid_can_pillage_a_consolidate_order() {
        // Placing a Consolidate Power order next to an enemy Raid order is the
        // only way to pillage; check the action exists and the Power moves.
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let winterfell = data.map.lookup("winterfell").unwrap();
        let white_harbor = data.map.lookup("white_harbor").unwrap();
        let raider = HouseId(2);
        let victim = HouseId(3);

        game.state.phase = Phase::Action;
        game.state.action_step = ActionStep::Raid;
        game.state.active_house = Some(raider);
        game.state.action_house_last = None;
        {
            let source = game.state.region_mut(winterfell);
            source.controller = Some(raider);
            source.order = Some(crate::enums::OrderKey::from_name("raid_a").unwrap());
            source.order_house = Some(raider);
            source.units[UnitType::Footman.index()] = 1;
        }
        {
            let target = game.state.region_mut(white_harbor);
            target.controller = Some(victim);
            target.order = Some(crate::enums::OrderKey::from_name("consolidate_a").unwrap());
            target.order_house = Some(victim);
            target.units[UnitType::Footman.index()] = 1;
        }
        if let Some(state) = game.state.house_mut(raider) {
            state.take_order(crate::enums::OrderKey::from_name("raid_a").unwrap());
        }
        if let Some(state) = game.state.house_mut(victim) {
            state.take_order(crate::enums::OrderKey::from_name("consolidate_a").unwrap());
        }
        let raider_power = game.state.house(raider).unwrap().power;
        let victim_power = game.state.house(victim).unwrap().power;

        let actions = game.legal_actions();
        let raid = actions
            .iter()
            .find(|action| action.target == Some(white_harbor))
            .copied()
            .expect("the adjacent enemy order can be raided");
        game.step(raid).unwrap();

        assert_eq!(game.state.house(raider).unwrap().power, raider_power + 1);
        assert_eq!(game.state.house(victim).unwrap().power, victim_power - 1);
        assert_eq!(game.state.region(white_harbor).order, None);
    }

    #[test]
    fn a_land_march_offers_every_subset_of_units() {
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let winterfell = data.map.lookup("winterfell").unwrap();
        let house = HouseId(2);
        game.state.phase = Phase::Action;
        game.state.action_step = ActionStep::March;
        game.state.active_house = Some(house);
        {
            let source = game.state.region_mut(winterfell);
            source.controller = Some(house);
            source.owner = Some(house);
            source.units = [0; 4];
            source.units[UnitType::Footman.index()] = 2;
            source.units[UnitType::Knight.index()] = 1;
        }
        let actions = game.march_actions(house, winterfell);
        // 2 footmen x 1 knight = 3 non-empty subsets, times the targets; every
        // action names a legal target and a non-empty unit set.
        assert!(!actions.is_empty());
        for action in &actions {
            let units = action.units.expect("a march always names its units");
            assert!(units.iter().any(|count| *count > 0));
            assert!(units[0] <= 2 && units[1] <= 1 && units[2] == 0);
            assert_ne!(action.target, Some(winterfell));
        }
    }

    #[test]
    fn a_march_into_an_empty_area_waits_for_further_moves() {
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let winterfell = data.map.lookup("winterfell").unwrap();
        let house = HouseId(2);
        let order = crate::enums::OrderKey::from_name("march").unwrap();
        game.state.phase = Phase::Action;
        game.state.action_step = ActionStep::March;
        game.state.active_house = Some(house);
        {
            let source = game.state.region_mut(winterfell);
            source.controller = Some(house);
            source.owner = Some(house);
            source.units = [0; 4];
            source.units[UnitType::Footman.index()] = 2;
            source.order = Some(order);
            source.order_house = Some(house);
        }
        if let Some(state) = game.state.house_mut(house) {
            state.take_order(order);
        }
        let actions = game.march_actions(house, winterfell);
        let march = actions
            .iter()
            .find(|action| action.units == Some([1, 0, 0, 0]))
            .copied()
            .expect("a one-footman march exists");
        let target = march.target.unwrap();
        game.step(march).unwrap();

        assert_eq!(
            game.state.region(winterfell).unit_count(UnitType::Footman),
            1
        );
        assert_eq!(game.state.region(target).unit_count(UnitType::Footman), 1);
        // The order stays on the board until the march is finished, which is what
        // lets one March Order reach several destinations.
        assert_eq!(game.state.region(winterfell).order, Some(order));
        assert!(game.state.pending_march.is_some());
        assert_eq!(game.current_player(), Some(house));
    }

    #[test]
    fn a_battle_waits_until_the_march_destinations_are_finished() {
        let mut game = game(1);
        let data = Arc::clone(&game.data);
        let winterfell = data.map.lookup("winterfell").unwrap();
        let white_harbor = data.map.lookup("white_harbor").unwrap();
        let moat_cailin = data.map.lookup("moat_cailin").unwrap();
        let house = HouseId(2);
        let enemy = HouseId(0);
        let order = crate::enums::OrderKey::from_name("march").unwrap();
        game.deferred_battles = true;
        game.state.phase = Phase::Action;
        game.state.action_step = ActionStep::March;
        game.state.active_house = Some(house);
        {
            let source = game.state.region_mut(winterfell);
            source.controller = Some(house);
            source.owner = Some(house);
            source.units = [0; 4];
            source.units[UnitType::Footman.index()] = 2;
            source.order = Some(order);
            source.order_house = Some(house);
            let enemy_region = game.state.region_mut(white_harbor);
            enemy_region.controller = Some(enemy);
            enemy_region.owner = Some(enemy);
            enemy_region.units = [0; 4];
            enemy_region.units[UnitType::Footman.index()] = 1;
            let empty = game.state.region_mut(moat_cailin);
            empty.controller = None;
            empty.owner = None;
            empty.units = [0; 4];
        }

        let actions = game.march_actions(house, winterfell);
        let attack = actions
            .iter()
            .find(|action| {
                action.target == Some(white_harbor) && action.units == Some([1, 0, 0, 0])
            })
            .copied()
            .expect("a one-footman attack is offered");
        game.step(attack).unwrap();
        assert_eq!(game.state.phase, Phase::Action);
        assert!(game.state.deferred_battle.is_some());
        assert!(game.state.pending_battle.is_none());

        let follow_up = game
            .legal_actions()
            .into_iter()
            .find(|action| {
                action.kind == crate::ActionKind::March
                    && action.target == Some(moat_cailin)
                    && action.units == Some([1, 0, 0, 0])
            })
            .expect("the second destination is still offered");
        game.step(follow_up).unwrap();

        assert_eq!(game.state.phase, Phase::Combat);
        assert!(game.state.deferred_battle.is_none());
        assert_eq!(
            game.state
                .pending_battle
                .as_ref()
                .map(|battle| battle.region),
            Some(white_harbor)
        );
    }
}
