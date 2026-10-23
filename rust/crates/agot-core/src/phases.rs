//! The Westeros, Bidding, Mustering and Wildling phases.
//!
//! Port of the corresponding sections of `src/agot/engine/game.py`. These phases
//! share one shape: the engine auto-advances until a player must decide, and a
//! card effect can hand control to a *different* phase (a bid, a Wildling
//! attack, a holder's choice) and resume afterwards.
//!
//! Card payloads stay as `serde_json::Value` (they live in `westeros.json` and
//! `wildlings.json`), so the effect switches read keys exactly as Python does.

use smallvec::SmallVec;

use crate::action::Action;
use crate::enums::{CardId, HouseId, Phase, RegionId, Track, UnitType, MUSTER_COST, SUPPLY_LIMITS};
use crate::game::{EngineError, Game};
use crate::state::{DeckState, WesterosChoice, WildlingAttack};

impl Game {
    // ==================================================================
    // Westeros phase
    // ==================================================================

    pub(crate) fn run_westeros_phase(&mut self) -> Result<(), EngineError> {
        if self.state.pending_westeros.is_none() {
            // Round one draws no Westeros cards.
            if self.state.round == 1 || !self.config.use_westeros_decks {
                self.state.phase = Phase::Planning;
                return Ok(());
            }
            self.state.order_restrictions.clear();
            let mut drawn: SmallVec<[u16; 3]> = SmallVec::new();
            for deck in 0..3 {
                let card = self.draw_westeros(deck);
                self.state.westeros_decks[deck].drawn = card;
                if let Some(card) = card {
                    drawn.push(card);
                }
            }
            // Every Wildling icon shown this round advances the threat token,
            // and an attack triggered at the cap resolves before the cards.
            let icons = drawn
                .iter()
                .filter(|card| {
                    self.data.westeros.card(**card)["advance_wildlings"]
                        .as_bool()
                        .unwrap_or(false)
                })
                .count() as u8;
            self.state.pending_westeros = Some(drawn);
            if icons > 0 {
                self.advance_wildlings(icons)?;
                if self.state.phase == Phase::WildlingAttack {
                    return Ok(());
                }
            }
        }
        self.drain_westeros()
    }

    /// Resolve the Westeros cards still waiting this round.
    ///
    /// A card may hand control to the bidding step, to a Wildling attack, or to
    /// an Influence token's holder. The unresolved cards stay in
    /// `pending_westeros` and are drained once that step finishes, because all
    /// three cards must still resolve.
    pub(crate) fn drain_westeros(&mut self) -> Result<(), EngineError> {
        while let Some(card_id) = self
            .state
            .pending_westeros
            .as_ref()
            .and_then(|queue| queue.first().copied())
        {
            let card = self.data.westeros.card(card_id).clone();
            let effect = card["effect"].as_str().unwrap_or("none").to_string();

            if effect == "influence_decision" {
                let holder = self.westeros_choice_holder(&card);
                let options: SmallVec<[u16; 3]> = card["options"]
                    .as_array()
                    .map(|list| {
                        list.iter()
                            .filter_map(|value| {
                                value.as_str().and_then(|id| {
                                    // "none" is a real option (the holder may
                                    // choose no effect); it has no card row in
                                    // the data, so it gets the sentinel.
                                    if id == "none" {
                                        Some(crate::state::NONE_OPTION)
                                    } else {
                                        self.data.westeros.card_index(id)
                                    }
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let usable = holder
                    .map(|house| {
                        !self
                            .state
                            .house(house)
                            .map(|state| state.eliminated)
                            .unwrap_or(true)
                    })
                    .unwrap_or(false);
                if !usable || options.is_empty() {
                    let name = card["name"].as_str().unwrap_or("card").to_string();
                    self.state
                        .note(format!("{name}: holder cannot choose; no effect"));
                    self.pop_pending_westeros();
                    continue;
                }
                let holder = holder.expect("checked above");
                self.pop_pending_westeros();
                self.state.westeros_choice = Some(WesterosChoice {
                    card: card_id,
                    holder,
                    options,
                });
                self.state.phase = Phase::WesterosChoice;
                return Ok(());
            }

            self.pop_pending_westeros();
            if self.apply_westeros_effect(card_id)? {
                return Ok(());
            }
        }

        self.state.pending_westeros = None;
        self.state.bid_tracks.clear();
        if self.any_mustering_points() {
            self.state.phase = Phase::Mustering;
        } else {
            self.state.phase = Phase::Planning;
        }
        Ok(())
    }

    fn pop_pending_westeros(&mut self) {
        if let Some(queue) = self.state.pending_westeros.as_mut() {
            if !queue.is_empty() {
                queue.remove(0);
            }
        }
    }

    /// Resolve one Westeros card's effect.
    ///
    /// Returns `true` when control passes to another phase; the caller must then
    /// stop draining.
    pub(crate) fn apply_westeros_effect(&mut self, card_id: u16) -> Result<bool, EngineError> {
        let card = self.data.westeros.card(card_id).clone();
        let effect = card["effect"].as_str().unwrap_or("none");
        let name = card["name"].as_str().unwrap_or("card").to_string();
        match effect {
            "mustering" => {
                self.grant_mustering_points();
            }
            "supply" => {
                self.recalculate_supply();
            }
            "gain_power_per_crown" => {
                self.gain_power_per_crown();
            }
            "bid_all_tracks" => {
                self.state.bid_tracks =
                    SmallVec::from_slice(&[Track::IronThrone, Track::Fiefdoms, Track::KingsCourt]);
                self.state.pending_bids.clear();
                self.state.phase = Phase::Bidding;
                return Ok(true);
            }
            "wildlings_attack" => {
                self.resolve_wildlings()?;
                if self.state.phase == Phase::WildlingAttack {
                    return Ok(true);
                }
            }
            "order_restriction" => {
                if let Some(restricted) = card["restricted_order"].as_str() {
                    if let Some(order) = crate::enums::OrderKey::from_name(restricted) {
                        self.state
                            .order_restrictions
                            .push(crate::state::Restriction::Key(order));
                    } else if let Some(order_type) = crate::enums::OrderType::from_name(restricted)
                    {
                        self.state
                            .order_restrictions
                            .push(crate::state::Restriction::Type(order_type));
                    }
                    self.state
                        .note(format!("{name}: {restricted} orders banned this round"));
                }
            }
            "none" => self.state.note(format!("{name}: nothing happens")),
            other => self.state.note(format!("{name}: unknown effect {other:?}")),
        }
        Ok(false)
    }

    /// The house that owns the Influence token named by a choice card.
    fn westeros_choice_holder(&self, card: &serde_json::Value) -> Option<HouseId> {
        match card["holder"].as_str()? {
            "iron_throne" => self.state.first_player(),
            "raven" => self.state.raven_holder(),
            "blade" => self.state.blade_holder(),
            _ => None,
        }
    }

    pub(crate) fn westeros_choice_actions(&self, house: HouseId) -> Vec<Action> {
        let Some(choice) = self.state.westeros_choice.as_ref() else {
            return Vec::new();
        };
        if choice.holder != house {
            return Vec::new();
        }
        choice
            .options
            .iter()
            .map(|card| Action::westeros_choice(house, CardId(*card as u8)))
            .collect()
    }

    pub(crate) fn apply_westeros_choice(&mut self, action: Action) -> Result<(), EngineError> {
        let Some(choice) = self.state.westeros_choice.take() else {
            return Ok(());
        };
        let picked = action
            .card
            .map(|card| card.0 as u16)
            .filter(|card| choice.options.contains(card))
            // The "none" option is carried as a sentinel, exactly like the
            // Python action's literal `card="none"`.
            .filter(|card| *card != crate::state::NONE_OPTION);
        let card_name = self.data.westeros.card(choice.card)["name"]
            .as_str()
            .unwrap_or("card")
            .to_string();
        match picked {
            None => self
                .state
                .note(format!("{card_name}: holder chooses no effect")),
            Some(picked) => {
                let picked_name = self.data.westeros.card(picked)["name"]
                    .as_str()
                    .unwrap_or("card")
                    .to_string();
                self.state
                    .note(format!("{card_name}: holder chooses {picked_name}"));
                self.apply_westeros_effect(picked)?;
            }
        }
        if self.state.phase == Phase::WesterosChoice {
            self.state.phase = Phase::Westeros;
        }
        Ok(())
    }

    fn any_mustering_points(&self) -> bool {
        self.state
            .region_ids()
            .any(|region| self.state.region(region).mustering_points > 0)
    }

    /// Advance the Wildling threat token and resolve an attack at the cap.
    pub(crate) fn advance_wildlings(&mut self, amount: u8) -> Result<(), EngineError> {
        let track_max = self.data.wildlings.track_max;
        self.state.wildling_threat = (self.state.wildling_threat + amount).min(track_max);
        if self.state.wildling_threat >= track_max {
            self.state.note(format!(
                "wildling threat reached {track_max}; the wildlings attack"
            ));
            self.resolve_wildlings()?;
        }
        Ok(())
    }

    /// Draw one Westeros card, reshuffling the deck when Winter is Coming.
    fn draw_westeros(&mut self, deck_index: usize) -> Option<u16> {
        for _ in 0..20 {
            let (position, len) = {
                let deck = &self.state.westeros_decks[deck_index];
                (deck.position, deck.cards.len())
            };
            if position >= len {
                return None;
            }
            let card_id = self.state.westeros_decks[deck_index].cards[position];
            self.state.westeros_decks[deck_index].position = position + 1;
            let effect = self.data.westeros.card(card_id)["effect"]
                .as_str()
                .unwrap_or("none")
                .to_string();
            if effect == "winter_is_coming" {
                // "Immediately shuffle this deck. Then draw and resolve a new card."
                let mut remaining: SmallVec<[u16; 12]> = self.state.westeros_decks[deck_index]
                    .cards[position + 1..]
                    .iter()
                    .copied()
                    .collect();
                remaining.push(card_id);
                self.rng.shuffle(&mut remaining);
                self.state.westeros_decks[deck_index] = DeckState {
                    cards: remaining,
                    position: 0,
                    drawn: None,
                };
                self.state
                    .note("Winter is Coming: deck reshuffled and redrawn".to_string());
                continue;
            }
            return Some(card_id);
        }
        None
    }

    fn grant_mustering_points(&mut self) {
        let houses = self.state.present_houses();
        for house in houses {
            if self
                .state
                .house(house)
                .map(|state| state.eliminated)
                .unwrap_or(true)
            {
                continue;
            }
            for region in self.state.controlled_regions(house) {
                let points = self.data.map.region(region).mustering;
                if points > 0 {
                    self.state.region_mut(region).mustering_points += points;
                }
            }
        }
        self.state
            .note("mustering points granted per area".to_string());
    }

    fn recalculate_supply(&mut self) {
        let houses = self.state.present_houses();
        for house in houses {
            if self
                .state
                .house(house)
                .map(|state| state.eliminated)
                .unwrap_or(true)
            {
                continue;
            }
            let supply = self.supply_from_regions(house);
            let ceiling = (SUPPLY_LIMITS.len() - 1) as u8;
            if let Some(state) = self.state.house_mut(house) {
                state.supply_level = supply.min(ceiling);
            }
            self.enforce_supply(house);
        }
        self.state.note("supply recalculated".to_string());
    }

    /// Supply a house draws from the areas it controls.
    pub(crate) fn supply_from_regions(&self, house: HouseId) -> u8 {
        self.state
            .controlled_regions(house)
            .iter()
            .map(|region| self.data.map.region(*region).supply)
            .sum()
    }

    /// Disband units until a house is within its supply limits.
    ///
    /// The real game asks the player which units to remove. Here the engine
    /// removes from the largest army, cheapest unit first, and will remove
    /// routed units as well (they still count towards Supply).
    pub(crate) fn enforce_supply(&mut self, house: HouseId) {
        let mut guard = 0;
        while !self.state.within_supply(house) && guard < 200 {
            guard += 1;
            let mut best: Option<(u32, RegionId)> = None;
            for region in self.state.controlled_regions(house) {
                let total = self.state.region(region).total(true);
                if total < 2 {
                    continue;
                }
                // Python's `max((size, region_id))` breaks ties on the id string.
                let key = (total, self.data.map.id_of(region).to_string());
                match &best {
                    None => best = Some((total, region)),
                    Some((best_total, best_region)) => {
                        let best_key = (*best_total, self.data.map.id_of(*best_region).to_string());
                        if key > best_key {
                            best = Some((total, region));
                        }
                    }
                }
            }
            let Some((_, region)) = best else {
                break;
            };
            if !self.remove_one_unit(region) {
                break;
            }
        }
    }

    /// Remove one unit, cheapest first, and settle control if the area empties.
    pub(crate) fn remove_one_unit(&mut self, region: RegionId) -> bool {
        let region_state = self.state.region_mut(region);
        let mut removed = false;
        for unit in [
            UnitType::Footman,
            UnitType::Siege,
            UnitType::Knight,
            UnitType::Ship,
        ] {
            if region_state.routed_count(unit) > 0 {
                region_state.routed[unit.index()] -= 1;
                removed = true;
                break;
            }
            if region_state.unit_count(unit) > 0 {
                region_state.units[unit.index()] -= 1;
                removed = true;
                break;
            }
        }
        if !removed {
            return false;
        }
        region_state.settle_control();
        true
    }

    fn gain_power_per_crown(&mut self) {
        let houses = self.state.present_houses();
        for house in houses {
            if self
                .state
                .house(house)
                .map(|state| state.eliminated)
                .unwrap_or(true)
            {
                continue;
            }
            let mut gained: u16 = self
                .state
                .controlled_regions(house)
                .iter()
                .map(|region| self.data.map.region(*region).power as u16)
                .sum();
            gained += self.port_power_bonus(house) as u16;
            if let Some(state) = self.state.house_mut(house) {
                state.power = state.power.saturating_add(gained.min(u8::MAX as u16) as u8);
            }
        }
        self.state.note("power gained per crown icon".to_string());
    }

    /// Extra Power from the "Game of Thrones" card.
    ///
    /// Each controlled Port containing at least one of the house's Ships grants
    /// one Power token, as long as the connected sea holds no enemy Ships.
    fn port_power_bonus(&self, house: HouseId) -> u8 {
        let mut bonus = 0u8;
        for (port_id, _host) in self.data.map.port_hosts.iter().copied() {
            let port = self.state.region(port_id);
            let own_ships = port.unit_count(UnitType::Ship) + port.routed_count(UnitType::Ship);
            if port.controller != Some(house) || own_ships == 0 {
                continue;
            }
            if let Some(sea_id) = self.data.map.region(port_id).sea {
                let sea = self.state.region(sea_id);
                let enemy_ships = sea.unit_count(UnitType::Ship) + sea.routed_count(UnitType::Ship);
                if sea.controller.is_some() && sea.controller != Some(house) && enemy_ships > 0 {
                    continue;
                }
            }
            bonus += 1;
        }
        bonus
    }

    // ==================================================================
    // Bidding phase
    // ==================================================================

    pub(crate) fn next_bidder(&self) -> Option<HouseId> {
        for house in self.state.track_order(Track::IronThrone) {
            if self
                .state
                .house(*house)
                .map(|state| state.eliminated)
                .unwrap_or(true)
            {
                continue;
            }
            if !self
                .state
                .pending_bids
                .iter()
                .any(|(bidder, _)| bidder == house)
            {
                return Some(*house);
            }
        }
        None
    }

    pub(crate) fn bidding_actions(&self, house: HouseId) -> Vec<Action> {
        let power = self
            .state
            .house(house)
            .map(|state| state.power)
            .unwrap_or(0);
        (0..=power)
            .map(|amount| Action::bid(house, amount))
            .collect()
    }

    pub(crate) fn apply_bid(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("bid without a house".into()))?;
        let power = self
            .state
            .house(house)
            .map(|state| state.power)
            .unwrap_or(0);
        let amount = action.amount.unwrap_or(0).min(power);
        if self.state.phase == Phase::WildlingAttack {
            self.apply_wildling_bid(house, amount);
        } else {
            match self
                .state
                .pending_bids
                .iter_mut()
                .find(|(bidder, _)| *bidder == house)
            {
                Some(entry) => entry.1 = amount,
                None => self.state.pending_bids.push((house, amount)),
            }
        }
        Ok(())
    }

    /// Resolve bidding for the next Influence track only.
    ///
    /// Bidding is sequential (Iron Throne, then Fiefdoms, then King's Court);
    /// all power bid is discarded, so later tracks are bid with what remains.
    /// Ties are broken by the current Iron Throne position.
    pub(crate) fn resolve_bids(&mut self) {
        if self.state.bid_tracks.is_empty() {
            self.resume_after_extra_step();
            return;
        }
        let track = self.state.bid_tracks.remove(0);
        let bids: SmallVec<[(HouseId, u8); 6]> = self.state.pending_bids.clone();
        for (house, amount) in &bids {
            if let Some(state) = self.state.house_mut(*house) {
                state.power = state.power.saturating_sub(*amount);
            }
        }
        let throne = self.state.tracks[Track::IronThrone.index()].clone();
        let current = self.state.tracks[track.index()].clone();
        let mut ordered: SmallVec<[HouseId; 6]> = current
            .iter()
            .copied()
            .filter(|house| {
                !self
                    .state
                    .house(*house)
                    .map(|state| state.eliminated)
                    .unwrap_or(true)
            })
            .collect();
        let throne_rank = |house: HouseId| {
            throne
                .iter()
                .position(|candidate| *candidate == house)
                .unwrap_or(99)
        };
        let previous = |house: HouseId| {
            current
                .iter()
                .position(|candidate| *candidate == house)
                .unwrap_or(0)
        };
        let bid_of = |house: HouseId| {
            bids.iter()
                .find(|(bidder, _)| *bidder == house)
                .map(|(_, amount)| *amount)
                .unwrap_or(0)
        };
        // Stable sort on the Python key: most power first, then Iron Throne, then
        // the house's previous position on this track.
        ordered.sort_by_key(|house| {
            (
                std::cmp::Reverse(bid_of(*house)),
                throne_rank(*house),
                previous(*house),
            )
        });
        for house in current.iter().copied() {
            if self
                .state
                .house(house)
                .map(|state| state.eliminated)
                .unwrap_or(false)
            {
                ordered.push(house);
            }
        }
        self.state.tracks[track.index()] = ordered;
        self.state
            .note(format!("bids resolved for {} ({bids:?})", track.as_str()));
        self.state.pending_bids.clear();
        if self.state.bid_tracks.is_empty() {
            self.resume_after_extra_step();
        }
    }

    /// Return to an interrupted Westeros phase, or move on if it is done.
    pub(crate) fn resume_after_extra_step(&mut self) {
        if self.state.pending_westeros.is_some() {
            self.state.phase = Phase::Westeros;
            return;
        }
        self.state.pending_westeros = None;
        if self.any_mustering_points() {
            self.state.phase = Phase::Mustering;
        } else {
            self.state.phase = Phase::Planning;
        }
    }

    /// Move a house to the top or bottom of an Influence track.
    pub(crate) fn move_influence(&mut self, track: Track, house: HouseId, to_top: bool) {
        let current = self.state.tracks[track.index()].clone();
        let rest: SmallVec<[HouseId; 6]> = current
            .iter()
            .copied()
            .filter(|candidate| *candidate != house)
            .collect();
        let alive: SmallVec<[HouseId; 6]> = rest
            .iter()
            .copied()
            .filter(|candidate| {
                !self
                    .state
                    .house(*candidate)
                    .map(|state| state.eliminated)
                    .unwrap_or(true)
            })
            .collect();
        let dead: SmallVec<[HouseId; 6]> = rest
            .iter()
            .copied()
            .filter(|candidate| {
                self.state
                    .house(*candidate)
                    .map(|state| state.eliminated)
                    .unwrap_or(false)
            })
            .collect();
        let mut next: SmallVec<[HouseId; 6]> = SmallVec::new();
        if to_top {
            next.push(house);
        }
        next.extend(alive);
        if !to_top {
            next.push(house);
        }
        next.extend(dead);
        self.state.tracks[track.index()] = next;
        let direction = if to_top { "top" } else { "bottom" };
        self.state.note(format!(
            "{house} moves to the {direction} of {}",
            track.as_str()
        ));
    }

    // ==================================================================
    // Mustering phase
    // ==================================================================

    pub(crate) fn next_musterer(&self) -> Option<HouseId> {
        for house in self.state.track_order(Track::IronThrone) {
            if self
                .state
                .house(*house)
                .map(|state| state.eliminated)
                .unwrap_or(true)
            {
                continue;
            }
            if self.state.has_mustering_points(*house) {
                return Some(*house);
            }
        }
        None
    }

    pub(crate) fn mustering_actions(&self, house: HouseId) -> Vec<Action> {
        let mut actions: Vec<Action> = Vec::new();
        let profile = self.supply_profile(house);
        for region in self.state.controlled_regions(house) {
            let static_region = self.data.map.region(region);
            let points = self.state.region(region).mustering_points;
            if points == 0 || !static_region.is_fortification() {
                continue;
            }
            if points >= MUSTER_COST[UnitType::Footman.index()]
                && self.muster_fits(house, region, UnitType::Footman, 1, Some(&profile))
            {
                actions.push(Action::muster(house, region, UnitType::Footman, None));
            }
            if points >= MUSTER_COST[UnitType::Knight.index()]
                && self.muster_fits(house, region, UnitType::Knight, 1, Some(&profile))
            {
                actions.push(Action::muster(house, region, UnitType::Knight, None));
            }
            if points >= MUSTER_COST[UnitType::Siege.index()]
                && self.muster_fits(house, region, UnitType::Siege, 1, Some(&profile))
            {
                actions.push(Action::muster(house, region, UnitType::Siege, None));
            }
            if points >= 1 && self.state.region(region).unit_count(UnitType::Footman) > 0 {
                actions.push(Action::upgrade(house, region, UnitType::Knight));
                actions.push(Action::upgrade(house, region, UnitType::Siege));
            }
            if points >= MUSTER_COST[UnitType::Ship.index()] {
                for destination in self.ship_muster_targets(house, region) {
                    if self.muster_fits(house, destination, UnitType::Ship, 1, Some(&profile)) {
                        actions.push(Action::muster(
                            house,
                            region,
                            UnitType::Ship,
                            Some(destination),
                        ));
                    }
                }
            }
        }
        actions.push(Action::pass(house));
        actions
    }

    /// Port or adjacent sea areas where a Ship may be mustered.
    fn ship_muster_targets(&self, house: HouseId, source: RegionId) -> SmallVec<[RegionId; 8]> {
        let mut targets: SmallVec<[RegionId; 8]> = SmallVec::new();
        if let Some(port_id) = self.data.map.port_of(source) {
            let port = self.state.region(port_id);
            if (port.controller.is_none() || port.controller == Some(house)) && port.total(true) < 3
            {
                targets.push(port_id);
            }
        }
        let mut seas: SmallVec<[RegionId; 8]> = self.data.map.sea_neighbours_of(source);
        // Python sorts these as strings; keep that order.
        let map = &self.data.map;
        seas.sort_unstable_by_key(|sea| map.id_of(*sea));
        for sea in seas {
            let region = self.state.region(sea);
            // Friendly and empty sea areas may receive a new Ship; only enemy
            // Ships block mustering there.
            if region.controller.is_some() && region.controller != Some(house) {
                continue;
            }
            targets.push(sea);
        }
        targets
    }

    /// Would adding `count` of `unit` in `destination` stay in Supply?
    pub(crate) fn muster_fits(
        &self,
        house: HouseId,
        destination: RegionId,
        _unit: UnitType,
        count: u8,
        profile: Option<&crate::action_phase::SupplyProfile>,
    ) -> bool {
        let owned;
        let profile = match profile {
            Some(profile) => profile,
            None => {
                owned = self.supply_profile(house);
                &owned
            }
        };
        let mut sizes: SmallVec<[u32; 8]> = SmallVec::new();
        let mut found = false;
        for (region, base) in profile.controlled_sizes() {
            let mut size = *base;
            if *region == destination {
                size += count as u32;
                found = true;
            }
            if size >= 2 {
                sizes.push(size);
            }
        }
        if !found && count >= 2 {
            sizes.push(count as u32);
        }
        sizes.sort_unstable_by(|left, right| right.cmp(left));
        sizes.iter().enumerate().all(|(index, size)| {
            index < profile.limit_count() && *size <= profile.limit(index) as u32
        })
    }

    pub(crate) fn apply_muster(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("muster without a house".into()))?;
        let source = action
            .region
            .ok_or_else(|| EngineError::IllegalAction("muster without a region".into()))?;
        let unit = action
            .unit
            .ok_or_else(|| EngineError::IllegalAction("muster without a unit".into()))?;
        let cost = MUSTER_COST[unit.index()];
        let destination = if unit == UnitType::Ship {
            action.target.unwrap_or(source)
        } else {
            source
        };
        {
            let placement = self.state.region_mut(destination);
            placement.units[unit.index()] += 1;
            placement.controller = Some(house);
        }
        self.state.region_mut(source).mustering_points = self
            .state
            .region(source)
            .mustering_points
            .saturating_sub(cost);
        let destination_text = self.data.map.id_of(destination).to_string();
        self.state
            .note(format!("{house} musters {unit} in {destination_text}"));
        Ok(())
    }

    pub(crate) fn apply_upgrade(&mut self, action: Action) -> Result<(), EngineError> {
        let house = action
            .house
            .ok_or_else(|| EngineError::IllegalAction("upgrade without a house".into()))?;
        let region = action
            .region
            .ok_or_else(|| EngineError::IllegalAction("upgrade without a region".into()))?;
        let unit = action
            .unit
            .ok_or_else(|| EngineError::IllegalAction("upgrade without a unit".into()))?;
        {
            let state = self.state.region_mut(region);
            state.units[UnitType::Footman.index()] =
                state.units[UnitType::Footman.index()].saturating_sub(1);
            state.units[unit.index()] += 1;
            state.mustering_points = state.mustering_points.saturating_sub(1);
        }
        let region_text = self.data.map.id_of(region).to_string();
        self.state.note(format!(
            "{house} upgrades a footman to {unit} in {region_text}"
        ));
        Ok(())
    }

    // ==================================================================
    // Wildling attacks
    // ==================================================================

    /// Start a Wildling attack whose strength is the current threat.
    pub(crate) fn resolve_wildlings(&mut self) -> Result<(), EngineError> {
        let strength = self.state.wildling_threat;
        self.setup_wildling_attack(strength, SmallVec::new());
        Ok(())
    }

    fn draw_wildling(&mut self) -> Option<u16> {
        if self.state.wildling_deck.cards.is_empty() {
            if self.state.wildling_discard.is_empty() {
                return None;
            }
            let mut deck: SmallVec<[u16; 12]> =
                self.state.wildling_discard.iter().copied().collect();
            self.state.wildling_discard.clear();
            self.rng.shuffle(&mut deck);
            self.state.wildling_deck = DeckState {
                cards: deck,
                position: 0,
                drawn: None,
            };
        }
        if self.state.wildling_deck.cards.is_empty() {
            return None;
        }
        Some(self.state.wildling_deck.cards.remove(0))
    }

    pub(crate) fn setup_wildling_attack(&mut self, strength: u8, excluded: SmallVec<[HouseId; 6]>) {
        let card = self.draw_wildling();
        self.state.wildling_attack = Some(WildlingAttack {
            strength,
            bids: SmallVec::new(),
            excluded,
            card,
            return_phase: Phase::Westeros,
            second_attack: None,
            second_strength: 6,
        });
        self.state
            .note(format!("wildlings attack with strength {strength}"));
        self.state.phase = Phase::WildlingAttack;
    }

    pub(crate) fn next_wildling_bidder(&self) -> Option<HouseId> {
        let attack = self.state.wildling_attack.as_ref()?;
        for house in self.state.track_order(Track::IronThrone) {
            if self
                .state
                .house(*house)
                .map(|state| state.eliminated)
                .unwrap_or(true)
            {
                continue;
            }
            if attack.excluded.contains(house) {
                continue;
            }
            if attack.bid_of(*house).is_none() {
                return Some(*house);
            }
        }
        None
    }

    pub(crate) fn wildling_bid_actions(&self, house: HouseId) -> Vec<Action> {
        let power = self
            .state
            .house(house)
            .map(|state| state.power)
            .unwrap_or(0);
        (0..=power)
            .map(|amount| Action::bid(house, amount))
            .collect()
    }

    pub(crate) fn apply_wildling_bid(&mut self, house: HouseId, amount: u8) {
        let power = self
            .state
            .house(house)
            .map(|state| state.power)
            .unwrap_or(0);
        if let Some(attack) = self.state.wildling_attack.as_mut() {
            attack.set_bid(house, amount.min(power));
        }
    }

    /// Compare the Night's Watch and Wildling strengths, then the card.
    pub(crate) fn resolve_wildling_attack(&mut self) -> Result<(), EngineError> {
        let Some(mut attack) = self.state.wildling_attack.take() else {
            self.state.phase = Phase::Westeros;
            return Ok(());
        };
        // All bid Power is discarded, win or lose.
        for (house, amount) in &attack.bids {
            if let Some(state) = self.state.house_mut(*house) {
                state.power = state.power.saturating_sub(*amount);
            }
        }
        let nights_watch: u32 = attack.bids.iter().map(|(_, amount)| *amount as u32).sum();
        let wildlings_win = nights_watch < attack.strength as u32;
        self.state.wildling_threat = if wildlings_win {
            attack.strength.saturating_sub(2)
        } else {
            0
        };
        self.state.note(format!(
            "wildling attack: Night's Watch {nights_watch} vs wildlings {}; {}",
            attack.strength,
            if wildlings_win {
                "wildlings win"
            } else {
                "Night watch holds"
            }
        ));

        if let Some(card_id) = attack.card {
            let participants: SmallVec<[HouseId; 6]> = self
                .state
                .track_order(Track::IronThrone)
                .iter()
                .copied()
                .filter(|house| attack.bid_of(*house).is_some())
                .collect();
            let card = self.data.wildlings.card(card_id).clone();
            let branch = if wildlings_win {
                "wildling_victory"
            } else {
                "nights_watch_victory"
            };
            let spec = card[branch].clone();
            let chosen = self.extreme_bidder(&participants, &attack, !wildlings_win);
            let others: SmallVec<[HouseId; 6]> = participants
                .iter()
                .copied()
                .filter(|house| Some(*house) != chosen)
                .collect();
            // The effect may amend *this* attack (Preemptive Raid adds a second
            // one); Python mutates the resolved card's local dict the same way.
            self.apply_wildling_effect(&spec, chosen, &others, &mut attack)?;
            self.state.wildling_discard.push(card_id);
            let name = card["name"].as_str().unwrap_or("card").to_string();
            let effect = spec["effect"].as_str().unwrap_or("nothing").to_string();
            self.state.note(format!(
                "{name} ({branch}): {effect} for {}",
                chosen
                    .map(|house| house.name().to_string())
                    .unwrap_or("nobody".into())
            ));
        }

        if let Some(second) = attack.second_attack.clone() {
            // "Preemptive Raid": the wildlings attack again, and the highest
            // bidder sits the second attack out.
            self.setup_wildling_attack(attack.second_strength, second);
            return Ok(());
        }
        self.state.phase = attack.return_phase;
        Ok(())
    }

    /// The highest (or lowest) bidder, breaking ties by Iron Throne seat.
    fn extreme_bidder(
        &self,
        participants: &[HouseId],
        attack: &WildlingAttack,
        highest: bool,
    ) -> Option<HouseId> {
        if participants.is_empty() {
            return None;
        }
        let value = |house: HouseId| attack.bid_of(house).unwrap_or(0);
        let target = if highest {
            participants.iter().map(|house| value(*house)).max()?
        } else {
            participants.iter().map(|house| value(*house)).min()?
        };
        let throne = self.state.track_order(Track::IronThrone);
        participants
            .iter()
            .copied()
            .filter(|house| value(*house) == target)
            .min_by_key(|house| {
                throne
                    .iter()
                    .position(|candidate| candidate == house)
                    .unwrap_or(99)
            })
    }

    /// Apply one Wildling card consequence.
    ///
    /// The printed cards let the rewarded player make several sub-choices; the
    /// engine resolves them with a deterministic default (largest army first,
    /// strongest House card, best Iron Throne seat).
    fn apply_wildling_effect(
        &mut self,
        spec: &serde_json::Value,
        chosen: Option<HouseId>,
        others: &[HouseId],
        attack: &mut WildlingAttack,
    ) -> Result<(), EngineError> {
        let effect = spec["effect"].as_str().unwrap_or("nothing");
        let Some(chosen) = chosen else {
            return Ok(());
        };
        if effect == "nothing" {
            return Ok(());
        }
        match effect {
            "replace_knights" => {
                let chosen_count = spec["chosen"].as_u64().map(|value| value as u8);
                self.replace_knights(chosen, chosen_count);
                let others_count = spec["others"].as_u64().map(|value| value as u8);
                for house in others {
                    self.replace_knights(*house, others_count);
                }
            }
            "upgrade_footmen" => {
                let count = spec["chosen"].as_u64().unwrap_or(0) as u8;
                self.upgrade_footmen(chosen, count);
            }
            "destroy_units" => {
                let count = spec["chosen"].as_u64().unwrap_or(0) as u8;
                self.destroy_units_anywhere(chosen, count);
                let others_count = spec["others"].as_u64().unwrap_or(0) as u8;
                for house in others {
                    self.destroy_units_anywhere(*house, others_count);
                }
            }
            "destroy_units_at_fort" => {
                let count = spec["chosen"].as_u64().unwrap_or(0) as u8;
                if !self.destroy_units_at_fortification(chosen, count) {
                    self.destroy_units_anywhere(chosen, count);
                }
                let others_count = spec["others"].as_u64().unwrap_or(0) as u8;
                for house in others {
                    self.destroy_units_anywhere(*house, others_count);
                }
            }
            "muster_at_fort" => self.auto_muster_at_best_fort(chosen),
            "influence_to_bottom" => {
                for name in influence_targets(&spec["chosen"]) {
                    if let Some(track) = Track::from_name(name) {
                        self.move_influence(track, chosen, false);
                    }
                }
                for house in others {
                    for name in influence_targets(&spec["others"]) {
                        if let Some(track) = Track::from_name(name) {
                            self.move_influence(track, *house, false);
                        }
                    }
                }
            }
            "influence_to_top" => {
                for name in influence_targets(&spec["chosen"]) {
                    if let Some(track) = Track::from_name(name) {
                        self.move_influence(track, chosen, true);
                    }
                }
            }
            "retrieve_house_card" => self.retrieve_house_card(chosen),
            "return_discard_pile" => self.return_discard_pile(chosen),
            "discard_house_cards" => {
                let mode = spec["chosen"].as_str().unwrap_or("one");
                self.discard_house_cards(chosen, mode);
                let others_mode = spec["others"].as_str().unwrap_or("one");
                for house in others {
                    self.discard_house_cards(*house, others_mode);
                }
            }
            "adjust_supply" => {
                let delta = spec["chosen"].as_i64().unwrap_or(0) as i16;
                self.adjust_supply(chosen, delta);
                let others_delta = spec["others"].as_i64().unwrap_or(0) as i16;
                for house in others {
                    self.adjust_supply(*house, others_delta);
                }
            }
            "discard_power" => {
                let amount = spec["chosen"].as_u64().unwrap_or(0) as u8;
                self.discard_power(chosen, amount);
                let others_amount = spec["others"].as_u64().unwrap_or(0) as u8;
                for house in others {
                    self.discard_power(*house, others_amount);
                }
            }
            "second_attack" => {
                let strength = spec["strength"].as_u64().unwrap_or(6) as u8;
                attack.second_attack = Some(SmallVec::from_slice(&[chosen]));
                attack.second_strength = strength;
            }
            "refund_bid" => {
                let refund = attack.bid_of(chosen).unwrap_or(0);
                if let Some(state) = self.state.house_mut(chosen) {
                    state.power = state.power.saturating_add(refund);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Destroy up to `count` units, taking from the largest force first.
    fn destroy_units_anywhere(&mut self, house: HouseId, count: u8) -> u8 {
        let mut destroyed = 0u8;
        while destroyed < count {
            let mut best: Option<(u32, RegionId)> = None;
            for region in self.state.controlled_regions(house) {
                let total = self.state.region(region).total(true);
                if total == 0 {
                    continue;
                }
                let key = (total, self.data.map.id_of(region).to_string());
                match &best {
                    None => best = Some((total, region)),
                    Some((best_total, best_region)) => {
                        let best_key = (*best_total, self.data.map.id_of(*best_region).to_string());
                        if key > best_key {
                            best = Some((total, region));
                        }
                    }
                }
            }
            let Some((_, region)) = best else {
                break;
            };
            if !self.remove_one_unit(region) {
                break;
            }
            destroyed += 1;
        }
        destroyed
    }

    fn destroy_units_at_fortification(&mut self, house: HouseId, count: u8) -> bool {
        let mut best: Option<(u32, RegionId)> = None;
        for region in self.state.controlled_regions(house) {
            if !self.data.map.region(region).is_fortification() {
                continue;
            }
            let total = self.state.region(region).total(true);
            if total == 0 {
                continue;
            }
            // Python's `max(..., key=total)` keeps the first maximal element.
            match &best {
                None => best = Some((total, region)),
                Some((best_total, _)) if total > *best_total => best = Some((total, region)),
                _ => {}
            }
        }
        let Some((_, region)) = best else {
            return false;
        };
        for _ in 0..count {
            if !self.remove_one_unit(region) {
                break;
            }
        }
        true
    }

    /// Swap Knights for Footmen in place; unit-pool limits are not modelled.
    fn replace_knights(&mut self, house: HouseId, count: Option<u8>) {
        let mut remaining = count;
        let mut regions: SmallVec<[RegionId; 32]> = self.state.controlled_regions(house);
        // Largest force first; Python's sort is stable, so map order breaks ties.
        regions.sort_by_key(|region| std::cmp::Reverse(self.state.region(*region).total(true)));
        for region in regions {
            let knights = self.state.region(region).unit_count(UnitType::Knight);
            if knights == 0 {
                continue;
            }
            let take = match remaining {
                None => knights,
                Some(left) => knights.min(left),
            };
            if take == 0 {
                continue;
            }
            {
                let state = self.state.region_mut(region);
                state.units[UnitType::Knight.index()] -= take;
                state.units[UnitType::Footman.index()] += take;
            }
            if let Some(left) = remaining.as_mut() {
                *left = left.saturating_sub(take);
                if *left == 0 {
                    return;
                }
            }
        }
    }

    /// Turn up to `count` Footmen into Knights (no Supply change).
    fn upgrade_footmen(&mut self, house: HouseId, count: u8) {
        let mut remaining = count;
        let mut regions: SmallVec<[RegionId; 32]> = self.state.controlled_regions(house);
        regions.sort_by_key(|region| std::cmp::Reverse(self.state.region(*region).total(true)));
        for region in regions {
            while remaining > 0 && self.state.region(region).unit_count(UnitType::Footman) > 0 {
                {
                    let state = self.state.region_mut(region);
                    state.units[UnitType::Footman.index()] -= 1;
                    state.units[UnitType::Knight.index()] += 1;
                }
                remaining -= 1;
            }
            if remaining == 0 {
                return;
            }
        }
    }

    /// Muster automatically in the house's richest Castle or Stronghold.
    fn auto_muster_at_best_fort(&mut self, house: HouseId) {
        let mut best: Option<(u8, RegionId)> = None;
        for region in self.state.controlled_regions(house) {
            if !self.data.map.region(region).is_fortification() {
                continue;
            }
            let mustering = self.data.map.region(region).mustering;
            match &best {
                None => best = Some((mustering, region)),
                Some((best_mustering, _)) if mustering > *best_mustering => {
                    best = Some((mustering, region))
                }
                _ => {}
            }
        }
        let Some((_, region)) = best else {
            return;
        };
        let mut points = self.data.map.region(region).mustering;
        while points > 0 {
            if points >= MUSTER_COST[UnitType::Knight.index()]
                && self.muster_fits(house, region, UnitType::Knight, 1, None)
            {
                self.state.region_mut(region).units[UnitType::Knight.index()] += 1;
                points -= MUSTER_COST[UnitType::Knight.index()];
            } else if self.muster_fits(house, region, UnitType::Footman, 1, None) {
                self.state.region_mut(region).units[UnitType::Footman.index()] += 1;
                points -= MUSTER_COST[UnitType::Footman.index()];
            } else {
                break;
            }
        }
        let region_text = self.data.map.id_of(region).to_string();
        self.state.note(format!(
            "{house} musters at {region_text} from a Wildling reward"
        ));
    }

    fn retrieve_house_card(&mut self, house: HouseId) {
        let Some(state) = self.state.house_mut(house) else {
            return;
        };
        if state.house_cards_used.is_empty() {
            return;
        }
        let card = state.house_cards_used.pop().expect("checked non-empty");
        state.house_cards_available.push(card);
        let card_text = self.data.house_cards.card(house, card).id.clone();
        self.state.note(format!(
            "{house} retrieves House card {card_text} from the discard"
        ));
    }

    fn return_discard_pile(&mut self, house: HouseId) {
        let Some(state) = self.state.house_mut(house) else {
            return;
        };
        let used: SmallVec<[CardId; 7]> = state.house_cards_used.iter().copied().collect();
        state.house_cards_available.extend(used);
        state.house_cards_used.clear();
        self.state.note(format!(
            "{house} returns its whole House card discard to hand"
        ));
    }

    fn discard_house_cards(&mut self, house: HouseId, mode: &str) {
        let hand: SmallVec<[CardId; 7]> = self
            .state
            .house(house)
            .map(|state| state.house_cards_available.clone())
            .unwrap_or_default();
        if hand.len() <= 1 {
            return;
        }
        let strength = |card: CardId| self.data.house_cards.card(house, card).strength;
        let discarded: SmallVec<[CardId; 7]> = if mode == "all_max" {
            let best = hand.iter().map(|card| strength(*card)).max().unwrap_or(0);
            hand.iter()
                .copied()
                .filter(|card| strength(*card) == best)
                .collect()
        } else {
            // `max(hand, key=strength)` keeps the first maximal element.
            let mut best: Option<CardId> = None;
            for card in hand.iter().copied() {
                match best {
                    None => best = Some(card),
                    Some(current) if strength(card) > strength(current) => best = Some(card),
                    _ => {}
                }
            }
            best.into_iter().collect()
        };
        if let Some(state) = self.state.house_mut(house) {
            for card in &discarded {
                if let Some(position) = state
                    .house_cards_available
                    .iter()
                    .position(|candidate| candidate == card)
                {
                    state.house_cards_available.remove(position);
                }
                state.house_cards_used.push(*card);
            }
        }
        let names: Vec<&str> = discarded
            .iter()
            .map(|card| self.data.house_cards.card(house, *card).id.as_str())
            .collect();
        self.state
            .note(format!("{house} discards House card(s) {names:?}"));
    }

    fn adjust_supply(&mut self, house: HouseId, delta: i16) {
        let ceiling = (SUPPLY_LIMITS.len() - 1) as i16;
        if let Some(state) = self.state.house_mut(house) {
            let level = state.supply_level as i16 + delta;
            state.supply_level = level.clamp(0, ceiling) as u8;
        }
        if delta < 0 {
            self.enforce_supply(house);
        }
    }

    fn discard_power(&mut self, house: HouseId, amount: u8) {
        if let Some(state) = self.state.house_mut(house) {
            state.power = state.power.saturating_sub(amount);
        }
    }
}

/// The tracks a Wildling card names: `"all"`, one track, or nothing.
fn influence_targets(value: &serde_json::Value) -> SmallVec<[&'static str; 3]> {
    let mut targets: SmallVec<[&'static str; 3]> = SmallVec::new();
    match value {
        serde_json::Value::Null => {}
        serde_json::Value::String(text) if text == "none" => {}
        serde_json::Value::String(text) if text == "all" => {
            targets.push("iron_throne");
            targets.push("fiefdoms");
            targets.push("kings_court");
        }
        serde_json::Value::String(text) => {
            targets.push(match text.as_str() {
                "iron_throne" => "iron_throne",
                "fiefdoms" => "fiefdoms",
                "kings_court" => "kings_court",
                _ => return SmallVec::new(),
            });
        }
        _ => {}
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionKind;
    use crate::map::DataSet;
    use crate::setup::GameConfig;
    use std::sync::Arc;

    fn game(seed: u64) -> Game {
        let data = Arc::new(DataSet::load_default().expect("bundled data must load"));
        Game::new(data, GameConfig::with_seed(seed)).expect("game must start")
    }

    #[test]
    fn game_of_thrones_grants_power_per_crown_and_controlled_port() {
        let mut game = game(1);
        let house = HouseId(2);
        let card = game
            .data
            .westeros
            .card_index("game_of_thrones")
            .expect("the card exists");
        let expected: u16 = game
            .state
            .controlled_regions(house)
            .iter()
            .map(|region| game.data.map.region(*region).power as u16)
            .sum::<u16>()
            + game.port_power_bonus(house) as u16;
        game.state.house_mut(house).unwrap().power = 0;
        game.state.pending_westeros = Some(SmallVec::from_slice(&[card]));
        game.state.phase = Phase::Westeros;

        game.drain_westeros().unwrap();

        assert!(expected > 0, "the setup house controls a crown icon");
        assert_eq!(game.state.house(house).unwrap().power, expected as u8);
        assert!(game.state.pending_westeros.is_none());
    }

    #[test]
    fn cleanup_recovers_routed_units_and_returns_orders() {
        let mut game = game(1);
        let house = game.current_player().unwrap();
        let action = game.legal_actions()[0];
        let region = action.region.unwrap();
        game.step(action).unwrap();
        assert!(game.state.region(region).order.is_some());

        let winterfell = game.data.map.lookup("winterfell").unwrap();
        game.state.region_mut(winterfell).routed[UnitType::Footman.index()] = 1;
        let ready_before = game.state.region(winterfell).unit_count(UnitType::Footman);

        game.state.phase = Phase::Cleanup;
        game.run_cleanup_phase().unwrap();

        // Routed units recover, orders go back to the pool, the round advances.
        assert_eq!(
            game.state.region(winterfell).unit_count(UnitType::Footman),
            ready_before + 1
        );
        assert_eq!(
            game.state
                .region(winterfell)
                .routed_count(UnitType::Footman),
            0
        );
        assert!(game.state.orders_on_board(house).is_empty());
        assert_eq!(game.state.house(house).unwrap().orders_available.len(), 15);
        assert_eq!(game.state.phase, Phase::Westeros);
        assert_eq!(game.state.round, 2);
    }

    #[test]
    fn cleanup_ends_the_game_after_the_last_round() {
        let mut game = game(2);
        game.state.round = game.state.max_rounds;
        game.state.phase = Phase::Cleanup;
        game.run_cleanup_phase().unwrap();
        assert_eq!(game.state.phase, Phase::GameOver);
        assert!(game.state.winner.is_some());
    }

    #[test]
    fn bids_resolve_by_power_then_iron_throne() {
        let mut game = game(1);
        let houses = game.state.present_houses();
        let throne_before = game.state.tracks[Track::IronThrone.index()].clone();
        let seat = |house: HouseId| {
            throne_before
                .iter()
                .position(|candidate| *candidate == house)
                .unwrap()
        };
        let (early, late) = if seat(houses[0]) < seat(houses[1]) {
            (houses[0], houses[1])
        } else {
            (houses[1], houses[0])
        };

        game.state.phase = Phase::Bidding;
        game.state.bid_tracks = SmallVec::from_slice(&[Track::IronThrone]);
        game.state.pending_bids.clear();
        game.state.pending_bids.push((early, 3));
        game.state.pending_bids.push((late, 3));
        for house in houses.iter().skip(2) {
            game.state.pending_bids.push((*house, 0));
        }
        let power_before = game.state.house(early).unwrap().power;

        game.resolve_bids();

        let after = game.state.tracks[Track::IronThrone.index()].clone();
        assert_eq!(
            after[0], early,
            "the tied pair leads, better throne seat first"
        );
        assert_eq!(after[1], late);
        assert_eq!(
            game.state.house(early).unwrap().power,
            power_before - 3,
            "all power bid is discarded"
        );
        assert!(game.state.pending_bids.is_empty());
    }

    #[test]
    fn mustering_spends_points_and_places_the_unit() {
        let mut game = game(1);
        let house = HouseId(2); // stark, whose home is Winterfell
        let winterfell = game.data.map.lookup("winterfell").unwrap();
        game.state.region_mut(winterfell).mustering_points = 2;

        let actions = game.mustering_actions(house);
        let knight = actions
            .iter()
            .find(|action| {
                action.kind == ActionKind::Muster && action.unit == Some(UnitType::Knight)
            })
            .copied()
            .expect("two points buy a knight");
        let before = game.state.region(winterfell).unit_count(UnitType::Knight);
        game.apply_muster(knight).unwrap();
        assert_eq!(
            game.state.region(winterfell).unit_count(UnitType::Knight),
            before + 1
        );
        assert_eq!(game.state.region(winterfell).mustering_points, 0);
    }

    #[test]
    fn upgrades_trade_a_footman_for_a_knight_for_one_point() {
        let mut game = game(1);
        let house = HouseId(2);
        let winterfell = game.data.map.lookup("winterfell").unwrap();
        game.state.region_mut(winterfell).mustering_points = 1;
        let footmen = game.state.region(winterfell).unit_count(UnitType::Footman);
        assert!(footmen > 0, "Winterfell starts with a footman");
        let upgrade = Action::upgrade(house, winterfell, UnitType::Knight);
        game.apply_upgrade(upgrade).unwrap();
        assert_eq!(
            game.state.region(winterfell).unit_count(UnitType::Footman),
            footmen - 1
        );
        assert_eq!(game.state.region(winterfell).mustering_points, 0);
    }

    #[test]
    fn wildling_attack_discards_bids_and_resets_the_threat() {
        let mut game = game(1);
        // A single known card keeps the consequence deterministic.
        let card = game
            .data
            .wildlings
            .card_index("crow_killers")
            .expect("the card exists");
        game.state.wildling_deck = DeckState {
            cards: SmallVec::from_slice(&[card]),
            position: 0,
            drawn: None,
        };
        game.state.wildling_threat = 6;
        game.resolve_wildlings().unwrap();
        assert_eq!(game.state.phase, Phase::WildlingAttack);

        let houses = game.state.present_houses();
        let bidder = houses[0];
        let power_before = game.state.house(bidder).unwrap().power;
        game.apply_wildling_bid(bidder, power_before);
        for house in houses.iter().skip(1) {
            game.apply_wildling_bid(*house, 0);
        }
        game.resolve_wildling_attack().unwrap();

        // The Night's Watch fell short, so the wildlings win and the threat drops
        // by two; every bid Power token is discarded either way.
        assert_eq!(game.state.wildling_threat, 4);
        assert_eq!(game.state.house(bidder).unwrap().power, 0);
        assert_eq!(game.state.phase, Phase::Westeros);
        assert!(game.state.wildling_attack.is_none());
    }

    #[test]
    fn a_westeros_choice_asks_only_its_holder() {
        let mut game = game(1);
        let holder = HouseId(1);
        let option = game
            .data
            .westeros
            .card_index("mustering")
            .expect("the card exists");
        game.state.westeros_choice = Some(WesterosChoice {
            card: option,
            holder,
            options: SmallVec::from_slice(&[option]),
        });
        let actions = game.westeros_choice_actions(holder);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, ActionKind::WesterosChoice);
        assert!(game.westeros_choice_actions(HouseId(2)).is_empty());
    }

    #[test]
    fn influence_moves_put_a_house_first_or_last() {
        let mut game = game(1);
        let house = HouseId(4);
        game.move_influence(Track::KingsCourt, house, true);
        assert_eq!(game.state.tracks[Track::KingsCourt.index()][0], house);
        game.move_influence(Track::KingsCourt, house, false);
        let track = &game.state.tracks[Track::KingsCourt.index()];
        assert_eq!(*track.last().unwrap(), house);
    }
}
