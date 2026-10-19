//! Root determinization for imperfect-information search (PIMC).
//!
//! Port of `src/agot/engine/hidden.py`. The engine tracks the *true* state,
//! including the commitments a real player cannot see: Order tokens placed face
//! down and secret Power bids. Perfect-information MCTS reads all of them, which
//! is an information leak — the agent would learn to play against knowledge it
//! will never have at the table.
//!
//! [`determinize`] returns an independent copy of a game in which every hidden
//! variable the `observer` cannot see is resampled from a plausible prior.
//! Running MCTS over several determinizations and aggregating the root visit
//! counts (Perfect-Information Monte Carlo) yields a policy that reflects the
//! observer's actual information set.
//!
//! Resampled here
//! --------------
//!
//! * **Order tokens** placed by another house while they are face down (Planning
//!   and Messenger Raven steps). The sampled keys respect the public Order
//!   restrictions in force and the house's starred-order allowance, so the
//!   determinization is consistent with everything the observer does know.
//! * **Power bids** already cast by other houses, during both a *Clash of Kings*
//!   (bidding for a track) and a Wildling attack.
//!
//! Not resampled here
//! ------------------
//!
//! House-card hands are already private in the observation encoding: the network
//! only ever sees the acting house's own cards. The one remaining leak is that,
//! inside a battle, the player who chooses second can see the first player's
//! card in the search tree; the physical game reveals both simultaneously. That
//! is tracked in `docs/methods.md`.
//!
//! The source game is never mutated: the port clones it, then rewrites the
//! clone. With the same generator state the result is deterministic, which keeps
//! tests and seeded self-play reproducible.

use smallvec::{smallvec, SmallVec};

use crate::enums::{HouseId, OrderKey, OrderType, Phase, RegionId, N_HOUSES};
use crate::game::Game;
use crate::rng::Rng;
use crate::state::{Region, State};

/// Copy `game` with the hidden state resampled as `observer` sees it.
///
/// The original game is never mutated.
pub fn determinize(game: &Game, observer: HouseId, rng: &mut Rng) -> Game {
    let mut clone = game.clone();
    resample_orders(&mut clone.state, observer, rng);
    resample_bids(&mut clone.state, observer, rng);
    clone
}

/// Removes and returns the first pool key that could legally sit in `region`.
///
/// A starred token needs allowance left; a March token needs units to march.
/// Python's `_take_plausible`.
fn take_plausible(pool: &mut Vec<OrderKey>, region: &Region, stars_left: i32) -> Option<OrderKey> {
    for index in 0..pool.len() {
        let key = pool[index];
        if key.starred() && stars_left <= 0 {
            continue;
        }
        if key.order_type() == OrderType::March && !region.has_units() {
            continue;
        }
        pool.remove(index);
        return Some(key);
    }
    None
}

/// Resamples every face-down Order token the observer did not place.
///
/// Once the orders are revealed (any phase after the Messenger Raven step) they
/// are public and nothing is resampled.
fn resample_orders(state: &mut State, observer: HouseId, rng: &mut Rng) {
    if state.orders_revealed() {
        return;
    }

    // The hidden tokens, grouped by owner. Python groups into a dict whose
    // insertion order follows the region list, so the grouping order (and with
    // it the generator's draw order) follows the board.
    let mut hidden: SmallVec<[(HouseId, SmallVec<[RegionId; 16]>); N_HOUSES]> = SmallVec::new();
    for region_id in state.region_ids() {
        let region = state.region(region_id);
        let (Some(_), Some(owner)) = (region.order, region.order_house) else {
            continue;
        };
        if owner == observer {
            continue;
        }
        match hidden.iter_mut().find(|(house, _)| *house == owner) {
            Some((_, regions)) => regions.push(region_id),
            None => hidden.push((owner, smallvec![region_id])),
        }
    }
    if hidden.is_empty() {
        return;
    }

    for (house, region_ids) in hidden {
        if state.house(house).is_none() {
            continue;
        }
        // The token pool: everything the public restrictions leave available.
        let mut pool: Vec<OrderKey> = OrderKey::ALL
            .into_iter()
            .filter(|key| !state.has_order_restriction(*key))
            .collect();
        rng.shuffle(&mut pool);
        let mut stars_left = state.star_allowance(house) as i32;
        let mut placed: SmallVec<[OrderKey; 15]> = SmallVec::new();
        for region_id in region_ids {
            let region = *state.region(region_id);
            let pick = match take_plausible(&mut pool, &region, stars_left) {
                Some(key) => {
                    if key.starred() {
                        stars_left -= 1;
                    }
                    key
                }
                // No plausible token left (should not happen with 15 tokens):
                // keep whatever was there rather than leaving the region empty.
                None => region.order.unwrap_or(OrderKey::ALL[0]),
            };
            state.region_mut(region_id).order = Some(pick);
            placed.push(pick);
        }
        // Keep the token-pool bookkeeping consistent with the resampled board.
        if let Some(house_state) = state.house_mut(house) {
            house_state.orders_placed = placed.clone();
            house_state.orders_available = OrderKey::ALL
                .into_iter()
                .filter(|key| !placed.contains(key))
                .collect();
        }
    }
}

/// Resamples other houses' secret Power bids, uniform over what they can pay.
fn resample_bids(state: &mut State, observer: HouseId, rng: &mut Rng) {
    if state.phase == Phase::WildlingAttack {
        let Some(attack) = state.wildling_attack.as_mut() else {
            return;
        };
        // Clone the bid list so the house records can be read while it is
        // rewritten (the port keeps them in separate fields of one state).
        let bids: SmallVec<[(HouseId, u8); N_HOUSES]> = attack.bids.clone();
        for (house, _) in bids {
            if house == observer {
                continue;
            }
            let power = state.house(house).map(|record| record.power).unwrap_or(0);
            let amount = rng.range_inclusive(0, power as i64) as u8;
            if let Some(attack) = state.wildling_attack.as_mut() {
                attack.set_bid(house, amount);
            }
        }
        return;
    }

    if state.phase == Phase::Bidding {
        let bids: SmallVec<[(HouseId, u8); N_HOUSES]> = state.pending_bids.clone();
        for (house, _) in bids {
            if house == observer {
                continue;
            }
            let power = state.house(house).map(|record| record.power).unwrap_or(0);
            let amount = rng.range_inclusive(0, power as i64) as u8;
            if let Some(entry) = state
                .pending_bids
                .iter_mut()
                .find(|(bidder, _)| *bidder == house)
            {
                entry.1 = amount;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::UnitType;
    use crate::setup::GameConfig;
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use crate::map::DataSet;
    use crate::state::Restriction;

    fn data() -> Arc<DataSet> {
        Arc::new(DataSet::load_default().expect("bundled data must load"))
    }

    const OBSERVER: HouseId = HouseId(2); // stark
    const RIVAL: HouseId = HouseId(0); // baratheon

    /// A planning position with every Order token cleared off the map.
    fn planning() -> Game {
        let mut game = Game::new(data(), GameConfig::with_seed(0)).expect("game must start");
        game.state.phase = Phase::Planning;
        game.state.order_restrictions.clear();
        let region_ids: Vec<RegionId> = game.state.region_ids().collect();
        for region_id in region_ids {
            let region = game.state.region_mut(region_id);
            region.order = None;
            region.order_house = None;
        }
        game
    }

    /// Places an order token directly, the way `test_hidden.py` does.
    fn place(game: &mut Game, region_id: &str, house: HouseId, order: OrderKey) {
        let region = game.data.map.lookup(region_id).expect("known region");
        let slot = game.state.region_mut(region);
        slot.units[UnitType::Footman.index()] = 1;
        slot.order = Some(order);
        slot.order_house = Some(house);
        if let Some(record) = game.state.house_mut(house) {
            if !record.orders_placed.contains(&order) {
                record.orders_placed.push(order);
            }
        }
    }

    fn hidden_orders(game: &Game, house: HouseId) -> Vec<(RegionId, OrderKey)> {
        game.state.orders_on_board(house).into_iter().collect()
    }

    #[test]
    fn determinize_does_not_mutate_the_original() {
        let mut game = planning();
        place(&mut game, "winterfell", OBSERVER, OrderKey(3)); // defend_1a
        place(&mut game, "kings_landing", RIVAL, OrderKey(0)); // march

        let _ = determinize(&game, OBSERVER, &mut Rng::from_seed(0));

        assert_eq!(
            game.state
                .region(game.data.map.lookup("winterfell").unwrap())
                .order,
            Some(OrderKey(3))
        );
        assert_eq!(
            game.state
                .region(game.data.map.lookup("kings_landing").unwrap())
                .order,
            Some(OrderKey(0))
        );
    }

    #[test]
    fn the_observers_orders_are_kept_and_the_rivals_are_resampled() {
        let mut game = planning();
        place(&mut game, "winterfell", OBSERVER, OrderKey(6)); // support_a
        place(&mut game, "kings_landing", RIVAL, OrderKey(0)); // march

        let mut sampled = BTreeSet::new();
        for seed in 0..50 {
            let world = determinize(&game, OBSERVER, &mut Rng::from_seed(seed));
            assert_eq!(
                world
                    .state
                    .region(game.data.map.lookup("winterfell").unwrap())
                    .order,
                Some(OrderKey(6)),
                "the observer's own token never moves"
            );
            let rival = world
                .state
                .region(game.data.map.lookup("kings_landing").unwrap())
                .order
                .unwrap();
            sampled.insert(rival);
        }
        assert!(
            sampled.len() > 1,
            "the rival's token is genuinely resampled"
        );
    }

    #[test]
    fn hidden_orders_stay_within_the_valid_pool() {
        let mut game = planning();
        game.state
            .order_restrictions
            .push(Restriction::Type(OrderType::March));
        place(&mut game, "kings_landing", RIVAL, OrderKey(0)); // march
        place(&mut game, "dragonstone", RIVAL, OrderKey(2)); // march_plus
        place(&mut game, "storms_end", RIVAL, OrderKey(1)); // march_minus

        for seed in 0..50 {
            let world = determinize(&game, OBSERVER, &mut Rng::from_seed(seed));
            let hidden = hidden_orders(&world, RIVAL);
            assert_eq!(hidden.len(), 3);
            let keys: BTreeSet<OrderKey> = hidden.iter().map(|(_, key)| *key).collect();
            assert_eq!(keys.len(), 3, "one house cannot place the same token twice");
            let mut starred = 0;
            for key in keys {
                assert_ne!(
                    key.order_type(),
                    OrderType::March,
                    "restrictions in force apply to the resample"
                );
                starred += u32::from(key.starred());
            }
            assert!(starred <= world.state.star_allowance(RIVAL) as u32);
        }
    }

    #[test]
    fn a_march_token_needs_units() {
        let mut game = planning();
        // A rival order in an area with no units: the resample must not keep the
        // March token there.
        let region_id = game.data.map.lookup("kings_landing").unwrap();
        let region = game.state.region_mut(region_id);
        region.units = [0; 4];
        region.order = Some(OrderKey(0));
        region.order_house = Some(RIVAL);

        for seed in 0..50 {
            let world = determinize(&game, OBSERVER, &mut Rng::from_seed(seed));
            let order = world.state.region(region_id).order.unwrap();
            assert_ne!(order.order_type(), OrderType::March);
        }
    }

    #[test]
    fn resampled_pool_bookkeeping_matches_the_board() {
        let mut game = planning();
        place(&mut game, "kings_landing", RIVAL, OrderKey(3)); // defend_1a
        place(&mut game, "dragonstone", RIVAL, OrderKey(6)); // support_a

        let world = determinize(&game, OBSERVER, &mut Rng::from_seed(4));
        let record = world.state.house(RIVAL).unwrap();
        let mut on_board: Vec<OrderKey> = hidden_orders(&world, RIVAL)
            .into_iter()
            .map(|(_, key)| key)
            .collect();
        on_board.sort_unstable();
        let mut placed: Vec<OrderKey> = record.orders_placed.iter().copied().collect();
        placed.sort_unstable();
        assert_eq!(placed, on_board);
        let mut available: Vec<OrderKey> = record.orders_available.iter().copied().collect();
        available.sort_unstable();
        let mut expected: Vec<OrderKey> = OrderKey::ALL
            .into_iter()
            .filter(|key| !record.orders_placed.contains(key))
            .collect();
        expected.sort_unstable();
        assert_eq!(available, expected);
    }

    #[test]
    fn revealed_orders_are_left_alone() {
        let mut game = planning();
        game.state.phase = Phase::Action;
        place(&mut game, "kings_landing", RIVAL, OrderKey(0)); // march

        let world = determinize(&game, OBSERVER, &mut Rng::from_seed(0));
        assert_eq!(
            world
                .state
                .region(game.data.map.lookup("kings_landing").unwrap())
                .order,
            Some(OrderKey(0))
        );
    }

    #[test]
    fn seeded_determinization_is_reproducible() {
        let mut game = planning();
        place(&mut game, "kings_landing", RIVAL, OrderKey(0));
        place(&mut game, "dragonstone", RIVAL, OrderKey(3));

        let first = determinize(&game, OBSERVER, &mut Rng::from_seed(7));
        let second = determinize(&game, OBSERVER, &mut Rng::from_seed(7));
        assert_eq!(hidden_orders(&first, RIVAL), hidden_orders(&second, RIVAL));
    }

    #[test]
    fn bids_of_other_houses_are_resampled() {
        let mut game = Game::new(data(), GameConfig::with_seed(0)).unwrap();
        game.state.phase = Phase::Bidding;
        game.state.pending_bids.clear();
        game.state.pending_bids.push((OBSERVER, 3));
        game.state.pending_bids.push((RIVAL, 5));
        game.state.pending_bids.push((HouseId(1), 0));

        let mut sampled = BTreeSet::new();
        for seed in 0..30 {
            let world = determinize(&game, OBSERVER, &mut Rng::from_seed(seed));
            let bid_of = |house: HouseId| {
                world
                    .state
                    .pending_bids
                    .iter()
                    .find(|(bidder, _)| *bidder == house)
                    .map(|(_, amount)| *amount)
                    .unwrap()
            };
            assert_eq!(
                bid_of(OBSERVER),
                3,
                "the observer's own bid is public to it"
            );
            assert!(bid_of(RIVAL) <= world.state.house(RIVAL).unwrap().power);
            sampled.insert(bid_of(RIVAL));
        }
        assert!(sampled.len() > 1, "the rival's bid is genuinely resampled");
        // The real game is untouched.
        assert_eq!(
            game.state
                .pending_bids
                .iter()
                .find(|(bidder, _)| *bidder == RIVAL)
                .unwrap()
                .1,
            5
        );
    }

    #[test]
    fn wildling_bids_are_resampled() {
        let mut game = Game::new(data(), GameConfig::with_seed(0)).unwrap();
        game.state.phase = Phase::WildlingAttack;
        let mut attack = crate::state::WildlingAttack::default();
        attack.set_bid(OBSERVER, 2);
        attack.set_bid(RIVAL, 4);
        game.state.wildling_attack = Some(attack);

        let mut sampled = BTreeSet::new();
        for seed in 0..30 {
            let world = determinize(&game, OBSERVER, &mut Rng::from_seed(seed));
            let attack = world.state.wildling_attack.as_ref().unwrap();
            assert_eq!(attack.bid_of(OBSERVER), Some(2));
            let rival = attack.bid_of(RIVAL).unwrap();
            assert!(rival <= world.state.house(RIVAL).unwrap().power);
            sampled.insert(rival);
        }
        assert!(sampled.len() > 1);
        assert_eq!(
            game.state.wildling_attack.as_ref().unwrap().bid_of(RIVAL),
            Some(4)
        );
    }
}
