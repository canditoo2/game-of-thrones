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
//! Also resampled here
//! -------------------
//!
//! * **The first chooser's House card**, while the second chooser is deciding.
//!   The physical game reveals both cards simultaneously; the engine asks the
//!   attacker first, so the defender's `pending_battle` copy replaces the known
//!   attacker card with a plausible card from the attacker's public pool.
//!   House-card hands themselves are already private in the observation
//!   encoding: the network only ever sees the acting house's own cards.
//! * **The order of the cards still face down** in the Westeros and Wildling
//!   decks. Sequential play makes this load-bearing: a mustering or bidding
//!   decision must not be valued with knowledge of the cards that will be
//!   revealed afterwards. The sampled decks keep their exact composition and
//!   only shuffle the positions after the current draw.
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
    resample_battle_card(&mut clone.state, observer, rng);
    // Last, so the existing order/bid/card samples keep their seeded stream.
    resample_decks(&mut clone.state, rng);
    clone
}

/// Shuffle the cards that are still face down, preserving each deck's
/// composition. Already-revealed (`drawn`) cards and the spent positions stay
/// put; this only removes knowledge of what comes next.
fn resample_decks(state: &mut State, rng: &mut Rng) {
    for deck in state.westeros_decks.iter_mut() {
        if deck.position < deck.cards.len() {
            rng.shuffle(&mut deck.cards[deck.position..]);
        }
    }
    if state.wildling_deck.position < state.wildling_deck.cards.len() {
        rng.shuffle(&mut state.wildling_deck.cards[state.wildling_deck.position..]);
    }
}

/// Hide the first chooser's House card from the second chooser.
///
/// The engine asks the attacker for a card first, but the physical game reveals
/// both cards simultaneously. The defender's PIMC worlds therefore replace the
/// already-chosen attacker card with a plausible card from the attacker's
/// public hand/discard pool. The actual game state is untouched; only search
/// branches see the sampled card.
fn resample_battle_card(state: &mut State, observer: HouseId, rng: &mut Rng) {
    if state.phase != Phase::Combat {
        return;
    }
    let Some(battle) = state.pending_battle.as_ref() else {
        return;
    };
    if battle.neutral
        || !battle.attacker_card_chosen
        || battle.defender_card_chosen
        || battle.attacker_card.is_none()
        || Some(observer) != battle.defender
    {
        return;
    }
    let Some(attacker) = state.house(battle.attacker) else {
        return;
    };
    // The just-played card is the last entry in the discard; earlier discards
    // are not plausible choices for this battle.
    let mut pool: Vec<crate::enums::CardId> =
        attacker.house_cards_available.iter().copied().collect();
    if let Some(card) = attacker.house_cards_used.last() {
        pool.push(*card);
    }
    pool.sort_unstable_by_key(|card| card.0);
    pool.dedup();
    if pool.is_empty() {
        return;
    }
    let sample = pool[rng.range_inclusive(0, pool.len() as i64 - 1) as usize];
    if let Some(battle) = state.pending_battle.as_mut() {
        battle.attacker_card = Some(sample);
    }
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
/// Once the orders are revealed (the step before the Messenger Raven) they are
/// public and nothing is resampled.
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
    // Once the tie is being broken, all bids are public to the chooser; do not
    // resample them under the holder's feet.
    if state.pending_wildling_tie.is_some() {
        return;
    }
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
    use crate::enums::{CardId, UnitType};
    use crate::setup::GameConfig;
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use crate::map::DataSet;
    use crate::state::{
        BattleStage, ChoiceKind, ChoiceResume, ChoiceTask, DeckState, PendingBattle, Restriction,
        UnitChoice,
    };

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

    #[test]
    fn determinize_hides_the_remaining_deck_order() {
        let mut game = planning();
        game.state.westeros_decks[0] = DeckState::new([10u16, 11, 12, 13]);
        game.state.westeros_decks[0].position = 1;
        game.state.westeros_decks[0].drawn = Some(10);
        game.state.wildling_deck = DeckState::new([20u16, 21, 22]);
        game.state.wildling_deck.position = 1;

        let mut orders = BTreeSet::new();
        for seed in 0..12 {
            let mut rng = Rng::from_seed(seed);
            let world = determinize(&game, OBSERVER, &mut rng);
            let deck = &world.state.westeros_decks[0];
            assert_eq!(deck.cards[0], 10, "the spent position must not move");
            assert_eq!(deck.drawn, Some(10), "the revealed card stays public");
            assert_eq!(deck.cards.len(), 4, "the composition is preserved");
            let mut remaining = deck.cards[1..].to_vec();
            remaining.sort_unstable();
            assert_eq!(remaining, vec![11, 12, 13]);
            orders.insert(deck.cards[1..].to_vec());
            assert_eq!(world.state.wildling_deck.cards[0], 20);
        }
        assert!(
            orders.len() > 1,
            "different determinizations must hide different future cards"
        );
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

    /// A defender's search must not see the card the attacker already chose.
    #[test]
    fn the_second_chooser_cannot_see_the_first_house_card() {
        let mut game = planning();
        let attacker = HouseId(0); // baratheon
        let defender = HouseId(2); // stark
        let actual: CardId = game
            .data
            .house_cards
            .card_index(attacker, "patchface")
            .expect("the card exists");
        let region = game.data.map.lookup("kingswood").expect("a region");
        game.state.phase = Phase::Combat;
        game.state.pending_battle = Some(PendingBattle {
            region,
            attacker,
            defender: Some(defender),
            neutral: false,
            origin: Some(game.data.map.lookup("dragonstone").unwrap()),
            attacker_order: None,
            attacker_units: [2, 0, 0, 0],
            defender_units: [1, 0, 0, 0],
            attacker_order_strength: 0,
            defender_order_strength: 0,
            attacker_card: Some(actual),
            defender_card: None,
            attacker_card_chosen: true,
            defender_card_chosen: false,
            pre_choices_done: false,
            doran_owner: None,
            doran_track: None,
            prepared: None,
            blade_done: false,
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
        });

        let actual_for_attacker = determinize(&game, attacker, &mut Rng::from_seed(1));
        assert_eq!(
            actual_for_attacker
                .state
                .pending_battle
                .as_ref()
                .and_then(|battle| battle.attacker_card),
            Some(actual),
            "the attacker knows their own committed card"
        );

        let mut sampled = std::collections::BTreeSet::new();
        for seed in 0..40 {
            let world = determinize(&game, defender, &mut Rng::from_seed(seed));
            sampled.insert(
                world
                    .state
                    .pending_battle
                    .as_ref()
                    .and_then(|battle| battle.attacker_card)
                    .expect("the attacker has a card"),
            );
        }
        assert!(
            sampled.len() > 1,
            "the defender must not be pinned to the first chooser's card: {sampled:?}"
        );
        assert!(
            sampled.iter().all(|card| {
                game.state
                    .house(attacker)
                    .map(|state| {
                        state.house_cards_available.contains(card)
                            || state.house_cards_used.contains(card)
                    })
                    .unwrap_or(false)
            }),
            "every sampled card must come from the attacker's public pool"
        );
    }

    /// PIMC resamples hidden information only. The decision the game is paused
    /// on — the generic choice queue, its resume point and the interactive
    /// flags — is part of the position and must reach every world intact;
    /// otherwise a resampled world would offer different actions than the real
    /// game and the root visits could not be aggregated.
    #[test]
    fn determinize_keeps_the_pending_choice_queue() {
        let mut game = planning();
        game.deferred_battles = true;
        game.sequential_westeros = true;
        game.choice_choices = true;
        game.state.choice_choices = true;
        game.state.choice_resume = Some(ChoiceResume::Wildling);
        let winterfell = game.data.map.lookup("winterfell").expect("winterfell");
        game.state.choice_queue.push(ChoiceTask {
            house: OBSERVER,
            kind: ChoiceKind::Unit {
                purpose: UnitChoice::Destroy,
                areas: smallvec::smallvec![winterfell],
                remaining: 2,
                allow_none: false,
            },
        });
        let actions = game.legal_actions();
        assert!(!actions.is_empty(), "the queue provides the legal actions");

        for observer in [OBSERVER, RIVAL] {
            for seed in 0..10 {
                let world = determinize(&game, observer, &mut Rng::from_seed(seed));
                assert_eq!(world.state.choice_queue, game.state.choice_queue);
                assert_eq!(world.state.choice_resume, Some(ChoiceResume::Wildling));
                assert!(world.state.choice_choices);
                assert!(world.choice_choices);
                assert!(world.deferred_battles && world.sequential_westeros);
                assert_eq!(
                    world.legal_actions(),
                    actions,
                    "a world must ask the same choice as the real game"
                );
            }
        }
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
