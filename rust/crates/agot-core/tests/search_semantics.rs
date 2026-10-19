//! Semantic gates for the MCTS port and the PIMC determinization.
//!
//! These pin *behaviour*, not numbers: every expectation below is derived by
//! hand from the PUCT rule in `agents/mcts.py`, so a refactor that changes the
//! selection rule, the PIMC aggregation or the temperature schedule fails here.

use std::sync::Arc;

use agot_core::action::Action;
use agot_core::enums::{HouseId, Phase, N_HOUSES};
use agot_core::fingerprint::state_fingerprint;
use agot_core::game::Game;
use agot_core::hidden::determinize;
use agot_core::map::DataSet;
use agot_core::rng::Rng;
use agot_core::search::{Evaluation, Evaluator, Mcts, SearchResult, UniformEvaluator};
use agot_core::setup::GameConfig;
use agot_core::state::WildlingAttack;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn data() -> Arc<DataSet> {
    Arc::new(DataSet::load_default().expect("bundled data must load"))
}

fn opening(seed: u64) -> Game {
    Game::new(data(), GameConfig::with_seed(seed)).expect("game must start")
}

/// A Planning position with the first planner's Order tokens face down on the
/// board, so determinization has something real to resample.
fn planning_with_a_rival_order() -> Game {
    let mut game = opening(1);
    let first = game
        .current_player()
        .expect("the opening is a Planning decision");
    while game.state.phase == Phase::Planning && game.current_player() == Some(first) {
        let actions = game.legal_actions();
        assert!(!actions.is_empty(), "a planner always has a legal action");
        game.step(actions[0])
            .expect("a legal planning action applies");
    }
    assert_eq!(
        game.state.phase,
        Phase::Planning,
        "the second planner is acting"
    );
    assert_ne!(game.current_player(), Some(first));
    game
}

/// A stub evaluator with strictly decreasing priors and one value for every
/// house, so the PUCT arithmetic below can be written out by hand.
struct Stub {
    value: f64,
}

impl Evaluator for Stub {
    fn evaluate(&mut self, _game: &Game, actions: &[Action]) -> Evaluation {
        let count = actions.len();
        Evaluation {
            priors: (0..count).map(|index| (count - index) as f64).collect(),
            value: [self.value; N_HOUSES],
        }
    }
}

/// Field paths where two fingerprints differ, `a/b/c` style.
fn differing_paths(a: &Value, b: &Value, prefix: &str, out: &mut Vec<String>) {
    let join = |prefix: &str, key: &str| {
        if prefix.is_empty() {
            key.to_string()
        } else {
            format!("{prefix}/{key}")
        }
    };
    match (a, b) {
        (Value::Object(left), Value::Object(right)) => {
            for (key, left_value) in left {
                let path = join(prefix, key);
                match right.get(key) {
                    Some(right_value) => differing_paths(left_value, right_value, &path, out),
                    None => out.push(path),
                }
            }
            for key in right.keys() {
                if !left.contains_key(key) {
                    out.push(join(prefix, key));
                }
            }
        }
        (Value::Array(left), Value::Array(right)) => {
            for index in 0..left.len().max(right.len()) {
                let path = format!("{prefix}/{index}");
                match (left.get(index), right.get(index)) {
                    (Some(left_value), Some(right_value)) => {
                        differing_paths(left_value, right_value, &path, out)
                    }
                    _ => out.push(path),
                }
            }
        }
        _ => {
            if a != b {
                out.push(prefix.to_string());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PUCT semantics
// ---------------------------------------------------------------------------

/// With one simulation the search expands the root and nothing else: the visit
/// distribution is all-zero and falls back to uniform.
#[test]
fn one_simulation_expands_only_the_root() {
    let game = opening(1);
    let actions = game.legal_actions();
    let mut search = Mcts::new(Stub { value: -1.0 });
    search.simulations = 1;
    search.determinizations = 1;
    search.dirichlet_weight = 0.0; // no root noise

    let result = search.search(&game).expect("the opening is searchable");

    assert_eq!(
        result.actions, actions,
        "the root's legal actions are the result"
    );
    assert!(
        result.visits.iter().all(|visits| *visits == 0.0),
        "one simulation expands the root but visits no child: {:?}",
        result.visits
    );
    assert!(result.q_values.iter().all(|q| *q == 0.0));
    assert_eq!(
        result.probabilities(1.0),
        vec![1.0 / actions.len() as f64; actions.len()]
    );
}

/// Two simulations: the first expands the root, the second visits exactly one
/// child — the one with the largest prior, because every q is still zero.
#[test]
fn two_simulations_visit_exactly_one_child() {
    let game = opening(1);
    let mut search = Mcts::new(Stub { value: -1.0 });
    search.simulations = 2;
    search.determinizations = 1;
    search.dirichlet_weight = 0.0;

    let result = search.search(&game).expect("the opening is searchable");

    assert_eq!(
        result.visits.iter().sum::<f64>(),
        1.0,
        "exactly one child visit"
    );
    assert_eq!(
        result.visits[0], 1.0,
        "u = c_puct * prior decides, and prior 0 is largest"
    );
    assert!(result.visits[1..].iter().all(|visits| *visits == 0.0));
    let probabilities = result.probabilities(1.0);
    assert_eq!(probabilities[0], 1.0);
    assert!(probabilities[1..].iter().all(|value| *value == 0.0));
}

/// Three simulations: the root is expanded, the second visits the child with the
/// largest prior, and the third must trade that child's prior edge against its
/// (bad) value. The expectation is computed by hand in the assertions below.
#[test]
fn puct_trades_prior_against_value() {
    let game = opening(1);
    let actions = game.legal_actions();
    let count = actions.len();
    let mut search = Mcts::new(Stub { value: -1.0 });
    search.simulations = 3;
    search.determinizations = 1;
    search.dirichlet_weight = 0.0;

    let result = search.search(&game).expect("the opening is searchable");

    // The stub returns priors `count - index`, clipped at 1e-8 and normalized.
    let prior_sum: f64 = (1..=count).map(|step| step as f64).sum();
    let prior = |index: usize| (count - index) as f64 / prior_sum;
    // When the third simulation selects, the root has two visits, the child
    // chosen by the second has one, and every other child none:
    //   score(visited) = -1 + c_puct * prior(0) * sqrt(2) / 2
    //   score(other)   =  0 + c_puct * prior(i) * sqrt(2) / 1
    let sqrt_total = 2.0f64.sqrt();
    let score_visited = -1.0 + search.c_puct * prior(0) * sqrt_total / 2.0;
    let score_second = search.c_puct * prior(1) * sqrt_total;
    assert!(
        score_second > score_visited,
        "the hand computation must prefer the next child: {score_second} <= {score_visited}"
    );
    for index in 2..count {
        assert!(
            search.c_puct * prior(index) * sqrt_total <= score_second,
            "later children have smaller priors"
        );
    }

    // The second simulation takes the largest prior, the third the next child.
    assert_eq!(result.visits[0], 1.0);
    assert_eq!(result.visits[1], 1.0);
    assert!(
        result.visits[2..].iter().all(|visits| *visits == 0.0),
        "only two children are visited: {:?}",
        result.visits
    );
    // q is the mean value of the action for the acting house.
    assert_eq!(result.q_values[0], -1.0);
    assert_eq!(result.q_values[1], -1.0);
    assert!(result.q_values[2..].iter().all(|q| *q == 0.0));
}

/// The uniform evaluator against the engine's own action list.
#[test]
fn the_uniform_evaluator_gives_a_uniform_distribution() {
    let game = opening(2);
    let actions = game.legal_actions();
    let mut search = Mcts::new(UniformEvaluator);
    search.simulations = 1;
    search.determinizations = 1;

    let result = search.search(&game).expect("the opening is searchable");

    assert_eq!(
        result.actions, actions,
        "the result mirrors the engine's list"
    );
    let probabilities = result.probabilities(1.0);
    assert_eq!(probabilities.len(), actions.len());
    for probability in probabilities {
        assert!((probability - 1.0 / actions.len() as f64).abs() < 1e-12);
    }
}

/// A batched driver: several worlds paused at their leaves, scored together,
/// then resumed.
#[test]
fn leaves_can_be_paused_and_scored_in_a_batch() {
    let game = planning_with_a_rival_order();
    let mut search = Mcts::new(UniformEvaluator);
    search.simulations = 6;
    search.determinizations = 3;
    let roots = search.open_search_worlds(&game);
    assert_eq!(roots.len(), 3, "one root per determinization");

    // Two rounds of the driver's loop: prepare every root, score every leaf,
    // then resume. Nothing is scored while another simulation is paused.
    for _ in 0..2 {
        let mut pending = Vec::new();
        for root in &roots {
            pending.push(
                search
                    .prepare_simulation(*root)
                    .expect("a world is searchable"),
            );
        }
        let mut evaluations = Vec::new();
        for simulation in &pending {
            match simulation.leaf {
                Some(_) => {
                    let leaf_game = search.leaf_game(simulation).expect("a leaf has a game");
                    evaluations.push(Some(
                        UniformEvaluator.evaluate(&leaf_game, &simulation.actions),
                    ));
                }
                None => evaluations.push(None),
            }
        }
        for (simulation, evaluation) in pending.into_iter().zip(evaluations) {
            search.finish_simulation(simulation, evaluation).unwrap();
        }
    }

    let result = search.aggregate_roots(&roots, game.current_player());
    assert_eq!(result.actions, game.legal_actions());
    assert_eq!(
        result.visits.iter().sum::<f64>(),
        3.0,
        "the second round visits one child in each world"
    );
}

// ---------------------------------------------------------------------------
// PIMC
// ---------------------------------------------------------------------------

/// Aggregated visits are the sum of the worlds' visits, action by action.
#[test]
fn pimc_aggregates_the_worlds_visits() {
    let game = planning_with_a_rival_order();
    let actions = game.legal_actions();
    let mut search = Mcts::new(UniformEvaluator);
    search.simulations = 6;
    search.determinizations = 2;
    search.dirichlet_weight = 0.0;

    let roots = search.open_search_worlds(&game);
    assert_eq!(roots.len(), 2, "two worlds");
    for root in &roots {
        search.run(*root, 3).unwrap(); // per_world = max(1, 6 / 2)
    }
    let per_world: Vec<SearchResult> = roots
        .iter()
        .map(|root| search.result_from_root(*root))
        .collect();
    for world in &per_world {
        assert_eq!(
            world.actions, actions,
            "determinization never changes the acting house's own moves"
        );
    }

    let aggregated = search.aggregate_roots(&roots, game.current_player());
    assert_eq!(aggregated.actions, actions);
    let mut total = 0.0;
    for (index, action) in aggregated.actions.iter().enumerate() {
        let expected: f64 = per_world
            .iter()
            .map(|world| {
                world
                    .actions
                    .iter()
                    .position(|candidate| candidate == action)
                    .map(|position| world.visits[position])
                    .unwrap_or(0.0)
            })
            .sum();
        assert_eq!(aggregated.visits[index], expected, "visits must add up");
        total += aggregated.visits[index];
    }
    assert_eq!(total, 4.0, "each world spends one simulation on its root");
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn the_same_seed_repeats_the_visits() {
    let game = planning_with_a_rival_order();
    let mut first = Mcts::with_seed(UniformEvaluator, 11);
    first.simulations = 8;
    first.determinizations = 2;
    let mut second = Mcts::with_seed(UniformEvaluator, 11);
    second.simulations = 8;
    second.determinizations = 2;

    let left = first.search(&game).expect("searchable");
    let right = second.search(&game).expect("searchable");
    assert_eq!(left.actions, right.actions);
    assert_eq!(left.visits, right.visits);
    assert_eq!(left.q_values, right.q_values);
}

#[test]
fn different_seeds_explore_differently() {
    // The seed drives the Dirichlet root noise (and, with PIMC, the resampled
    // worlds), so two seeds must not produce the same visit vector.
    let game = planning_with_a_rival_order();
    let mut first = Mcts::with_seed(UniformEvaluator, 11);
    first.simulations = 8;
    first.determinizations = 2;
    let mut second = Mcts::with_seed(UniformEvaluator, 12);
    second.simulations = 8;
    second.determinizations = 2;

    let left = first.search(&game).expect("searchable");
    let right = second.search(&game).expect("searchable");
    assert_ne!(left.visits, right.visits);
}

// ---------------------------------------------------------------------------
// Temperature
// ---------------------------------------------------------------------------

#[test]
fn temperature_shapes_the_distribution() {
    // Hand-built visits, so the expected distribution is arithmetic: visits
    // raised to 1 / temperature, normalized.
    let result = SearchResult {
        actions: vec![
            Action::pass(HouseId(0)),
            Action::bid(HouseId(0), 1),
            Action::bid(HouseId(0), 2),
            Action::bid(HouseId(0), 3),
        ],
        visits: vec![3.0, 1.0, 1.0, 0.0],
        q_values: vec![0.0; 4],
    };

    // A temperature at or below 1e-3 is a one-hot target on the most visited
    // action (ties go to the earliest, like numpy's argmax).
    let one_hot = result.probabilities(1e-3);
    assert_eq!(one_hot, vec![1.0, 0.0, 0.0, 0.0]);
    let tied = SearchResult {
        actions: result.actions.clone(),
        visits: vec![1.0, 3.0, 3.0, 0.0],
        q_values: vec![0.0; 4],
    };
    assert_eq!(tied.probabilities(0.0), vec![0.0, 1.0, 0.0, 0.0]);

    // A low but valid temperature sharpens without becoming one-hot.
    let sharp = result.probabilities(0.5);
    assert!((sharp[0] - 9.0 / 11.0).abs() < 1e-12);
    assert!(sharp[0] > 0.5, "sharp");
    assert_eq!(
        sharp.iter().filter(|value| **value > 0.0).count(),
        3,
        "not one-hot"
    );
    assert!((sharp.iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // And a search result follows the same rule.
    let game = opening(1);
    let mut search = Mcts::new(Stub { value: -1.0 });
    search.simulations = 3;
    search.dirichlet_weight = 0.0;
    let searched = search.search(&game).expect("searchable");
    let best = searched.argmax_visits().expect("a decision was found");
    let target = searched.probabilities(1e-3);
    assert_eq!(target[best], 1.0);
    assert_eq!(target.iter().filter(|value| **value > 0.0).count(), 1);
}

// ---------------------------------------------------------------------------
// Determinization
// ---------------------------------------------------------------------------

#[test]
fn determinize_leaves_the_source_untouched() {
    let game = planning_with_a_rival_order();
    let observer = game.current_player().expect("a planner is acting");
    let placed_house = game
        .state
        .present_houses()
        .into_iter()
        .find(|house| *house != observer && !game.state.orders_on_board(*house).is_empty())
        .expect("the first planner's tokens are face down on the board");

    let before = state_fingerprint(&game.state, &data());
    let mut rng = Rng::from_seed(3);
    let mut resampled_a_token = false;
    for _ in 0..8 {
        let world = determinize(&game, observer, &mut rng);

        // The source game is untouched by any number of resamplings.
        assert_eq!(
            state_fingerprint(&game.state, &data()),
            before,
            "determinize must not mutate its input"
        );

        // Ownership never moves, and the observer's own tokens never change.
        for region_id in game.state.region_ids() {
            let source = game.state.region(region_id);
            let sampled = world.state.region(region_id);
            assert_eq!(sampled.order_house, source.order_house);
            if source.order_house == Some(observer) {
                assert_eq!(sampled.order, source.order);
            }
        }
        assert_eq!(
            world.state.orders_on_board(placed_house).len(),
            game.state.orders_on_board(placed_house).len()
        );

        // Everything that differs is an Order token or its pool bookkeeping.
        let mut paths = Vec::new();
        differing_paths(
            &before,
            &state_fingerprint(&world.state, &data()),
            "",
            &mut paths,
        );
        for path in &paths {
            assert!(
                path.contains("/order"),
                "only Order tokens may move: {path}"
            );
            if let Some(rest) = path.strip_prefix("houses/") {
                let house = rest.split('/').next().unwrap_or("");
                assert_ne!(house, observer.name(), "the observer's own pool changed");
            }
            if path.ends_with("/order") {
                resampled_a_token = true;
            }
        }
    }
    assert!(
        resampled_a_token,
        "the rival's hidden tokens must be resampled"
    );
}

#[test]
fn determinize_resamples_other_houses_bids() {
    let mut game = opening(2);
    let observer = HouseId(2);
    let rival = HouseId(0);
    game.state.phase = Phase::Bidding;
    game.state.pending_bids.clear();
    game.state.pending_bids.push((observer, 3));
    game.state.pending_bids.push((rival, 5));

    let before = state_fingerprint(&game.state, &data());
    let mut rng = Rng::from_seed(1);
    let mut sampled = std::collections::BTreeSet::new();
    for _ in 0..30 {
        let world = determinize(&game, observer, &mut rng);
        let bid_of = |house: HouseId| {
            world
                .state
                .pending_bids
                .iter()
                .find(|(bidder, _)| *bidder == house)
                .map(|(_, amount)| *amount)
                .expect("every bidder keeps a bid")
        };
        assert_eq!(
            bid_of(observer),
            3,
            "the observer's own bid is never resampled"
        );
        assert!(bid_of(rival) <= world.state.house(rival).unwrap().power);
        sampled.insert(bid_of(rival));
        assert_eq!(state_fingerprint(&game.state, &data()), before);
    }
    assert!(sampled.len() > 1, "the rival's bid is genuinely resampled");

    // A Wildling attack carries its bids inside the pending attack.
    game.state.phase = Phase::WildlingAttack;
    game.state.pending_bids.clear();
    let mut attack = WildlingAttack::default();
    attack.set_bid(observer, 2);
    attack.set_bid(rival, 4);
    game.state.wildling_attack = Some(attack);

    let mut sampled = std::collections::BTreeSet::new();
    for _ in 0..30 {
        let world = determinize(&game, observer, &mut rng);
        let attack = world
            .state
            .wildling_attack
            .as_ref()
            .expect("still attacking");
        assert_eq!(attack.bid_of(observer), Some(2));
        let rival_bid = attack.bid_of(rival).expect("the rival bid");
        assert!(rival_bid <= world.state.house(rival).unwrap().power);
        sampled.insert(rival_bid);
    }
    assert!(sampled.len() > 1);
}
