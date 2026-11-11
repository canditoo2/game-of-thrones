//! Driver-level gates for the per-phase simulation budgets.
//!
//! The PyO3 driver in `agot-py` cannot be built in the core test loop, so these
//! tests reproduce its two load-bearing pieces exactly: the budget it computes
//! per decision ([`decision_budget`]) and the batched pass loop it runs over the
//! PIMC worlds (one simulation per root per pass, leaves scored in one batch).
//! Everything else is the real engine, the real search and a counting
//! evaluator, so a change in the budget or in the division by determinizations
//! shows up here as a changed leaf count.
//!
//! "Rows" below means leaves scored by the evaluator, which is what the
//! measured 61%-in-Planning figure counted.

use std::sync::Arc;

use agot_core::action::Action;
use agot_core::budget::{decision_budget, PhaseBudgets};
use agot_core::enums::{Phase, N_HOUSES, N_PHASES};
use agot_core::game::Game;
use agot_core::map::DataSet;
use agot_core::search::{Evaluation, Evaluator, Mcts, SearchResult};
use agot_core::setup::GameConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn data() -> Arc<DataSet> {
    Arc::new(DataSet::load_default().expect("bundled data must load"))
}

/// A game configured like training: every optional decision enabled.
fn training_game(seed: u64) -> Game {
    let mut game = Game::new(data(), GameConfig::with_seed(seed)).expect("game must start");
    game.deferred_battles = true;
    game.sequential_westeros = true;
    game.doran_choice = true;
    game.wildling_tie_choice = true;
    game.bid_tie_choice = true;
    game.choice_choices = true;
    game.state.choice_choices = true;
    game
}

/// A uniform evaluator that counts the leaves it was asked to score.
#[derive(Default)]
struct Counting {
    leaves: usize,
}

impl Evaluator for Counting {
    fn evaluate(&mut self, _game: &Game, actions: &[Action]) -> Evaluation {
        self.leaves += 1;
        let count = actions.len();
        Evaluation {
            priors: vec![1.0 / count.max(1) as f64; count],
            value: [0.0; N_HOUSES],
        }
    }
}

/// One decision's search: `per_world` batched passes over `roots`.
///
/// This is the driver's loop with the Python callback replaced by the searcher's
/// own evaluator: prepare one simulation per root, score every paused leaf in
/// one batch, resume them all. Returns the number of scored leaves.
fn run_passes(search: &mut Mcts<Counting>, roots: &[u32], per_world: u32) -> usize {
    let mut scored = 0;
    for _ in 0..per_world {
        let mut pending = Vec::with_capacity(roots.len());
        for root in roots {
            pending.push(
                search
                    .prepare_simulation(*root)
                    .expect("a root is always searchable"),
            );
        }
        let mut evaluations = Vec::with_capacity(pending.len());
        for simulation in &pending {
            match simulation.leaf {
                Some(_) => {
                    let leaf = search.leaf_game(simulation).expect("a leaf has a game");
                    scored += 1;
                    evaluations.push(Some(search.evaluator.evaluate(&leaf, &simulation.actions)));
                }
                None => evaluations.push(None),
            }
        }
        for (simulation, evaluation) in pending.into_iter().zip(evaluations) {
            search
                .finish_simulation(simulation, evaluation)
                .expect("a scored leaf expands its node");
        }
    }
    scored
}

/// Search one decision the way the driver does, reporting rows and visits.
fn search_decision(
    game: &Game,
    budgets: Option<&PhaseBudgets>,
    simulations: u32,
    determinizations: u32,
    seed: u64,
) -> (SearchResult, usize) {
    let determinizations = determinizations.max(1);
    let actions = game.legal_actions().len();
    let budget = decision_budget(budgets, game.state.phase, actions, simulations);
    let per_world = (budget / determinizations).max(1);

    let mut search = Mcts::with_seed(Counting::default(), seed);
    search.determinizations = determinizations;
    let roots = search.open_search_worlds(game);
    assert!(!roots.is_empty(), "the position must be a decision");
    let scored = run_passes(&mut search, &roots, per_world);
    let result = if roots.len() == 1 {
        search.result_from_root(roots[0])
    } else {
        search.aggregate_roots(&roots, game.current_player())
    };
    (result, scored)
}

/// Scored leaves per phase across a whole game, as the driver would count them.
#[derive(Clone, Default)]
struct Tally {
    decisions: [usize; N_PHASES],
    leaves: [usize; N_PHASES],
}

impl Tally {
    fn record(&mut self, phase: Phase, leaves: usize) {
        self.decisions[phase.index()] += 1;
        self.leaves[phase.index()] += leaves;
    }

    fn total_decisions(&self) -> usize {
        self.decisions.iter().sum()
    }

    fn total_leaves(&self) -> usize {
        self.leaves.iter().sum()
    }

    fn report(&self, label: &str) {
        println!(
            "  {label:<9} decisions {:>5}  rows {:>7}  rows/decision {:.2}",
            self.total_decisions(),
            self.total_leaves(),
            self.total_leaves() as f64 / self.total_decisions().max(1) as f64
        );
        for phase in Phase::ALL {
            let decisions = self.decisions[phase.index()];
            if decisions == 0 {
                continue;
            }
            println!(
                "    {:<15} decisions {:>4}  rows {:>6}  rows/decision {:.2}",
                phase.as_str(),
                decisions,
                self.leaves[phase.index()],
                self.leaves[phase.index()] as f64 / decisions as f64
            );
        }
    }
}

/// Plays one full self-play game with a counting evaluator and tallies rows.
fn play_game(
    budgets: Option<&PhaseBudgets>,
    simulations: u32,
    determinizations: u32,
    seed: u64,
) -> Tally {
    let determinizations = determinizations.max(1);
    let mut game = training_game(seed);
    let mut search = Mcts::with_seed(Counting::default(), seed);
    search.determinizations = determinizations;
    let mut tally = Tally::default();
    let mut steps = 0u32;
    while !game.is_terminal() && steps < 20_000 {
        // A sequential Westeros reveal is not a decision: resolve the card
        // before the following one is drawn, exactly like the driver.
        if game.state.phase == Phase::Westeros && game.state.westeros_waiting {
            game.continue_westeros().expect("the reveal resolves");
            continue;
        }
        if game.current_player().is_none() {
            break;
        }
        let phase = game.state.phase;
        let actions = game.legal_actions().len();
        let budget = decision_budget(budgets, phase, actions, simulations);
        let per_world = (budget / determinizations).max(1);
        let roots = search.open_search_worlds(&game);
        if roots.is_empty() {
            break;
        }
        tally.record(phase, run_passes(&mut search, &roots, per_world));
        let result = if roots.len() == 1 {
            search.result_from_root(roots[0])
        } else {
            search.aggregate_roots(&roots, game.current_player())
        };
        let (action, _) = search
            .pick(&result, Some(1.0))
            .expect("a decision has actions");
        game.step(action).expect("a legal action applies");
        steps += 1;
    }
    tally
}

fn play_games(
    budgets: Option<&PhaseBudgets>,
    simulations: u32,
    determinizations: u32,
    games: u64,
) -> Tally {
    let mut total = Tally::default();
    for game in 0..games {
        let tally = play_game(budgets, simulations, determinizations, 7 + game);
        for phase in Phase::ALL {
            total.decisions[phase.index()] += tally.decisions[phase.index()];
            total.leaves[phase.index()] += tally.leaves[phase.index()];
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The budget is honoured by the batched loop: a Planning decision scores one
/// leaf per simulation without budgets and only its cap with them.
#[test]
fn a_planning_decision_scores_the_budgeted_number_of_leaves() {
    let game = training_game(1);
    assert_eq!(
        game.state.phase,
        Phase::Planning,
        "the opening position is a Planning decision"
    );
    let actions = game.legal_actions().len();
    let (fixed_result, fixed_rows) = search_decision(&game, None, 128, 1, 3);
    let (budgeted_result, budgeted_rows) =
        search_decision(&game, Some(&PhaseBudgets::DEFAULT), 128, 1, 3);

    assert_eq!(
        fixed_rows, 128,
        "every simulation of a Planning root pauses at a scoreable leaf"
    );
    // The expected budget, spelled out from the preset: 32 for Planning, raised
    // to one simulation per candidate plus the root's expansion.
    let expected = 32.max(actions + 1);
    assert!(expected < 128, "the opening has {actions} candidates");
    assert_eq!(budgeted_rows, expected, "Planning is capped at 32");
    assert!(
        budgeted_rows * 2 <= fixed_rows,
        "budgeted {budgeted_rows} rows vs fixed {fixed_rows}"
    );
    // The action list is unchanged: a cap changes depth, never the choices.
    assert_eq!(budgeted_result.actions, fixed_result.actions);
    println!(
        "planning decision: {actions} candidates, fixed {fixed_rows} rows, budgeted {budgeted_rows} rows"
    );
}

/// A cap is a per-decision number: PIMC divides it over its worlds exactly as
/// the fixed budget is divided, so both runs get the same passes per world.
#[test]
fn a_cap_is_divided_over_the_pimc_worlds_like_the_fixed_budget() {
    let game = training_game(1);
    let actions = game.legal_actions().len();
    let planning = 32u32.max(actions as u32 + 1);
    for determinizations in [1u32, 2, 4] {
        let worlds = determinizations; // one root per determinized world
        let (_, fixed_rows) = search_decision(&game, None, 64, determinizations, 5);
        let (_, capped_rows) =
            search_decision(&game, Some(&PhaseBudgets::DEFAULT), 64, determinizations, 5);
        assert_eq!(
            fixed_rows,
            ((64 / determinizations).max(1) * worlds) as usize,
            "{determinizations} worlds of the fixed budget"
        );
        assert_eq!(
            capped_rows,
            ((planning / determinizations).max(1) * worlds) as usize,
            "{determinizations} worlds of the Planning cap"
        );
    }
    // The Wildling cap is smaller, and a cap below the world count still gives
    // every world one simulation, exactly like `simulations < determinizations`.
    let wildling = decision_budget(Some(&PhaseBudgets::DEFAULT), Phase::WildlingAttack, 6, 128);
    assert_eq!(wildling, 16);
    assert_eq!((wildling / 4).max(1), 4);
    assert_eq!((wildling / 32).max(1), 1);
}

/// `phase_budgets = null` is the old fixed loop: same seeds, same passes, same
/// visits, same row count.
#[test]
fn no_phase_budgets_reproduces_the_fixed_row_count_exactly() {
    /// The pre-feature driver: one shared per-world budget for every decision,
    /// `simulations / determinizations` passes over every root.
    fn fixed_reference(
        game: &Game,
        simulations: u32,
        determinizations: u32,
        seed: u64,
    ) -> (SearchResult, usize) {
        let determinizations = determinizations.max(1);
        let per_world = (simulations / determinizations).max(1);
        let mut search = Mcts::with_seed(Counting::default(), seed);
        search.determinizations = determinizations;
        let roots = search.open_search_worlds(game);
        assert!(!roots.is_empty(), "the position must be a decision");
        let scored = run_passes(&mut search, &roots, per_world);
        let result = if roots.len() == 1 {
            search.result_from_root(roots[0])
        } else {
            search.aggregate_roots(&roots, game.current_player())
        };
        (result, scored)
    }

    let game = training_game(1);
    for determinizations in [1u32, 2, 4] {
        for simulations in [1u32, 5, 16, 45, 128] {
            let (fixed, fixed_rows) =
                search_decision(&game, None, simulations, determinizations, 11);
            let (reference, reference_rows) =
                fixed_reference(&game, simulations, determinizations, 11);
            // Identical visits, not only identical counts: no budget is not a
            // different search with the same depth.
            assert_eq!(
                fixed, reference,
                "{determinizations} worlds, {simulations} sims"
            );
            assert_eq!(fixed_rows, reference_rows);
            // And the rows are what the old loop produced: one simulation per
            // root per pass, with `simulations / determinizations` passes
            // floored at one.
            let per_world = (simulations / determinizations).max(1);
            assert_eq!(fixed_rows, (per_world * determinizations) as usize);
        }
    }

    // The decision budget itself is the plain ceiling whenever the table is
    // absent, for every phase and every candidate count.
    for phase in Phase::ALL {
        for actions in [0usize, 1, 7, 45] {
            for simulations in [1u32, 16, 128] {
                assert_eq!(
                    decision_budget(None, phase, actions, simulations),
                    simulations
                );
            }
        }
    }
}

/// The measurement the preset was tuned on, replayed on two real games: at 16
/// simulations the caps are above the ceiling, so the feature is a no-op; at 64
/// it removes the Planning leaves that carry no signal.
#[test]
fn measured_rows_per_decision_on_two_real_games() {
    for simulations in [16u32, 64] {
        println!("two games at {simulations} simulations, 2 determinizations:");
        let fixed = play_games(None, simulations, 2, 2);
        let budgeted = play_games(Some(&PhaseBudgets::DEFAULT), simulations, 2, 2);
        fixed.report("fixed");
        budgeted.report("budgeted");
        let saved = fixed.total_leaves() - budgeted.total_leaves();
        println!(
            "  rows saved {saved} of {} ({:.1}%)\n",
            fixed.total_leaves(),
            100.0 * saved as f64 / fixed.total_leaves().max(1) as f64
        );
        if simulations <= 16 {
            // Every cap is at or above a 16-simulation ceiling, so the run must
            // be bit-identical: same decisions, same rows, same games.
            assert_eq!(
                fixed.total_leaves(),
                budgeted.total_leaves(),
                "every cap is at or above a 16-simulation ceiling"
            );
            for phase in Phase::ALL {
                assert_eq!(
                    fixed.decisions[phase.index()],
                    budgeted.decisions[phase.index()],
                    "{phase}: the budget changes depth, not the decision stream"
                );
            }
        } else {
            // The games diverge once the shallower search picks a different
            // action, so the decision counts differ too; the rows are what the
            // preset promises to cut.
            assert!(
                budgeted.total_leaves() * 3 <= fixed.total_leaves() * 2,
                "the 64-simulation run must drop at least a third of its rows: \
                 {} -> {}",
                fixed.total_leaves(),
                budgeted.total_leaves()
            );
        }
    }
}
