//! Kill-gate bench: how fast are the operations the Python engine is slowest at?
//!
//! Python baselines measured on this machine (see the port plan):
//!
//! | operation                    | Python        |
//! |------------------------------|---------------|
//! | `GameState.clone()`          | ~5 900 ns     |
//! | `Game.clone()`               | ~12 000 ns    |
//! | random game (603 decisions)  | 0.107 s       |
//! | uniform MCTS                 | ~7 500 sims/s |
//!
//! Each measured closure hides its input behind `black_box` and returns a tiny
//! value: without that, the optimiser hoists the clone out of the loop and the
//! number stops meaning anything.
//!
//! Run with `cargo bench --bench clone_step` (the bench profile is optimised).

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use agot_core::game::Game;
use agot_core::map::DataSet;
use agot_core::setup::GameConfig;
use agot_core::state::State;

/// Runs `body` `iterations` times and returns nanoseconds per iteration.
fn time<T>(iterations: u32, mut body: impl FnMut() -> T) -> f64 {
    // Warm up so first-touch allocation does not land in the measurement.
    for _ in 0..(iterations / 20).max(1) {
        black_box(body());
    }
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(body());
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn main() {
    let data = Arc::new(DataSet::load_default().expect("bundled data must load"));
    let config = GameConfig::with_seed(1);
    let game = Game::new(Arc::clone(&data), config.clone()).expect("game must start");
    let state: &State = &game.state;

    println!("agot-core clone/step bench (release)");
    println!(
        "board: {} regions, {} houses, State is {} bytes",
        state.region_count,
        state.present_houses().len(),
        std::mem::size_of::<State>()
    );
    println!();

    let clone_state = time(1_000_000, || {
        let source = black_box(state);
        let copy = source.clone();
        (copy.round, copy.region_count, copy.houses[0].is_some())
    });
    println!("State::clone            {clone_state:9.1} ns   (Python GameState.clone ~5 900 ns)");

    let clone_game = time(1_000_000, || {
        let source = black_box(&game);
        let copy = source.clone();
        (copy.state.round, copy.state.region_count)
    });
    println!("Game::clone             {clone_game:9.1} ns   (Python Game.clone ~12 000 ns)");

    let actions = game.legal_actions();
    let legal_actions = time(200_000, || {
        let source = black_box(&game);
        source.legal_actions().len()
    });
    println!(
        "Game::legal_actions     {legal_actions:9.1} ns   ({} actions in the opening)",
        actions.len()
    );

    let action = actions[0];
    let clone_and_step = time(200_000, || {
        let mut child = black_box(&game).clone();
        let stepped = child.step(action).is_ok();
        (stepped, child.state.region_count)
    });
    println!("clone + step            {clone_and_step:9.1} ns   (the MCTS inner loop)");

    // A full Planning Phase: how many decisions per second can the engine drive?
    let phase_iterations = 2_000;
    let start = Instant::now();
    let mut decisions = 0u64;
    for _ in 0..phase_iterations {
        let mut child = Game::new(Arc::clone(&data), config.clone()).unwrap();
        let mut guard = 0;
        while child.state.phase == agot_core::Phase::Planning {
            guard += 1;
            if guard > 500 {
                break;
            }
            let actions = child.legal_actions();
            if actions.is_empty() {
                break;
            }
            let _ = child.step(actions[0]);
            decisions += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!();
    println!(
        "planning phase          {:.0} decisions/s ({} phases in {:.3} s)",
        decisions as f64 / elapsed,
        phase_iterations,
        elapsed
    );

    // Python walks a whole random game at ~5 700 decisions/s; the planning phase
    // is its heaviest legal-action generator.
    println!();
    println!(
        "speedups vs Python: State::clone x{:.0}, Game::clone x{:.0}",
        5_900.0 / clone_state,
        12_000.0 / clone_game
    );
}
