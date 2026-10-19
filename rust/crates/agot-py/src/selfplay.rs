//! Batched self-play driven from Rust.
//!
//! The training hot path crosses the language boundary as rarely as possible:
//! whole games are played here, the search runs in Rust, and the neural network
//! is called once per *batch of leaves* rather than once per decision. Python
//! supplies one callback — a pure torch call — and receives packed arrays.
//!
//! The protocol mirrors `agot.training.selfplay.batched_self_play`:
//!
//! * several games advance in lockstep (`batch_games`), so one forward pass
//!   scores many leaves;
//! * every decision is recorded as the state features, the candidate-action
//!   matrix and the visit distribution the search produced at the scheduled
//!   temperature — the target the network is trained to imitate;
//! * `determinizations > 1` gives each seat PIMC, aggregating the root visits of
//!   that many resampled worlds.
//!
//! The evaluator callback receives
//! `(state (B, STATE_DIM), actions (T, ACTION_DIM), index (T,), mask (T,))` as
//! float32/int32 arrays and returns `(logits (T,), values (B, N_HOUSES))`. Rust
//! applies the per-leaf softmax, so the callback stays a single torch expression.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use agot_core::enums::{HouseId, N_HOUSES};
use agot_core::game::Game;
use agot_core::map::DataSet;
use agot_core::observation::{self, ACTION_DIM, STATE_DIM};
use agot_core::search::{Evaluation, Mcts, SearchResult, Simulation, UniformEvaluator};
use agot_core::setup::GameConfig;

/// One recorded decision, before the game's outcome is known.
struct Record {
    state: Vec<f32>,
    actions: Vec<f32>,
    policy: Vec<f32>,
    player: u8,
}

struct FinishedGame {
    examples: Vec<Record>,
    winner: Option<HouseId>,
    steps: u32,
    rounds: u8,
}

fn spec_value<'py, T>(spec: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    match spec.get_item(key)? {
        Some(value) => Ok(Some(value.extract()?)),
        None => Ok(None),
    }
}

fn spec_or<'py, T>(spec: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    Ok(spec_value::<T>(spec, key)?.unwrap_or(default))
}

/// The temperature for a round, from a `(last_round, temperature)` schedule.
fn scheduled_temperature(schedule: &[(u8, f64)], round: u8) -> f64 {
    for (last_round, temperature) in schedule {
        if round <= *last_round {
            return *temperature;
        }
    }
    schedule
        .last()
        .map(|(_, temperature)| *temperature)
        .unwrap_or(1.0)
}

/// Per-leaf softmax over a flat logit vector, one segment per leaf.
fn segment_softmax(logits: &[f32], index: &[i64], rows: usize) -> Vec<f32> {
    let mut maxima = vec![f32::NEG_INFINITY; rows];
    for (slot, row) in index.iter().enumerate() {
        let row = *row as usize;
        if logits[slot] > maxima[row] {
            maxima[row] = logits[slot];
        }
    }
    let mut totals = vec![0.0f32; rows];
    let mut out = vec![0.0f32; logits.len()];
    for (slot, row) in index.iter().enumerate() {
        let row = *row as usize;
        let value = (logits[slot] - maxima[row]).exp();
        out[slot] = value;
        totals[row] += value;
    }
    for (slot, row) in index.iter().enumerate() {
        let total = totals[*row as usize];
        out[slot] = if total > 0.0 { out[slot] / total } else { 0.0 };
    }
    out
}

fn array2(py: Python<'_>, rows: usize, cols: usize, flat: Vec<f32>) -> PyResult<Py<PyArray2<f32>>> {
    let array = numpy::ndarray::Array2::from_shape_vec((rows, cols), flat)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py).unbind())
}

/// Plays `spec["games"]` games, batching every network call.
#[pyfunction]
pub fn selfplay<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
    evaluator: Py<PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let games: usize = spec_or(spec, "games", 1)?;
    let simulations: u32 = spec_or(spec, "simulations", 32)?;
    let determinizations: u32 = spec_or(spec, "determinizations", 1)?.max(1);
    let batch_games: usize = spec_or(spec, "batch_games", 1)?.max(1);
    let seed: u64 = spec_or(spec, "seed", 0)?;
    let temperature: f64 = spec_or(spec, "temperature", 1.0)?;
    let temperature_rounds: u8 = spec_or(spec, "temperature_rounds", 3)?;
    let final_temperature: f64 = spec_or(spec, "final_temperature", 0.25)?;
    let max_rounds: u8 = spec_or(spec, "max_rounds", 10)?;
    let victory_castles: u8 = spec_or(spec, "victory_castles", 7)?;
    let use_westeros_decks: bool = spec_or(spec, "use_westeros_decks", true)?;
    let houses: Option<Vec<String>> = spec_value(spec, "houses")?;
    let data_dir: Option<String> = spec_value(spec, "data_dir")?;
    let step_limit: u32 = spec_or(spec, "step_limit", 20_000)?;

    if games == 0 {
        return Ok(PyTuple::empty(py));
    }

    let data = Arc::new(
        match data_dir {
            Some(dir) => DataSet::load(std::path::Path::new(&dir)),
            None => DataSet::load_default(),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))?,
    );

    let house_list = match houses {
        Some(names) => names
            .iter()
            .map(|name| {
                HouseId::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))
            })
            .collect::<PyResult<smallvec::SmallVec<[HouseId; 6]>>>()?,
        None => HouseId::ALL.into_iter().collect(),
    };

    let schedule: Vec<(u8, f64)> = vec![
        (temperature_rounds, temperature),
        (u8::MAX, final_temperature),
    ];
    let per_world = (simulations / determinizations).max(1);

    let mut finished: Vec<FinishedGame> = Vec::with_capacity(games);
    for start in (0..games).step_by(batch_games) {
        let chunk: Vec<usize> = (start..(start + batch_games).min(games)).collect();
        let mut configs = Vec::with_capacity(chunk.len());
        for index in &chunk {
            configs.push(GameConfig {
                houses: house_list.clone(),
                max_rounds,
                victory_castles,
                seed: seed + *index as u64,
                use_westeros_decks,
            });
        }

        let mut states: Vec<Game> = configs
            .iter()
            .map(|config| {
                Game::new(Arc::clone(&data), config.clone())
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
            })
            .collect::<PyResult<Vec<Game>>>()?;

        // One searcher per game, so PIMC worlds and noise stay independent.
        let mut searchers: Vec<Mcts<UniformEvaluator>> = chunk
            .iter()
            .map(|index| Mcts::with_seed(UniformEvaluator, seed + *index as u64 * 101))
            .collect();
        let mut histories: Vec<Vec<Record>> = (0..chunk.len()).map(|_| Vec::new()).collect();
        let mut steps = vec![0u32; chunk.len()];
        let mut winners: Vec<Option<HouseId>> = vec![None; chunk.len()];

        loop {
            // Which games still need a decision, and their roots for this one.
            let mut active: Vec<(usize, Vec<u32>)> = Vec::new();
            for (slot, game) in states.iter().enumerate() {
                if game.is_terminal() || steps[slot] >= step_limit {
                    if winners[slot].is_none() {
                        winners[slot] = game.winner();
                    }
                    continue;
                }
                if game.current_player().is_none() {
                    winners[slot] = game.winner();
                    continue;
                }
                let roots = searchers[slot].open_search_worlds(game);
                if roots.is_empty() {
                    winners[slot] = game.winner();
                    continue;
                }
                active.push((slot, roots));
            }
            if active.is_empty() {
                break;
            }

            // Search every world of every active game, batching the leaves.
            for _ in 0..per_world {
                let mut pending: Vec<(usize, Simulation)> = Vec::new();
                for (slot, roots) in &active {
                    for root in roots {
                        let simulation = searchers[*slot]
                            .prepare_simulation(*root)
                            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                        pending.push((*slot, simulation));
                    }
                }

                // Encode every leaf that needs a network score.
                let mut leaves: Vec<usize> = Vec::new();
                let mut leaf_states: Vec<f32> = Vec::new();
                let mut leaf_actions: Vec<f32> = Vec::new();
                // int64 because the callback feeds this straight to torch's
                // advanced indexing, exactly as `predict_many` does.
                let mut leaf_index: Vec<i64> = Vec::new();
                for (position, (slot, simulation)) in pending.iter().enumerate() {
                    if simulation.leaf.is_none() {
                        continue;
                    }
                    let row = leaves.len();
                    let game = searchers[*slot]
                        .leaf_game(simulation)
                        .ok_or_else(|| PyRuntimeError::new_err("a leaf without a position"))?;
                    let mut cache = observation::RegionCache::new();
                    leaf_states.extend_from_slice(&observation::encode_state(
                        &game.state,
                        &data,
                        None,
                    ));
                    for action in &simulation.actions {
                        let mut buf = vec![0.0f32; ACTION_DIM];
                        observation::encode_action_into(
                            &game.state,
                            &data,
                            action,
                            None,
                            &mut buf,
                            &mut cache,
                        );
                        leaf_actions.extend_from_slice(&buf);
                        leaf_index.push(row as i64);
                    }
                    leaves.push(position);
                }

                // One callback for the whole batch.
                let mut evaluations: Vec<Option<Evaluation>> = pending
                    .iter()
                    .map(|(_, simulation)| {
                        if simulation.leaf.is_some() {
                            None
                        } else {
                            Some(Evaluation {
                                priors: Vec::new(),
                                value: simulation.value.unwrap_or([0.0; N_HOUSES]),
                            })
                        }
                    })
                    .collect();

                if !leaves.is_empty() {
                    let rows = leaves.len();
                    let total = leaf_index.len();
                    let state_array = array2(py, rows, STATE_DIM, leaf_states)?;
                    let action_array = array2(py, total, ACTION_DIM, leaf_actions)?;
                    let index_array = numpy::ndarray::Array1::from_vec(leaf_index.clone())
                        .into_pyarray(py)
                        .unbind();
                    let mask = vec![1.0f32; total];
                    let mask_array = PyArray1::from_vec(py, mask).unbind();

                    let reply = evaluator
                        .call1(py, (state_array, action_array, index_array, mask_array))?;
                    let (logits, values): (PyReadonlyArray1<f32>, PyReadonlyArray2<f32>) =
                        reply.extract(py)?;
                    let logits = logits.as_slice()?;
                    let values = values.as_slice()?;
                    if logits.len() != total {
                        return Err(PyValueError::new_err(format!(
                            "evaluator returned {} logits for {total} candidates",
                            logits.len()
                        )));
                    }
                    if values.len() != rows * N_HOUSES {
                        return Err(PyValueError::new_err(format!(
                            "evaluator returned {} values for {rows} leaves",
                            values.len()
                        )));
                    }
                    let priors = segment_softmax(logits, &leaf_index, rows);

                    // Scatter each leaf's slice back to its simulation.
                    let mut bounds = Vec::with_capacity(rows + 1);
                    bounds.push(0usize);
                    for position in &leaves {
                        bounds.push(
                            bounds.last().copied().unwrap_or(0)
                                + pending[*position].1.actions.len(),
                        );
                    }
                    for (row, position) in leaves.iter().enumerate() {
                        let mut value = [0.0f64; N_HOUSES];
                        for (slot, entry) in value.iter_mut().enumerate() {
                            *entry = values[row * N_HOUSES + slot] as f64;
                        }
                        evaluations[*position] = Some(Evaluation {
                            priors: priors[bounds[row]..bounds[row + 1]]
                                .iter()
                                .map(|value| *value as f64)
                                .collect(),
                            value,
                        });
                    }
                }

                for (slot, simulation) in pending {
                    searchers[slot]
                        .finish_simulation(simulation, evaluations.remove(0))
                        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                }
            }

            // Aggregate the worlds, pick, record, step.
            for (slot, roots) in &active {
                let observer = states[*slot].current_player();
                let result: SearchResult = if roots.len() == 1 {
                    searchers[*slot].result_from_root(roots[0])
                } else {
                    searchers[*slot].aggregate_roots(roots, observer)
                };
                let round = states[*slot].state.round;
                let temperature = scheduled_temperature(&schedule, round);
                let (action, target) = searchers[*slot]
                    .pick(&result, Some(temperature))
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

                let observation =
                    observation::observe(&states[*slot].state, &data, &result.actions, observer);
                histories[*slot].push(Record {
                    state: observation.state_features,
                    actions: observation.action_features,
                    policy: target,
                    player: observer.map(|house| house.index() as u8).unwrap_or(0),
                });

                states[*slot]
                    .step(action)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                steps[*slot] += 1;
            }
        }

        for (slot, game) in states.iter().enumerate() {
            winners[slot] = winners[slot].or_else(|| game.winner());
            finished.push(FinishedGame {
                examples: std::mem::take(&mut histories[slot]),
                winner: winners[slot],
                steps: steps[slot],
                rounds: game.state.round,
            });
        }
    }

    // Pack the examples into flat buffers the replay buffer can slice.
    let total_examples: usize = finished.iter().map(|game| game.examples.len()).sum();
    let total_rows: usize = finished
        .iter()
        .flat_map(|game| game.examples.iter())
        .map(|record| record.policy.len())
        .sum();

    let mut states_flat: Vec<f32> = Vec::with_capacity(total_examples * STATE_DIM);
    let mut actions_flat: Vec<f32> = Vec::with_capacity(total_rows * ACTION_DIM);
    let mut policy_flat: Vec<f32> = Vec::with_capacity(total_rows);
    let mut offsets: Vec<u32> = Vec::with_capacity(total_examples + 1);
    let mut players: Vec<u8> = Vec::with_capacity(total_examples);
    let mut outcomes: Vec<f32> = Vec::with_capacity(total_examples * N_HOUSES);
    let mut winners: Vec<Option<String>> = Vec::with_capacity(finished.len());
    let mut steps: Vec<u32> = Vec::with_capacity(finished.len());
    let mut rounds: Vec<u32> = Vec::with_capacity(finished.len());

    offsets.push(0);
    for game in &finished {
        let mut outcome = [0.0f64; N_HOUSES];
        let present = states_present(&game.examples, &data);
        for (slot, entry) in outcome.iter_mut().enumerate() {
            let house = HouseId(slot as u8);
            if !present[slot] {
                continue;
            }
            *entry = if Some(house) == game.winner {
                1.0
            } else {
                -1.0
            };
        }
        for record in &game.examples {
            states_flat.extend_from_slice(&record.state);
            actions_flat.extend_from_slice(&record.actions);
            policy_flat.extend_from_slice(&record.policy);
            players.push(record.player);
            for entry in outcome {
                outcomes.push(entry as f32);
            }
            offsets.push(offsets.last().copied().unwrap_or(0) + record.policy.len() as u32);
        }
        winners.push(game.winner.map(|house| house.name().to_string()));
        steps.push(game.steps);
        rounds.push(game.rounds as u32);
    }

    let items: Vec<Py<PyAny>> = vec![
        array2(py, total_examples, STATE_DIM, states_flat)?.into_any(),
        array2(py, total_rows, ACTION_DIM, actions_flat)?.into_any(),
        numpy::ndarray::Array1::from_vec(offsets)
            .into_pyarray(py)
            .unbind()
            .into_any(),
        PyArray1::from_vec(py, policy_flat).unbind().into_any(),
        PyArray1::from_vec(py, players).unbind().into_any(),
        array2(py, total_examples, N_HOUSES, outcomes)?.into_any(),
        winners.into_pyobject(py)?.unbind().into_any(),
        steps.into_pyobject(py)?.unbind().into_any(),
        rounds.into_pyobject(py)?.unbind().into_any(),
    ];
    Ok(PyTuple::new(py, items)?)
}

/// Which canonical houses take part, inferred from the recorded examples.
///
/// The outcome vector gives `+1`/`-1` to the houses in the game and leaves the
/// absent ones at zero, which is what `outcome_vector` does in Python.
fn states_present(examples: &[Record], _data: &DataSet) -> [bool; N_HOUSES] {
    let mut present = [false; N_HOUSES];
    for record in examples {
        present[record.player as usize] = true;
    }
    // A house that never acted may still be in the game; fall back to all houses
    // when nothing was recorded, which keeps short-handed games from silently
    // dropping a seat.
    if !present.iter().any(|flag| *flag) {
        present = [true; N_HOUSES];
    }
    present
}

/// The terminal reward vector for a finished game, as Python's
/// `outcome_vector` computes it.
#[pyfunction]
pub fn game_outcome(winner: Option<String>, houses: Vec<String>) -> PyResult<Vec<f32>> {
    let winner = match winner {
        Some(name) => Some(
            HouseId::from_name(&name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))?,
        ),
        None => None,
    };
    let mut present = [false; N_HOUSES];
    for name in &houses {
        let house = HouseId::from_name(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))?;
        present[house.index()] = true;
    }
    let mut out = vec![0.0f32; N_HOUSES];
    for (slot, entry) in out.iter_mut().enumerate() {
        if present[slot] {
            *entry = if Some(HouseId(slot as u8)) == winner {
                1.0
            } else {
                -1.0
            };
        }
    }
    Ok(out)
}
