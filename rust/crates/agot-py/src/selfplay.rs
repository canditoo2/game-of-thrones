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

/// The leaves of one batch that belong to the same evaluator.
///
/// A game may mix models by seat (net-vs-net evaluation); each model gets its
/// own callback instead of every leaf going through one shared model.
struct LeafGroup {
    model: usize,
    positions: Vec<usize>,
    states: Vec<f32>,
    actions: Vec<f32>,
    index: Vec<i64>,
}

impl LeafGroup {
    fn new(model: usize) -> LeafGroup {
        LeafGroup {
            model,
            positions: Vec::new(),
            states: Vec::new(),
            actions: Vec::new(),
            index: Vec::new(),
        }
    }
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
///
/// Seats named in `spec["scripted"]` are driven by `scripted_decider` once per
/// decision instead of by search; its call is `(game_index, house, game_view)`
/// and it returns `(action_index, target | None)`, where `target` is only
/// required when `spec["record_opponents"]` is true. Search seats remain fully
/// native; only the scripted seat's own decisions cross the boundary.
#[pyfunction]
#[pyo3(signature = (spec, evaluator, scripted_decider=None, extra_evaluators=None))]
pub fn selfplay<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
    evaluator: Py<PyAny>,
    scripted_decider: Option<Py<PyAny>>,
    extra_evaluators: Option<Vec<Py<PyAny>>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut evaluators: Vec<Py<PyAny>> = Vec::with_capacity(
        1 + extra_evaluators
            .as_ref()
            .map(|extra| extra.len())
            .unwrap_or(0),
    );
    evaluators.push(evaluator);
    if let Some(extra) = extra_evaluators {
        evaluators.extend(extra);
    }
    // Optional per-game, per-house model index. Absent means every search seat
    // uses evaluator 0, which is the normal self-play request.
    let seat_models: Vec<Vec<Option<usize>>> = spec_or(spec, "seat_models", Vec::new())?;

    let games: usize = spec_or(spec, "games", 1)?;
    let simulations: u32 = spec_or(spec, "simulations", 32)?;
    let determinizations: u32 = spec_or(spec, "determinizations", 1)?.max(1);
    let batch_games: usize = spec_or(spec, "batch_games", 1)?.max(1);
    let seed: u64 = spec_or(spec, "seed", 0)?;
    // Optional explicit per-game seeds. Without them the driver keeps the old
    // `seed + index` derivation, which is only correct when one worker sees a
    // contiguous run of games.
    let seeds: Vec<u64> = spec_or(spec, "seeds", Vec::new())?;
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

    // `scripted` is one list of house names per game; a missing entry is an
    // all-search game, which keeps the common request free of empty vectors.
    let scripted_names: Vec<Vec<String>> = spec_or(spec, "scripted", Vec::new())?;
    let mut scripted_houses: Vec<Vec<HouseId>> = Vec::with_capacity(games);
    for index in 0..games {
        let mut houses = Vec::new();
        if let Some(names) = scripted_names.get(index) {
            for name in names {
                houses.push(HouseId::from_name(name).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown scripted house {name:?}"))
                })?);
            }
        }
        scripted_houses.push(houses);
    }
    let record_opponents: bool = spec_or(spec, "record_opponents", false)?;

    let schedule: Vec<(u8, f64)> = vec![
        (temperature_rounds, temperature),
        (u8::MAX, final_temperature),
    ];
    let per_world = (simulations / determinizations).max(1);

    let mut finished: Vec<FinishedGame> = Vec::with_capacity(games);
    for start in (0..games).step_by(batch_games) {
        let chunk: Vec<usize> = (start..(start + batch_games).min(games)).collect();
        let game_seed =
            |index: usize| -> u64 { seeds.get(index).copied().unwrap_or(seed + index as u64) };
        let mut configs = Vec::with_capacity(chunk.len());
        for index in &chunk {
            configs.push(GameConfig {
                houses: house_list.clone(),
                max_rounds,
                victory_castles,
                seed: game_seed(*index),
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
            .map(|index| Mcts::with_seed(UniformEvaluator, game_seed(*index)))
            .collect();
        let mut histories: Vec<Vec<Record>> = (0..chunk.len()).map(|_| Vec::new()).collect();
        let mut steps = vec![0u32; chunk.len()];
        let mut winners: Vec<Option<HouseId>> = vec![None; chunk.len()];

        loop {
            // Advance every game to its next decision. Scripted seats decide
            // immediately (possibly several in a row); search seats open their
            // roots and wait for the shared leaf batch below.
            let mut active: Vec<(usize, Vec<u32>)> = Vec::new();
            let mut acted = false;
            for slot in 0..states.len() {
                loop {
                    if states[slot].is_terminal() || steps[slot] >= step_limit {
                        if winners[slot].is_none() {
                            winners[slot] = states[slot].winner();
                        }
                        break;
                    }
                    let Some(house) = states[slot].current_player() else {
                        winners[slot] = states[slot].winner();
                        break;
                    };
                    let global = start + slot;
                    let is_scripted = scripted_houses
                        .get(global)
                        .map(|houses| houses.contains(&house))
                        .unwrap_or(false);
                    if !is_scripted {
                        let roots = searchers[slot].open_search_worlds(&states[slot]);
                        if roots.is_empty() {
                            winners[slot] = states[slot].winner();
                        } else {
                            active.push((slot, roots));
                        }
                        break;
                    }

                    let decider = scripted_decider.as_ref().ok_or_else(|| {
                        PyRuntimeError::new_err(
                            "spec names a scripted seat but no scripted_decider was supplied",
                        )
                    })?;
                    let view = Py::new(py, crate::PyGame::from_game(states[slot].clone()))?;
                    let reply = decider.call1(py, (global, house.name(), view))?;
                    let (index, target): (usize, Option<Vec<f32>>) = reply.extract(py)?;
                    let actions = states[slot].legal_actions();
                    if index >= actions.len() {
                        return Err(PyValueError::new_err(format!(
                            "scripted policy chose action {index} of {}",
                            actions.len()
                        )));
                    }
                    if record_opponents {
                        let target = target.ok_or_else(|| {
                            PyValueError::new_err(
                                "record_opponents needs the scripted policy's target",
                            )
                        })?;
                        if target.len() != actions.len() {
                            return Err(PyValueError::new_err(format!(
                                "scripted target has {} entries for {} actions",
                                target.len(),
                                actions.len()
                            )));
                        }
                        let observation =
                            observation::observe(&states[slot].state, &data, &actions, Some(house));
                        histories[slot].push(Record {
                            state: observation.state_features,
                            actions: observation.action_features,
                            policy: target,
                            player: house.index() as u8,
                        });
                    }
                    states[slot]
                        .step(actions[index])
                        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                    steps[slot] += 1;
                    acted = true;
                }
            }
            if active.is_empty() {
                if acted {
                    continue;
                }
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

                // Encode every leaf, grouped by the model that owns its seat.
                // One cache for the whole batch: each leaf clears it, which is
                // far cheaper than allocating (and zeroing) 64 entries per leaf.
                let mut cache = observation::RegionCache::new();
                let mut groups: Vec<LeafGroup> = Vec::new();
                for (position, (slot, simulation)) in pending.iter().enumerate() {
                    if simulation.leaf.is_none() {
                        continue;
                    }
                    let game = searchers[*slot]
                        .leaf_game(simulation)
                        .ok_or_else(|| PyRuntimeError::new_err("a leaf without a position"))?;
                    let house = game.current_player().ok_or_else(|| {
                        PyRuntimeError::new_err("a search leaf without an acting house")
                    })?;
                    let model = seat_models
                        .get(start + slot)
                        .and_then(|models| models.get(house.index()))
                        .copied()
                        .flatten()
                        .unwrap_or(0);
                    if model >= evaluators.len() {
                        return Err(PyValueError::new_err(format!(
                            "seat model {model} is out of range for {} evaluators",
                            evaluators.len()
                        )));
                    }
                    let group_index = match groups.iter().position(|group| group.model == model) {
                        Some(index) => index,
                        None => {
                            groups.push(LeafGroup::new(model));
                            groups.len() - 1
                        }
                    };
                    let group = &mut groups[group_index];
                    let row = group.positions.len();
                    cache.clear();
                    let state_start = group.states.len();
                    group.states.resize(state_start + STATE_DIM, 0.0);
                    observation::encode_state_into(
                        &game.state,
                        &data,
                        None,
                        &mut group.states[state_start..],
                    );
                    for action in &simulation.actions {
                        let action_start = group.actions.len();
                        group.actions.resize(action_start + ACTION_DIM, 0.0);
                        observation::encode_action_into(
                            &game.state,
                            &data,
                            action,
                            None,
                            &mut group.actions[action_start..],
                            &mut cache,
                        );
                        group.index.push(row as i64);
                    }
                    group.positions.push(position);
                }

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

                for group in groups {
                    let rows = group.positions.len();
                    if rows == 0 {
                        continue;
                    }
                    let total = group.index.len();
                    let state_array = array2(py, rows, STATE_DIM, group.states)?;
                    let action_array = array2(py, total, ACTION_DIM, group.actions)?;
                    let index_array = numpy::ndarray::Array1::from_vec(group.index.clone())
                        .into_pyarray(py)
                        .unbind();
                    let mask = vec![1.0f32; total];
                    let mask_array = PyArray1::from_vec(py, mask).unbind();

                    let reply = evaluators[group.model]
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
                    let priors = segment_softmax(logits, &group.index, rows);

                    // Scatter each leaf's slice back to its simulation.
                    let mut bounds = Vec::with_capacity(rows + 1);
                    bounds.push(0usize);
                    for position in &group.positions {
                        bounds.push(
                            bounds.last().copied().unwrap_or(0)
                                + pending[*position].1.actions.len(),
                        );
                    }
                    for (row, position) in group.positions.iter().enumerate() {
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

                for ((slot, simulation), evaluation) in pending.into_iter().zip(evaluations) {
                    searchers[slot]
                        .finish_simulation(simulation, evaluation)
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
    let mut examples_per_game: Vec<u32> = Vec::with_capacity(finished.len());
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
        // Scripted seats are not recorded unless asked, so `steps` is the
        // number of decisions while this is the number of examples. They are
        // equal for a pure-search game and must be kept separate in the
        // protocol: `steps` is also the game length the trainer logs.
        examples_per_game.push(game.examples.len() as u32);
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
        examples_per_game.into_pyobject(py)?.unbind().into_any(),
        rounds.into_pyobject(py)?.unbind().into_any(),
    ];
    PyTuple::new(py, items)
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
