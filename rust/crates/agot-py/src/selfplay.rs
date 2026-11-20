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
//!   that many resampled worlds;
//! * `phase_budgets` (default `null`) caps the simulations a decision may spend
//!   by phase, so a Planning decision stops buying leaves the target cannot use;
//!   the cap is per decision and is divided by the determinizations exactly as
//!   the fixed `simulations` setting is, so an existing run keeps its split;
//! * bit-identical leaves are evaluated once per batch (`dedupe_leaves`,
//!   default on): the driver identifies each leaf by its state row plus its
//!   ordered action rows, scores the unique leaves, and scatters the logits
//!   and values back to the original packed positions, so the search, the
//!   seeds and the recorded examples are unchanged.
//!
//! The evaluator callback receives
//! `(state (B, STATE_DIM), actions (T, ACTION_DIM), index (T,), mask (T,))` as
//! float32/int32 arrays and returns `(logits (T,), values (B, N_HOUSES))`. Rust
//! applies the per-leaf softmax, so the callback stays a single torch expression.

use std::sync::Arc;
use std::time::{Duration, Instant};

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

use agot_core::budget::{decision_budget, PhaseBudgets};
use agot_core::enums::{HouseId, Phase, N_HOUSES};
use agot_core::game::Game;
use agot_core::leaf_cache::{self, LeafCache, LeafKey, LeafKeyMap};
use agot_core::leaf_dedupe::LeafDedupe;
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

/// Leaf-evaluation counters for one `selfplay()` call.
///
/// Read from Python with `selfplay_stats()` after the call returns; the
/// driver's return tuple is unchanged, so existing consumers keep working.
#[derive(Clone, Copy, Debug, Default)]
struct DriverStats {
    /// Leaves that needed a network score.
    leaves: u64,
    /// Candidate rows of those leaves.
    slots: u64,
    /// Leaves answered by the cross-pass cache.
    cache_hits: u64,
    /// Candidate rows of those leaves.
    cache_hit_slots: u64,
    /// Leaves merged into an identical leaf of the same pass.
    pass_duplicates: u64,
    /// Evaluator callbacks made.
    evaluator_calls: u64,
    /// Passes whose leaves were all answered without a callback.
    calls_skipped: u64,
    /// Leaves and candidate rows actually sent to the evaluator.
    rows_sent: u64,
    slots_sent: u64,
    /// Cache bookkeeping, summed over the per-evaluator caches.
    cache_inserts: u64,
    cache_evicted: u64,
    cache_rotations: u64,
    cache_verified: u64,
    cache_peak_entries: u64,
    cache_peak_logits: u64,
    /// Wall time by section (a few clock reads per pass, never per leaf).
    time_total: Duration,
    /// Advancing games to their next decision, scripted seats included.
    time_advance: Duration,
    /// Of which: the scripted decider's Python calls.
    time_scripted: Duration,
    /// `prepare_simulation` (tree descent and child materialization).
    time_prepare: Duration,
    /// Leaf positions and their network-input encoding.
    time_encode: Duration,
    /// Leaf evaluation: cache/dedupe, the callback, softmax and scatter.
    time_evaluate: Duration,
    /// Of which: the evaluator callback (array handover, Python, extraction).
    time_callback: Duration,
    /// `finish_simulation` (expansion and backup).
    time_finish: Duration,
    /// Aggregate, pick, record (observe) and step each decision.
    time_record: Duration,
}

impl DriverStats {
    fn to_dict<'py>(self, py: Python<'py>, leaf_cache: bool) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("leaf_cache", leaf_cache)?;
        dict.set_item("leaves", self.leaves)?;
        dict.set_item("slots", self.slots)?;
        dict.set_item("cache_hits", self.cache_hits)?;
        dict.set_item("cache_hit_slots", self.cache_hit_slots)?;
        dict.set_item("pass_duplicates", self.pass_duplicates)?;
        dict.set_item("evaluator_calls", self.evaluator_calls)?;
        dict.set_item("calls_skipped", self.calls_skipped)?;
        dict.set_item("rows_sent", self.rows_sent)?;
        dict.set_item("slots_sent", self.slots_sent)?;
        dict.set_item("cache_inserts", self.cache_inserts)?;
        dict.set_item("cache_evicted", self.cache_evicted)?;
        dict.set_item("cache_rotations", self.cache_rotations)?;
        dict.set_item("cache_verified", self.cache_verified)?;
        dict.set_item("cache_peak_entries", self.cache_peak_entries)?;
        dict.set_item("cache_peak_logits", self.cache_peak_logits)?;
        for (name, time) in [
            ("seconds_total", self.time_total),
            ("seconds_advance", self.time_advance),
            ("seconds_scripted", self.time_scripted),
            ("seconds_prepare", self.time_prepare),
            ("seconds_encode", self.time_encode),
            ("seconds_evaluate", self.time_evaluate),
            ("seconds_callback", self.time_callback),
            ("seconds_finish", self.time_finish),
            ("seconds_record", self.time_record),
        ] {
            dict.set_item(name, time.as_secs_f64())?;
        }
        Ok(dict)
    }
}

/// The counters of the most recent completed `selfplay()` call in this
/// process, with whether its cache was on.
static LAST_STATS: std::sync::Mutex<Option<(DriverStats, bool)>> = std::sync::Mutex::new(None);

/// Leaf-evaluation counters of the last completed `selfplay()` call in this
/// process (`None` before the first one).
///
/// `leaves` counts the leaves that needed a score; `cache_hits` were answered
/// from an earlier pass of the same call, `pass_duplicates` were merged with
/// an identical leaf of their own pass, and `rows_sent` / `slots_sent` are what
/// actually reached the evaluator in `evaluator_calls` callbacks.
#[pyfunction]
pub fn selfplay_stats(py: Python<'_>) -> PyResult<Option<Bound<'_, PyDict>>> {
    let last = *LAST_STATS
        .lock()
        .map_err(|_| PyRuntimeError::new_err("selfplay stats lock poisoned"))?;
    match last {
        Some((stats, leaf_cache)) => Ok(Some(stats.to_dict(py, leaf_cache)?)),
        None => Ok(None),
    }
}

/// Score one model group without the cache: the whole group in one callback,
/// optionally merged by the in-pass [`LeafDedupe`]. This is the path every run
/// took before the cache, kept for A/B runs (`leaf_cache = False`).
fn evaluate_direct(
    py: Python<'_>,
    evaluator: &Py<PyAny>,
    group: &LeafGroup,
    dedupe_leaves: bool,
    stats: &mut DriverStats,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let rows = group.positions.len();
    let total = group.index.len();

    // Index the batch by exact contents. A leaf is identified by its state row
    // *and* its ordered action rows, so two positions with the same board but
    // different candidates stay separate. The evaluator never sees a duplicate.
    let dedupe = if dedupe_leaves {
        Some(LeafDedupe::from_packed(
            &group.states,
            STATE_DIM,
            &group.actions,
            ACTION_DIM,
            &group.index,
            rows,
        ))
    } else {
        None
    };
    let (scored_rows, scored_slots) = match &dedupe {
        Some(dedupe) => (dedupe.unique_rows(), dedupe.unique_slots()),
        None => (rows, total),
    };
    let (scored_states, scored_actions, scored_index) = match &dedupe {
        Some(dedupe) => (
            dedupe.states().to_vec(),
            dedupe.actions().to_vec(),
            dedupe.index().to_vec(),
        ),
        None => (
            group.states.clone(),
            group.actions.clone(),
            group.index.clone(),
        ),
    };
    stats.pass_duplicates += (rows - scored_rows) as u64;
    stats.evaluator_calls += 1;
    stats.rows_sent += scored_rows as u64;
    stats.slots_sent += scored_slots as u64;

    let callback_started = Instant::now();
    let state_array = array2(py, scored_rows, STATE_DIM, scored_states)?;
    let action_array = array2(py, scored_slots, ACTION_DIM, scored_actions)?;
    let index_array = numpy::ndarray::Array1::from_vec(scored_index)
        .into_pyarray(py)
        .unbind();
    let mask = vec![1.0f32; scored_slots];
    let mask_array = PyArray1::from_vec(py, mask).unbind();

    let reply = evaluator.call1(py, (state_array, action_array, index_array, mask_array))?;
    let (logits, values): (PyReadonlyArray1<f32>, PyReadonlyArray2<f32>) = reply.extract(py)?;
    let logits = logits.as_slice()?;
    let values = values.as_slice()?;
    stats.time_callback += callback_started.elapsed();
    if logits.len() != scored_slots {
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} logits for {scored_slots} candidates",
            logits.len()
        )));
    }
    if values.len() != scored_rows * N_HOUSES {
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} values for {scored_rows} leaves",
            values.len()
        )));
    }

    // Restore the full packed layout before the per-leaf softmax: each
    // duplicate gets exactly the bits its own forward pass would have produced.
    Ok(match &dedupe {
        Some(dedupe) => (
            dedupe.scatter_logits(logits),
            dedupe.scatter_values(values, N_HOUSES),
        ),
        None => (logits.to_vec(), values.to_vec()),
    })
}

/// Score one model group through the cross-pass evaluation cache.
///
/// Every leaf is looked up by its exact network input. Hits are copied from
/// the cache; misses are packed in first-occurrence order (an input repeated
/// inside the pass is sent once) and sent in one callback, or no callback at
/// all when nothing is new. Returns the raw logits in the group's packed slot
/// order and the value rows in its leaf order, exactly the layout the
/// evaluator would have returned for the whole group.
fn evaluate_cached(
    py: Python<'_>,
    evaluator: &Py<PyAny>,
    cache: &mut LeafCache,
    verify: bool,
    group: &LeafGroup,
    bounds: &[usize],
    stats: &mut DriverStats,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let rows = group.positions.len();
    let total = group.index.len();
    let mut logits = vec![0.0f32; total];
    let mut values = vec![0.0f32; rows * N_HOUSES];

    let mismatch = |error: leaf_cache::CacheMismatch| PyRuntimeError::new_err(error.to_string());
    let state_of = |row: usize| &group.states[row * STATE_DIM..(row + 1) * STATE_DIM];
    let actions_of =
        |row: usize| &group.actions[bounds[row] * ACTION_DIM..bounds[row + 1] * ACTION_DIM];

    // The misses, packed as the evaluator wants them.
    let mut miss_keys: Vec<LeafKey> = Vec::new();
    let mut miss_rows: Vec<usize> = Vec::new();
    let mut miss_states: Vec<f32> = Vec::new();
    let mut miss_actions: Vec<f32> = Vec::new();
    let mut miss_index: Vec<i64> = Vec::new();
    let mut in_pass: LeafKeyMap<usize> = LeafKeyMap::default();
    // (row, miss) for leaves that repeat a miss of this pass.
    let mut waiting: Vec<(usize, usize)> = Vec::new();

    for row in 0..rows {
        let state = state_of(row);
        let actions = actions_of(row);
        let count = bounds[row + 1] - bounds[row];
        let key = cache.key(state, actions, count);
        if let Some((hit_logits, hit_value)) = cache
            .lookup(&key, count, state, actions)
            .map_err(mismatch)?
        {
            logits[bounds[row]..bounds[row + 1]].copy_from_slice(hit_logits);
            values[row * N_HOUSES..(row + 1) * N_HOUSES].copy_from_slice(hit_value);
            stats.cache_hits += 1;
            stats.cache_hit_slots += count as u64;
            continue;
        }
        if let Some(&miss) = in_pass.get(&key) {
            if verify {
                let first = miss_rows[miss];
                let same = |left: &[f32], right: &[f32]| {
                    left.len() == right.len()
                        && left
                            .iter()
                            .zip(right)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                };
                if !same(state_of(first), state) || !same(actions_of(first), actions) {
                    return Err(mismatch(leaf_cache::CacheMismatch));
                }
            }
            waiting.push((row, miss));
            stats.pass_duplicates += 1;
            continue;
        }
        let miss = miss_rows.len();
        in_pass.insert(key, miss);
        miss_keys.push(key);
        miss_rows.push(row);
        miss_states.extend_from_slice(state);
        miss_actions.extend_from_slice(actions);
        miss_index.extend(std::iter::repeat(miss as i64).take(count));
    }

    if miss_rows.is_empty() {
        stats.calls_skipped += 1;
        return Ok((logits, values));
    }

    let scored_rows = miss_rows.len();
    let scored_slots = miss_index.len();
    stats.evaluator_calls += 1;
    stats.rows_sent += scored_rows as u64;
    stats.slots_sent += scored_slots as u64;

    let callback_started = Instant::now();
    let state_array = array2(py, scored_rows, STATE_DIM, miss_states)?;
    let action_array = array2(py, scored_slots, ACTION_DIM, miss_actions)?;
    let index_array = numpy::ndarray::Array1::from_vec(miss_index)
        .into_pyarray(py)
        .unbind();
    let mask_array = PyArray1::from_vec(py, vec![1.0f32; scored_slots]).unbind();
    let reply = evaluator.call1(py, (state_array, action_array, index_array, mask_array))?;
    let (reply_logits, reply_values): (PyReadonlyArray1<f32>, PyReadonlyArray2<f32>) =
        reply.extract(py)?;
    let reply_logits = reply_logits.as_slice()?;
    let reply_values = reply_values.as_slice()?;
    stats.time_callback += callback_started.elapsed();
    if reply_logits.len() != scored_slots {
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} logits for {scored_slots} candidates",
            reply_logits.len()
        )));
    }
    if reply_values.len() != scored_rows * N_HOUSES {
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} values for {scored_rows} leaves",
            reply_values.len()
        )));
    }

    // Each miss's slice of the reply, in reply order.
    let mut miss_bounds = Vec::with_capacity(scored_rows + 1);
    miss_bounds.push(0usize);
    for row in &miss_rows {
        let last = *miss_bounds.last().unwrap_or(&0);
        miss_bounds.push(last + bounds[row + 1] - bounds[*row]);
    }
    for (miss, row) in miss_rows.iter().enumerate() {
        let miss_logits = &reply_logits[miss_bounds[miss]..miss_bounds[miss + 1]];
        let miss_value = &reply_values[miss * N_HOUSES..(miss + 1) * N_HOUSES];
        logits[bounds[*row]..bounds[row + 1]].copy_from_slice(miss_logits);
        values[row * N_HOUSES..(row + 1) * N_HOUSES].copy_from_slice(miss_value);
        cache.insert(
            miss_keys[miss],
            miss_logits,
            miss_value,
            state_of(*row),
            actions_of(*row),
        );
    }
    for (row, miss) in waiting {
        logits[bounds[row]..bounds[row + 1]]
            .copy_from_slice(&reply_logits[miss_bounds[miss]..miss_bounds[miss + 1]]);
        values[row * N_HOUSES..(row + 1) * N_HOUSES]
            .copy_from_slice(&reply_values[miss * N_HOUSES..(miss + 1) * N_HOUSES]);
    }
    Ok((logits, values))
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

/// Parse the `phase_budgets` spec key into a cap table.
///
/// `None` (the default when the key is absent or `null`) keeps the fixed
/// simulation budget, which is what every existing run and test expects. A
/// mapping such as `{"planning": 32, "wildling_attack": 16}` starts from
/// [`PhaseBudgets::DEFAULT`] and replaces only the phases it names, so a caller
/// spells out the ones it wants to differ. Keys are the engine's own phase
/// names (`planning`, `wildling_attack`, ...); an unknown name is an error
/// rather than a typo that silently does nothing.
fn parse_phase_budgets(value: &Bound<'_, PyAny>) -> PyResult<Option<PhaseBudgets>> {
    if value.is_none() {
        return Ok(None);
    }
    let dict = value.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(
            "phase_budgets must be a mapping of phase name to simulation cap, or None",
        )
    })?;
    let mut budgets = PhaseBudgets::DEFAULT;
    for (key, cap) in dict.iter() {
        let name: String = key.extract()?;
        let phase = Phase::from_name(&name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown phase {name:?} in phase_budgets"))
        })?;
        budgets.set_cap(phase, cap.extract()?);
    }
    Ok(Some(budgets))
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
///
/// Rule options travel in the spec: `deferred_battles`, `sequential_westeros`,
/// `doran_choice`, `wildling_tie_choice` and `choice_choices`. Each defaults to
/// `false`, which is the raw engine behaviour the frozen fixtures pin, so a
/// caller that omits them keeps the old games.
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
    let lannister_port_ship: bool = spec_or(spec, "lannister_port_ship", false)?;
    let deferred_battles: bool = spec_or(spec, "deferred_battles", false)?;
    let sequential_westeros: bool = spec_or(spec, "sequential_westeros", false)?;
    let doran_choice: bool = spec_or(spec, "doran_choice", false)?;
    let wildling_tie_choice: bool = spec_or(spec, "wildling_tie_choice", false)?;
    let bid_tie_choice: bool = spec_or(spec, "bid_tie_choice", false)?;
    let choice_choices: bool = spec_or(spec, "choice_choices", false)?;
    // Exact duplicate leaves (the same 97-float state and candidate rows) are
    // common under PIMC and across games. The evaluator is pure, so scoring
    // each unique leaf once is exact; the switch exists for A/B runs.
    let dedupe_leaves: bool = spec_or(spec, "dedupe_leaves", true)?;
    // Cross-pass evaluation cache (see `agot_core::leaf_cache`): a leaf whose
    // exact network input was scored earlier in this call reuses that output.
    // Exact for an evaluator that is a pure row-wise function of its input;
    // the cache also merges in-pass repeats, so it subsumes `dedupe_leaves`.
    // `leaf_cache_verify` keeps every input and compares it on each hit.
    let use_leaf_cache: bool = spec_or(spec, "leaf_cache", true)?;
    let leaf_cache_verify: bool = spec_or(spec, "leaf_cache_verify", false)?;
    let leaf_cache_entries: usize =
        spec_or(spec, "leaf_cache_entries", leaf_cache::DEFAULT_MAX_ENTRIES)?;
    let leaf_cache_logits: usize =
        spec_or(spec, "leaf_cache_logits", leaf_cache::DEFAULT_MAX_LOGITS)?;
    // Per-phase caps on the simulations one decision may spend. Absent or
    // `null` is the fixed budget, so existing specs are unchanged.
    let phase_budgets = match spec.get_item("phase_budgets")? {
        Some(value) => parse_phase_budgets(&value)?,
        None => None,
    };
    let houses: Option<Vec<String>> = spec_value(spec, "houses")?;
    let data_dir: Option<String> = spec_value(spec, "data_dir")?;
    let step_limit: u32 = spec_or(spec, "step_limit", 20_000)?;

    if games == 0 {
        if let Ok(mut last) = LAST_STATS.lock() {
            *last = Some((DriverStats::default(), use_leaf_cache));
        }
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

    // One cache per evaluator, created on first use and dropped with this
    // call: a network never sees another network's outputs, and the next call
    // (new weights) starts empty.
    let mut caches: Vec<Option<LeafCache>> = (0..evaluators.len()).map(|_| None).collect();
    let mut stats = DriverStats::default();
    let started_total = Instant::now();

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
                lannister_port_ship,
            });
        }

        let mut states: Vec<Game> = configs
            .iter()
            .map(|config| {
                let mut game = Game::new(Arc::clone(&data), config.clone())
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                // Rule options that change what the engine asks for. They must
                // match the Python-side config, otherwise a training run records
                // decisions the reference driver would never have posed.
                game.deferred_battles = deferred_battles;
                game.sequential_westeros = sequential_westeros;
                game.doran_choice = doran_choice;
                game.wildling_tie_choice = wildling_tie_choice;
                game.bid_tie_choice = bid_tie_choice;
                game.choice_choices = choice_choices;
                game.state.choice_choices = choice_choices;
                Ok(game)
            })
            .collect::<PyResult<Vec<Game>>>()?;

        // One searcher per game, so PIMC worlds and noise stay independent.
        // `determinizations` is a spec key, so the searcher has to be told about
        // it: without this the driver ran one perfect-information tree per
        // decision while `per_world` divided the simulations as if PIMC were on.
        let mut searchers: Vec<Mcts<UniformEvaluator>> = chunk
            .iter()
            .map(|index| {
                let mut searcher = Mcts::with_seed(UniformEvaluator, game_seed(*index));
                searcher.determinizations = determinizations;
                searcher
            })
            .collect();
        let mut histories: Vec<Vec<Record>> = (0..chunk.len()).map(|_| Vec::new()).collect();
        let mut steps = vec![0u32; chunk.len()];
        let mut winners: Vec<Option<HouseId>> = vec![None; chunk.len()];

        // Each game's open decision: its roots, its simulations per PIMC
        // world (the phase budget, or `simulations`, divided by the
        // determinizations) and how many passes it has run. A game that has
        // spent its budget picks, records and steps right after the pass, and
        // opens its next decision before the following pass, so the passes stay
        // full instead of waiting for the deepest decision of the batch. Every
        // game still runs exactly its own sequence of simulations on its own
        // searcher; only which games share a leaf callback changes.
        let mut searching: Vec<Option<(Vec<u32>, u32, u32)>> = vec![None; chunk.len()];
        let mut finished_slot = vec![false; chunk.len()];
        let mut cache = observation::RegionCache::new();

        loop {
            // Advance every game that is neither searching nor over to its
            // next decision. Scripted seats decide immediately (possibly
            // several in a row); search seats open their roots and join the
            // shared leaf batch below.
            let mark = Instant::now();
            for slot in 0..states.len() {
                if finished_slot[slot] || searching[slot].is_some() {
                    continue;
                }
                loop {
                    if states[slot].is_terminal() || steps[slot] >= step_limit {
                        if winners[slot].is_none() {
                            winners[slot] = states[slot].winner();
                        }
                        finished_slot[slot] = true;
                        break;
                    }
                    if states[slot].state.phase == Phase::Westeros
                        && states[slot].state.westeros_waiting
                    {
                        // Reveal/resolve the next Westeros card. This is not a
                        // decision, but the card and its mustering/supply step
                        // must resolve before the following card is drawn.
                        states[slot]
                            .continue_westeros()
                            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                        continue;
                    }
                    let Some(house) = states[slot].current_player() else {
                        winners[slot] = states[slot].winner();
                        finished_slot[slot] = true;
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
                            // Nothing to search: the position can never move
                            // again, so the game is over for this driver.
                            winners[slot] = states[slot].winner();
                            finished_slot[slot] = true;
                        } else {
                            let budget = decision_budget(
                                phase_budgets.as_ref(),
                                states[slot].state.phase,
                                states[slot].legal_actions().len(),
                                simulations,
                            );
                            searching[slot] = Some((roots, (budget / determinizations).max(1), 0));
                        }
                        break;
                    }

                    let decider = scripted_decider.as_ref().ok_or_else(|| {
                        PyRuntimeError::new_err(
                            "spec names a scripted seat but no scripted_decider was supplied",
                        )
                    })?;
                    let view = Py::new(py, crate::PyGame::from_game(states[slot].clone()))?;
                    let scripted_started = Instant::now();
                    let reply = decider.call1(py, (global, house.name(), view))?;
                    let (index, target): (usize, Option<Vec<f32>>) = reply.extract(py)?;
                    stats.time_scripted += scripted_started.elapsed();
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
                }
            }
            stats.time_advance += mark.elapsed();
            if searching.iter().all(Option::is_none) {
                break;
            }

            // One pass: one simulation on every root of every searching game,
            // in slot order and, inside a game, in root order (the order the
            // fixed loop used), with the leaves batched across games.
            let mark = Instant::now();
            let mut pending: Vec<(usize, Simulation)> = Vec::new();
            for (slot, open) in searching.iter().enumerate() {
                let Some((roots, _, _)) = open else {
                    continue;
                };
                for root in roots {
                    let simulation = searchers[slot]
                        .prepare_simulation(*root)
                        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                    pending.push((slot, simulation));
                }
            }

            // Encode every leaf, grouped by the model that owns its seat.
            // One region cache for the whole call (declared with the batch):
            // each leaf clears it, which only bumps its generation.
            stats.time_prepare += mark.elapsed();
            let mark = Instant::now();
            let mut groups: Vec<LeafGroup> = Vec::new();
            for (position, (slot, simulation)) in pending.iter().enumerate() {
                if simulation.leaf.is_none() {
                    continue;
                }
                // Borrowed from the arena: the encoder only reads the position,
                // so the leaf is not copied out of its node.
                let leaf_state = searchers[*slot]
                    .leaf_state(simulation)
                    .ok_or_else(|| PyRuntimeError::new_err("a leaf without a position"))?;
                let house = leaf_state.active_house.ok_or_else(|| {
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
                    leaf_state,
                    &data,
                    None,
                    &mut group.states[state_start..],
                );
                for action in &simulation.actions {
                    let action_start = group.actions.len();
                    group.actions.resize(action_start + ACTION_DIM, 0.0);
                    observation::encode_action_into(
                        leaf_state,
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

            stats.time_encode += mark.elapsed();
            let mark = Instant::now();
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
                stats.leaves += rows as u64;
                stats.slots += group.index.len() as u64;

                // Each leaf's candidate-slot range in the packed layout.
                let mut bounds = Vec::with_capacity(rows + 1);
                bounds.push(0usize);
                for position in &group.positions {
                    bounds.push(
                        bounds.last().copied().unwrap_or(0) + pending[*position].1.actions.len(),
                    );
                }

                let (logits, values) = if use_leaf_cache {
                    let cache = caches[group.model].get_or_insert_with(|| {
                        LeafCache::new(leaf_cache_entries, leaf_cache_logits, leaf_cache_verify)
                    });
                    evaluate_cached(
                        py,
                        &evaluators[group.model],
                        cache,
                        leaf_cache_verify,
                        &group,
                        &bounds,
                        &mut stats,
                    )?
                } else {
                    evaluate_direct(
                        py,
                        &evaluators[group.model],
                        &group,
                        dedupe_leaves,
                        &mut stats,
                    )?
                };
                debug_assert_eq!(logits.len(), group.index.len());
                debug_assert_eq!(values.len(), rows * N_HOUSES);
                let priors = segment_softmax(&logits, &group.index, rows);

                // Scatter each leaf's slice back to its simulation.
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

            stats.time_evaluate += mark.elapsed();
            let mark = Instant::now();
            for ((slot, simulation), evaluation) in pending.into_iter().zip(evaluations) {
                searchers[slot]
                    .finish_simulation(simulation, evaluation)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            }
            stats.time_finish += mark.elapsed();

            // A game whose decision has spent its budget aggregates its worlds,
            // picks, records and steps; it opens its next decision in the next
            // advance.
            let mark = Instant::now();
            for slot in 0..searching.len() {
                let Some((roots, per_world, done)) = searching[slot].as_mut() else {
                    continue;
                };
                *done += 1;
                if *done < *per_world {
                    continue;
                }
                let observer = states[slot].current_player();
                let result: SearchResult = if roots.len() == 1 {
                    searchers[slot].result_from_root(roots[0])
                } else {
                    searchers[slot].aggregate_roots(roots, observer)
                };
                searching[slot] = None;
                let round = states[slot].state.round;
                let temperature = scheduled_temperature(&schedule, round);
                let (action, target) = searchers[slot]
                    .pick(&result, Some(temperature))
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

                let observation =
                    observation::observe(&states[slot].state, &data, &result.actions, observer);
                histories[slot].push(Record {
                    state: observation.state_features,
                    actions: observation.action_features,
                    policy: target,
                    player: observer.map(|house| house.index() as u8).unwrap_or(0),
                });

                states[slot]
                    .step(action)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                steps[slot] += 1;
            }
            stats.time_record += mark.elapsed();
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

    stats.time_total += started_total.elapsed();
    for cache in caches.iter().flatten() {
        let cache_stats = cache.stats();
        stats.cache_inserts += cache_stats.inserts;
        stats.cache_evicted += cache_stats.evicted;
        stats.cache_rotations += cache_stats.rotations;
        stats.cache_verified += cache_stats.verified;
        stats.cache_peak_entries += cache_stats.peak_entries;
        stats.cache_peak_logits += cache_stats.peak_logits;
    }
    if let Ok(mut last) = LAST_STATS.lock() {
        *last = Some((stats, use_leaf_cache));
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
