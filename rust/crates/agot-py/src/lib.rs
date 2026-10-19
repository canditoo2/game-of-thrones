//! Python bindings for `agot-core`.
//!
//! The extension is deliberately thin and comes in two tiers:
//!
//! * **session tier** — [`Game`] plus [`Action`], for the CLI, the demo and the
//!   gauntlet: construct a position, list the legal actions, step, observe.
//!   Cheap per call (a few hundred nanoseconds of engine work), so the Python
//!   overhead around it does not matter.
//! * **batch tier** — whole self-play games driven in Rust with one batched
//!   evaluator callback per leaf batch, so the training hot path crosses the
//!   boundary once per batch rather than once per decision.
//!
//! The Python engine stays the reference implementation while the port is
//! finished; this module is additive and changes nothing on its own.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

use numpy::IntoPyArray;

use agot_core::action::{ActionKind, Side};
use agot_core::enums::{HouseId, UnitType, N_HOUSES};
use agot_core::game::{EngineError, Game as CoreGame};
use agot_core::map::DataSet;
use agot_core::observation;
use agot_core::setup::GameConfig;

/// Which engine implementation is loaded. Logged by training runs so a run's
/// timings can be attributed to the right backend.
#[pyfunction]
fn backend() -> &'static str {
    "rust"
}

fn engine_error(error: EngineError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// One decision, mirroring the Python `Action` dataclass.
///
/// Payload fields are resolved to the same strings the Python engine uses, so
/// existing call sites (`str(action)`, comparisons, the CLI) keep working.
#[pyclass(name = "Action", module = "agot._core", frozen)]
#[derive(Clone)]
pub struct PyAction {
    inner: agot_core::action::Action,
    data: Arc<DataSet>,
}

impl PyAction {
    fn new(inner: agot_core::action::Action, data: Arc<DataSet>) -> PyAction {
        PyAction { inner, data }
    }
}

#[pymethods]
impl PyAction {
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind.as_str()
    }

    #[getter]
    fn house(&self) -> Option<&'static str> {
        self.inner.house.map(|house| house.name())
    }

    #[getter]
    fn region(&self) -> Option<&str> {
        self.inner
            .region
            .map(|region| self.data.map.id_of(region))
    }

    #[getter]
    fn target(&self) -> Option<&str> {
        self.inner
            .target
            .map(|target| self.data.map.id_of(target))
    }

    #[getter]
    fn order(&self) -> Option<&'static str> {
        self.inner.order.map(|order| order.key())
    }

    #[getter]
    fn card(&self) -> Option<&str> {
        match (self.inner.house, self.inner.card) {
            (Some(house), Some(card)) => Some(self.data.house_cards.card(house, card).id.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn unit(&self) -> Option<&'static str> {
        self.inner.unit.map(|unit| unit.as_str())
    }

    #[getter]
    fn side(&self) -> Option<&'static str> {
        self.inner.side.map(|side| side.as_str())
    }

    #[getter]
    fn amount(&self) -> Option<u8> {
        self.inner.amount
    }

    #[getter]
    fn units(&self) -> Option<(u8, u8, u8, u8)> {
        self.inner
            .units
            .map(|units| (units[0], units[1], units[2], units[3]))
    }

    #[getter]
    fn establish(&self) -> bool {
        self.inner.establish
    }

    fn __repr__(&self) -> String {
        let mut parts = vec![self.inner.kind.as_str().to_string()];
        for (name, value) in [
            ("house", self.house().map(str::to_string)),
            ("region", self.region().map(str::to_string)),
            ("target", self.target().map(str::to_string)),
            ("order", self.order().map(str::to_string)),
            ("card", self.card().map(str::to_string)),
            ("unit", self.unit().map(str::to_string)),
            ("side", self.side().map(str::to_string)),
            ("amount", self.inner.amount.map(|amount| amount.to_string())),
            ("units", self.units().map(|units| format!("{units:?}"))),
        ] {
            if let Some(value) = value {
                parts.push(format!("{name}={value}"));
            }
        }
        if self.inner.establish {
            parts.push("establish=True".to_string());
        }
        format!("Action({})", parts.join(" "))
    }

    /// Actions compare by value, which is what `action in game.legal_actions()`
    /// needs.
    fn __eq__(&self, other: &PyAction) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// Game
// ---------------------------------------------------------------------------

/// A single game, driven one decision at a time.
#[pyclass(name = "Game", module = "agot._core")]
pub struct PyGame {
    game: CoreGame,
}

impl PyGame {
    fn wrap(&self, action: agot_core::action::Action) -> PyAction {
        PyAction::new(action, Arc::clone(&self.game.data))
    }
}

#[pymethods]
impl PyGame {
    /// `Game.new(houses=None, seed=0, max_rounds=10, victory_castles=7,
    /// use_westeros_decks=True, data_dir=None)`.
    #[classmethod]
    #[pyo3(signature = (houses=None, seed=0, max_rounds=10, victory_castles=7, use_westeros_decks=true, data_dir=None))]
    fn new(
        _cls: &Bound<'_, PyType>,
        houses: Option<Vec<String>>,
        seed: u64,
        max_rounds: u8,
        victory_castles: u8,
        use_westeros_decks: bool,
        data_dir: Option<String>,
    ) -> PyResult<PyGame> {
        let data = match data_dir {
            Some(dir) => DataSet::load(std::path::Path::new(&dir)),
            None => DataSet::load_default(),
        }
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let data = Arc::new(data);

        let houses = match houses {
            Some(names) => names
                .iter()
                .map(|name| {
                    HouseId::from_name(name)
                        .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))
                })
                .collect::<PyResult<smallvec::SmallVec<[HouseId; 6]>>>()?,
            None => HouseId::ALL.into_iter().collect(),
        };

        let config = GameConfig {
            houses,
            max_rounds,
            victory_castles,
            seed,
            use_westeros_decks,
        };
        let game = CoreGame::new(data, config).map_err(engine_error)?;
        Ok(PyGame { game })
    }

    /// All actions the current player may take.
    fn legal_actions(&self) -> Vec<PyAction> {
        self.game
            .legal_actions()
            .into_iter()
            .map(|action| self.wrap(action))
            .collect()
    }

    fn step(&mut self, action: &PyAction) -> PyResult<()> {
        self.game.step(action.inner).map_err(engine_error)
    }

    fn is_terminal(&self) -> bool {
        self.game.is_terminal()
    }

    fn current_player(&self) -> Option<&'static str> {
        self.game.current_player().map(|house| house.name())
    }

    fn winner(&self) -> Option<&'static str> {
        self.game.winner().map(|house| house.name())
    }

    #[getter]
    fn round(&self) -> u8 {
        self.game.state.round
    }

    #[getter]
    fn phase(&self) -> &'static str {
        self.game.state.phase.as_str()
    }

    #[getter]
    fn action_step(&self) -> &'static str {
        self.game.state.action_step.as_str()
    }

    /// Structural copy, used by tree search and by the differential tests.
    fn clone(&self) -> PyGame {
        PyGame {
            game: self.game.clone(),
        }
    }

    /// The decision point as network input.
    ///
    /// Returns `(actor, state_features, action_features)` — the same triple
    /// `agot.engine.observation.observe` produces, with the action matrix
    /// row-aligned with [`PyGame::legal_actions`]. A terminal position yields an
    /// empty matrix rather than raising, because the caller is already looping on
    /// `is_terminal`.
    fn observe(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Option<&'static str>, Py<numpy::PyArray1<f32>>, Py<numpy::PyArray2<f32>>)> {
        let actions = self.game.legal_actions();
        let observation =
            observation::observe(&self.game.state, &self.game.data, &actions, None);
        let rows = observation.len();
        let actor = observation.actor.map(|house| house.name());

        let state = numpy::PyArray1::from_vec(py, observation.state_features).unbind();
        let matrix = numpy::ndarray::Array2::from_shape_vec(
            (rows, observation::ACTION_DIM),
            observation.action_features,
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py)
        .unbind();
        Ok((actor, state, matrix))
    }

    /// Compact description of the position, as in the Python engine.
    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = self.game.summary();
        let text = serde_json::to_string(&value)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let json = py.import("json")?;
        Ok(json.call_method1("loads", (text,))?.into())
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// The kind names the encoder and the CLI use, exposed for introspection.
#[pyfunction]
fn action_kinds() -> Vec<&'static str> {
    ActionKind::ALL
        .into_iter()
        .map(|kind| kind.as_str())
        .collect()
}

/// `(unit, side)` vocabulary, mirroring `observation.UNIT_ORDER` and
/// `SUPPORT_SIDES`.
#[pyfunction]
fn vocabulary() -> (Vec<&'static str>, Vec<&'static str>, usize) {
    (
        UnitType::ALL
            .into_iter()
            .map(|unit| unit.as_str())
            .collect(),
        Side::ALL.into_iter().map(|side| side.as_str()).collect(),
        N_HOUSES,
    )
}

/// The observation layout, so the Python side can size its buffers without
/// hard-coding numbers that must agree with the encoder.
#[pyfunction]
fn observation_dims() -> (usize, usize) {
    (observation::STATE_DIM, observation::ACTION_DIM)
}

#[pymodule]
fn agot_core_ext(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(backend, module)?)?;
    module.add_function(wrap_pyfunction!(action_kinds, module)?)?;
    module.add_function(wrap_pyfunction!(vocabulary, module)?)?;
    module.add_function(wrap_pyfunction!(observation_dims, module)?)?;
    module.add_class::<PyAction>()?;
    module.add_class::<PyGame>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
