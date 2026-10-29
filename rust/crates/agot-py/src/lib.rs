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
use pyo3::types::{PyDict, PyList, PyType};

use numpy::IntoPyArray;

use agot_core::action::{ActionKind, Side};
use agot_core::enums::{CardId, HouseId, UnitType, N_HOUSES};
use agot_core::game::{EngineError, Game as CoreGame};
use agot_core::map::DataSet;
use agot_core::observation;
use agot_core::setup::GameConfig;

mod selfplay;

/// Which engine implementation is loaded. Logged by training runs so a run's
/// timings can be attributed to the right backend.
#[pyfunction]
fn backend() -> &'static str {
    "rust"
}

fn engine_error(error: EngineError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// A `serde_json` value as native Python data.
///
/// The snapshot used to travel as JSON text and be parsed by Python's `json`
/// module, which cost a serialization and a parse on every step. The recursive
/// conversion writes the same dicts/lists/ints/strings directly; numbers keep
/// their integer-vs-float distinction, matching `json.loads`.
fn to_python(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value;

    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(flag) => (*flag).into_pyobject(py)?.to_owned().unbind().into_any(),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                value.into_pyobject(py)?.unbind().into_any()
            } else if let Some(value) = number.as_u64() {
                value.into_pyobject(py)?.unbind().into_any()
            } else {
                number
                    .as_f64()
                    .unwrap_or(0.0)
                    .into_pyobject(py)?
                    .unbind()
                    .into_any()
            }
        }
        Value::String(text) => text.as_str().into_pyobject(py)?.unbind().into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_python(py, item)?)?;
            }
            list.unbind().into_any()
        }
        Value::Object(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, to_python(py, item)?)?;
            }
            dict.unbind().into_any()
        }
    })
}

// -- field parsing for the position builder ------------------------------

fn optional_house(value: &Bound<'_, PyAny>) -> PyResult<Option<HouseId>> {
    match value.extract::<Option<String>>()? {
        Some(name) => Ok(Some(HouseId::from_name(&name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown house {name:?}"))
        })?)),
        None => Ok(None),
    }
}

fn optional_order(value: &Bound<'_, PyAny>) -> PyResult<Option<agot_core::enums::OrderKey>> {
    match value.extract::<Option<String>>()? {
        Some(key) => Ok(Some(
            agot_core::enums::OrderKey::from_name(&key)
                .ok_or_else(|| PyValueError::new_err(format!("unknown order {key:?}")))?,
        )),
        None => Ok(None),
    }
}

/// `{"footman": 2, "ship": 1}` to a count per unit slot.
fn parse_units(value: &Bound<'_, PyAny>) -> PyResult<[u8; 4]> {
    let dict = value.cast::<PyDict>()?;
    let mut out = [0u8; 4];
    for (key, count) in dict.iter() {
        let name: String = key.extract()?;
        let unit = UnitType::from_name(&name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown unit {name:?}")))?;
        out[unit.index()] = count.extract()?;
    }
    Ok(out)
}

fn parse_orders(
    value: &Bound<'_, PyAny>,
) -> PyResult<smallvec::SmallVec<[agot_core::enums::OrderKey; 15]>> {
    let keys: Vec<String> = value.extract()?;
    keys.iter()
        .map(|key| {
            agot_core::enums::OrderKey::from_name(key)
                .ok_or_else(|| PyValueError::new_err(format!("unknown order {key:?}")))
        })
        .collect()
}

fn parse_cards(
    value: &Bound<'_, PyAny>,
    house: HouseId,
    data: &DataSet,
) -> PyResult<smallvec::SmallVec<[agot_core::enums::CardId; 7]>> {
    let ids: Vec<String> = value.extract()?;
    ids.iter()
        .map(|id| {
            data.house_cards
                .card_index(house, id)
                .ok_or_else(|| PyValueError::new_err(format!("{house} has no house card {id:?}")))
        })
        .collect()
}

/// Resolve a printed House-card name to its slot and owning deck.
///
/// `preferred` is the deck the caller named (or the action's own house); a card
/// that is not in it is looked up across the six decks, because a generic card
/// choice can name another house's card: Patchface discards from the opponent's
/// hand. Printed names are unique across the decks, so the search is unambiguous.
fn resolve_house_card(
    data: &DataSet,
    preferred: Option<HouseId>,
    id: &str,
) -> PyResult<(CardId, HouseId)> {
    if let Some(house) = preferred {
        if let Some(card) = data.house_cards.card_index(house, id) {
            return Ok((card, house));
        }
    }
    HouseId::ALL
        .into_iter()
        .find_map(|house| {
            data.house_cards
                .card_index(house, id)
                .map(|card| (card, house))
        })
        .ok_or_else(|| PyValueError::new_err(format!("unknown house card {id:?}")))
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
        self.inner.region.map(|region| self.data.map.id_of(region))
    }

    #[getter]
    fn target(&self) -> Option<&str> {
        self.inner.target.map(|target| self.data.map.id_of(target))
    }

    #[getter]
    fn order(&self) -> Option<&'static str> {
        self.inner.order.map(|order| order.key())
    }

    /// The synthetic generic-choice context tag, when `order` holds one.
    ///
    /// Generic choices pack a context tag into `order` (values `0..=14`, see
    /// `agot_core::action::choice_context`). Only `PlaceOrder` and `RavenSwap`
    /// carry a real order token, so every other kind with `order` set holds a
    /// tag; it is reported here as the integer `agot.choice_context` expects so
    /// a caller never has to reverse an order key name.
    #[getter]
    fn choice_tag(&self) -> Option<u8> {
        if matches!(
            self.inner.kind,
            ActionKind::PlaceOrder | ActionKind::RavenSwap
        ) {
            return None;
        }
        self.inner.order.map(|order| order.0)
    }

    /// The house whose deck `card` belongs to, when it is not `house`'s own.
    ///
    /// Patchface's controller discards an opponent's card, so the action belongs
    /// to the Patchface player while the card id indexes the target's deck.
    #[getter]
    fn card_house(&self) -> Option<&'static str> {
        self.inner.card_house.map(|house| house.name())
    }

    #[getter]
    fn card(&self) -> Option<&str> {
        // A holders'-choice action stores a *Westeros* card index in the same
        // field; resolve it against that deck instead of the house deck. The
        // out-of-range sentinel is the "no effect" option.
        if self.inner.kind == ActionKind::WesterosChoice {
            let card = self.inner.card?;
            if card.0 == u8::MAX {
                return Some("none");
            }
            return Some(self.data.westeros.card_id(card.0 as u16));
        }
        // A card id is a slot in one house's deck. `card_house` names the deck
        // when it is not the acting house's (Patchface discards an opponent's
        // card), so printing the action names the card the chooser really saw
        // instead of the same slot in their own hand.
        let house = observation::card_owner(&self.inner)?;
        self.inner
            .card
            .and_then(|card| self.data.house_cards.deck(house).get(card.0 as usize))
            .map(|spec| spec.id.as_str())
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

    /// Wrap an owned core game, for the self-play driver's scripted-seat view.
    pub(crate) fn from_game(game: CoreGame) -> PyGame {
        PyGame { game }
    }
}

#[pymethods]
impl PyGame {
    /// `Game.new(houses=None, seed=0, max_rounds=10, victory_castles=7,
    /// use_westeros_decks=True, deferred_battles=False,
    /// sequential_westeros=False, doran_choice=False,
    /// wildling_tie_choice=False, choice_choices=False, data_dir=None)`.
    ///
    /// The rule options are optional keyword arguments, so an existing caller
    /// keeps working unchanged; each defaults to the raw engine behaviour the
    /// frozen fixtures pin.
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (houses=None, seed=0, max_rounds=10, victory_castles=7, use_westeros_decks=true, deferred_battles=false, sequential_westeros=false, doran_choice=false, wildling_tie_choice=false, choice_choices=false, data_dir=None))]
    fn new(
        _cls: &Bound<'_, PyType>,
        houses: Option<Vec<String>>,
        seed: u64,
        max_rounds: u8,
        victory_castles: u8,
        use_westeros_decks: bool,
        deferred_battles: bool,
        sequential_westeros: bool,
        doran_choice: bool,
        wildling_tie_choice: bool,
        choice_choices: bool,
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
        let mut game = CoreGame::new(data, config).map_err(engine_error)?;
        game.deferred_battles = deferred_battles;
        game.sequential_westeros = sequential_westeros;
        game.doran_choice = doran_choice;
        game.wildling_tie_choice = wildling_tie_choice;
        game.choice_choices = choice_choices;
        game.state.choice_choices = choice_choices;
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

    /// The top card of the Wildling deck, for the Messenger Raven's look.
    fn peek_wildling_card(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some(index) = self.game.state.wildling_deck.cards.first().copied() else {
            return Ok(py.None());
        };
        let card = self.game.data.wildlings.card(index);
        let text_of = |side: &str| card[side]["text"].as_str().unwrap_or_default().to_string();
        let payload = serde_json::json!({
            "name": card["name"].as_str().unwrap_or("Wildling card"),
            "wildling": text_of("wildling_victory"),
            "nights_watch": text_of("nights_watch_victory"),
        });
        let text = serde_json::to_string(&payload)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let json = py.import("json")?;
        Ok(json.call_method1("loads", (text,))?.unbind())
    }

    /// The Messenger Raven looks at one rival order token.
    fn peek_raven_order(&mut self, house: &str, region: &str) -> PyResult<()> {
        let house_id = agot_core::enums::HouseId::from_name(house)
            .ok_or_else(|| PyValueError::new_err(format!("unknown house {house:?}")))?;
        let region_id = self
            .game
            .data
            .map
            .lookup(region)
            .ok_or_else(|| PyValueError::new_err(format!("unknown region {region:?}")))?;
        self.game
            .peek_raven_order(house_id, region_id)
            .map_err(engine_error)
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
    fn max_rounds(&self) -> u8 {
        self.game.state.max_rounds
    }

    #[getter]
    fn victory_castles(&self) -> u8 {
        self.game.state.victory_castles
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

    /// A copy with the hidden state resampled as `observer` sees it.
    ///
    /// This is `agot.engine.hidden.determinize` for the Rust engine: the caller
    /// supplies a seed so the resampling stays reproducible without sharing the
    /// Python generator's stream.
    fn determinize(&self, observer: &str, seed: u64) -> PyResult<PyGame> {
        let observer = HouseId::from_name(observer)
            .ok_or_else(|| PyValueError::new_err(format!("unknown house {observer:?}")))?;
        let mut rng = agot_core::rng::Rng::from_seed(seed);
        Ok(PyGame {
            game: agot_core::hidden::determinize(&self.game, observer, &mut rng),
        })
    }

    /// The decision point as network input.
    ///
    /// Returns `(actor, state_features, action_features)` — the same triple
    /// `agot.engine.observation.observe` produces, with the action matrix
    /// row-aligned with [`PyGame::legal_actions`]. A terminal position yields an
    /// empty matrix rather than raising, because the caller is already looping on
    /// `is_terminal`.
    ///
    /// The return type is the point of the method, so the tuple is spelled out.
    #[allow(clippy::type_complexity)]
    fn observe(
        &self,
        py: Python<'_>,
    ) -> PyResult<(
        Option<&'static str>,
        Py<numpy::PyArray1<f32>>,
        Py<numpy::PyArray2<f32>>,
    )> {
        let actions = self.game.legal_actions();
        let observation = observation::observe(&self.game.state, &self.game.data, &actions, None);
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

    /// The network's state features for this position.
    ///
    /// `actor` defaults to the acting house, as `observation.encode_state`
    /// does; the encoder itself lives in the core crate.
    fn encode_state(
        &self,
        py: Python<'_>,
        actor: Option<&str>,
    ) -> PyResult<Py<numpy::PyArray1<f32>>> {
        let actor = actor
            .map(|name| {
                HouseId::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))
            })
            .transpose()?;
        let features = observation::encode_state(&self.game.state, &self.game.data, actor);
        Ok(numpy::PyArray1::from_vec(py, features).unbind())
    }

    /// One feature row per candidate action, row-aligned with `actions`.
    ///
    /// The row format is `encode_action_into`'s: callers pass actions built by
    /// [`PyGame::action`], so an action from the Python facade encodes exactly
    /// as the engine would encode its own.
    fn encode_actions(
        &self,
        py: Python<'_>,
        actions: Vec<PyRef<'_, PyAction>>,
        actor: Option<&str>,
    ) -> PyResult<Py<numpy::PyArray2<f32>>> {
        let actor = actor
            .map(|name| {
                HouseId::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))
            })
            .transpose()?;
        let rows = actions.len();
        let mut matrix = vec![0.0f32; rows * observation::ACTION_DIM];
        let mut cache = observation::RegionCache::new();
        for (row, action) in actions.iter().enumerate() {
            observation::encode_action_into(
                &self.game.state,
                &self.game.data,
                &action.inner,
                actor,
                &mut matrix[row * observation::ACTION_DIM..(row + 1) * observation::ACTION_DIM],
                &mut cache,
            );
        }
        let array = numpy::ndarray::Array2::from_shape_vec((rows, observation::ACTION_DIM), matrix)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py).unbind())
    }

    /// The whole position as plain Python data.
    ///
    /// This is the same structure the differential corpus records, so a Rust
    /// position and a Python one can be compared field by field from Python —
    /// which is what the engine tests need in order to read a position without
    /// caring which implementation produced it.
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = agot_core::fingerprint::state_fingerprint(&self.game.state, &self.game.data);
        let text = serde_json::to_string(&value)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let json = py.import("json")?;
        Ok(json.call_method1("loads", (text,))?.unbind())
    }

    /// Whether the desktop-sequenced Westeros phase is waiting for Continue.
    fn westeros_waiting(&self) -> bool {
        self.game.state.westeros_waiting
    }

    /// Resolve the next step of a desktop-sequenced Westeros phase.
    fn continue_westeros(&mut self) -> PyResult<()> {
        self.game.continue_westeros().map_err(engine_error)
    }

    /// Apply Doran Martell's chosen Influence track in the current battle.
    fn choose_doran_track(&mut self, track: &str) -> PyResult<()> {
        let track = agot_core::enums::Track::from_name(track)
            .ok_or_else(|| PyValueError::new_err(format!("unknown track {track:?}")))?;
        self.game.choose_doran_track(track).map_err(engine_error)
    }

    /// The parked battle, as the dict the Python engine keeps.
    ///
    /// The fingerprint only records whether a battle is pending; a caller that
    /// scores support or reads the two sides needs the contents, so they are
    /// serialized here rather than in the fixture format.
    fn pending_battle(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some(battle) = self.game.state.pending_battle.as_ref() else {
            return Ok(py.None());
        };
        let units = |counts: &[u8; 4]| {
            let mut map = serde_json::Map::new();
            for unit in agot_core::enums::UnitType::ALL {
                map.insert(
                    unit.as_str().to_string(),
                    serde_json::json!(counts[unit.index()]),
                );
            }
            serde_json::Value::Object(map)
        };
        let support = agot_core::combat::support_from_choices(
            &self.game.state,
            &self.game.data.map,
            battle.region,
            &battle.support_choices,
        );
        let target = self.game.state.region(battle.region);
        let target_is_fortification = self.game.data.map.region(battle.region).is_fortification();
        let attacker_unit_strength =
            agot_core::combat::units_strength(&battle.attacker_units, target_is_fortification)
                as i16;
        let defender_unit_strength =
            agot_core::combat::units_strength(&battle.defender_units, false) as i16;
        let attacker_total = attacker_unit_strength
            + battle.attacker_order_strength
            + support.attacker_strength as i16;
        let defender_total = defender_unit_strength
            + battle.defender_order_strength
            + support.defender_strength as i16
            + target.garrison as i16;
        let value = serde_json::json!({
            "region": self.game.data.map.id_of(battle.region),
            "attacker": battle.attacker.name(),
            "defender": battle.defender.map(|house| house.name()),
            "neutral": battle.neutral,
            "origin": battle.origin.map(|region| self.game.data.map.id_of(region)),
            "stage": format!("{:?}", battle.stage),
            "attacker_units": units(&battle.attacker_units),
            "defender_units": units(&battle.defender_units),
            // Both House cards stay hidden until both players have committed
            // (FFG secrecy ruling); the physical game reveals them together.
            "attacker_card": if battle.attacker_card_chosen && battle.defender_card_chosen {
                battle.attacker_card.map(|card| {
                    self.game.data.house_cards.card(battle.attacker, card).id.as_str()
                })
            } else {
                None
            },
            "defender_card": if battle.attacker_card_chosen && battle.defender_card_chosen {
                match (battle.defender, battle.defender_card) {
                    (Some(house), Some(card)) => {
                        Some(self.game.data.house_cards.card(house, card).id.as_str())
                    }
                    _ => None,
                }
            } else {
                None
            },
            "doran_owner": battle.doran_owner.map(|house| house.name()),
            "attacker_units_strength": attacker_unit_strength,
            "defender_units_strength": defender_unit_strength,
            "attacker_order_strength": battle.attacker_order_strength,
            "defender_order_strength": battle.defender_order_strength,
            "attacker_support": support.attacker_strength,
            "defender_support": support.defender_strength,
            "defender_garrison": target.garrison,
            "neutral_force": target.neutral_force,
            "attacker_total": attacker_total,
            "defender_total": defender_total,
            "blade_done": battle.blade_done,
            "attacker_blade": battle.attacker_blade,
            "defender_blade": battle.defender_blade,
            "casualties_remaining": battle.casualties_remaining,
        });
        to_python(py, &value)
    }

    /// The generic choice the game is waiting on, or `None`.
    ///
    /// Read-only decision context for the desktop game and the training loop:
    /// the network layout is frozen and `state_fingerprint` must not grow a
    /// choice block, so the pending task is exposed here instead. The dict is
    /// `observation::pending_choice`'s: `kind`, `purpose`, `house`, `queued`,
    /// `resume` when the queue resumes a battle/Wildling sequence, plus the
    /// variant's own lists (`areas`, `cards`, `regions`, `tracks`, `options`)
    /// and counts (`remaining`, `allow_none`, `points`, `count`, `target`).
    fn pending_choice(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match observation::pending_choice(&self.game.state, &self.game.data) {
            Some(value) => to_python(py, &value),
            None => Ok(py.None()),
        }
    }

    /// The Wildling attack in progress, as the dict the Python engine keeps.
    fn wildling_attack(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some(attack) = self.game.state.wildling_attack.as_ref() else {
            return Ok(py.None());
        };
        let bids: serde_json::Map<String, serde_json::Value> = attack
            .bids
            .iter()
            .map(|(house, amount)| (house.name().to_string(), serde_json::json!(amount)))
            .collect();
        let tie = self.game.state.pending_wildling_tie.as_ref();
        let value = serde_json::json!({
            "strength": attack.strength,
            "bids": bids,
            "excluded": attack
                .excluded
                .iter()
                .map(|house| house.name())
                .collect::<Vec<_>>(),
            "card": attack.card.map(|card| self.game.data.wildlings.card_id(card)),
            "return_phase": attack.return_phase.as_str(),
            "tie_highest": tie.map(|tie| tie.highest),
            "tie_chooser": tie.map(|tie| tie.chooser.name()),
            "tie_candidates": tie.map(|tie| {
                tie.candidates.iter().map(|house| house.name()).collect::<Vec<_>>()
            }),
        });
        to_python(py, &value)
    }

    /// Compact description of the position, as in the Python engine.
    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = self.game.summary();
        let text = serde_json::to_string(&value)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let json = py.import("json")?;
        Ok(json.call_method1("loads", (text,))?.into())
    }

    // -- building positions ----------------------------------------------
    //
    // The engine tests set up specific positions before exercising a rule, and
    // the Python engine let them do that by writing fields directly. These are
    // the same writes, named: enough to build a position, no more. They are
    // deliberately not a general view tier — a test that needs to *read* a
    // position uses `snapshot()`.

    /// `game.set_region("winterfell", controller="stark", units={"footman": 1})`.
    #[pyo3(signature = (region_id, **fields))]
    fn set_region(&mut self, region_id: &str, fields: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let region = self
            .game
            .data
            .map
            .lookup(region_id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown region {region_id:?}")))?;
        let Some(fields) = fields else {
            return Ok(());
        };
        for (key, value) in fields.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "controller" => {
                    self.game.state.region_mut(region).controller = optional_house(&value)?
                }
                "owner" => self.game.state.region_mut(region).owner = optional_house(&value)?,
                "influence_token" => {
                    self.game.state.region_mut(region).influence_token = optional_house(&value)?
                }
                "order_house" => {
                    self.game.state.region_mut(region).order_house = optional_house(&value)?
                }
                "units" => self.game.state.region_mut(region).units = parse_units(&value)?,
                "routed" => self.game.state.region_mut(region).routed = parse_units(&value)?,
                "order" => {
                    self.game.state.region_mut(region).order = optional_order(&value)?;
                }
                "garrison" => self.game.state.region_mut(region).garrison = value.extract()?,
                "neutral_force" => {
                    self.game.state.region_mut(region).neutral_force = value.extract()?
                }
                "mustering_points" => {
                    self.game.state.region_mut(region).mustering_points = value.extract()?
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown region field {other:?}"
                    )))
                }
            }
        }
        Ok(())
    }

    /// `game.set_house("stark", power=3, supply_level=1)`.
    #[pyo3(signature = (house, **fields))]
    fn set_house(&mut self, house: &str, fields: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let house = HouseId::from_name(house)
            .ok_or_else(|| PyValueError::new_err(format!("unknown house {house:?}")))?;
        let Some(fields) = fields else {
            return Ok(());
        };
        for (key, value) in fields.iter() {
            let key: String = key.extract()?;
            let state = self
                .game
                .state
                .house_mut(house)
                .ok_or_else(|| PyValueError::new_err(format!("{house} is not in this game")))?;
            match key.as_str() {
                "power" => state.power = value.extract()?,
                "supply_level" => state.supply_level = value.extract()?,
                "orders_available" => {
                    state.orders_available = parse_orders(&value)?;
                }
                "orders_placed" => state.orders_placed = parse_orders(&value)?,
                "house_cards_available" => {
                    state.house_cards_available = parse_cards(&value, house, &self.game.data)?;
                }
                "house_cards_used" => {
                    state.house_cards_used = parse_cards(&value, house, &self.game.data)?;
                }
                "blade_used" => state.blade_used = value.extract()?,
                "passed" => state.passed = value.extract()?,
                "eliminated" => state.eliminated = value.extract()?,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown house field {other:?}"
                    )))
                }
            }
        }
        Ok(())
    }

    /// `game.set_track("iron_throne", ["stark", "greyjoy", ...])`.
    fn set_track(&mut self, track: &str, houses: Vec<String>) -> PyResult<()> {
        let track = agot_core::enums::Track::from_name(track)
            .ok_or_else(|| PyValueError::new_err(format!("unknown track {track:?}")))?;
        let order = houses
            .iter()
            .map(|name| {
                HouseId::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))
            })
            .collect::<PyResult<smallvec::SmallVec<[HouseId; 6]>>>()?;
        self.game.state.tracks[track.index()] = order;
        Ok(())
    }

    /// Declare a winner (`None` clears it), as the engine does when the victory
    /// condition is met. The clearing form exists for tests that build a
    /// finished position and then look again.
    fn set_winner(&mut self, house: Option<&str>) -> PyResult<()> {
        match house {
            Some(name) => {
                let house = HouseId::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))?;
                self.game.state.set_winner(house);
            }
            None => self.game.state.winner = None,
        }
        Ok(())
    }

    /// The trace log, empty while it is switched off (search).
    fn log(&self) -> Vec<String> {
        self.game.state.log.clone().unwrap_or_default()
    }

    /// Build an action from names: `game.action("place_order", region="winterfell", order="march")`.
    ///
    /// `card_house` names the deck a `card` belongs to when it is not `house`'s
    /// own (Patchface discards an opponent's card); without it the card is
    /// looked up in `house`'s deck first and then across the decks by printed
    /// name, so a GUI can rebuild any legal action it was offered.
    #[pyo3(signature = (kind, *, house=None, region=None, target=None, order=None, card=None, card_house=None, unit=None, side=None, amount=None, units=None, establish=false))]
    #[allow(clippy::too_many_arguments)]
    fn action(
        &self,
        kind: &str,
        house: Option<String>,
        region: Option<String>,
        target: Option<String>,
        order: Option<String>,
        card: Option<String>,
        card_house: Option<String>,
        unit: Option<String>,
        side: Option<String>,
        amount: Option<u8>,
        units: Option<(u8, u8, u8, u8)>,
        establish: bool,
    ) -> PyResult<PyAction> {
        let data = &self.game.data;
        let kind = ActionKind::from_name(kind)
            .ok_or_else(|| PyValueError::new_err(format!("unknown action kind {kind:?}")))?;
        let house_id = house
            .as_deref()
            .map(|name| {
                HouseId::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))
            })
            .transpose()?;
        let named_card_house = card_house
            .as_deref()
            .map(|name| {
                HouseId::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown house {name:?}")))
            })
            .transpose()?;
        let resolve = |id: &str| {
            data.map
                .lookup(id)
                .ok_or_else(|| PyValueError::new_err(format!("unknown region {id:?}")))
        };
        let region_id = region.as_deref().map(resolve).transpose()?;
        let target_id = target.as_deref().map(resolve).transpose()?;
        let order_key = order
            .as_deref()
            .map(|key| {
                agot_core::enums::OrderKey::from_name(key)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown order {key:?}")))
            })
            .transpose()?;
        let (card_id, action_card_house) = match (kind, card.as_deref()) {
            // A holders'-choice action stores a card id from the Westeros deck,
            // with the literal "none" as the no-effect sentinel.
            (ActionKind::WesterosChoice, Some("none")) => {
                (Some(CardId(agot_core::state::NONE_OPTION as u8)), None)
            }
            (ActionKind::WesterosChoice, Some(id)) => {
                let index = data.westeros.card_index(id).ok_or_else(|| {
                    PyValueError::new_err(format!("unknown Westeros card {id:?}"))
                })?;
                (Some(CardId(index as u8)), None)
            }
            (_, Some(id)) => {
                let (card, deck) = resolve_house_card(data, named_card_house.or(house_id), id)?;
                // Mirror `Action::with_card_house`: only a *foreign* deck is
                // recorded, so a rebuilt action compares equal to the one the
                // engine offered.
                let card_house = if Some(deck) == house_id {
                    None
                } else {
                    Some(deck)
                };
                (Some(card), card_house)
            }
            _ => (None, None),
        };
        let unit_id = unit
            .as_deref()
            .map(|name| {
                UnitType::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown unit {name:?}")))
            })
            .transpose()?;
        let side_id = side
            .as_deref()
            .map(|name| {
                Side::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown side {name:?}")))
            })
            .transpose()?;

        Ok(PyAction::new(
            agot_core::action::Action {
                kind,
                house: house_id,
                region: region_id,
                target: target_id,
                order: order_key,
                card: card_id,
                unit: unit_id,
                side: side_id,
                amount,
                units: units.map(|counts| [counts.0, counts.1, counts.2, counts.3]),
                establish,
                card_house: action_card_house,
            },
            Arc::clone(&self.game.data),
        ))
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
#[pyo3(name = "_core")]
fn agot_core_ext(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(backend, module)?)?;
    module.add_function(wrap_pyfunction!(action_kinds, module)?)?;
    module.add_function(wrap_pyfunction!(vocabulary, module)?)?;
    module.add_function(wrap_pyfunction!(observation_dims, module)?)?;
    module.add_function(wrap_pyfunction!(selfplay::selfplay, module)?)?;
    module.add_function(wrap_pyfunction!(selfplay::game_outcome, module)?)?;
    module.add_class::<PyAction>()?;
    module.add_class::<PyGame>()?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
