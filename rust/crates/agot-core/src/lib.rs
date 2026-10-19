//! Rules engine, Monte-Carlo tree search and observation encoder for
//! *A Game of Thrones: The Board Game (2nd Edition)*.
//!
//! This crate is the Rust port of the Python package's `engine/`, `agents/mcts.py`
//! and `engine/observation.py`. It is deliberately free of any Python or torch
//! dependency: the Python layer drives it through a thin extension module, and
//! the training loop keeps owning the neural network.
//!
//! Porting rules:
//!
//! * behaviour is ported verbatim from the Python reference until the
//!   differential harness (`tests/planning_parity.rs` and its successors) is
//!   green; refactoring happens only afterwards;
//! * the board is owned by value (no copy-on-write), so a clone cannot alias a
//!   source region — the invariant Python's `GameState.mut` enforced by hand.

pub mod action;
pub mod action_phase;
pub mod combat;
pub mod enums;
pub mod fingerprint;
pub mod game;
pub mod hidden;
pub mod house_cards;
pub mod map;
pub mod observation;
pub mod phases;
pub mod rng;
#[cfg(test)]
mod rules_fixes;
pub mod search;
pub mod setup;
pub mod state;

pub use action::{Action, ActionKind, Side};
pub use combat::{
    apply_battle_result, apply_casualties, destroy_army, legal_retreats, order_modifier,
    potential_support_strength, resolve_and_apply, resolve_battle, retreat, retreat_keeps_supply,
    retreat_options, support_detail, support_from_choices, support_orders, support_strength,
    unit_strength, units_strength, BattleRequest, BattleResult, BattleSide, SupportTotals,
};
pub use enums::{
    ActionStep, CardId, HouseId, OrderKey, OrderType, Phase, RegionId, Terrain, Track, UnitType,
};
pub use game::{EngineError, Game};
pub use hidden::determinize;
pub use house_cards::BattleContext;
pub use map::{DataSet, GameMap, MapDataError};
pub use observation::{
    encode_action, encode_action_into, encode_state, observe, Observation, RegionCache,
};
pub use rng::Rng;
pub use search::{
    outcome_vector, Evaluation, Evaluator, Mcts, SearchResult, Simulation, UniformEvaluator,
};
pub use setup::GameConfig;
pub use state::State;
