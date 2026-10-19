//! Monte-Carlo tree search with a pluggable evaluator (PUCT).
//!
//! Port of `src/agot/agents/mcts.py`. The engine supports any number of
//! players, so a two-player negamax value (the usual AlphaZero sign flip) is not
//! enough: every node stores a *value vector* with one entry per canonical house
//! (`value_sums[house]` = summed reward for that house over the node's visits),
//! and a node compares its children through the entry of *the house that is
//! choosing* (`node.player`). That keeps the search correct for 2-6 players
//! with no sign bookkeeping anywhere.
//!
//! Four structural decisions, and why:
//!
//! * **Arena.** Python's tree is a web of `_Node` objects; here it is a
//!   `Vec<Node>` addressed by `u32` indices, and a node's children are a
//!   contiguous range of that vector. A node holds its `State` by value (~490 ns
//!   to clone, no allocation), so a descent walks a flat array and never chases
//!   a pointer.
//! * **Lazy children.** Expanding a node records one *pending* child per legal
//!   action and only steps into a child when the search actually visits it. Most
//!   children of a 45-action Planning decision are never visited, and the Python
//!   original defers the same way (`_Node(pending=(parent, action))`).
//! * **Split leaf evaluation.** [`Mcts::prepare_simulation`] descends to a leaf
//!   without scoring it; [`Mcts::finish_simulation`] expands the leaf and backs
//!   the value up. A driver can therefore pause many games at their leaves,
//!   score every leaf in one batched network call, and only then resume — which
//!   is what makes batched self-play fast.
//! * **Plain value arrays.** Python allocates a node's `value_sums` on its first
//!   visit; here it is a `[f64; N_HOUSES]` field. 48 bytes inline remove an
//!   allocation and a branch from the backup path, and an unvisited node's sums
//!   are never read (its q is defined as zero).
//!
//! Hidden information is handled by PIMC: with `determinizations > 1` the search
//! runs on several resampled worlds ([`crate::hidden::determinize`]) and
//! aggregates the root visit counts.
//!
//! One port boundary to be aware of: a phase that has not been ported yet
//! produces no legal actions, so a descent stops there and treats the position
//! as terminal for backup purposes (every house in play scores `-1`). That
//! cannot happen in the Python reference, which has no unported phases.

use std::collections::HashMap;
use std::sync::Arc;

use crate::action::Action;
use crate::enums::{HouseId, Phase, N_HOUSES};
use crate::game::{EngineError, Game};
use crate::hidden::determinize;
use crate::map::DataSet;
use crate::rng::Rng;
use crate::setup::GameConfig;
use crate::state::State;

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// Canonical value-vector slot of a house, `0` when there is none.
///
/// Port of Python's `node_value_index`.
pub fn node_value_index(house: Option<HouseId>) -> usize {
    house.map(|house| house.index()).unwrap_or(0)
}

/// Per-house terminal reward: `+1` for the winner, `-1` for the rest.
///
/// Houses that are not part of this game keep a zero, so the vector always has
/// [`N_HOUSES`] entries. Port of `agents/base.py::outcome_vector`; it lives here
/// because the search is its only caller. "Part of this game" means "has a house
/// record in the state", which is exactly the set Python's `config.houses`
/// holds.
pub fn outcome_vector(state: &State) -> [f64; N_HOUSES] {
    let mut result = [0.0; N_HOUSES];
    for house in HouseId::ALL {
        if state.house(house).is_none() {
            continue;
        }
        result[house.index()] = if state.winner == Some(house) {
            1.0
        } else {
            -1.0
        };
    }
    result
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// What an evaluator returns for one position.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Prior over the position's legal actions, unnormalized: the search clips
    /// them at `1e-8` and normalizes.
    pub priors: Vec<f64>,
    /// Predicted value, indexed by [`HouseId`].
    pub value: [f64; N_HOUSES],
}

/// Supplies a policy prior and a value vector for a position.
///
/// A trait rather than a boxed closure, so the Python binding can implement it
/// with a struct that collects leaves across games; the blanket implementation
/// below covers plain closures.
pub trait Evaluator {
    fn evaluate(&mut self, game: &Game, actions: &[Action]) -> Evaluation;
}

impl<F> Evaluator for F
where
    F: FnMut(&Game, &[Action]) -> Evaluation,
{
    fn evaluate(&mut self, game: &Game, actions: &[Action]) -> Evaluation {
        self(game, actions)
    }
}

/// A network-free evaluator: uniform priors, neutral values.
///
/// The default of the Python `MCTS`, and what lets the engine and the search be
/// exercised without a network.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&mut self, _game: &Game, actions: &[Action]) -> Evaluation {
        let count = actions.len();
        Evaluation {
            priors: vec![1.0 / count.max(1) as f64; count],
            value: [0.0; N_HOUSES],
        }
    }
}

// ---------------------------------------------------------------------------
// Result and simulation
// ---------------------------------------------------------------------------

/// What a completed search exposes to its caller.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchResult {
    /// The root's legal actions, in engine order.
    pub actions: Vec<Action>,
    /// Visits per action, aggregated over the PIMC worlds.
    pub visits: Vec<f64>,
    /// Mean value of each action *for the acting house*.
    pub q_values: Vec<f64>,
}

impl SearchResult {
    /// The result of a search over a terminal (or actionless) position.
    pub fn empty() -> SearchResult {
        SearchResult::default()
    }

    /// Index of the most visited action, ties going to the earliest.
    ///
    /// `None` only for an empty result. All-zero visits (a search that only
    /// expanded the root) select the first action, like `np.argmax`.
    pub fn argmax_visits(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (index, value) in self.visits.iter().enumerate() {
            match best {
                None => best = Some(index),
                Some(current) if *value > self.visits[current] => best = Some(index),
                _ => {}
            }
        }
        best
    }

    /// Visit distribution at a temperature, the policy target of self-play.
    ///
    /// `temperature <= 1e-3` returns a one-hot vector on the most visited
    /// action; otherwise visits are raised to `1 / temperature` and normalized.
    /// An all-zero visit vector falls back to uniform. Python's default
    /// temperature is `1.0`.
    pub fn probabilities(&self, temperature: f64) -> Vec<f64> {
        let count = self.actions.len();
        if count == 0 {
            return Vec::new();
        }
        let total: f64 = self.visits.iter().sum();
        if total <= 0.0 {
            return vec![1.0 / count as f64; count];
        }
        if temperature <= 1e-3 {
            let mut one_hot = vec![0.0; count];
            if let Some(index) = self.argmax_visits() {
                one_hot[index] = 1.0;
            }
            return one_hot;
        }
        let scaled: Vec<f64> = self
            .visits
            .iter()
            .map(|visits| visits.powf(1.0 / temperature))
            .collect();
        let scaled_total: f64 = scaled.iter().sum();
        if scaled_total <= 0.0 {
            return vec![1.0 / count as f64; count];
        }
        scaled
            .into_iter()
            .map(|value| value / scaled_total)
            .collect()
    }
}

/// One MCTS simulation paused between descent and leaf evaluation.
///
/// `value` is set when the descent stopped at a terminal (or actionless)
/// position. Otherwise `leaf` is the node that still needs a network score and
/// `actions` are its legal moves; a batching driver collects these across games,
/// scores them in one forward pass, then calls [`Mcts::finish_simulation`].
#[derive(Clone, Debug, PartialEq)]
pub struct Simulation {
    /// Arena indices from the root down to the leaf, inclusive.
    pub path: Vec<u32>,
    /// The node awaiting an evaluation, or `None` for a finished position.
    pub leaf: Option<u32>,
    /// The leaf's legal actions (empty when there is no leaf).
    pub actions: Vec<Action>,
    /// The terminal value, when there is no leaf to evaluate.
    pub value: Option<[f64; N_HOUSES]>,
    /// Whether Dirichlet noise applies: true only when the leaf is the root.
    pub apply_noise: bool,
}

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

/// One position in the search tree.
///
/// A node owns the static tables as well as its position, so every operation
/// that needs a `Game` (legal actions, the evaluator, stepping into a child) can
/// rebuild one without the caller handing the game back. Both handles are cheap:
/// an `Arc` bump and an inline `GameConfig` copy.
struct Node {
    data: Arc<DataSet>,
    config: GameConfig,
    /// The position, or `None` while the node is still pending.
    ///
    /// Boxed deliberately: most nodes are pending children created by an
    /// expansion and never visited, and an inline `State` is ~3.2 KB. At 45
    /// candidates per expansion that padding dominated the search's memory
    /// traffic (a 128-simulation decision wrote ~19 MB of never-read state).
    state: Option<Box<State>>,
    /// Stream of the node's own game; a child forks from it when it is stepped.
    rng: Rng,
    /// `(parent, action)` until the node is first stepped into.
    pending: Option<(u32, Action)>,
    /// The action that led here, `None` for a root. This is the parent's action
    /// list, so no separate action arena is needed.
    action: Option<Action>,
    /// Prior the parent assigned to this child, after clipping and noise.
    prior: f64,
    /// House that chooses here; set when the node is expanded.
    player: Option<HouseId>,
    /// Children are `nodes[first_child .. first_child + child_count]`.
    first_child: u32,
    child_count: u32,
    /// Visits to this node.
    visits: u32,
    /// Summed per-house reward over those visits.
    value_sums: [f64; N_HOUSES],
}

impl Node {
    /// A materialized root, the analogue of Python's `_Node(game=game.clone())`.
    fn root(game: &Game) -> Node {
        // `Game::clone` forks the stream from a copy, leaving the source alone;
        // do the same without copying the board a second time.
        let mut source = game.rng.clone();
        Node {
            data: Arc::clone(&game.data),
            config: game.config.clone(),
            state: Some(Box::new(game.state.clone())),
            rng: source.fork(),
            pending: None,
            action: None,
            prior: 0.0,
            player: None,
            first_child: 0,
            child_count: 0,
            visits: 0,
            value_sums: [0.0; N_HOUSES],
        }
    }

    /// Whether the node has been expanded (Python's `_Node.expanded`).
    #[inline]
    fn expanded(&self) -> bool {
        self.child_count > 0
    }
}

/// Puts a moved-out position back when it goes out of scope.
struct RestoreGame<'a> {
    slot: &'a mut Option<Box<State>>,
    game: Option<Game>,
}

impl Drop for RestoreGame<'_> {
    fn drop(&mut self) {
        if let Some(game) = self.game.take() {
            *self.slot = Some(Box::new(game.state));
        }
    }
}

/// Runs `body` with a node's position as a `Game`.
///
/// The state is *moved* into the temporary game and moved back when the body
/// returns — even if it panics — so no copy of the board is made and the arena
/// keeps the only copy throughout. The node must be materialized.
fn with_node_game<R>(nodes: &mut [Node], index: u32, body: impl FnOnce(&Game) -> R) -> R {
    let node = &mut nodes[index as usize];
    let game = Game {
        data: Arc::clone(&node.data),
        // Move the state out of its box: `with_node_game` hands out a `Game`,
        // which owns a plain `State`. The guard re-boxes it on the way out, so
        // only materialized nodes ever pay this allocation.
        state: *node
            .state
            .take()
            .expect("a node must be materialized before it is used"),
        config: node.config.clone(),
        rng: node.rng.clone(),
    };
    let restore = RestoreGame {
        slot: &mut nodes[index as usize].state,
        game: Some(game),
    };
    body(restore.game.as_ref().expect("the guard owns the game"))
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// PUCT search that turns an evaluator into a policy.
///
/// The arena belongs to the searcher: [`Mcts::open_search`] and
/// [`Mcts::open_search_worlds`] clear it, so roots and paused simulations from
/// an earlier decision become invalid (Python's tree is garbage collected
/// instead).
pub struct Mcts<E: Evaluator = UniformEvaluator> {
    /// Simulations per decision. Python's default is 32.
    pub simulations: u32,
    /// Exploration constant of the PUCT rule. Python's default is 1.5.
    pub c_puct: f64,
    /// Temperature used by [`Mcts::pick`] when the caller passes none.
    pub temperature: f64,
    /// Dirichlet concentration of the root noise. Python's default is 0.3.
    pub dirichlet_alpha: f64,
    /// How much of the root prior the noise replaces. Python's default is 0.25.
    pub dirichlet_weight: f64,
    /// Resampled worlds searched per decision. `<= 1` means plain
    /// perfect-information search; higher values enable PIMC.
    pub determinizations: u32,
    /// The evaluator this searcher drives.
    pub evaluator: E,
    /// Stream for the root noise and for sampling in [`Mcts::pick`].
    rng: Rng,
    /// Separate stream for [`determinize`], so PIMC worlds do not perturb the
    /// noise stream (Python keeps `_det_rng` for the same reason).
    det_rng: Rng,
    /// The arena: nodes, with each node's children stored contiguously.
    nodes: Vec<Node>,
}

impl<E: Evaluator> Mcts<E> {
    /// A searcher with Python's defaults and a fresh stream.
    pub fn new(evaluator: E) -> Mcts<E> {
        Mcts::with_seed(evaluator, 0)
    }

    /// A searcher seeded like `MCTS(..., seed=seed)`.
    pub fn with_seed(evaluator: E, seed: u64) -> Mcts<E> {
        Mcts {
            simulations: 32,
            c_puct: 1.5,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            determinizations: 1,
            evaluator,
            rng: Rng::from_seed(seed),
            // Python seeds two different generators with the same seed; two
            // xoshiro streams must not start from the same state, so derive one.
            det_rng: Rng::from_seed(seed ^ 0x9E37_79B9_7F4A_7C15),
            nodes: Vec::new(),
        }
    }

    /// Reseeds the noise/sampling stream and the determinization stream.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Rng::from_seed(seed);
        self.det_rng = Rng::from_seed(seed ^ 0x9E37_79B9_7F4A_7C15);
    }

    /// Drops every node. Roots and paused simulations from before are invalid.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    // -- whole-decision entry points -------------------------------------

    /// Runs a whole search for the game's current decision.
    ///
    /// This is Python's `MCTS.search`: one perfect-information tree, or PIMC
    /// over `determinizations` worlds. The arena is cleared first.
    pub fn search(&mut self, game: &Game) -> Result<SearchResult, EngineError> {
        self.clear();
        let observer = game.current_player();
        let pimc = self.determinizations > 1 && observer.is_some();
        let roots = self.open_worlds(game, observer);
        if roots.is_empty() {
            return Ok(SearchResult::empty());
        }
        let per_world = if pimc {
            (self.simulations / self.determinizations.max(1)).max(1)
        } else {
            self.simulations
        };
        for root in &roots {
            self.run(*root, per_world)?;
        }
        if roots.len() == 1 {
            Ok(self.result_from_root(roots[0]))
        } else {
            Ok(self.aggregate_roots(&roots, observer))
        }
    }

    /// Creates (but does not run) a search root, for stepwise drivers.
    ///
    /// The arena is cleared first. `None` means the position is already over.
    pub fn open_search(&mut self, game: &Game) -> Option<u32> {
        self.clear();
        self.push_root(game)
    }

    /// Roots for one stepwise decision: one per determinized world.
    ///
    /// With `determinizations <= 1` this is a single perfect-information root.
    /// Otherwise each root is a resampled world the acting house cannot
    /// distinguish; a stepwise driver runs simulations on all of them and
    /// aggregates with [`Mcts::aggregate_roots`]. The arena is cleared first.
    pub fn open_search_worlds(&mut self, game: &Game) -> Vec<u32> {
        self.clear();
        self.open_worlds(game, game.current_player())
    }

    /// A deterministic greedy action: Python's `MCTS.act`.
    pub fn act(&mut self, game: &Game) -> Result<Action, EngineError> {
        let result = self.search(game)?;
        match result.argmax_visits() {
            Some(index) => Ok(result.actions[index]),
            None => Err(EngineError::IllegalAction(
                "MCTS was asked to move in a terminal position".to_string(),
            )),
        }
    }

    /// Samples an action and returns it with its (float32) target.
    ///
    /// Python's `MCTS.pick`; `temperature` overrides the searcher's own value
    /// for this one call. A temperature at or below `1e-3` takes the most
    /// visited action instead of sampling.
    pub fn pick(
        &mut self,
        result: &SearchResult,
        temperature: Option<f64>,
    ) -> Result<(Action, Vec<f32>), EngineError> {
        if result.actions.is_empty() {
            return Err(EngineError::IllegalAction(
                "MCTS was asked to move in a terminal position".to_string(),
            ));
        }
        let temperature = temperature.unwrap_or(self.temperature);
        let probabilities = result.probabilities(temperature);
        let index = if temperature <= 1e-3 {
            result.argmax_visits().unwrap_or(0)
        } else {
            self.sample_index(&probabilities)
        };
        let target = probabilities.iter().map(|value| *value as f32).collect();
        Ok((result.actions[index], target))
    }

    /// Search and pick in one call, Python's `MCTS.decide`.
    pub fn decide(&mut self, game: &Game) -> Result<(Action, Vec<f32>), EngineError> {
        let result = self.search(game)?;
        self.pick(&result, None)
    }

    // -- stepwise API ----------------------------------------------------

    /// Descends to this simulation's leaf without evaluating it.
    ///
    /// Every node on the path is materialized, so the leaf's position can be
    /// scored; children created by an expansion stay pending until they are
    /// visited.
    pub fn prepare_simulation(&mut self, root: u32) -> Result<Simulation, EngineError> {
        let mut path = vec![root];
        let mut index = root;
        loop {
            self.materialize(index)?;
            if self.state_of(index).phase == Phase::GameOver {
                let value = outcome_vector(self.state_of(index));
                return Ok(Simulation {
                    path,
                    leaf: None,
                    actions: Vec::new(),
                    value: Some(value),
                    apply_noise: false,
                });
            }
            if !self.nodes[index as usize].expanded() {
                let actions = self.legal_actions_at(index);
                if actions.is_empty() {
                    let value = outcome_vector(self.state_of(index));
                    return Ok(Simulation {
                        path,
                        leaf: None,
                        actions,
                        value: Some(value),
                        apply_noise: false,
                    });
                }
                return Ok(Simulation {
                    path,
                    leaf: Some(index),
                    actions,
                    value: None,
                    apply_noise: index == root,
                });
            }
            index = self.select_child(index);
            path.push(index);
        }
    }

    /// Expands the leaf (if any) and backs the value up along the path.
    ///
    /// `evaluation` is required when the simulation has a leaf, ignored when it
    /// has none; Python raises `ValueError` in the first case and accepts an
    /// overriding value in the second.
    pub fn finish_simulation(
        &mut self,
        simulation: Simulation,
        evaluation: Option<Evaluation>,
    ) -> Result<(), EngineError> {
        let value = match simulation.leaf {
            Some(leaf) => {
                let evaluation = evaluation.ok_or_else(|| {
                    EngineError::IllegalAction(
                        "finish_simulation: an unexpanded leaf needs priors and a value"
                            .to_string(),
                    )
                })?;
                if evaluation.priors.len() != simulation.actions.len() {
                    return Err(EngineError::IllegalAction(format!(
                        "finish_simulation: {} priors for {} actions",
                        evaluation.priors.len(),
                        simulation.actions.len()
                    )));
                }
                self.expand_with(
                    leaf,
                    &simulation.actions,
                    evaluation.priors,
                    evaluation.value,
                    simulation.apply_noise,
                )
            }
            None => evaluation
                .map(|evaluation| evaluation.value)
                .or(simulation.value)
                .ok_or_else(|| {
                    EngineError::IllegalAction(
                        "finish_simulation: a terminal simulation needs a value".to_string(),
                    )
                })?,
        };
        for index in &simulation.path {
            let node = &mut self.nodes[*index as usize];
            node.visits += 1;
            for (slot, contribution) in node.value_sums.iter_mut().zip(value) {
                *slot += contribution;
            }
        }
        Ok(())
    }

    /// Runs `simulations` simulations on a root (Python's `_run`).
    pub fn run(&mut self, root: u32, simulations: u32) -> Result<(), EngineError> {
        for _ in 0..simulations.max(1) {
            self.simulate(root)?;
        }
        Ok(())
    }

    /// One simulation with the searcher's own evaluator: Python's `_simulate`.
    pub fn simulate(&mut self, root: u32) -> Result<(), EngineError> {
        let simulation = self.prepare_simulation(root)?;
        let evaluation = match simulation.leaf {
            None => None,
            Some(leaf) => {
                // Split the borrow: the evaluator field is disjoint from the
                // arena, so the leaf's game can be handed to it directly.
                let Mcts {
                    nodes, evaluator, ..
                } = self;
                Some(with_node_game(nodes, leaf, |game| {
                    evaluator.evaluate(game, &simulation.actions)
                }))
            }
        };
        self.finish_simulation(simulation, evaluation)
    }

    // -- results ---------------------------------------------------------

    /// Packages a finished root into a [`SearchResult`].
    ///
    /// A root that was never expanded (a position with no legal actions) gives
    /// an empty result.
    pub fn result_from_root(&self, root: u32) -> SearchResult {
        let node = &self.nodes[root as usize];
        if !node.expanded() {
            return SearchResult::empty();
        }
        let mut actions = Vec::with_capacity(node.child_count as usize);
        let mut visits = Vec::with_capacity(node.child_count as usize);
        for child_index in node.first_child..node.first_child + node.child_count {
            let child = &self.nodes[child_index as usize];
            actions.push(
                child
                    .action
                    .expect("a child carries the action that made it"),
            );
            visits.push(child.visits as f64);
        }
        SearchResult {
            actions,
            visits,
            q_values: self.root_q(root),
        }
    }

    /// Merges finished PIMC roots into one visit distribution.
    ///
    /// The action list is the same in every world (determinization never changes
    /// the acting house's own legal moves), so visits simply add up.
    pub fn aggregate_roots(&self, roots: &[u32], observer: Option<HouseId>) -> SearchResult {
        if roots.is_empty() {
            return SearchResult::empty();
        }
        if roots.len() == 1 {
            return self.result_from_root(roots[0]);
        }
        let first = &self.nodes[roots[0] as usize];
        if !first.expanded() {
            return SearchResult::empty();
        }
        let actions: Vec<Action> = (first.first_child..first.first_child + first.child_count)
            .map(|child| {
                self.nodes[child as usize]
                    .action
                    .expect("a child carries the action that made it")
            })
            .collect();
        let index_of: HashMap<Action, usize> = actions
            .iter()
            .enumerate()
            .map(|(index, action)| (*action, index))
            .collect();

        let mut visits = vec![0.0; actions.len()];
        let mut value_sums = vec![[0.0; N_HOUSES]; actions.len()];
        for root in roots {
            let node = &self.nodes[*root as usize];
            for offset in 0..node.child_count {
                let child = &self.nodes[(node.first_child + offset) as usize];
                let Some(action) = child.action else { continue };
                let Some(position) = index_of.get(&action).copied() else {
                    continue;
                };
                if child.visits == 0 {
                    continue;
                }
                visits[position] += child.visits as f64;
                for (slot, contribution) in value_sums[position].iter_mut().zip(child.value_sums) {
                    *slot += contribution;
                }
            }
        }
        let value_index = node_value_index(observer.or(self.nodes[roots[0] as usize].player));
        let q_values = (0..actions.len())
            .map(|index| {
                if visits[index] > 0.0 {
                    value_sums[index][value_index] / visits[index]
                } else {
                    0.0
                }
            })
            .collect();
        SearchResult {
            actions,
            visits,
            q_values,
        }
    }

    /// A copy of a paused simulation's leaf position, for a batched driver.
    ///
    /// Python hands the driver `simulation.leaf.game` directly. The arena keeps
    /// the state, so the driver gets a `Game` clone instead (~0.5 µs, no
    /// allocation) and the paused simulation stays valid.
    pub fn leaf_game(&self, simulation: &Simulation) -> Option<Game> {
        let leaf = simulation.leaf?;
        let node = &self.nodes[leaf as usize];
        Some(Game {
            data: Arc::clone(&node.data),
            state: node.state.as_deref()?.clone(),
            config: node.config.clone(),
            rng: node.rng.clone(),
        })
    }

    // -- internals -------------------------------------------------------

    /// Appends a root for `game` without clearing the arena.
    fn push_root(&mut self, game: &Game) -> Option<u32> {
        if game.is_terminal() {
            return None;
        }
        let index = self.nodes.len() as u32;
        self.nodes.push(Node::root(game));
        Some(index)
    }

    /// Roots for one decision: one per determinized world, or a single root.
    fn open_worlds(&mut self, game: &Game, observer: Option<HouseId>) -> Vec<u32> {
        match observer {
            Some(observer) if self.determinizations > 1 => {
                let mut roots = Vec::new();
                for _ in 0..self.determinizations {
                    let world = determinize(game, observer, &mut self.det_rng);
                    if let Some(root) = self.push_root(&world) {
                        roots.push(root);
                    }
                }
                roots
            }
            _ => self.push_root(game).into_iter().collect(),
        }
    }

    /// Steps a pending node into its position, once.
    ///
    /// The parent's state is copied and the action applied exactly like Python's
    /// `clone = parent_game.clone(); clone.step(action)`.
    fn materialize(&mut self, index: u32) -> Result<(), EngineError> {
        let Some((parent, action)) = self.nodes[index as usize].pending else {
            return Ok(());
        };
        // One direct child game: Python does `parent_game.clone(); step`, but
        // the intermediate clone is redundant here because the child owns its
        // own state as soon as it exists.
        let mut child_game = {
            let parent = &self.nodes[parent as usize];
            Game {
                data: Arc::clone(&parent.data),
                state: parent
                    .state
                    .as_deref()
                    .expect("a pending child's parent is materialized")
                    .clone(),
                config: parent.config.clone(),
                rng: parent.rng.clone(),
            }
        };
        match child_game.step(action) {
            // The action was applied and the engine then stopped at a phase that
            // has not been ported yet: the position is the best the port can
            // produce, and the search treats it as actionless.
            Ok(()) | Err(EngineError::UnportedPhase(_)) => {}
            Err(error) => return Err(error),
        }
        let node = &mut self.nodes[index as usize];
        node.state = Some(Box::new(child_game.state));
        node.rng = child_game.rng;
        node.pending = None;
        Ok(())
    }

    /// The node's position; panics when the node is still pending.
    fn state_of(&self, index: u32) -> &State {
        self.nodes[index as usize]
            .state
            .as_deref()
            .expect("a node must be materialized before it is used")
    }

    /// The node's legal actions, through a temporary `Game`.
    fn legal_actions_at(&mut self, index: u32) -> Vec<Action> {
        with_node_game(&mut self.nodes, index, |game| game.legal_actions())
    }

    /// PUCT selection among an expanded node's children.
    fn select_child(&self, index: u32) -> u32 {
        let node = &self.nodes[index as usize];
        debug_assert!(node.expanded(), "only an expanded node selects a child");
        let sqrt_total = (node.visits.max(1) as f64).sqrt();
        let value_index = node_value_index(node.player);
        let mut best = node.first_child;
        let mut best_score = f64::NEG_INFINITY;
        for child_index in node.first_child..node.first_child + node.child_count {
            let child = &self.nodes[child_index as usize];
            let q = if child.visits > 0 {
                child.value_sums[value_index] / child.visits as f64
            } else {
                0.0
            };
            let u = self.c_puct * child.prior * sqrt_total / (1.0 + child.visits as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = child_index;
            }
        }
        best
    }

    /// Mean value of the root's children for the root's choosing house.
    fn root_q(&self, root: u32) -> Vec<f64> {
        let node = &self.nodes[root as usize];
        let value_index = node_value_index(node.player);
        (node.first_child..node.first_child + node.child_count)
            .map(|child_index| {
                let child = &self.nodes[child_index as usize];
                if child.visits > 0 {
                    child.value_sums[value_index] / child.visits as f64
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Expands a node: clips and normalizes the priors, adds the root noise and
    /// creates one pending child per action. Python's `_expand_with`.
    fn expand_with(
        &mut self,
        index: u32,
        actions: &[Action],
        priors: Vec<f64>,
        value: [f64; N_HOUSES],
        apply_noise: bool,
    ) -> [f64; N_HOUSES] {
        debug_assert!(!actions.is_empty(), "only an actionful node is expanded");
        let mut priors = priors;
        for prior in priors.iter_mut() {
            *prior = prior.max(1e-8);
        }
        let total: f64 = priors.iter().sum();
        if total > 0.0 {
            for prior in priors.iter_mut() {
                *prior /= total;
            }
        }
        if apply_noise && actions.len() > 1 {
            let noise = self.dirichlet(actions.len());
            let weight = self.dirichlet_weight;
            for (prior, sample) in priors.iter_mut().zip(noise.iter()) {
                *prior = (1.0 - weight) * *prior + weight * sample;
            }
        }
        let (data, config, rng) = {
            let parent = &self.nodes[index as usize];
            (
                Arc::clone(&parent.data),
                parent.config.clone(),
                parent.rng.clone(),
            )
        };
        let first_child = self.nodes.len() as u32;
        for (offset, action) in actions.iter().enumerate() {
            self.nodes.push(Node {
                data: Arc::clone(&data),
                config: config.clone(),
                state: None,
                rng: rng.clone(),
                pending: Some((index, *action)),
                action: Some(*action),
                prior: priors[offset],
                player: None,
                first_child: 0,
                child_count: 0,
                visits: 0,
                value_sums: [0.0; N_HOUSES],
            });
        }
        let parent = &mut self.nodes[index as usize];
        parent.player = parent.state.as_ref().and_then(|state| state.active_house);
        parent.first_child = first_child;
        parent.child_count = actions.len() as u32;
        value
    }

    /// A Dirichlet sample of `size` entries, `dirichlet_alpha` concentration.
    ///
    /// numpy's `Generator.dirichlet` draws one gamma per entry and normalizes;
    /// the two streams differ, only the distribution has to match.
    fn dirichlet(&mut self, size: usize) -> Vec<f64> {
        let alpha = self.dirichlet_alpha;
        let mut samples = Vec::with_capacity(size);
        for _ in 0..size {
            samples.push(sample_gamma(&mut self.rng, alpha));
        }
        let total: f64 = samples.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return vec![1.0 / size as f64; size];
        }
        for sample in samples.iter_mut() {
            *sample /= total;
        }
        samples
    }

    /// Draws an index from a categorical distribution (numpy's
    /// `rng.choice(n, p=...)`).
    fn sample_index(&mut self, probabilities: &[f64]) -> usize {
        let draw = self.rng.next_f64();
        let mut cumulative = 0.0;
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if draw < cumulative {
                return index;
            }
        }
        probabilities.len().saturating_sub(1)
    }
}

impl Default for Mcts<UniformEvaluator> {
    fn default() -> Self {
        Mcts::new(UniformEvaluator)
    }
}

/// Accessors the unit tests need to see through the arena.
#[cfg(test)]
impl<E: Evaluator> Mcts<E> {
    fn node_state(&self, index: u32) -> Option<&State> {
        self.nodes[index as usize].state.as_deref()
    }

    fn child_indices(&self, index: u32) -> Vec<u32> {
        let node = &self.nodes[index as usize];
        (node.first_child..node.first_child + node.child_count).collect()
    }
}

/// One standard normal sample (Box-Muller, the second sample discarded).
fn sample_normal(rng: &mut Rng) -> f64 {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// One gamma sample (Marsaglia-Tsang, with the `alpha < 1` boost).
fn sample_gamma(rng: &mut Rng, alpha: f64) -> f64 {
    if alpha < 1.0 {
        let boost = rng.next_f64().max(f64::MIN_POSITIVE);
        return sample_gamma(rng, alpha + 1.0) * boost.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.033_1 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::state_fingerprint;
    use crate::setup::GameConfig;

    fn data() -> Arc<DataSet> {
        Arc::new(DataSet::load_default().expect("bundled data must load"))
    }

    fn opening(seed: u64) -> Game {
        Game::new(data(), GameConfig::with_seed(seed)).expect("game must start")
    }

    #[test]
    fn a_node_does_not_inline_the_whole_board() {
        // Pending children are ~97% of an expansion's nodes; the board must be
        // behind the box or the arena writes megabytes it never reads.
        assert!(
            std::mem::size_of::<Node>() < std::mem::size_of::<State>() / 4,
            "Node is {} bytes, State is {}",
            std::mem::size_of::<Node>(),
            std::mem::size_of::<State>()
        );
    }

    #[test]
    fn outcome_vector_marks_the_winner() {
        let mut game = opening(5);
        let winner = HouseId(2);
        game.state.set_winner(winner);
        let vector = outcome_vector(&game.state);
        for house in HouseId::ALL {
            let expected = if house == winner { 1.0 } else { -1.0 };
            assert_eq!(vector[house.index()], expected);
        }

        // Houses that are not in the game keep a zero in their slot.
        let mut config = GameConfig::with_seed(5);
        config.houses.clear();
        config.houses.push(HouseId(2));
        config.houses.push(HouseId(3));
        let mut short = Game::new(data(), config).expect("two houses are a game");
        short.state.set_winner(HouseId(3));
        let vector = outcome_vector(&short.state);
        assert_eq!(vector[HouseId(2).index()], -1.0);
        assert_eq!(vector[HouseId(3).index()], 1.0);
        for house in HouseId::ALL {
            if house != HouseId(2) && house != HouseId(3) {
                assert_eq!(vector[house.index()], 0.0);
            }
        }
    }

    #[test]
    fn node_value_index_follows_the_canonical_order() {
        assert_eq!(node_value_index(Some(HouseId(0))), 0);
        assert_eq!(node_value_index(Some(HouseId(5))), 5);
        assert_eq!(node_value_index(None), 0);
    }

    #[test]
    fn probabilities_cover_the_edge_cases() {
        let actions = vec![
            Action::bid(HouseId(0), 0),
            Action::bid(HouseId(0), 1),
            Action::bid(HouseId(0), 2),
            Action::bid(HouseId(0), 3),
        ];
        let result = SearchResult {
            actions,
            visits: vec![3.0, 1.0, 1.0, 0.0],
            q_values: vec![0.0; 4],
        };

        // A one-hot target at a low temperature.
        assert_eq!(result.probabilities(1e-3), vec![1.0, 0.0, 0.0, 0.0]);
        // Temperature 1 is the raw visit distribution, zeros included.
        let flat = result.probabilities(1.0);
        assert!((flat[0] - 0.6).abs() < 1e-12);
        assert!((flat[1] - 0.2).abs() < 1e-12);
        assert_eq!(flat[3], 0.0);
        // A lower temperature sharpens but stays a distribution.
        let sharp = result.probabilities(0.5);
        assert!((sharp[0] - 9.0 / 11.0).abs() < 1e-12);
        assert!(sharp[0] > 0.5);
        assert_eq!(sharp.iter().filter(|value| **value > 0.0).count(), 3);

        // Zero visits fall back to uniform, an empty result to nothing.
        let unvisited = SearchResult {
            actions: vec![Action::pass(HouseId(1))],
            visits: vec![0.0],
            q_values: vec![0.0],
        };
        assert_eq!(unvisited.probabilities(1.0), vec![1.0]);
        assert!(SearchResult::empty().probabilities(1.0).is_empty());
    }

    #[test]
    fn argmax_visits_keeps_the_first_maximum() {
        let mut result = SearchResult {
            actions: vec![Action::pass(HouseId(0)); 3],
            visits: vec![1.0, 3.0, 3.0],
            q_values: vec![0.0; 3],
        };
        assert_eq!(result.argmax_visits(), Some(1));
        result.visits = vec![0.0, 0.0, 0.0];
        assert_eq!(result.argmax_visits(), Some(0));
        assert_eq!(SearchResult::empty().argmax_visits(), None);
    }

    #[test]
    fn dirichlet_is_a_distribution() {
        let mut search = Mcts::new(UniformEvaluator);
        for size in [2usize, 5, 45] {
            let samples = search.dirichlet(size);
            assert_eq!(samples.len(), size);
            let total: f64 = samples.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "size {size}: {samples:?}");
            assert!(samples.iter().all(|value| *value > 0.0));
        }
    }

    #[test]
    fn uniform_evaluator_is_uniform_and_neutral() {
        let game = opening(1);
        let actions = game.legal_actions();
        let evaluation = UniformEvaluator.evaluate(&game, &actions);
        assert_eq!(evaluation.value, [0.0; N_HOUSES]);
        assert_eq!(evaluation.priors.len(), actions.len());
        for prior in &evaluation.priors {
            assert!((*prior - 1.0 / actions.len() as f64).abs() < 1e-12);
        }
    }

    #[test]
    fn children_are_materialized_lazily_and_match_a_direct_step() {
        let game = opening(1);
        let mut search = Mcts::new(UniformEvaluator);
        search.simulations = 1;
        let root = search
            .open_search(&game)
            .expect("the opening is not terminal");
        search.run(root, 1).unwrap();

        let children = search.child_indices(root);
        assert_eq!(children.len(), game.legal_actions().len());
        assert!(
            children
                .iter()
                .all(|child| search.node_state(*child).is_none()),
            "unvisited children stay pending"
        );

        // The next descent materializes the child PUCT picks, and it must be the
        // position the engine itself produces from that action.
        let simulation = search.prepare_simulation(root).unwrap();
        let leaf = simulation
            .leaf
            .expect("the root's child is not expanded yet");
        let action = search.nodes[leaf as usize].action.unwrap();
        let mut expected = game.clone();
        expected
            .step(action)
            .expect("a legal planning action applies");
        assert_eq!(
            state_fingerprint(search.node_state(leaf).unwrap(), &game.data),
            state_fingerprint(&expected.state, &game.data)
        );
    }

    #[test]
    fn finish_simulation_rejects_a_leaf_without_an_evaluation() {
        let game = opening(1);
        let mut search = Mcts::new(UniformEvaluator);
        search.simulations = 1;
        let root = search.open_search(&game).unwrap();
        let simulation = search.prepare_simulation(root).unwrap();
        assert!(simulation.leaf.is_some());
        assert!(matches!(
            search.finish_simulation(simulation, None),
            Err(EngineError::IllegalAction(_))
        ));
    }

    #[test]
    fn leaf_game_is_the_leaf_position() {
        let game = opening(1);
        let mut search = Mcts::new(UniformEvaluator);
        search.simulations = 1;
        let root = search.open_search(&game).unwrap();
        let simulation = search.prepare_simulation(root).unwrap();
        let leaf = search.leaf_game(&simulation).expect("a leaf has a game");
        assert_eq!(
            state_fingerprint(&leaf.state, &game.data),
            state_fingerprint(&game.state, &game.data),
            "the first leaf of a fresh root is the root itself"
        );
        assert_eq!(leaf.legal_actions(), simulation.actions);
    }
}
