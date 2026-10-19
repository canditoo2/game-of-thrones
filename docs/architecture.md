# Architecture

The project is three loosely-coupled layers. Nothing above the engine may leak
into the layer below it, which keeps the rules code testable and torch-free.

```
            ┌──────────────────────────────────────────────┐
            │  training/   self-play · replay · trainer     │
            └───────────────┬──────────────────────────────┘
                            │ uses
            ┌───────────────▼──────────────┐   ┌───────────────────────┐
            │  agents/   random · greedy   │   │  nn/  scorer net     │
            │            PUCT MCTS         │──▶│       (optional torch) │
            └───────────────┬──────────────┘   └───────────────────────┘
                            │ drives
            ┌───────────────▼──────────────────────────────────────┐
            │  engine/   Game facade · state view · actions · map    │
            └───────────────┬──────────────────────────────────────┘
                            │ calls
            ┌───────────────▼──────────────────────────────────────┐
            │  rust/crates/agot-core   rules · encoder · search      │
            │  rust/crates/agot-py     PyO3 session + batch tiers    │
            └────────────────────────────────────────────────────────┘
                            numpy only on the Python side
```

## The engine

The rules live in `rust/crates/agot-core`; Python owns the layer above it.

Rust:

* `map.rs` loads the board from JSON (the files remain in `src/agot/data`) and
  validates adjacency symmetry, ports and home areas.
* `state.rs` owns the board *by value* (`State::clone` is a structural copy; the
  old copy-on-write machinery is gone), and `setup.rs` deals the opening.
* `game.rs`, `phases.rs` and `action_phase.rs` own phase progression.
  `Game::step` applies one player decision and **auto-advances** through every
  phase that needs no input, stopping only when a player must decide. Legal
  actions are non-empty only then.
* `combat.rs` resolves battles from two *explicit* unit pools (casualty removal
  must know whose units are whose), with support fed in as per-area
  declarations. `house_cards.rs` implements all 42 abilities; the pure part
  records deferred effects on a `BattleContext`, and the apply step commits
  them.
* `hidden.rs` resamples face-down Orders and secret bids for PIMC;
  `observation.rs` is the encoder; `search.rs` is PUCT search with a pluggable
  evaluator and PIMC aggregation.
* `agot-py` is a thin binding in two tiers: a **session tier** (`Game`,
  `legal_actions`, `step`, `observe`, `encode_state`, `encode_actions`,
  `determinize`) and a **batch tier** (`selfplay`) that runs whole games in Rust
  and calls the network once per leaf batch.

Python:

* `engine/game.py` is the facade: `GameConfig` and `Game = RustGame`.
* `engine/rust.py` presents the objects the rest of the package expects: Python
  `Action` dataclasses, a `state` whose regions/houses have attribute access,
  the static map, `clone`/`step`/`observe`. The view caches its snapshot per
  step and invalidates it on `step`, so reading a position is cheap and cannot
  go stale.
* `engine/observation.py` keeps the layout constants and forwards encoding to
  the extension (one call per leaf); `engine/hidden.py` forwards determinization.
* `engine/map.py` loads the static board data the facade and the agents read.

## Observation encoding

`engine/observation.py` keeps the layout and forwards to `observation.rs`:

* `encode_state(game) -> (97,)` — per-house blocks (castles, power, supply,
  track positions, unit counts, cards in hand) plus a global block (round,
  phase one-hot, action-step one-hot, wildling threat).
* `encode_action(game, action) -> (141,)` — a *relational* description of one
  candidate action: kind one-hot, actor, and, for the region and target,
  terrain, controller relation, unit counts and order state. March actions also
  carry which units move and whether control is established, and Support
  declarations carry which side they help. It deliberately
  does **not** one-hot region ids, so the encoding transfers to any board.

Houses always appear in a fixed canonical order (`HOUSE_ORDER`), with unused
slots zeroed for games with fewer than six players. The network's input size is
therefore constant regardless of table size.

## Policies

`agents/base.py` defines the interface every agent implements: `decide(game)`
returns the chosen action *and a target distribution* over the legal actions.
That single method is enough for both play and training-data generation.

* `RandomPolicy` — uniform, the correctness baseline.
* `GreedyPolicy` — a hand-written one-step heuristic; the yardstick a trained
  network should beat.
* `MCTS` — PUCT search with a pluggable `Evaluator`. The default
  `UniformEvaluator` needs no network, which is what lets the engine and search
  be tested without PyTorch.

### Why a value *vector*

Standard AlphaZero flips the sign of a scalar value each ply. That only works
for two players. Here each search node stores `value_sums[house]` for all six
houses, and a node selecting among its children looks up the entry for *the
house that is choosing*. This is correct for any number of players and needs no
sign bookkeeping. The network's value head mirrors this by emitting six numbers.

## The network

`nn/model.py` is a **scorer network**. A trunk embeds the position to `h`; a
second trunk embeds each candidate action to `a_i`; a small MLP reads the
concatenation `[h, a_i]` and emits the logit for that action. Logits for illegal
actions are masked to a large negative number. The value head maps `h` to six
tanh-bounded house values.

The scorer is deliberately an MLP rather than the dot product `h · a_i` it
started as. A dot product is a product of two learned tensors, so it has a zero
fixed point: once either factor reaches zero the gradient through the other is
zero too and the whole policy branch stops training permanently. That is what
happened to the first two runs — the policy emitted a uniform distribution from
the first iteration to the last. See the *Notable fixes* section of
`docs/roadmap.md`.

Batched inference (`predict_many`, used by the lockstep self-play driver) scores
many leaves in one pass. It concatenates every leaf's candidates into a single
flat tensor rather than padding them up to the widest action set in the batch —
on real batches that padding made the action and scorer trunks do 2.5× the rows
they needed — and normalises each leaf's own slice with `segment_softmax`, so a
leaf's distribution cannot depend on what else shares its batch. Each leaf's
input comes from one call to the Rust encoder; there is no Python-side memo to
go stale.

Training (in `training/trainer.py`) minimises

```
policy cross-entropy  +  value_weight · masked MSE over the six house values
```

where the value mask restricts the loss to houses actually in the game. The
optimiser is `AdamW`, with *decoupled* weight decay applied to weight matrices
only — biases and other one-dimensional parameters are never decayed. Decoupled
decay shrinks a weight geometrically instead of adding `weight_decay · parameter`
to its gradient, which is what let the old optimiser (plain `Adam` plus an L2
penalty) grind the policy parameters to zero.

## Data flow of one self-play game

1. `Game.new(config)` deals a fresh position.
2. For each decision: the seat's `Policy.decide` returns `(action, target)`.
3. `observe(game)` encodes the position and every candidate action; the example
   is stored with the actor and a placeholder outcome.
4. The action is applied and the engine advances.
5. When the game ends, `outcome_vector` labels every stored example with the
   final per-house reward.
6. Examples flow into the replay buffer and then into the trainer.

## Testing strategy

The rules, combat, hidden-information and encoder gates live in the Rust crate;
the Python suite covers the layer above the engine (facade, agents, training,
CLI). The retired Python copies of the rules/combat/encoder tests were ported
one-for-one first, so no audit entry lost its regression. The differential
fixture and encoder golden vectors are frozen pins: they were recorded from the
Python reference before it was deleted, and are never regenerated by hand.

Rust (`cd rust && cargo test --release`):

* `src/rules_fixes.rs` — one regression per audited rule fix: Action-Phase turn
  order, per-order decline, consolidate on open water, port capacity and
  capture, supply on marches, establishing control, Special Consolidate Power
  mustering, the Messenger Raven, the Valyrian Steel Blade, neutral forces,
  Wildling icon ordering, Wildling bidding and card effects, the end-of-game
  tiebreakers, House Stark's starting Supply, the three holders'-choice Westeros
  cards, and every House-card ability.
* `tests/combat_semantics.rs` — strength arithmetic, the Fiefdoms tie-break,
  `swords − forts` casualties, retreats, and the engine-level halves of
  cross-house support and player-chosen casualties.
* `src/support_semantics.rs` — the phase-machine halves: who is asked, what
  they may declare, neutral forces never receiving support, and the Support
  stage opening on a real march.
* `src/hidden.rs` and `tests/search_semantics.rs` — determinization (Order
  tokens, secret bids, pool bookkeeping, seeding) and PIMC search/decide.
* `src/observation.rs` and `tests/encoder_parity.rs` — encoder layout, hidden
  orders, slot initialisation, bounded actions, and bit-exact golden vectors.
* `tests/engine_parity.rs` — complete games replayed against the frozen
  reference fingerprints, field for field.

Python:

* `test_map.py` — board data invariants.
* `test_engine.py` — full games terminate, legal actions are never empty,
  clones are independent, round 1 skips Westeros.
* `test_agents.py` — every policy returns actions aligned with its targets.
* `test_training.py` — examples are labelled, replay round-trips, matches run.
* `test_batched_selfplay.py` — the batched/parallel self-play drivers.
* `test_nn.py` — network shapes, save/load, masking, one training step, and the
  policy-head health checks (candidates scored differently, every parameter
  receiving gradient, cross-entropy reaching a learnable target's entropy).
  Skipped automatically when torch is absent.
* `test_rust_engine.py` — the extension's acceptance gate: actions, stepping,
  snapshots, the position builder, the encoder wrapper, the batch self-play
  driver and the training-loop adapter.
