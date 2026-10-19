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
            │  engine/   map · state · actions · combat · game       │
            │            house_cards · observation (features)        │
            └────────────────────────────────────────────────────────┘
                            numpy only
```

## The engine

`engine/` is a plain, deterministic simulator with no ML dependency:

* `map.py` loads the board from JSON and runs structural validation on every
  import of the data (symmetry of adjacency, ports next to their host and sea,
  unique home areas, ...).
* `state.py` is a dataclass tree (`GameState` → `HouseState`, `RegionState`)
  designed to be deep-copied cheaply, because MCTS clones it constantly.
* `game.py` owns phase progression. `Game.step(action)` applies one player
  decision and then **auto-advances** through every phase that needs no input,
  stopping only when a player must decide (or the game ends). That keeps the
  agent loop simple: ask for legal actions, pick one, repeat.
* `GameState.orders_revealed` tracks hidden information: Order tokens are face
  down during Planning and the Raven step and revealed when the Action Phase
  begins. `observation.py` masks the identity of another house's hidden order
  (its presence stays visible). MCTS still searches the true state; a PIMC root
  determinisation is the follow-up.
* `combat.py` resolves battles from two *explicit* unit pools. Keeping the
  attacker's and defender's units separate is essential: casualty removal has to
  know whose units are whose. Support is fed in as explicit per-area
  declarations (`support_from_choices`); `resolve_battle` also accepts the old
  self-directed derivation when no declarations are supplied.
* `house_cards.py` applies each House card's special ability around a battle.
  `resolve_battle` is kept free of board mutation: it *records* deferred effects
  (Order removals, Influence moves, card swaps) on a `BattleContext`, and
  `resolve_and_apply` commits them and runs the post-combat abilities.

`Game.clone()` is the backbone of search. It shallow-copies the state's region
dict and gives the clone a fresh RNG seeded from the parent, so a cloned game is
independent. Regions are shared between a state and the states cloned from it
until someone writes one: the engine fetches a region to mutate with
`GameState.mut(region_id)`, which takes a private copy on the first write (see
the copy-on-write note in `state.py`). Cloning also drops the *source's* record of
which regions it had taken private, because those objects are shared again from
that moment — both sides re-copy on their next write. A step writes about six of
the 58 regions, so this costs a fraction of copying the board per node. Two tests
pin the invariant, one per direction: stepping a clone must not touch the source,
and stepping the source must not touch a clone already taken. The pending-effect
dicts (`wildling_attack`, `westeros_choice`, `pending_battle`, `pending_march`)
are rebuilt structurally rather than with `copy.deepcopy`, and the copy is handed
a memo that rebinds any back-reference to the state: a resolved battle's
`BattleResult` carries a context whose `state` is read *and written* by the
post-combat House-card abilities, so it has to point at the clone. A
clone does not re-run the phase machine: the source position is already settled
and advancing would move the Action Phase turn-order cursor.

## Observation encoding

`engine/observation.py` is the bridge to the network. It defines:

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
leaf's distribution cannot depend on what else shares its batch. Encoding shares
one region-block memo across the leaves, keyed on region object identity, since
the leaves are usually the same position seen through several hidden worlds.

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

The rules, combat, hidden-information and encoder gates have moved into the Rust
crate; the Python suite covers the layer above the engine (agents, training,
CLI). The retired Python copies of the rules/combat/encoder tests were ported
one-for-one first, so no audit entry lost its regression.

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
* `tests/engine_parity.rs` — complete games replayed against the Python
  reference fingerprint for fingerprint.

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
* `test_rust_engine.py` — the extension's acceptance gate (both engines play
  identical games, the encoder agrees bit for bit through the binding).
