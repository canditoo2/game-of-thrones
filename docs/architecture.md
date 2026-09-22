# Architecture

The project is three loosely-coupled layers. Nothing above the engine may leak
into the layer below it, which keeps the rules code testable and torch-free.

```
            ┌──────────────────────────────────────────────┐
            │  training/   self-play · replay · trainer     │
            └───────────────┬──────────────────────────────┘
                            │ uses
            ┌───────────────▼──────────────┐   ┌───────────────────────┐
            │  agents/   random · greedy   │   │  nn/  pointer net     │
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
* `combat.py` resolves battles from two *explicit* unit pools. Keeping the
  attacker's and defender's units separate is essential: casualty removal has to
  know whose units are whose.
* `house_cards.py` applies each House card's special ability around a battle.
  `resolve_battle` is kept free of board mutation: it *records* deferred effects
  (Order removals, Influence moves, card swaps) on a `BattleContext`, and
  `resolve_and_apply` commits them and runs the post-combat abilities.

`Game.clone()` is the backbone of search. It deep-copies the state and gives the
clone a fresh RNG seeded from the parent, so a cloned game is independent. A
clone does not re-run the phase machine: the source position is already settled
and advancing would move the Action Phase turn-order cursor.

## Observation encoding

`engine/observation.py` is the bridge to the network. It defines:

* `encode_state(game) -> (97,)` — per-house blocks (castles, power, supply,
  track positions, unit counts, cards in hand) plus a global block (round,
  phase one-hot, action-step one-hot, wildling threat).
* `encode_action(game, action) -> (136,)` — a *relational* description of one
  candidate action: kind one-hot, actor, and, for the region and target,
  terrain, controller relation, unit counts and order state. March actions also
  carry which units move and whether control is established. It deliberately
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

`nn/model.py` is a **pointer network**. A trunk embeds the position to `h`; a
second trunk embeds each candidate action to `a_i`; the policy logit for action
`i` is `(a_i · W h) / sqrt(d)`. Logits for illegal actions are masked to a large
negative number. The value head maps `h` to six tanh-bounded house values.

Training (in `training/trainer.py`) minimises

```
policy cross-entropy  +  value_weight · masked MSE over the six house values
```

where the value mask restricts the loss to houses actually in the game.

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

* `test_map.py` — board data invariants.
* `test_engine.py` — full games terminate, legal actions are never empty,
  clones are independent, round 1 skips Westeros.
* `test_combat.py` — strength arithmetic, the tie-break, `swords − forts`
  casualties, and a regression test that the defender's units are never
  credited to the attacker.
* `test_rules_fixes.py` — one regression test per audited rule fix: Action-Phase
  turn order, per-order decline, consolidate on open water, port capacity and
  capture, supply on marches, establishing control, Special Consolidate Power
  mustering, the Messenger Raven, the Valyrian Steel Blade, neutral forces,
  Wildling icon ordering, Wildling bidding and card effects, the end-of-game
  tiebreakers, House Stark's starting Supply, the three holders'-choice Westeros
  cards, and a representative ability for every House-card mechanic.
* `test_observation.py` — fixed dimensions, determinism, one-hot blocks.
* `test_agents.py` — every policy returns actions aligned with its targets.
* `test_training.py` — examples are labelled, replay round-trips, matches run.
* `test_nn.py` — network shapes, save/load, masking, and one training step.
  Skipped automatically when torch is absent.
