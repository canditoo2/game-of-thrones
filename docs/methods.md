# How to make the agent strong: the decision record

This document is the working plan for turning the engine into a player that can
win a six-player table game. It records **which methods were considered, why the
chosen path was chosen, and what is measured**. It is deliberately separate from
[`roadmap.md`](roadmap.md) (what the engine still lacks) and
[`rulebook-audit.md`](rulebook-audit.md) (rules fidelity).

## What "unbeatable" can and cannot mean

A Game of Thrones is a **six-player free-for-all with one winner**, hidden
simultaneous commitments, no dice, and explicit diplomacy. That rules out the
usual guarantees:

* There is no two-player zero-sum equilibrium to solve. Six-player general-sum
  Nash is intractable, so "perfect play" is not computable.
* The other five players can always collude against the leader. Strong play can
  *cause* a dogpile.
* There is no public human game database and no standard benchmark.

So the goals are stated as measurable win rates against explicit fields:

* **G1** — beat the existing heuristics (`GreedyPolicy`, random) over many games.
* **G2** — beat a league of past checkpoints (robust, not exploitable).
* **G3** — win a live six-player table against humans, including negotiation.

Everything below is scored with the gauntlet in
[`../src/agot/training/evaluate.py`](../src/agot/training/evaluate.py):
`evaluate_vs_field` puts the candidate in one rotating seat and a field in the
other five, and reports the single-seat win rate with a Wilson 95% interval.
The all-else-equal baseline is **1/6 ≈ 16.7%**.

## Measured starting point

Run with `agot gauntlet --candidate X --opponents Y --games 60`.

| Candidate | Field | Win rate | 95% CI |
| --- | --- | --- | --- |
| greedy | 5 × random | 38.3% | 27–51% |
| random | 5 × greedy | 3.3% | 1–11% |
| greedy | 5 × greedy | 23.3% | 14–35% |

The greedy heuristic already beats five random opponents comfortably, so it is a
real yardstick: **a trained agent must clear ~38% against random, and a
meaningfully superhuman agent should approach the ceiling against the greedy
field.**

These anchors predate the sequential Westeros model. The default now resolves
each card's mustering/supply step before the next reveal, and PIMC resamples the
cards still face down, so re-run the same gauntlet before comparing a new run's
numbers with this table.

### Why run3's best checkpoint was iteration 10

This is worth recording because it is not only evaluation noise. In `run3` the
30-game eval at iteration 10 was 66.7% (CI 48.8?80.8%) and at iteration 200 was
60.0% (CI 42.3?75.4%); those intervals overlap. The larger 200-game validation is
decisive:

| Checkpoint | vs mixed random+greedy | vs 5?random |
| --- | --- | --- |
| `agot-best.pt` (iteration 10) | 52.0% | 73.0% |
| `agot.pt` (iteration 200) | 44.0% | 49.0% |

A direct 200-game match of the final against the best ended **52?148** (final
26%, best 74%). So the later model really was weaker against fixed opponents,
not just unlucky in a 30-game sample.

The logs show the mechanism: policy loss fell from 1.82 at iteration 10 to 0.59
at 200, and value loss from 0.14 to 0.10. Lower loss only means the network fits
its **own** search targets better. There is no league or old-opponent sampling,
no KL anchor to the previous policy, a constant learning rate, and a replay
buffer that is swept in full and resets whenever a Colab session resumes. The
policy became sharper and more confident on a narrowing self-play distribution
while its strength against random/greedy fell. The 30-game eval also selects the
maximum over many noisy evaluations, which biases `agot-best` upward.

Run4 addresses this with a **relative checkpoint gate** instead of choosing the
best on the random/greedy anchor:

* every 10 iterations, 60 games of the current net (3 seats) against
  `agot-best.pt` (3 seats), swapping seats each game;
* a new checkpoint replaces the best only when the overall win rate clears
  `--gate-threshold` **and** every house is within tolerance of the previous
  best in that seat (`candidate + --gate-house-tolerance >= previous_best +
  --gate-house-margin`; `--gate-min-house` is an optional absolute floor);
* the tolerance (default `0.10`) exists because each house is played about 30
  times in the default 60-game gate, so its win rate carries a binomial
  standard error of roughly 9% (`sqrt(0.25 / 30)`); requiring all six houses to
  be non-negative stacks six independent ~1-sigma constraints and rejects
  clearly stronger networks. Run4 showed this at iteration 50: a candidate won
  70% of its 60 gate games overall and was still rejected by the per-house
  rule. `--gate-house-tolerance 0` restores the old strict comparison, and
  `--gate-house-margin` remains a *required* advantage applied on top of the
  tolerance;
* the first gate seeds the best, and three consecutive failures stop the run
  with the best preserved;
* the 12-game random/greedy anchor remains in the log for continuity, but it
  no longer chooses the best checkpoint.

### How the weights change between versions

Every iteration is one AlphaZero-style update, not an interpolation between two
checkpoints:

1. the current network plays 48 self-play games through MCTS (128 simulations,
   2 PIMC determinizations by default);
2. each decision is stored with the search's temperature-scaled visit
   distribution as the policy target and the final per-house outcome as the
   value target;
3. the examples go into the replay buffer (80k on Colab); one full shuffled pass
   over the buffer is trained with cross-entropy policy loss plus MSE value loss;
4. the optimizer is AdamW with learning rate `1e-3`, batch 64, gradient clipping
   `1.0`, one epoch, and decoupled weight decay `1e-4` on weight matrices only;
5. `agot.pt` is overwritten with the result every iteration; `agot.state.pt`
   carries the optimizer/iteration for resume; `agot-best.pt` is the
   gate-approved checkpoint, not a moving average.

There is no EMA/SWA, no learning-rate schedule, no KL penalty and no population
of past checkpoints in the current trainer. Those are the next levers if the
gate alone is not enough: a small old-opponent fraction in self-play, an EMA
evaluation copy, or a KL/behavior-cloning anchor to the previous best.

## Methods considered

Kept short; each entry is *what it is → pro → con → verdict*.

### Search and game theory

* **AlphaZero-style self-play (MCTS + scorer net).** The existing scaffold.
  General and needs no human data, but it assumes two-player perfect
  information; this game is neither. Rough pure-Python throughput made it
  impractical, which is what drove the Rust port (now the engine, search and
  encoder). **Chosen backbone, after the speed and hidden-information fixes.**
* **Perfect-information MCTS with root determinization (PIMC).** Sample a
  consistent hidden state, search it, average. Cheap to add and removes the
  "search sees face-down orders" leak, but suffers strategy fusion and needs
  several samples per move. **Chosen first hidden-information step.**
* **Information-set MCTS (SO-ISMCTS).** Correct handling of hidden
  information; more complex and slower, but the principled upgrade once PIMC
  proves insufficient. **Planned second step.**
* **Belief-state / opponent modelling.** Predict enemy orders, support and bids
  from observed behaviour. High ceiling for diplomacy, data-hungry. **Phase 3.**
* **CFR / Deep CFR / ReBeL.** Principled for hidden information, but equilibrium
  computation is intractable for six-player general-sum. **Rejected for the full
  game.**
* **Exact solving / endgame tables.** Perfect where it applies, but the full
  state space is astronomical. **Only ever a small endgame oracle.**
* **Exact combat search / shallow expectimax.** Combat is deterministic, so a
  short search is exact and cheap. **Adopt as a component.**

### Learning

* **Self-play deep RL, no search (PPO/IMPALA).** Higher throughput, weaker
  tactics. **Useful pre-trainer.**
* **Value-based RL (DQN-style).** Poor fit for a huge variable action space and
  multiplayer bootstrapping. **Rejected.**
* **League / population-based training.** Trains against a pool of past selves
  and exploiters; avoids self-play cycling and targets win rate directly.
  **Essential for G2.**
* **Evolutionary / neuroevolution.** Too sample-hungry for large nets.
  **Rejected.**
* **Imitation learning.** Needs data that does not exist publicly. **Not a
  primary path.**

### Knowledge and human tables

* **Hand-written expert system + opening book.** Cheap, interpretable, already
  beats casual play, and doubles as a yardstick and reward-shaping signal.
  **Keep permanently.**
* **Hybrid search + scripted tactics + learned evaluation.** Combines exact
  tactics with learned strategy. **Strong pragmatic route.**
* **Meta-game / kingmaking doctrine.** Explicit rules for coalition management,
  not looking like the leader, selling support. **Needed for G3.**
* **LLM diplomacy layer.** Negotiation over an engine choice. A real
  differentiator at a table, but unverifiable in self-play. **Phase 3, G3.**

### Enablers

* **Engine speed** (this was the binding constraint), **parallel/batched
  self-play**, **history/belief encoders**, **reward shaping and a fixed
  evaluation suite**, and finally a **playable table interface** for G3.

## Staged plan

1. **Evaluation harness.** Done: fixed-seed gauntlet with confidence intervals,
   seat rotation and the baseline numbers above.
2. **Engine speed.** *(done)* Hand-written `GameState.clone`, copy-on-write
   sharing of regions between a state and its clones, and single-pass
   observation encoding gave ~2–4× more simulations per second, and self-play is
   now batched (several games in lockstep share one network forward pass) and
   parallel (one game-running process per core, measured ~8× on 12 workers).
   Details in `roadmap.md`. The machine this was built on has no NVIDIA GPU, so
   the planned "batched GPU inference" is realised as batched *CPU* inference
   plus multi-process self-play.
3. **Hidden information.** Root determinization (PIMC) in MCTS: `engine/hidden.py`
   resamples face-down Order tokens, secret Power bids and the first chooser's
   House card while the second chooser is deciding, and the search aggregates
   over `determinizations` worlds. The physical game reveals both House cards
   simultaneously; the GUI/snapshot hides them until both commitments are made.
   Implemented and tested. SO-ISMCTS remains the principled upgrade if PIMC
   proves insufficient.
4. **Scaled AlphaZero.** *(in progress)* Batched + parallel self-play with PIMC,
   scripted-opponent mixing (`--opponent-rate`, a first step toward a league),
   and periodic gauntlet evaluation with best-checkpoint selection are done. A
   self-play run is stable over many iterations. Runs are now resumable: each
   checkpoint gets a companion `*.state.pt` (next iteration, best eval, optimizer
   state) and a `*.log.jsonl` history, and `--resume` restores them, so a long run
   survives a crash or restart. Saved checkpoints are now first-class policies
   (`net:<path>` candidates) and `agot round-robin` compares them in a league.
   What remains is scale (games, width) plus the history/belief encoder, reward
   shaping and a checkpoint registry.
5. **Table play (G3).** Meta/kingmaking rules and, for live negotiation, an LLM
   diplomacy layer over a strong engine, plus an interface to play one seat.

## Runbook: the next credible run (run4)

Heavy runs happen on the training machine; the build machine only prepares and
verifies. The commands below are the whole procedure. On Google Colab,
`colab/agot-colab.ipynb` automates them (private-repo token in Colab secrets,
checkpoints on Drive, resume across disconnects).

0. **Smoke first.** Five iterations on a small budget:

   ```bash
   agot train --iterations 5 --games 12 --workers 4 --batch-games 3 \
     --eval-games 12 --eval-every 2 --checkpoint checkpoints/smoke/agot.pt
   ```

   Gate: policy loss must **move and trend down**; the absolute value depends on
   the number of legal candidates, and the generic choice actions raised it
   above the old ~2.0 signature. Kill the run once and restart it with
   `--resume` to prove recovery, and render
   `agot report --checkpoint checkpoints/smoke/agot.pt`.

1. **Calibrate the field.** The periodic gauntlet fills five seats round-robin
   from random and greedy, so measure that mixed field for the trend line:

   ```bash
   agot gauntlet --candidate greedy --opponents random,greedy --games 60
   agot gauntlet --candidate random --opponents random,greedy --games 60
   ```

2. **Launch run4.**

   ```bash
   agot train --iterations 120 --games 48 --workers 12 --batch-games 4 \
     --opponent-rate 0.2 --train-threads 4 \
     --temperature-rounds 3 --final-temperature 0.25 \
     --eval-games 12 --eval-every 10 \
     --gate-games 60 --gate-threshold 0.5 --gate-house-margin 0.0 --gate-patience 3 \
     --checkpoint checkpoints/run4/agot.pt
   ```

   `agot-best.pt` is replaced only by a checkpoint that wins the 3v3 gate and
   stays within per-house tolerance of the previous best in every house. It is
   the checkpoint to use even when the last iteration is weaker.

   Watch it in another terminal:

   ```bash
   agot report --checkpoint checkpoints/run4/agot.pt --iterations 120 --watch 15 --open
   ```

   Interrupt it at any time; continue with the same command plus
   `--resume checkpoints/run4/agot.pt`.

3. **Decision gates.** By iteration ~20 the policy loss must be trending down,
   not flat. By iteration ~50-70 the gauntlet win rate must sit above the greedy
   mixed-field baseline with a CI that excludes it. If both hold, extend the
   run; if not, stop and diagnose before spending more hours.

4. **Verify the result.** Measure the best checkpoint directly:

   ```bash
   agot gauntlet --candidate net:checkpoints/run4/agot-best.pt --opponents random --games 200
   ```

   G1 is met when this clears the greedy baseline (38.3% against five random
   opponents) with a CI that excludes it. For the league view:

   ```bash
   agot round-robin --checkpoints checkpoints/run4/agot.pt checkpoints/run4/agot-best.pt --games 30
   ```
