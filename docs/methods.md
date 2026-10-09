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

## Methods considered

Kept short; each entry is *what it is → pro → con → verdict*.

### Search and game theory

* **AlphaZero-style self-play (MCTS + scorer net).** The existing scaffold.
  General and needs no human data, but it assumes two-player perfect
  information; this game is neither. Rough pure-Python throughput makes it
  impractical without an engine speed-up. **Chosen backbone, after the speed and
  hidden-information fixes.**
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
   resamples face-down Order tokens and secret Power bids, and the search
   aggregates over `determinizations` worlds. Implemented and tested; the one
   remaining leak is the first chooser's House card inside a battle (chosen and
   revealed simultaneously in the physical game). SO-ISMCTS remains the
   principled upgrade if PIMC proves insufficient.
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

## Runbook: the first credible run (run3)

Heavy runs happen on the training machine; the build machine only prepares and
verifies. The commands below are the whole procedure. On Google Colab,
`colab/agot-colab.ipynb` automates them (private-repo token in Colab secrets,
checkpoints on Drive, resume across disconnects).

0. **Smoke first.** Five iterations on a small budget:

   ```bash
   agot train --iterations 5 --games 12 --workers 4 --batch-games 3 \
     --eval-games 12 --eval-every 2 --checkpoint checkpoints/smoke/agot.pt
   ```

   Gate: policy loss must fall **below ~2.0** (the log of the candidate count)
   and keep falling. A flat ~2.0 for every iteration is the old broken-head
   signature, not convergence. Kill the run once and restart it with `--resume`
   to prove recovery, and render
   `agot report --checkpoint checkpoints/smoke/agot.pt`.

1. **Calibrate the field.** The periodic gauntlet fills five seats round-robin
   from random and greedy, so measure that mixed field for the trend line:

   ```bash
   agot gauntlet --candidate greedy --opponents random,greedy --games 60
   agot gauntlet --candidate random --opponents random,greedy --games 60
   ```

2. **Launch run3.**

   ```bash
   agot train --iterations 120 --games 48 --workers 12 --batch-games 4 \
     --opponent-rate 0.2 --train-threads 4 \
     --temperature-rounds 3 --final-temperature 0.25 \
     --eval-games 30 --eval-every 10 \
     --checkpoint checkpoints/run3/agot.pt
   ```

   Watch it in another terminal:

   ```bash
   agot report --checkpoint checkpoints/run3/agot.pt --iterations 120 --watch 15 --open
   ```

   Interrupt it at any time; continue with the same command plus
   `--resume checkpoints/run3/agot.pt`.

3. **Decision gates.** By iteration ~20 the policy loss must be trending down,
   not flat. By iteration ~50-70 the gauntlet win rate must sit above the greedy
   mixed-field baseline with a CI that excludes it. If both hold, extend the
   run; if not, stop and diagnose before spending more hours.

4. **Verify the result.** Measure the best checkpoint directly:

   ```bash
   agot gauntlet --candidate net:checkpoints/run3/agot-best.pt --opponents random --games 200
   ```

   G1 is met when this clears the greedy baseline (38.3% against five random
   opponents) with a CI that excludes it. For the league view:

   ```bash
   agot round-robin --checkpoints checkpoints/run3/agot.pt checkpoints/run3/agot-best.pt --games 30
   ```
