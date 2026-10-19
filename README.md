# Game of Thrones — a learning agent for the board game

An AlphaZero-style reinforcement-learning project for
**A Game of Thrones: The Board Game (Second Edition)**.

The goal is a neural network that learns to play the full **six-player** game by
playing against itself, without any human game records. This repository is the
foundation for that: a fast, test-covered rules engine, a Monte-Carlo tree
search, a policy/value network, and the self-play training loop that ties them
together.

> **Status: pre-alpha.** The engine plays complete six-player games. The network
> and training loop run, but no meaningful playing strength has been trained yet:
> the first two runs were void — the policy head could not learn — and the fixed
> pipeline has not had its first long run. Known rule simplifications are listed
> below and tracked in [`docs/roadmap.md`](docs/roadmap.md).

## Why does this need a custom engine?

A Game of Thrones is an unusually awkward fit for textbook AlphaZero:

* **Six players, not two.** There is no single win/lose value to back up. The
  network predicts one value *per house*, and the search stores a value vector
  at each node.
* **Open-ended action space.** A planning turn can offer dozens of
  region/order pairs; a bid offers a range of integers. There is no fixed move
  list, so the policy is a **scorer network**: it embeds each candidate action
  and feeds `[position, action]` through a small MLP that emits one logit per
  candidate. (It began as a dot product `position · action`; that collapsed to
  zero and never learned — see the *Notable fixes* in `docs/roadmap.md`.)
* **Hidden information.** Order tokens are placed face down and Power bids are
  secret. The search does not peek: it resamples the information it cannot see
  (PIMC, see below) and averages over those worlds before choosing.

## Layout

```
rust/crates/agot-core/   rules engine, observation encoder and PUCT search
rust/crates/agot-py/     the abi3 Python extension (agot._core)
src/agot/
  engine/        Python facade: Game/GameConfig, the state view, actions and map data
  agents/        policies: random baseline, greedy heuristic, PUCT MCTS
  nn/            the policy/value network (needs PyTorch)
  training/      self-play, replay buffer, trainer, evaluation
  cli.py         command-line entry point
tests/           engine facade, map, agents, training, network and the extension
docs/            architecture, methods, roadmap and the rulebook fidelity audit
```

The engine and search are Rust; the Python layer needs only `numpy` (PyTorch is
the optional `.[rl]` extra). The build produces one abi3 extension, so a wheel
built once works on Python 3.10 and later.

## Install

Building from source needs a Rust toolchain (1.75 or newer); installing a
published `agot` wheel does not.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"         # builds the Rust extension + tests + linter
pip install -e ".[rl]"          # optional: adds PyTorch
```

## Quickstart

```bash
# Play one full game with the greedy baseline.
agot demo --policy greedy --seed 1

# Head-to-head over several games (random vs greedy).
agot evaluate --a random --b greedy --games 20

# Strength check: candidate's single-seat win rate vs a field (1/6 = parity).
agot gauntlet --candidate greedy --opponents random --games 60

# Evaluate a saved checkpoint: any policy argument accepts net:<path>, which
# searches with MCTS over that network at the given --simulations budget.
agot gauntlet --candidate net:checkpoints/run3/agot-best.pt --opponents random --games 200

# League table: each checkpoint plays one rotating seat against the other
# checkpoints (plus random/greedy anchors), with a Wilson 95% interval.
agot round-robin --checkpoints checkpoints/run3/agot.pt checkpoints/run3/agot-best.pt --games 30

# See the legal actions at the start of a game.
agot legal

# Self-play training loop (requires PyTorch). The defaults are the ones that
# matter: 128 simulations over 2 determinizations (--determinizations > 1 is what
# keeps the search and its targets honest about face-down information), and a
# temperature that holds for the opening rounds then cools, because the recorded
# target is built from it. --workers spreads games over CPU cores; --eval-games
# turns on the periodic gauntlet and best net (its budget follows --simulations
# unless --eval-simulations says otherwise). --train-threads caps torch in the
# training process (small mini-batches are faster on a few threads than on every
# core; the workers stay single-threaded).
agot train --iterations 100 --games 48 --workers 12 --batch-games 4 \
  --opponent-rate 0.2 --train-threads 4 \
  --temperature-rounds 3 --final-temperature 0.25 \
  --eval-games 30 --eval-every 10 --checkpoint checkpoints/agot.pt

# Continue a run after a crash or restart: reloads the weights, iteration
# counter, best-eval and optimizer state from the checkpoint's companion file.
agot train --iterations 100 --resume checkpoints/agot.pt \
  --checkpoint checkpoints/agot.pt

# Watch a run: renders the companion *.log.jsonl as a self-contained HTML page
# (loss curves, seconds per iteration, eval trend vs parity/greedy, ETA). No
# extra dependencies. The page is a snapshot; --watch regenerates it so an open
# tab keeps up (Ctrl-C stops it).
agot report --checkpoint checkpoints/agot.pt --iterations 100 --watch 15 --open
```

From Python:

```python
from agot.engine.game import Game, GameConfig
from agot.agents import GreedyPolicy

game = Game.new(GameConfig(seed=0))
policies = {house: GreedyPolicy(seed=i) for i, house in enumerate(game.config.houses)}
while not game.is_terminal():
    house = game.current_player()
    action, _ = policies[house].decide(game)
    game.step(action)
print(game.summary())
```

Run the Python layer with `pytest -q`, the engine, search and encoder gates with
`cargo test --release` from `rust/`, and the linter with
`ruff check src tests`.

## Run it on Google Colab

[`colab/agot-colab.ipynb`](colab/agot-colab.ipynb) is a ready-to-run notebook:
it clones this repository, mounts Drive, builds (or reuses) the abi3 wheel and
then launches a resumable training run whose checkpoints are written to Google
Drive, so a disconnected runtime continues from the last iteration. The wheel
is cached per commit, so a reconnected session installs it in seconds instead
of rebuilding Rust; the first session for a new commit takes a couple of
minutes. A CPU runtime is fine — the search is CPU-bound, so a GPU adds little.
Open it in Colab (upload the file, or *File → Open notebook → GitHub*) and run
the cells top to bottom.

The repository is private, so add a fine-grained read-only token as a Colab
secret named `GITHUB_TOKEN` (🔑 panel in the left sidebar) before running the
clone cell; make the repository public to skip that step.

## How the learning works

1. **Self-play.** Each seat is played by MCTS guided by the current network.
   At every decision the search's visit counts become a policy target, and the
   position is stored with the final per-house outcome. Hidden information is
   respected: when `determinizations > 1`, MCTS resamples the face-down Order
   tokens and secret Power bids it cannot see (PIMC) and averages over those
   worlds before producing the target. Games are run in lockstep batches and
   across CPU processes so the network is fed large forward-pass batches.
2. **Replay.** Examples go into a bounded buffer
   ([`training/replay.py`](src/agot/training/replay.py)).
3. **Training.** The network is trained to match the search's policy
   (cross-entropy) and predict the outcome for every house (mean squared error).
4. **Repeat.** The improved network guides the next round of self-play.

Because the game is multiplayer, value targets are *vectors* over the six
houses, not a single scalar. That is what lets the same machinery handle 2-6
players.

## Documented simplifications

These are real deviations from the physical game. They are tracked in
[`docs/roadmap.md`](docs/roadmap.md) rather than hidden:

* **Hidden information is sampled, not solved.** Order tokens are placed face
  down and Power bids are secret; the engine tracks reveal timing and the
  observation masks what a house may not see. The search uses PIMC
  (`--determinizations`): it resamples the face-down Orders and secret bids from
  the observer's belief and averages the root visits over those worlds. The one
  remaining leak is the first chooser's House card inside a battle, which the
  physical game reveals simultaneously.
* **Support is a real choice across houses.** A Support order may back either
  side of an adjacent battle, and its owner decides in Iron Throne order.
* **Casualty and retreat choices are player decisions.** The defeated player
  picks their own casualties and, when defending, the retreat area. Only the
  non-interactive `combat.resolve_and_apply` path falls back to cheapest-first.
* **Wildling attacks** model the secret Power bid, the Night's Watch strength
  and all nine cards' win/loss consequences. Sub-choices inside a consequence
  (which units are destroyed, which track is moved, which House card is
  discarded) use a deterministic default, and bidder ties are broken by Iron
  Throne position rather than asking the holder.
* **House-card special abilities are implemented**, but the sub-choices inside
  an ability (which Footman to upgrade, which Order to remove, which alternate
  card to play, which retreat area to pick) are auto-resolved with
  deterministic defaults; see
  [`house_cards.rs`](rust/crates/agot-core/src/house_cards.rs).
* **Siege Engines** contribute their printed strength 4 only when attacking (or
  supporting an attack against) an area containing a Castle or Stronghold.
* **Neutral forces** (King's Landing 5, The Eyrie 6) defend with their printed
  strength; ties go to the attacker and they never use a House card or the
  Valyrian Steel Blade.
* **Multi-destination marches are not offered.** A March Order moves to a
  single destination; the real game allows splitting across several areas (at
  most one of which is a battle).

The board data is also partly unverified; see [`PROVENANCE.md`](PROVENANCE.md)
for exactly which values are trusted and which need checking against the
physical board.

## License

MIT. This is an unofficial fan project and is not affiliated with or endorsed by
Fantasy Flight Games or George R. R. Martin.
