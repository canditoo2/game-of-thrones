# Game of Thrones — a learning agent for the board game

An AlphaZero-style reinforcement-learning project for
**A Game of Thrones: The Board Game (Second Edition)**.

The goal is a neural network that learns to play the full **six-player** game by
playing against itself, without any human game records. This repository is the
foundation for that: a fast, test-covered rules engine, a Monte-Carlo tree
search, a policy/value network, and the self-play training loop that ties them
together.

> **Status: pre-alpha.** The engine plays complete six-player games; the network
> and training loop run, but no meaningful playing strength has been trained yet.
> Known rule simplifications are listed below and tracked in
> [`docs/roadmap.md`](docs/roadmap.md).

## Why does this need a custom engine?

A Game of Thrones is an unusually awkward fit for textbook AlphaZero:

* **Six players, not two.** There is no single win/lose value to back up. The
  network predicts one value *per house*, and the search stores a value vector
  at each node.
* **Open-ended action space.** A planning turn can offer dozens of
  region/order pairs; a bid offers a range of integers. There is no fixed move
  list, so the policy is a **pointer network**: it scores each candidate action
  by embedding it and taking a dot product with the position embedding.
* **Hidden information.** Order tokens are placed face down. The current engine
  models them face up, which is a deliberate simplification (see below).

## Layout

```
src/agot/
  engine/        pure-Python, numpy-only rules engine
    map.py         board loading and structural validation
    state.py       mutable game state (unit pools, tracks, supply)
    actions.py     the action record and its kinds
    combat.py      battle resolution and retreat
    house_cards.py House card ability effects
    game.py        setup, legal actions, phase progression, victory
    observation.py state and per-action feature encoding
  agents/        policies: random baseline, greedy heuristic, PUCT MCTS
  nn/            the policy/value network (needs PyTorch)
  training/      self-play, replay buffer, trainer, evaluation
  cli.py         command-line entry point
tests/           map, engine, combat, observation, agents, training, network
docs/            architecture, roadmap and the rulebook fidelity audit
```

The **rules engine depends only on `numpy`**. PyTorch is an optional extra
(`.[rl]`), so the engine can be simulated and tested on a machine without it.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"         # engine + tests + linter
pip install -e ".[rl]"          # optional: adds PyTorch
```

## Quickstart

```bash
# Play one full game with the greedy baseline.
agot demo --policy greedy --seed 1

# Head-to-head over several games (random vs greedy).
agot evaluate --a random --b greedy --games 20

# See the legal actions at the start of a game.
agot legal

# Self-play training loop (requires PyTorch).
agot train --games 50 --simulations 32 --epochs 2 --checkpoint checkpoints/agot.pt
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

Run the tests with `pytest -q` and the linter with `ruff check src tests`.

## How the learning works

1. **Self-play.** Each seat is played by MCTS guided by the current network.
   At every decision the search's visit counts become a policy target, and the
   position is stored with the final per-house outcome.
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

* **Order tokens are face up.** The real game hides them until the reveal, so
  the engine currently has perfect information.
* **Support is automatic and self-directed.** A house only supports its own
  battles; the real game lets any house support either side.
* **Casualty and retreat choices are automatic** (cheapest unit first, first
  legal destination).
* **Wildling attacks** model the secret Power bid, the Night's Watch strength
  and all nine cards' win/loss consequences. Sub-choices inside a consequence
  (which units are destroyed, which track is moved, which House card is
  discarded) use a deterministic default, and bidder ties are broken by Iron
  Throne position rather than asking the holder.
* **House-card special abilities are implemented**, but the sub-choices inside
  an ability (which Footman to upgrade, which Order to remove, which alternate
  card to play, which retreat area to pick) are auto-resolved with
  deterministic defaults; see
  [`engine/house_cards.py`](src/agot/engine/house_cards.py).
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
