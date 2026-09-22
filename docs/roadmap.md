# Roadmap

This document is the honest list of what the engine does *not* yet do, ordered by
how much it matters for learning a strong agent. It doubles as the project plan.

## Current state

**Done**

* Full board, six houses, starting units, 20 fortifications.
* Round loop: Westeros (rounds 2-10), bidding, mustering, planning, the three
  action steps (raid, march, consolidate power), combat, cleanup, victory.
* Combat resolution with March/Defend modifiers, support, House cards, the
  Valyrian Steel Blade, `swords − forts` casualties and Fiefdom tie-breaks.
* All 42 House cards' special abilities (see `engine/house_cards.py`); the
  player sub-choices inside an ability are auto-resolved with documented
  deterministic defaults.
* Supply recalculation and enforcement, including on marches.
* Action-Phase turn order cycles one order per house; declining an individual
  order, choosing which units march, establishing control with a Power token,
  Special Consolidate Power mustering, the Messenger Raven and the Valyrian
  Steel Blade are all modelled as explicit player decisions.
* Westeros holders' choice cards (A Throne of Blades, Dark Wings Dark Words,
  Put to the Sword) are modelled as real decisions for the named Influence
  token's holder, and *Winter is Coming* reshuffles its deck and redraws.
* Ports: three-Ship capacity, capture of a port with its host area, the "Game of
  Thrones" port Power bonus, and routed Ships transporting land units.
* End-of-game tiebreakers follow the FAQ order (areas, Strongholds, Supply,
  Power, Iron Throne).
* Observation encoding, PUCT MCTS, pointer policy/value network, replay buffer,
  trainer, evaluation, CLI.
* A test suite that runs full games and pins the combat rules.

**Not started / approximate** — see below.

## Priority 1 — correctness that changes strategy

These are simplifications that materially change what good play looks like, so
they should be fixed before expecting a strong agent.

1. **Face-down orders.** Order tokens are placed secretly and revealed
   simultaneously. The engine stores them face up. Options: give the network a
   belief over hidden orders, or (simpler) search with a determinised sample of
   the hidden orders. Until then, search is optimistic.

2. **Cross-house support.** A Support order may help *either* side in an
   adjacent battle, and the supporting player chooses. Currently support is
   automatic and only ever helps its owner. This is where most of the game's
   diplomacy lives.

3. **Player-chosen casualties and retreats.** Currently automatic
   (cheapest-first, first legal region). The full game makes losing a battle an
   active decision (which unit to lose, where to retreat) and can be decisive.

4. **House-card abilities.** *(done)* All 42 cards' text effects are modelled
   in `engine/house_cards.py`: passive strength/icon modifiers, immediate
   destruction/Order/track effects, and post-combat effects. Where a card lets
   its owner choose (which Footman, which Order, which alternate card, which
   retreat), the engine picks a deterministic default; the exact defaults are
   documented in the module docstring.

5. **Wildlings.** *(done)* The track, the per-round icon advance (all icons
   counted before card effects resolve), the secret Power bid and all nine
   Wildling cards' consequences are modelled. Sub-choices *inside* a card's
   consequence (which units are destroyed, which Influence track is moved to,
   which House card is discarded) use a deterministic default, and a tie for
   highest/lowest bidder is broken by Iron Throne position rather than asking
   the holder.

6. **Holders' choice Westeros cards.** *(done)* The three cards whose text
   names an Influence token (A Throne of Blades, Dark Wings Dark Words, Put to
   the Sword) pause play so the token's holder can pick one of their listed
   options. *Winter is Coming* reshuffles its deck and redraws.

7. **Tides of Battle / optional modules.** Not modelled (base game only).

## Priority 2 — engine completeness

8. **Neutral forces.** *(done)* King's Landing (strength 5) and The Eyrie
   (strength 6) are placed at setup and defend with their printed strength.
   Ties go to the attacker, and neither House cards nor the Valyrian Steel
   Blade may be used against them (rulebook page 26). The 3-player
   "insurmountable" tokens are out of scope for the six-player base game.
9. **Siege Engines.** *(done)* Modelled at their printed strength 4 when
   attacking an area with a Castle or Stronghold.
10. **Order restrictions.** *(done)* Cards such as *Web of Lies* and *Sea of
    Storms* actually forbid placing the restricted order type.
11. **Recycling and deck exhaustion.** The physical decks are finite across ten
    rounds; reshuffling behaviour should be verified.
12. **Multi-destination marches.** A March Order may move units to several
    destinations (at most one of them a battle); the engine offers one
    destination per order. Ports, port capture and ship mustering are done.

## Priority 3 — learning quality

13. **No trained weights yet.** The pipeline runs; no meaningful strength has
    been produced. First credible target: a net that reliably beats
    `GreedyPolicy` over a few hundred games.
14. **Search efficiency.** MCTS clones `Game` per node and is slow in pure
    Python. Options: an incremental state representation, batched network
    evaluation, or a faster engine core. This is currently the binding
    constraint on training throughput.
15. **Self-play diversity.** Temperature schedules, Dirichlet noise
    (implemented) and league/opponent sampling should be tuned once training
    actually produces a non-trivial policy.
16. **Evaluation protocol.** A fixed suite of seeds and a round-robin across
    checkpoints, so improvements are measured rather than assumed.

## Data verification

17. **Check the flagged board data.** The six-player influence tracks are the
    highest-impact unverified value; see [`../PROVENANCE.md`](../PROVENANCE.md).
18. **Westeros deck composition.** *(done)* Confirmed against
    `Rulebook/Westeros_card.txt`; each of the three decks holds exactly ten
    cards.

## Reproducibility and tooling

19. **CI** already runs the numpy-only test suite on two Python versions and the
    RL suite against a CPU torch wheel. The RL job is allowed to fail so a torch
    packaging problem never blocks the engine.
20. **Checkpoint registry and run logs** for training (currently a single
    `--checkpoint` path and printed losses).
