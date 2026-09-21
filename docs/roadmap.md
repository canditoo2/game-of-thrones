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
* Supply recalculation and enforcement.
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

4. **House-card abilities.** Only printed strength, swords and forts are used.
   Every card's text effect (e.g. extra units, cancelled swords, special wins)
   is ignored.

5. **Wildlings.** The track and the "Wildlings Attack" trigger exist, but an
   attack resolves as a threshold check with no bidding, no consequences and no
   per-card effects.

6. **Tides of Battle / optional modules.** Not modelled (base game only).

## Priority 2 — engine completeness

7. **Neutral forces.** Regions with neutral force tokens are not defended in a
   fight; they are simply occupied.
8. **Siege Engines.** Modelled with combat strength 0; their printed role
   (ignoring fortifications in some contexts) is not implemented, so the agent
   undervalues them.
9. **Order restrictions with real teeth.** Cards such as *Web of Lies* or
   *Sea of Storms* are drawn and logged but do not actually forbid placing the
   restricted order type.
10. **"Winter is Coming" and "Throne of Blades".** Effects unimplemented.
11. **Recycling and deck exhaustion.** The physical decks are finite across ten
    rounds; reshuffling behaviour should be verified.
12. **Ports and naval movement edge cases**, e.g. retreating fleets, mustering
    ships in ports, and what happens when a port's host is captured.

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

17. **Check the flagged board data.** The per-region icons and the six-player
    influence tracks are the highest-impact unverified values; see
    [`../PROVENANCE.md`](../PROVENANCE.md).
18. **Westeros deck composition** is a reconstruction and should be corrected
    from the rulebook.

## Reproducibility and tooling

19. **CI** already runs the numpy-only test suite on two Python versions and the
    RL suite against a CPU torch wheel. The RL job is allowed to fail so a torch
    packaging problem never blocks the engine.
20. **Checkpoint registry and run logs** for training (currently a single
    `--checkpoint` path and printed losses).
