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
* Cross-house support: combat opens with a Support step, each adjacent Support
  order is offered to its owner in Iron Throne order, and owners may back either
  side or refuse. A player may never support an opponent against their own
  units, and Support orders are not consumed by a battle.
* All 42 House cards' special abilities (see `engine/house_cards.py`); the
  player sub-choices inside an ability are auto-resolved with documented
  deterministic defaults.
* Supply recalculation and enforcement, including on marches.
* Action-Phase turn order cycles one order per house; declining an individual
  order, choosing which units march, establishing control with a Power token,
  Special Consolidate Power mustering, the Messenger Raven and the Valyrian
  Steel Blade are all modelled as explicit player decisions.
* The defeated player chooses their own casualties and (when defending) the
  retreat destination; Robb Stark lets the winner pick the retreat instead.
* Westeros holders' choice cards (A Throne of Blades, Dark Wings Dark Words,
  Put to the Sword) are modelled as real decisions for the named Influence
  token's holder, and *Winter is Coming* reshuffles its deck and redraws.
* Ports: three-Ship capacity, capture of a port with its host area, the "Game of
  Thrones" port Power bonus, and routed Ships transporting land units.
* End-of-game tiebreakers follow the FAQ order (areas, Strongholds, Supply,
  Power, Iron Throne).
* Observation encoding, PUCT MCTS, policy/value network, replay buffer,
  trainer, evaluation, CLI.
* A test suite that runs full games and pins the combat rules.

**Not started / approximate** — see below.

**Notable fixes**

* *Uninitialised observation features.* `_region_block` filled an `np.empty`
  array but left several slots unwritten (the controller block, the hidden-order
  one-hot, and the order-strength slot). Those slots occasionally held garbage
  floats (~1e24) from freed memory, which made the network's logits and the
  training loss explode at random. `_region_block` now writes every slot, and
  regression tests pin the empty case and bound every encoded action.

* *A policy head that could not learn anything.* The first two runs produced no
  policy at all: the network emitted a uniform distribution from the first
  iteration to the last, and the swings in the evaluation numbers (0.125-0.333
  against a 1/6 parity line) were measurement noise, not progress. The cause was
  the head. It scored an action with the dot product `h · a_i`, a product of two
  learned tensors whose zero is a fixed point: if either factor reaches zero the
  gradient through the other vanishes, so the branch is stuck there for good. It
  did reach zero — `action_trunk` and `policy_proj` sat at the float32 floor
  (~1e-37) while `state_trunk` and `value_head` stayed healthy. The pull came
  from the optimiser, which was `Adam` with an L2 penalty: a penalty adds
  `weight_decay · parameter` to the gradient, and adaptive scaling turns that
  into a roughly constant step toward zero, enough to erase a parameter whose
  task gradient is weak. The task gradient *was* weak, partly because the search
  targets were themselves near-uniform (32 simulations split over four
  determinizations left about one visit per candidate). The head is now a scorer
  MLP over `[h, a_i]`, so the position embedding reaches the head
  unconditionally and the branch cannot switch itself off, and the optimiser is
  `AdamW` with decoupled decay on weight matrices only. `tests/test_nn.py` pins
  the three properties that were violated: a fresh head scores candidates
  differently, every parameter receives a non-zero gradient, and the head drives
  cross-entropy down to the entropy of a learnable target. The old checkpoints
  are refused on load with an explanatory error rather than loading silently.

* *A copy-on-write region that could be written through a clone.* `GameState`
  shares region objects with its clones and copies a region only when the engine
  writes to it: `mut(region_id)` skips the copy once the id is in the state's
  private set. The bookkeeping was only half right — cloning reset the *clone's*
  set but left the source's, so once the source had taken a region private, the
  next clone shared that exact object and the source's *next* write landed in the
  clone. Measured on real games: stepping a game changed 1,317 of 1,751 clones
  taken before the step, badly enough in one case to leave a clone holding a
  region with no legal action in it. Tree search happened to be safe because it
  never steps a node after cloning it, but PIMC steps the true game while worlds
  built from it are still live, and any future code that holds a clone across a
  step would have corrupted silently. `clone` now clears ownership on *both*
  sides, so each re-copies on its next write; this is free in the hot path, where
  parents are never stepped. The old suite only checked one direction (stepping a
  clone must not touch the source); it now checks the other one too, and a direct
  test pins the mechanism rather than just the symptom.

* *A cloned battle resolved against a ghost board.* A resolved battle parks its
  `BattleResult` on `pending_battle`, and the result carries a `BattleContext`
  whose `state` field is not just a reference: `house_cards.apply_post_battle`
  writes Power (Tywin), unit upgrades (Renly), Order removals (Cersei) and card
  returns (Roose, Patchface) *through it*. On the live game that field is the
  game's own state, so those abilities worked. `GameState.clone` copied the
  pending battle with a generic `deepcopy`, which turned the back-reference into
  a detached copy of the state: a clone that then finalised the battle applied
  the post-combat abilities to a board nobody would ever see and read neighbour
  regions off stale data. Direct play (the game object stepped in place) stayed
  correct, so MCTS and the recorded games disagreed about the rules — the worst
  version of this bug, because the search would happily march into a fight whose
  consequences it had mis-modelled. `clone` now deep-copies the pending dicts
  with a memo seeded as `{id(self): new}`, so the back-reference is rebound to
  the clone while the state itself is not copied at all. The isolation
  fingerprint was extended to expand dataclasses and to record whether a pending
  value points at its own state, and a test that finds a real parked battle pins
  the invariant directly.

## Priority 1 — correctness that changes strategy

These are simplifications that materially change what good play looks like, so
they should be fixed before expecting a strong agent.

1. **Face-down orders.** *(done)* Order tokens are placed secretly and
   revealed simultaneously. The engine tracks reveal timing via
   `GameState.orders_revealed`: orders are secret during Planning and the
   Messenger Raven step and public once the Action Phase begins, and the
   observation masks the *identity* of another house's hidden order while still
   showing that a face-down token is present. The search no longer sees hidden
   information either: `engine/hidden.py:determinize` resamples face-down orders
   and secret Power bids from the observer's belief, respecting public Order
   restrictions and the starred-token allowance, and `MCTS(determinizations=N)`
   runs PIMC over N worlds and aggregates the root visits. The one known
   remaining leak is the first chooser's House card inside a battle, which the
   physical game reveals simultaneously.

2. **Cross-house support.** *(done)* A Support order may help *either* side in
   an adjacent battle, and the supporting player chooses. Combat opens with a
   Support step: each adjacent Support order is offered to its owner in Iron
   Throne order, and the owner declares `attacker`, `defender` or `refuse`.
   A player may never support an opponent against their own units; a third
   house may back either side. The Support order is not consumed and may help
   several battles in a round. Attacking a Neutral Force opens the same step,
   but the token itself can never be supported (rulebook page 26). The
   pre-march legality check for a Neutral Force uses the maximum possible
   support, so an attack that only becomes legal once allies declare is still
   offered and is cancelled if the support is refused. This is where most of
   the game's diplomacy lives.

3. **Player-chosen casualties and retreats.** *(done)* After the strength
   comparison the Combat phase pauses: the defeated player destroys one unit at a
   time of their own choosing, then (if defending) picks the legal adjacent area
   to retreat to. A defeated attacker always returns to where it marched from.
   Robb Stark's controller picks the defender's retreat instead, restricted to
   areas that keep the loser within Supply. The non-interactive
   `combat.resolve_and_apply` path still removes cheapest-first for callers that
   do not run the phase machine.

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

13. **Training must produce a non-trivial policy.** The pipeline runs, but the
    first two runs are **void**: their policy head had collapsed to a zero fixed
    point and never learned anything (see *Notable fixes*). The architecture and
    the optimiser are now fixed, with regression tests, so the next long run is
    the first whose numbers can be read as evidence. First credible target: a net
    that reliably beats `GreedyPolicy` over a few hundred games.

    The search side has now been changed so that the next run has something to
    learn from. Training defaults are **128 simulations across 2
    determinizations** instead of 32 across 4 — about 64 visits per world instead
    of 8, and still PIMC because the count stays above one — and the eval gauntlet
    defaults to the training budget instead of a cheaper 16, so it measures the
    policy that is actually being trained. Self-play temperature is no longer a
    flat 1.0: `--temperature` holds for `--temperature-rounds` (3) and then cools
    to `--final-temperature` (0.25), and the recorded target is built from the
    sharpened distribution, because a flat 1.0 with a handful of visits per
    candidate trains on something close to noise. Top-K pruning of candidates by
    prior was considered and left out: the worlds of one decision are searched
    separately, so they would prune to different action sets, and aggregating the
    roots then silently drops whichever action another world kept. It needs one
    shared top-K computed before the worlds diverge.
14. **Search efficiency.** *(done)* `GameState.clone` no longer uses
    `copy.deepcopy` (a hand-written structural copy, ~8× faster), the state
    encoding is single-pass, and static region/card features are cached. MCTS
    went from ~850 to ~3,500 simulations/s and a full random game from 0.51 s to
    0.25 s. Self-play is now both *batched* (`batched_self_play` runs several
    games in lockstep and scores all their leaves in one forward pass, and can
    use PIMC at the same time) and *parallel* (`parallel_self_play` runs games in
    one process per core; measured ~8× on 12 workers). An incremental state
    representation is the remaining idea.

    A second pass removed repeated full-board scans. Victory checking now groups
    the fortifications once instead of calling `castle_count` (a full scan) per
    house on every `_advance` tick — that was the large majority of all
    `controlled_regions` calls (216k → 14k over the profile run). Action-Phase
    step handling builds one house→orders index per step, march and muster Supply
    checks reuse a single per-decision profile instead of rescanning per
    candidate move, `_planning_actions` filters starred/restricted orders once per
    house rather than once per (region, order) pair, and MCTS resolves the
    acting house's value index once per selection instead of per child.
    Deterministic call counts fell about 40% (a random game: 7.2M → 4.3M calls;
    the MCTS search profile 496k → 388k) with no rule changes.

    A third pass attacked the two costs that dominate *neural* self-play (a
    cProfile of four games went from 99 s to 51 s). The network evaluator
    encodes one action per candidate and was allocating a dozen arrays per call:
    `encode_action_into` now fills a preallocated, already-zeroed row in place
    (shared by `predict_actions`, `predict_many` and `observe`), one-hot blocks
    are written with index maps instead of tuple scans, region blocks are
    memoised per leaf, and the "no region" case reuses one read-only block. The
    state dataclasses use `slots=True` for cheaper attribute access and clone
    construction. Measured: 48 games across 12 workers fell from 67.0 s to ~45 s
    (12 workers already saturates the machine — its 22 logical threads are 11
    physical cores, and 16 or 20 workers measured no faster), and training one
    full 120k-buffer pass from 26.7 s to 15.8 s after pinning
    the training process to four torch threads (`--train-threads`; small
    mini-batches are faster on a few threads than on every core). Training also
    reuses one worker pool across iterations (`create_executor`; the CLI keeps it
    for the whole run), so it no longer pays process spawn plus `torch` import
    every iteration — a tiny two-worker run showed the second iteration at 0.6 s
    against 3.6 s for the cold first one. A full iteration (48 self-play games,
    training, and a 24-game gauntlet) now runs in ~75-80 s.

    A fourth pass removed the remaining per-candidate allocations. The region and
    target lookup is inlined into `encode_action_into`, and an absent region is
    left as the row's existing zeros instead of copying an all-zero block; MCTS
    nodes allocate their per-house reward buffer only on the first visit (a fresh
    expansion creates one node per candidate action and most are never visited);
    and batched inference runs under `torch.inference_mode()`. Nested profile
    calls fell another ~9% per example and `numpy.zeros` dropped out of the hot
    list entirely. Search results are arithmetically identical — only allocation
    and dispatch were removed. `Game.clone`'s float-seeded RNG
    (`random.Random(self.rng.random())`) is also ~1% but changing its seeding
    would alter RNG streams, so it is left alone.

    A fifth pass added copy-on-write for regions, which was the last large clone
    cost. A `GameState` now keeps its regions in a per-state dict that *shares*
    the `RegionState` objects with the state it was cloned from, and the engine
    takes a private copy only the first time it writes one. `GameState.mut` is
    the single write path, used throughout `game.py`, `combat.py`,
    `house_cards.py` and `hidden.py`; fresh states (and `_initial_state`) keep
    writing directly because nothing else can see their regions yet. Measured: a
    step changes on average 1.16 of 58 regions, so a clone now copies ~2 regions
    instead of all 58 — `GameState.clone` fell from 56 µs to 5.9 µs (~10×) and
    `Game.clone` from 70 µs to 12 µs. The invariant that makes it safe (cloning a
    game and stepping the clone never mutates the source) is pinned by
    `tests/test_clone_isolation.py`, which clones and steps *every* legal action
    at every decision of several seeded games; a wider sweep of 213k such checks
    found no violation, and the full test suite is unchanged. A missed `mut` call
    on a rarely-taken path was the residual risk, which is why the isolation test
    exists — and it turned out to be the wrong direction to watch. See the
    *Notable fixes* entry on the ownership bookkeeping and the source side of a
    clone: the audit of every `RegionState` write site in `engine/` found no
    missed `mut`, so the bookkeeping, not the call sites, was at fault.

    A sixth pass removed the waste in the batched forward and the duplicate work
    in the encoder. `predict_many` padded every leaf's action set up to the widest
    one in the batch, and on real self-play batches that padded **2.47×** the rows
    the action trunk and the scorer had to see (12× on the worst batch). It now
    concatenates the candidates of every leaf into one `(sum N_i, action_dim)`
    tensor, runs the trunks and the scorer over that once, and normalises each
    leaf's slice with a segmented softmax (`forward_packed` and
    `segment_softmax`), so a leaf costs exactly its own candidates and no more.
    Measured 1.39× on a batch that padded 1.67×, and the real distribution pads
    more than that. End to end (4 games, 32 simulations, 4 worlds, one thread),
    the paired runs went 30.2 s → 21.5 s and 37.9 s → 33.0 s on the repeat; the
    two repeats of one configuration differ by ~25% on this machine, so the
    micro-benchmark is the trustworthy number. The region-block memo used to be
    rebuilt for every leaf in a batch; it is now shared across the whole batch,
    keyed on region *object identity* rather than region id — two leaves can hold
    different objects under the same id, because a determinization swaps in a copy
    for the regions it resamples — which drops the block computations from ~36 per
    call to ~16. Both paths are pinned against their per-leaf equivalents by
    tests, and a dead `RegionState.clear_units` (the only writer that could have
    bypassed `mut`) was removed. The pending-effect dicts are no longer copied
    with `copy.deepcopy` either: containers are rebuilt structurally and only an
    opaque value falls through to `deepcopy` — a resolved battle parks a
    `BattleResult` whose context back-references the state, and sharing that
    corrupted the board immediately. That investigation turned up a real
    correctness bug in the old copier (see *Notable fixes*), so the fallback now
    deep-copies with a memo that rebinds the context to the clone, which both
    fixes the bug and skips copying the state. The extended isolation fingerprint
    caught the first, wrong version of the copier within a minute, which is why
    the fingerprint now expands dataclasses and covers all four pending dicts
    instead of only the board. Profiling the same self-play workload before and
    after the pass (4 games, 32 simulations, 4 worlds, 4 torch threads) puts the
    profiled time at 67.1 s → 31.8 s and `torch._C._nn.linear` at 13.9 s → 3.2 s,
    with `copy.deepcopy` gone from the profile entirely and `_region_block` calls
    down from 184k to 76k.

15. **Self-play diversity.** Temperature schedules and Dirichlet noise
    (implemented) are in place; `--opponent-rate` now mixes scripted greedy seats
    into every game, which is the first piece of a league. Full league/opponent
    sampling should be tuned once training produces a non-trivial policy.
16. **Evaluation protocol.** *(partly done)* `agot gauntlet` measures a
    candidate's single-seat win rate against a field over fixed seeds, with a
    Wilson 95% interval and a rotating seat (`evaluate_vs_field`). Baseline:
    greedy wins 38.3% against five random opponents. The train loop now runs that
    same gauntlet periodically (`--eval-games`, `--eval-every`) and saves the
    best-evaluated net (`--best-checkpoint`). Saved checkpoints are first-class
    candidates (`net:<path>` in `gauntlet` and `evaluate`) and `agot round-robin`
    builds a league table across checkpoints. Still to do: a curated seed suite.

## Data verification

17. **Check the flagged board data.** *(done)* The six-player influence tracks,
    the starred-order allowance and House Martell's starting units are all
    confirmed. See [`../PROVENANCE.md`](../PROVENANCE.md).
18. **Westeros deck composition.** *(done)* Confirmed against
    `Rulebook/Westeros_card.txt`; each of the three decks holds exactly ten
    cards.

## Reproducibility and tooling

19. **CI** already runs the numpy-only test suite on two Python versions and the
    RL suite against a CPU torch wheel. The RL job is allowed to fail so a torch
    packaging problem never blocks the engine.
20. **Checkpoint registry and run logs.** *(partly done)* A run now writes a
    companion `*.state.pt` next to the checkpoint (next iteration, best eval and
    optimizer state) and a `*.log.jsonl` line per iteration, and `--resume`
    restores all of it, so a long run survives a crash or restart. The log can be
    watched without installing anything: `agot report --checkpoint <ckpt>` renders
    it to a self-contained HTML page (`src/agot/training/report.py`) with the loss
    curves, seconds per iteration, and the eval trend drawn against the parity and
    greedy baselines, plus an ETA against a planned iteration count. The page is a
    snapshot of the log, so `--watch N` re-renders it every `N` seconds and an open
    browser tab (which reloads itself) then follows a live run; the module is pure
    standard library, so it works on a torch-free machine. Still to do: a registry
    of named runs (round-robin evaluation now takes explicit checkpoint paths).

## Rust engine port

20. **Move the engine, search and encoder to Rust.** *(in progress)* The engine
    is the binding constraint on training throughput: profiling neural self-play
    on this machine puts ~61% of wall time in the Python engine, search and
    observation encoder, and only ~39% in the torch matmuls. The port targets
    that 61% with a Rust core (`rust/crates/agot-core`) driven from Python through
    a thin extension module, while the training loop keeps owning the network.

    Design: the board is owned by value (a `State` clone is a memcpy), which
    deletes the copy-on-write machinery, `_owned` bookkeeping and the
    deepcopy-with-memo fallback that the Python engine needs — the alias bugs
    those exist to prevent cannot be expressed. Pending effects are enums rather
    than `dict[str, Any]`, and the per-node log is off during search.

    Validation: `tools/dump_fixture.py` records action scripts and full state
    fingerprints from the Python engine, and the parity tests replay them against
    Rust, comparing the acting house, the canonical legal-action list at every
    decision and the whole position at the end. Mismatches name the scenario, the
    decision index and the first differing field.

    Measured (`cargo bench --bench clone_step`, release, this machine): `State`
    clone 364 ns against Python's 5 900 ns (16×), `Game` clone 432 ns against
    12 000 ns (28×), planning-phase legal actions 364 ns for 45 candidates, and
    clone-plus-step — the MCTS inner loop — 539 ns against ~14 000 ns (26×).

    Verified: the whole rules engine, the observation encoder and MCTS are ported
    and green. `tests/engine_parity.rs` replays five complete games (905
    decisions) recorded from the Python engine and compares the acting house, the
    canonical legal-action list and the full position at every step, with zero
    mismatches; `tests/encoder_parity.rs` compares 1 914 state and action vectors
    bit for bit; `tests/combat_semantics.rs` and `tests/search_semantics.rs` pin
    the battle rules and the search's semantic gates.

    The Python side is wired up too: `rust/crates/agot-py` builds an abi3 wheel
    (one artefact for the local venv, CI and Colab) exposing a session tier
    (`Game`, `legal_actions`, `step`, `observe`) and a batch tier
    (`selfplay(spec, evaluator)`), and `agot.training.rust_engine` adapts the
    batch tier to the training loop. `batched_self_play` takes the Rust driver
    whenever the request is one it can serve and falls back to the reference
    driver otherwise, recording which ran in `selfplay.LAST_BACKEND`.
    `tests/test_rust_engine.py` is the acceptance gate: the two engines play
    identical games move for move, the encoder agrees to the last bit through the
    binding, and the driver's packed output unpacks into the `Example` objects the
    trainer already consumes.

    Two reference-engine defects were found and fixed by the port, both
    reproducibility bugs rather than rule bugs: `GameMap.neighbours_of` returned a
    `frozenset`, so the *order* of the legal actions a Raid offers (and of the
    greedy baseline's tie-breaks) depended on the interpreter's string-hash seed
    and changed from process to process; and the corpus is recorded with the
    Westeros decks off, because the two engines shuffle differently and nothing
    may be drawn if the games are to stay comparable move for move.

    The rule-fidelity audit moved with the engine: every regression in
    `tests/test_rules_fixes.py` now has a Rust counterpart in
    `crates/agot-core/src/rules_fixes.rs` (53 tests, named after the Python
    originals so `docs/rulebook-audit.md` still tracks across the port). Porting
    them found two Rust-side defects the parity corpus had never exercised: the
    "Preemptive Raid" second Wildling attack was applied to an attack struct
    that resolution had already taken out of the state, so the card silently did
    nothing; and a holders'-choice card's `"none"` option was dropped while the
    option list was parsed, leaving the holder unable to decline. Both are
    fixed, and `none` is carried as an out-of-range card id so the observation
    encoder still sets its "a card field is present" marker exactly as Python
    does.

    The read views over the Rust position were extended while the regressions
    were ported: the state exposes the pieces the rest of the package reads
    (`orders_revealed`, `fortifications`, `action_step`, the pending dicts, the
    Wildling attack, `max_rounds`/`victory_castles`, a settable `winner` and the
    log), `RustGame` keeps the `GameConfig` it was built with, and the session
    tier grew `pending_battle`, `wildling_attack`, `log` and `max_rounds` /
    `victory_castles` getters. With the ported regressions landed,
    `AGOT_ENGINE=rust` plays a full greedy demo to a winner; the remaining
    failures are the tests that build positions by writing into
    `state.regions[...]` (they need the same builder the extension already has,
    wired into the view) and the PIMC path, which still resamples hidden state in
    Python (`hidden.py` writes through `state.mut`).

    Remaining: finishing the write tier over the Rust state so the Python tests
    that mutate a position can run against the extension, then the maturin
    packaging switch, changing the default engine, and finally deleting the
    Python engine.
