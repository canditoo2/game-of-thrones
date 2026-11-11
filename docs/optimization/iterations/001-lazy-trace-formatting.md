# 001 — stop formatting trace lines that nobody reads

- Track: HYGIENE (removes measured work)
- Status: kept
- Commit: see the branch head (`rust/crates/agot-core`)

## Hypothesis

`State::note` is called at 86 sites, 73 of them as `state.note(format!(...))`. The `format!`
runs at the call site, so every step builds a `String` (allocation + formatting) whether or
not `state.log` is `Some`. Search, self-play and every CLI path except the desktop trace run
with `log == None` — the string is allocated and dropped.

`state.rs:1000` kept the guard *inside* `note`, which cannot skip work the caller already did.

## Prediction (written before the change)

- Metric: `cargo bench --bench clone_step`, `full game` line added for this iteration, and
  `clone + step`.
- Expected: 10–25 % on full games (the planning phase alone has few notes, so the planning
  line should barely move); `clone + step` −5…−20 %.
- Falsified if: full-game throughput moves less than 3 %, or the trace text changes at all.

## Change

- `state.rs`: `note(String)` becomes a thin wrapper over a new
  `note_args(std::fmt::Arguments)`, which formats only when a log exists.
- 73 call sites: `note(format!(...))` → `note_args(format_args!(...))` (identical syntax, no
  allocation until the message is actually stored).
- `benches/clone_step.rs`: new `full game (log off/log on)` lines. The old bench only walked
  the planning phase, which is exactly the phase with the fewest trace calls; there was no
  harness for the metric this change targets.

Behaviour is unchanged when a log exists: `note_args` formats with the same prefix
(`R{round} {phase}: `) and the same message.

## Gates

`cargo fmt --all -- --check` clean · `cargo clippy --all-targets -- -D warnings` clean ·
`cargo test --release` 292 passed.

## Measurement

Interleaved A/B in one session (stash the `src` change, rebuild, bench, restore, rebuild):
two runs per side, full-game line, `200 games` each.

| metric | before | after | delta |
|---|---|---|---|
| full game, trace off | 531 k / 533 k decisions/s | 722 k / 723 k | **1.36x** |
| full game, trace on | 430 k / 455 k decisions/s | 582 k / 583 k | 1.30x |
| clone + step | 648 / 630 ns | 527 / 514 ns | **1.22x** |
| planning phase | 1.98–2.02 M/s | 1.75–2.30 M/s | no measurable effect |
| State::clone | 384 ns | 373 ns | noise |

The trace-on case improved too: the old path formatted the message, then `note` formatted it
again into the log line — two allocations per note.

## Verdict

Actual (+36 % full game, +22 % on clone+step) is above the predicted range and well outside
the ~5 % run-to-run spread. Kept. The planning-phase line is the control: it did not move,
which is what the hypothesis predicted for a phase with almost no trace calls.
