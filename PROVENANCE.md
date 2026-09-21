# Data provenance

Every rule constant and every piece of board data in `src/agot/data/` is either
**verified** against a source or explicitly **flagged as uncertain**. This file
records which is which, so that nobody trains on bad data (or trusts a balance
conclusion drawn from it) without knowing the risk.

The guiding policy for this project is: *implement something honest and mark
what is uncertain, rather than invent a plausible-sounding rule.*

## Confidence summary

| Data | Confidence | How it was established |
| --- | --- | --- |
| Board adjacency (which regions border which) | **High** | Cross-checked against two independent community implementations |
| Unit combat strengths (footman 1, knight 2, ship 1) | **High** | Stated in the official 2E rulebook |
| Casualty rule (swords on the *winner's* card minus forts on the *loser's*) | **High** | Official 2E rulebook |
| Tie-break (higher Fiefdoms position wins) | **High** | Official 2E rulebook |
| Supply / army limits | **High** | Cross-verified identically in two independent references |
| Base-game setup (supply 2, 5 power tokens) | **High** | Standard base-game setup |
| Round structure (round 1 has no Westeros phase; rounds 2-10 draw one card from each deck) | **High** | Official 2E rulebook |
| Per-region power / supply / mustering icons | **Medium** | Single reference; **must be checked against the physical board** |
| The 15 Westeros card *names* | **High** | Match the physical component list |
| Which Westeros card sits in which of the three decks | **Low / unverified** | Best-effort reconstruction |
| The 42 House card names for the five original houses | **High** | Community data sets |
| House Martell's House cards | **Low / unverified** | One reference omits Martell entirely |
| House Martell's starting units | **Low / unverified** | Inferred by pattern |
| Six-player influence track order | **Low / unverified** | Reference does not fully model Martell; Martell placed last by inference |
| Starred-order allowance by King's Court position `(3, 3, 2, 1, 0, 0)` | **Medium** | Encodes the known "top two get three/four stars" behaviour; exact table unverified |
| Wildling card names (9) | **High** | Match the physical component list |
| Wildling card effects | **Not implemented** | Resolution is a stub |

A machine-readable copy of these flags lives inside the JSON files themselves
(under `verified`, `*_confidence` and `notes` keys). Those keys are the source of
truth for tooling; the table above is a human summary.

## What to verify first

If you have the physical board and rulebook to hand, checking these in order
would do the most to improve the fidelity of the engine:

1. **Per-region icons** in `map_base_6p.json` (`power`, `supply`, `mustering`).
   These directly drive consolidation income, supply recalculation and
   mustering, so errors change the balance of the whole game.
2. **The six-player influence tracks** in `houses.json`
   (`initial_tracks.iron_throne`, `.fiefdoms`, `.kings_court`). Turn order,
   tie-breaks and starred orders all depend on them.
3. **The Westeros deck composition** in `westeros.json` (`composition`). The
   card list is right; the deal into decks I, II and III is a reconstruction.
4. **House Martell** cards and starting units in `house_cards.json` and
   `houses.json`.

## Sources

* **Official rulebook** for A Game of Thrones: The Board Game (Second Edition) —
  combat, casualties, tie-breaks, supply, round structure and setup.
* `chibimagic/got-board-game` — board adjacency (cross-reference 1).
* `Robin59/Got-BoardGame` — region attributes, starting units, supply limits,
  House card names (cross-reference 2).
* `buhanec/pygot` — unit statistics cross-check.

Local copies of the rulebook scans are kept out of version control (see
`.gitignore`) because they are copyrighted; only the facts extracted from them
are stored here.

## How to correct the data

The data is deliberately external to the code. To fix a value:

1. Edit the relevant file under `src/agot/data/`.
2. Flip its `verified` flag (or update the `notes`) to record that it is now
   checked.
3. Run `pytest -q`; the map validator and data tests will catch structural
   mistakes immediately.

No engine code needs to change for a data correction.
