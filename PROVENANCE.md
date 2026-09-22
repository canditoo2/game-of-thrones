# Data provenance

Every rule constant and every piece of board data in `src/agot/data/` is either
**verified** against a source or explicitly **flagged as uncertain**. This file
records which is which, so that nobody trains on bad data (or trusts a balance
conclusion drawn from it) without knowing the risk.

The guiding policy for this project is: *implement something honest and mark
what is uncertain, rather than invent a plausible-sounding rule.*

For the audit trail behind these flags — what was checked, corrected or left
open, and the deterministic defaults used for multi-choice effects — see
[`docs/rulebook-audit.md`](docs/rulebook-audit.md).

## Confidence summary

| Data | Confidence | How it was established |
| --- | --- | --- |
| Board adjacency (which regions border which) | **High** | Audited against the bundled `board.jpg` and cross-checked against two independent community implementations |
| Unit combat strengths (footman 1, knight 2, ship 1) | **High** | Stated in the official 2E rulebook |
| Casualty rule (swords on the *winner's* card minus forts on the *loser's*) | **High** | Official 2E rulebook |
| Tie-break (higher Fiefdoms position wins) | **High** | Official 2E rulebook |
| Supply / army limits | **High** | Cross-verified identically in two independent references |
| Base-game setup (5 power tokens) | **High** | Standard base-game setup |
| Starting Supply per house | **High** | `houses.json` gives House Stark 1 and every other house 2, confirmed by the user against the physical player screen. Stark is the only house to open with both a Stronghold and a Castle |
| Round structure (round 1 has no Westeros phase; rounds 2-10 draw one card from each deck) | **High** | Official 2E rulebook |
| Per-region power / supply / mustering icons | **High** | Audited against `board.jpg`; one error (The Mountains of the Moon is a Supply area) was corrected |
| The 15 Westeros card *names* and their per-deck deal | **High** | Confirmed against the user-supplied `Rulebook/Westeros_card.txt`; each deck holds ten cards |
| Which Westeros cards carry a Wildling icon | **High** | Confirmed against the same file and the physical component list |
| Holders'-choice Westeros cards (A Throne of Blades, Dark Wings Dark Words, Put to the Sword) | **High** | Text confirmed against `Rulebook/Westeros_card.txt`; modelled as holder decisions |
| The 42 House card names and printed stats | **High** | Transcribed from the user-supplied `Rulebook/151060549-House-Cards-Reference-Sheet.txt` and cross-checked against community data sets |
| House card ability texts | **High** | Transcribed from the same reference sheet; player sub-choices use a documented deterministic default |
| House Martell's House cards | **High** | Included in the reference sheet; the earlier file was missing Darkstar and had wrong strengths |
| House Martell's starting units | **High** | Confirmed by the user against the physical board (Sunspear 1 Footman + 1 Knight, Salt Shore 1 Footman, 1 Ship in the Sea of Dorne) |
| Six-player influence track order | **High** | Corrected by the user against the physical board |
| Starred-order allowance by King's Court position `(3, 3, 2, 1, 0, 0)` | **High** | Matches chibimagic's `KingsCourtTrack::FIVE_SIX_PLAYER_SPECIAL_ORDERS`; also printed on the board |
| Wildling card names (9) | **High** | Match the physical component list |
| Wildling card effects | **High** | Transcribed from the bundled scans (`Rulebook/wildling-cards1..3.webp`); player sub-choices use a documented deterministic default |
| Wildling attack procedure (bid, pooled strength, track reset) | **High** | Official 2E rulebook pages 22-23 |
| Neutral forces (King's Landing 5, The Eyrie 6) | **High** | Official 2E rulebook page 26 and rulebook page 4 setup diagram |

A machine-readable copy of these flags lives inside the JSON files themselves
(under `verified`, `*_confidence` and `notes` keys). Those keys are the source of
truth for tooling; the table above is a human summary.

## What to verify first

Nothing is currently flagged. The starting positions, the six-player Influence
tracks, the King's Court starred-order allowance, House Stark's starting Supply,
the House card data and the Westeros deck deal have all been confirmed.

If you spot a discrepancy in play, re-check the relevant file under
`src/agot/data/` and update its `verified` flag and `notes` as described below.

## Sources

* **Official rulebook** for A Game of Thrones: The Board Game (Second Edition) —
  combat, casualties, tie-breaks, supply, round structure, setup, Wildling
  attacks (pp. 22-23) and Neutral Forces (p. 26).
* **Bundled rulebook scans** (`Rulebook/board.jpg`, `Rulebook/wildling-cards*.webp`,
  `Rulebook/Westeros_card.txt`, `Rulebook/151060549-House-Cards-Reference-Sheet.txt`,
  `Rulebook/book_image/*`) — the board-icon audit, the Wildling card texts, the
  Westeros deck deal and all 42 House card stats/abilities were read from these.
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
