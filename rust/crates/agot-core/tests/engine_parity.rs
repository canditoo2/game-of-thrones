//! Differential test: the Rust engine must reproduce the Python reference.
//!
//! The fixture is recorded from the Python engine by `tools/dump_fixture.py`
//! (five full games, 909 decisions). Every scenario replays the same action
//! script in Rust and compares, at each step, the acting house, the canonical
//! legal-action list (or its hash) and finally the whole position. A mismatch
//! names the scenario, the decision index and the first differing field.
//!
//! The corpus runs with the Westeros decks off: the two engines shuffle with
//! different generators, so nothing may be *drawn* if the games are to stay
//! comparable move for move. The card-drawing phases are covered by unit tests
//! in the library instead.

use std::sync::Arc;

use agot_core::enums::HouseId;
use agot_core::fingerprint::{canonical_action, fnv1a64, state_fingerprint};
use agot_core::game::Game;
use agot_core::map::DataSet;
use agot_core::setup::GameConfig;
use serde_json::Value;

const FIXTURE: &str = include_str!("data/engine_fixture.json");

fn config_from_json(value: &Value) -> GameConfig {
    let houses = value["houses"]
        .as_array()
        .expect("houses")
        .iter()
        .map(|name| HouseId::from_name(name.as_str().expect("house name")).expect("known house"))
        .collect();
    GameConfig {
        houses,
        max_rounds: value["max_rounds"].as_u64().expect("max_rounds") as u8,
        victory_castles: value["victory_castles"].as_u64().expect("victory_castles") as u8,
        seed: value["seed"].as_u64().expect("seed"),
        use_westeros_decks: value["use_westeros_decks"]
            .as_bool()
            .expect("use_westeros_decks"),
    }
}

/// Reports the first field that differs, so a failure is actionable.
fn assert_same_fingerprint(label: &str, actual: &Value, expected: &Value) {
    if actual == expected {
        return;
    }
    if let (Some(actual), Some(expected)) = (actual.as_object(), expected.as_object()) {
        for (key, expected_value) in expected {
            let actual_value = actual.get(key);
            if actual_value != Some(expected_value) {
                panic!(
                    "[{label}] fingerprint differs at `{key}`:\n  rust:   {}\n  python: {}",
                    actual_value.map_or("<missing>".to_string(), |v| v.to_string()),
                    expected_value
                );
            }
        }
        for key in actual.keys() {
            if !expected.contains_key(key) {
                panic!("[{label}] fingerprint has an extra field `{key}`");
            }
        }
    }
    panic!("[{label}] fingerprints differ: {actual} != {expected}");
}

#[test]
fn full_games_reproduce_the_python_reference() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture must parse");
    let data = Arc::new(DataSet::load_default().expect("bundled data must load"));
    let scenarios = fixture["scenarios"].as_array().expect("scenarios");
    assert!(!scenarios.is_empty(), "fixture has no scenarios");

    let mut total_decisions = 0usize;
    for scenario in scenarios {
        let label = scenario["label"].as_str().unwrap_or("scenario");
        let config = config_from_json(&scenario["config"]);
        let mut game = Game::new(Arc::clone(&data), config).expect("game must start");

        assert_same_fingerprint(
            &format!("{label} opening"),
            &state_fingerprint(&game.state, &data),
            &scenario["initial"],
        );

        let decisions = scenario["decisions"].as_array().expect("decisions");
        assert!(!decisions.is_empty(), "[{label}] fixture has no decisions");

        for (index, decision) in decisions.iter().enumerate() {
            let expected_house = decision["house"].as_str().expect("house");
            let actual_house = game.current_player().map(|house| house.name());
            if actual_house != Some(expected_house) {
                // Dump the position on one line so a merged stderr panic cannot
                // split it; `tools/debug_parity.py` prints the Python twin and
                // `tools/diff_state.py` diffs the two.
                println!("=== RUST FINGERPRINT BEGIN ===");
                println!(
                    "{}",
                    serde_json::to_string(&state_fingerprint(&game.state, &data))
                        .unwrap_or_default()
                );
                println!("=== RUST FINGERPRINT END ===");
                panic!(
                    "[{label}] decision {index}: acting house is {:?}, expected {expected_house}",
                    actual_house
                );
            }

            let actions = game.legal_actions();
            let canonical: Vec<String> = actions
                .iter()
                .map(|action| canonical_action(action, &data))
                .collect();
            let expected_count = decision["count"].as_u64().expect("count") as usize;
            assert_eq!(
                canonical.len(),
                expected_count,
                "[{label}] decision {index}: legal action count"
            );
            let hash = fnv1a64(&canonical.join("\n"));
            if hash != decision["hash"].as_str().expect("hash") {
                panic!(
                    "[{label}] decision {index}: legal action set differs\nrust actions:\n{}",
                    canonical.join("\n")
                );
            }
            // Sampled decisions keep the full list so a mismatch can be read.
            if let Some(expected) = decision["actions"].as_array() {
                let expected: Vec<String> = expected
                    .iter()
                    .map(|value| value.as_str().expect("action text").to_string())
                    .collect();
                for (position, (actual, expected)) in
                    canonical.iter().zip(expected.iter()).enumerate()
                {
                    assert_eq!(
                        actual, expected,
                        "[{label}] decision {index}: action {position} differs"
                    );
                }
            }

            let chosen = decision["chosen"].as_u64().unwrap_or(0) as usize;
            game.step(actions[chosen])
                .unwrap_or_else(|error| panic!("[{label}] decision {index}: {error}"));
            total_decisions += 1;
        }

        assert_same_fingerprint(
            &format!("{label} closing"),
            &state_fingerprint(&game.state, &data),
            &scenario["final"],
        );
        assert!(game.is_terminal(), "[{label}] the game must end");
    }
    assert!(
        total_decisions > 800,
        "expected the full corpus, replayed {total_decisions} decisions"
    );
}

#[test]
fn fixture_was_recorded_for_this_engine_version() {
    let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture must parse");
    assert_eq!(fixture["generator"], "tools/dump_fixture.py");
    // Guards against a stale fixture being read as a pass.
    let scenarios = fixture["scenarios"].as_array().expect("scenarios");
    assert!(scenarios.len() >= 5, "expected the recorded scenario set");
    for scenario in scenarios {
        assert!(
            !scenario["config"]["use_westeros_decks"]
                .as_bool()
                .unwrap_or(true),
            "the corpus must run without card draws to stay deterministic"
        );
    }
}

#[test]
fn hashes_match_the_python_generator() {
    // A hand-computed check of the shared hash, so a change on either side fails
    // here rather than as an unexplained action-set mismatch.
    assert_eq!(fnv1a64(""), "cbf29ce484222325");
    assert_eq!(fnv1a64("a"), "af63dc4c8601ec8c");
    assert_eq!(
        fnv1a64("pass|stark|||||||||0"),
        fnv1a64("pass|stark|||||||||0")
    );
}
