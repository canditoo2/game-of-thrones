//! Differential test: the Rust encoder must reproduce the Python reference
//! **bit for bit**.
//!
//! The golden data is recorded from the Python engine by
//! `tools/dump_encoder_golden.py`. Every vector is a hex string of raw
//! little-endian float32 bytes, so the comparison is exact (`f32::to_bits`)
//! and a mismatch reports the first differing index with both values — a single
//! changed last bit is a failure, not a rounding detail.
//!
//! Two sections are checked:
//!
//! * `scenarios` — seeded games replayed through `Game::new` and `Game::step`
//!   (Planning and the Messenger Raven step, the phases the Rust engine can
//!   replay today): the state vector and every candidate action vector at every
//!   decision;
//! * `positions` — positions sampled from full games, rebuilt directly from a
//!   recorded state dump so that battles, bids, mustering and the
//!   holders'-choice cards are covered without depending on the rules port.

use std::sync::Arc;

use agot_core::action::{Action, ActionKind, Side};
use agot_core::enums::{
    ActionStep, CardId, HouseId, OrderKey, Phase, RegionId, Track, UnitType, HOUSE_ORDER,
    ORDER_SPECS,
};
use agot_core::game::Game;
use agot_core::map::DataSet;
use agot_core::observation::{self, ACTION_DIM, STATE_DIM};
use agot_core::setup::GameConfig;
use agot_core::state::{House, State};
use serde_json::Value;

const GOLDEN: &str = include_str!("data/encoder_golden.json");

/// Smallest number of vectors the recorded set must contain (a truncated
/// fixture should not read as a pass).
const MIN_VECTORS: usize = 1500;

// ---------------------------------------------------------------------------
// Golden data helpers
// ---------------------------------------------------------------------------

fn house(name: &str) -> HouseId {
    HouseId::from_name(name).unwrap_or_else(|| panic!("unknown house {name:?}"))
}

fn optional_house(value: &Value) -> Option<HouseId> {
    match value {
        Value::Null => None,
        Value::String(name) => Some(house(name)),
        other => panic!("expected a house name or null, got {other}"),
    }
}

fn optional_region(value: &str, data: &DataSet) -> Option<RegionId> {
    if value.is_empty() {
        return None;
    }
    Some(
        data.map
            .lookup(value)
            .unwrap_or_else(|| panic!("unknown region {value:?}")),
    )
}

fn optional_order(value: &str) -> Option<OrderKey> {
    if value.is_empty() {
        return None;
    }
    Some(OrderKey::from_name(value).unwrap_or_else(|| panic!("unknown order {value:?}")))
}

/// Decodes the raw little-endian float32 bytes the generator recorded.
fn hex_to_f32s(hex: &str) -> Vec<f32> {
    assert_eq!(hex.len() % 8, 0, "hex must hold whole float32 words");
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&hex[index..index + 2], 16)
                .unwrap_or_else(|_| panic!("bad hex at {index}: {:?}", &hex[index..index + 2]))
        })
        .collect();
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for word in bytes.chunks(4) {
        let word: [u8; 4] = word.try_into().expect("whole float32 words");
        values.push(f32::from_le_bytes(word));
    }
    values
}

/// Compares bit patterns, naming the first difference with both values.
fn assert_bits_equal(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "[{label}] vector length");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            panic!(
                "[{label}] index {index}: rust {} (0x{:08x}) != python {} (0x{:08x})",
                actual,
                actual.to_bits(),
                expected,
                expected.to_bits()
            );
        }
    }
}

fn config_from_json(value: &Value) -> GameConfig {
    let houses = value["houses"]
        .as_array()
        .expect("houses")
        .iter()
        .map(|name| house(name.as_str().expect("house name")))
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

/// Rebuilds one action from its canonical `kind|house|...` record.
///
/// A card id that is not in the action house's deck becomes an out-of-range
/// `CardId`, which is the encoder's "unknown card" fallback — exactly what the
/// Python reference does when the id is missing from its card table (a
/// holders'-choice Westeros card, say).
fn parse_canonical_action(text: &str, data: &DataSet) -> Action {
    let fields: Vec<&str> = text.split('|').collect();
    assert_eq!(fields.len(), 11, "canonical action field count: {text:?}");
    let mut action = Action::new(
        ActionKind::from_name(fields[0]).unwrap_or_else(|| panic!("unknown kind {text:?}")),
    );
    action.house = optional_house(&Value::String(fields[1].to_string()));
    action.region = optional_region(fields[2], data);
    action.target = optional_region(fields[3], data);
    action.order = optional_order(fields[4]);
    action.card = match (fields[5], action.house) {
        ("", _) => None,
        (id, Some(house_id)) => Some(
            data.house_cards
                .card_index(house_id, id)
                .unwrap_or(CardId(u8::MAX)),
        ),
        (id, None) => panic!("card {id:?} without a house in {text:?}"),
    };
    action.unit = match fields[6] {
        "" => None,
        name => Some(UnitType::from_name(name).unwrap_or_else(|| panic!("unknown unit {name:?}"))),
    };
    action.side = match fields[7] {
        "" => None,
        name => Some(Side::from_name(name).unwrap_or_else(|| panic!("unknown side {name:?}"))),
    };
    action.amount = match fields[8] {
        "" => None,
        value => Some(
            value
                .parse()
                .unwrap_or_else(|_| panic!("bad amount {value:?}")),
        ),
    };
    action.units = match fields[9] {
        "" => None,
        counts => {
            let values: Vec<u8> = counts
                .split(',')
                .map(|count| {
                    count
                        .parse()
                        .unwrap_or_else(|_| panic!("bad unit count {count:?}"))
                })
                .collect();
            assert_eq!(values.len(), 4, "march unit vector: {counts:?}");
            Some([values[0], values[1], values[2], values[3]])
        }
    };
    action.establish = match fields[10] {
        "0" => false,
        "1" => true,
        other => panic!("bad establish flag {other:?}"),
    };
    action
}

/// Rebuilds a `State` from the generator's encoder-facing dump.
///
/// `fortifications` is not in the dump: every game starts with the map's
/// castles and strongholds, so it is taken from the data set.
fn state_from_dump(value: &Value, data: &DataSet) -> State {
    let mut state = State::blank();
    state.max_rounds = value["max_rounds"].as_u64().expect("max_rounds") as u8;
    state.victory_castles = value["victory_castles"].as_u64().expect("victory_castles") as u8;
    state.round = value["round"].as_u64().expect("round") as u8;
    state.phase = Phase::from_name(value["phase"].as_str().expect("phase"))
        .unwrap_or_else(|| panic!("unknown phase {:?}", value["phase"]));
    state.action_step = ActionStep::from_name(value["action_step"].as_str().expect("action_step"))
        .unwrap_or_else(|| panic!("unknown step {:?}", value["action_step"]));
    state.active_house = optional_house(&value["active_house"]);
    state.wildling_threat = value["wildling_threat"].as_u64().expect("wildling_threat") as u8;
    state.fortifications = data.map.fortifications.clone();

    for (track, key) in [
        (Track::IronThrone, "iron_throne"),
        (Track::Fiefdoms, "fiefdoms"),
        (Track::KingsCourt, "kings_court"),
    ] {
        state.tracks[track.index()] = value["tracks"][key]
            .as_array()
            .unwrap_or_else(|| panic!("track {key:?}"))
            .iter()
            .map(|name| house(name.as_str().expect("house name")))
            .collect();
    }

    for (name, entry) in value["houses"].as_object().expect("houses") {
        let id = house(name);
        let mut house_state = House::new(
            data.house_cards.deck_size(id),
            entry["power"].as_u64().expect("power") as u8,
            entry["supply_level"].as_u64().expect("supply_level") as u8,
        );
        house_state.orders_placed = entry["orders_placed"]
            .as_array()
            .expect("orders_placed")
            .iter()
            .map(|key| optional_order(key.as_str().expect("order key")).expect("order key"))
            .collect();
        house_state.house_cards_available = entry["cards_available"]
            .as_array()
            .expect("cards_available")
            .iter()
            .map(|card| {
                let card_id = card.as_str().expect("card id");
                data.house_cards
                    .card_index(id, card_id)
                    .unwrap_or_else(|| panic!("{id} has no card {card_id:?}"))
            })
            .collect();
        house_state.eliminated = entry["eliminated"].as_bool().expect("eliminated");
        state.houses[id.index()] = Some(house_state);
    }

    let regions = value["regions"].as_array().expect("regions");
    assert_eq!(
        regions.len(),
        data.map.region_count(),
        "the dump must cover the whole board"
    );
    state.region_count = regions.len() as u8;
    for (index, entry) in regions.iter().enumerate() {
        let region = &mut state.regions[index];
        let units = entry["units"].as_array().expect("units");
        let routed = entry["routed"].as_array().expect("routed");
        assert_eq!(units.len(), 4, "unit vector");
        assert_eq!(routed.len(), 4, "routed vector");
        for (slot, count) in units.iter().enumerate() {
            region.units[slot] = count.as_u64().expect("unit count") as u8;
        }
        for (slot, count) in routed.iter().enumerate() {
            region.routed[slot] = count.as_u64().expect("routed count") as u8;
        }
        region.controller = optional_house(&entry["controller"]);
        region.owner = optional_house(&entry["owner"]);
        region.influence_token = optional_house(&entry["influence_token"]);
        region.order = match &entry["order"] {
            Value::Null => None,
            Value::String(key) => optional_order(key),
            other => panic!("bad order {other}"),
        };
        region.order_house = optional_house(&entry["order_house"]);
        region.garrison = entry["garrison"].as_u64().expect("garrison") as u8;
        region.neutral_force = entry["neutral_force"].as_u64().expect("neutral_force") as u8;
        region.mustering_points = entry["mustering_points"]
            .as_u64()
            .expect("mustering_points") as u8;
    }

    state
}

/// Encodes every recorded candidate action of one decision and compares.
fn compare_actions(
    label: &str,
    state: &State,
    data: &DataSet,
    actor: Option<HouseId>,
    texts: &[Value],
    vectors: &[Value],
    cache: &mut observation::RegionCache,
) -> usize {
    assert_eq!(texts.len(), vectors.len(), "[{label}] action/vector count");
    for (position, (text, vector)) in texts.iter().zip(vectors.iter()).enumerate() {
        let action = parse_canonical_action(text.as_str().expect("action text"), data);
        let mut out = vec![0.0f32; ACTION_DIM];
        observation::encode_action_into(state, data, &action, actor, &mut out, cache);
        let expected = hex_to_f32s(vector.as_str().expect("vector hex"));
        assert_bits_equal(&format!("{label} action {position}"), &out, &expected);
    }
    texts.len()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn replay_decisions_match_python_bit_for_bit() {
    let golden: Value = serde_json::from_str(GOLDEN).expect("golden data must parse");
    let data = Arc::new(DataSet::load_default().expect("bundled data must load"));
    let mut compared = 0usize;

    for scenario in golden["scenarios"].as_array().expect("scenarios") {
        let label = scenario["label"].as_str().unwrap_or("scenario");
        let config = config_from_json(&scenario["config"]);
        let mut game = Game::new(Arc::clone(&data), config).expect("game must start");

        for (index, decision) in scenario["decisions"]
            .as_array()
            .expect("decisions")
            .iter()
            .enumerate()
        {
            let actor = optional_house(&decision["actor"]);
            assert_eq!(
                game.state.active_house, actor,
                "[{label}] decision {index}: acting house"
            );

            let expected = hex_to_f32s(decision["state"].as_str().expect("state hex"));
            let actual = observation::encode_state(&game.state, &data, actor);
            assert_bits_equal(
                &format!("{label} decision {index} state"),
                &actual,
                &expected,
            );
            compared += 1;

            let mut cache = observation::RegionCache::new();
            compared += compare_actions(
                &format!("{label} decision {index}"),
                &game.state,
                &data,
                actor,
                decision["actions"].as_array().expect("actions"),
                decision["vectors"].as_array().expect("vectors"),
                &mut cache,
            );

            // The scripted path continues through the recorded action unless
            // the generator stopped at the port boundary.
            if let Some(chosen) = decision["chosen"].as_u64() {
                let text = decision["actions"][chosen as usize]
                    .as_str()
                    .expect("chosen action");
                let action = parse_canonical_action(text, &data);
                game.step(action)
                    .unwrap_or_else(|error| panic!("[{label}] decision {index}: {error}"));
            }
        }
    }

    assert!(
        compared >= MIN_VECTORS,
        "the golden set must not shrink: {compared} vectors"
    );
}

#[test]
fn sampled_positions_match_python_bit_for_bit() {
    let golden: Value = serde_json::from_str(GOLDEN).expect("golden data must parse");
    let data = DataSet::load_default().expect("bundled data must load");
    let mut compared = 0usize;

    for (index, position) in golden["positions"]
        .as_array()
        .expect("positions")
        .iter()
        .enumerate()
    {
        let label = format!(
            "{} position {index} ({} decision {})",
            position["label"].as_str().unwrap_or("?"),
            position["phase"].as_str().unwrap_or("?"),
            position["decision"].as_u64().unwrap_or(0)
        );
        let state = state_from_dump(&position["state_dump"], &data);
        let actor = optional_house(&position["actor"]);
        assert_eq!(state.active_house, actor, "[{label}] acting house");

        let expected = hex_to_f32s(position["state"].as_str().expect("state hex"));
        let actual = observation::encode_state(&state, &data, actor);
        assert_bits_equal(&format!("{label} state"), &actual, &expected);
        compared += 1;

        let mut cache = observation::RegionCache::new();
        compared += compare_actions(
            &label,
            &state,
            &data,
            actor,
            position["actions"].as_array().expect("actions"),
            position["vectors"].as_array().expect("vectors"),
            &mut cache,
        );
    }

    assert!(
        compared >= 60,
        "the sampled set must not shrink: {compared} vectors"
    );
}

#[test]
fn golden_was_recorded_for_this_encoder_version() {
    let golden: Value = serde_json::from_str(GOLDEN).expect("golden data must parse");
    assert_eq!(golden["generator"], "tools/dump_encoder_golden.py");
    // Guards against a stale fixture being read as a pass.
    assert_eq!(
        golden["state_dim"].as_u64().expect("state_dim"),
        STATE_DIM as u64
    );
    assert_eq!(
        golden["action_dim"].as_u64().expect("action_dim"),
        ACTION_DIM as u64
    );
    let houses: Vec<&str> = golden["house_order"]
        .as_array()
        .expect("house_order")
        .iter()
        .map(|name| name.as_str().expect("house"))
        .collect();
    assert_eq!(houses, HOUSE_ORDER.to_vec());
    let orders: Vec<&str> = golden["order_keys"]
        .as_array()
        .expect("order_keys")
        .iter()
        .map(|key| key.as_str().expect("order"))
        .collect();
    assert_eq!(
        orders,
        ORDER_SPECS.iter().map(|spec| spec.key).collect::<Vec<_>>()
    );
    assert!(!golden["scenarios"]
        .as_array()
        .expect("scenarios")
        .is_empty());
    assert!(!golden["positions"]
        .as_array()
        .expect("positions")
        .is_empty());
}
