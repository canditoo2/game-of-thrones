//! `Game.snapshot()` built directly as Python objects.
//!
//! The snapshot is `agot_core::fingerprint::state_fingerprint` as Python data.
//! Going through `serde_json::Value` first allocated a `String` per key and a
//! tree node per value before any Python object existed; the scripted greedy
//! seats read one snapshot per decision, so that detour was most of their
//! per-decision cost. This module writes the same structure straight into
//! Python objects. It must stay field-for-field identical to the fingerprint:
//! the same keys, every object's keys in sorted (byte) order as a
//! `serde_json::Map` keeps them, `int` for numbers, `bool` for flags and `None`
//! for nulls. `PyGame._snapshot_reference` keeps the old conversion so a test
//! can compare the two on real positions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use agot_core::enums::{HouseId, Track, UnitType};
use agot_core::map::{DataSet, WesterosData};
use agot_core::state::{Restriction, State, NONE_OPTION};

type Item<'py> = (&'static str, Bound<'py, PyAny>);

/// A dict from `(key, value)` pairs, inserted in sorted key order.
///
/// A stable sort keeps a repeated key's pairs in their original order, so the
/// later value wins at the first one's position, as a `BTreeMap` insert does.
fn sorted_dict<'py, K: AsRef<str>>(
    py: Python<'py>,
    mut items: Vec<(K, Bound<'py, PyAny>)>,
) -> PyResult<Bound<'py, PyAny>> {
    items.sort_by(|left, right| left.0.as_ref().cmp(right.0.as_ref()));
    let dict = PyDict::new(py);
    for (key, value) in items {
        dict.set_item(key.as_ref(), value)?;
    }
    Ok(dict.into_any())
}

fn int<'py>(py: Python<'py>, value: u64) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.into_pyobject(py)?.into_any())
}

fn flag<'py>(py: Python<'py>, value: bool) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.into_pyobject(py)?.to_owned().into_any())
}

fn text<'py>(py: Python<'py>, value: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(value.into_pyobject(py)?.into_any())
}

fn optional_text<'py>(py: Python<'py>, value: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Some(value) => text(py, value),
        None => Ok(py.None().into_bound(py)),
    }
}

fn house_name<'py>(py: Python<'py>, house: Option<HouseId>) -> PyResult<Bound<'py, PyAny>> {
    optional_text(py, house.map(|house| house.name()))
}

fn text_list<'py, 'a>(
    py: Python<'py>,
    values: impl IntoIterator<Item = &'a str>,
) -> PyResult<Bound<'py, PyAny>> {
    let list = PyList::empty(py);
    for value in values {
        list.append(value)?;
    }
    Ok(list.into_any())
}

/// Sorted, as the fingerprint sorts multisets it has no order for.
fn sorted_text_list<'py, 'a>(
    py: Python<'py>,
    values: impl IntoIterator<Item = &'a str>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut values: Vec<&str> = values.into_iter().collect();
    values.sort_unstable();
    text_list(py, values)
}

fn unit_counts<'py>(
    py: Python<'py>,
    count: impl Fn(UnitType) -> u8,
) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Item<'py>> = Vec::with_capacity(UnitType::ALL.len());
    for unit in UnitType::ALL {
        items.push((unit.as_str(), int(py, u64::from(count(unit)))?));
    }
    sorted_dict(py, items)
}

/// `state_fingerprint(state, data)` as Python objects.
pub fn snapshot<'py>(
    py: Python<'py>,
    state: &State,
    data: &DataSet,
) -> PyResult<Bound<'py, PyAny>> {
    let mut root: Vec<Item<'py>> = Vec::with_capacity(24);

    root.push(("round", int(py, u64::from(state.round))?));
    root.push(("phase", text(py, state.phase.as_str())?));
    root.push(("action_step", text(py, state.action_step.as_str())?));
    root.push(("active_house", house_name(py, state.active_house)?));
    root.push(("winner", house_name(py, state.winner)?));
    root.push((
        "action_house_last",
        house_name(py, state.action_house_last)?,
    ));
    root.push(("raven_used", flag(py, state.raven_used)?));
    root.push((
        "return_phase",
        optional_text(py, state.return_phase.map(|phase| phase.as_str()))?,
    ));
    root.push((
        "pending_planning",
        text_list(py, state.pending_planning.iter().map(|house| house.name()))?,
    ));
    root.push((
        "order_restrictions",
        text_list(
            py,
            state
                .order_restrictions
                .iter()
                .map(|restriction| match restriction {
                    Restriction::Key(key) => key.key(),
                    Restriction::Type(order_type) => order_type.as_str(),
                }),
        )?,
    ));
    let mut pending_bids: Vec<(&str, Bound<'py, PyAny>)> = Vec::new();
    for (house, amount) in state.pending_bids.iter() {
        pending_bids.push((house.name(), int(py, u64::from(*amount))?));
    }
    root.push(("pending_bids", sorted_dict(py, pending_bids)?));
    root.push((
        "bid_tracks",
        text_list(py, state.bid_tracks.iter().map(|track| track.as_str()))?,
    ));
    root.push((
        "fortifications",
        sorted_text_list(
            py,
            state
                .fortifications
                .iter()
                .map(|region| data.map.id_of(*region)),
        )?,
    ));

    let mut tracks: Vec<Item<'py>> = Vec::with_capacity(Track::ALL.len());
    for track in Track::ALL {
        tracks.push((
            track.as_str(),
            text_list(
                py,
                state.tracks[track.index()].iter().map(|house| house.name()),
            )?,
        ));
    }
    root.push(("tracks", sorted_dict(py, tracks)?));

    let mut houses: Vec<(&str, Bound<'py, PyAny>)> = Vec::with_capacity(HouseId::ALL.len());
    for house in HouseId::ALL {
        let Some(house_state) = state.house(house) else {
            continue;
        };
        let card_id =
            |card: &agot_core::enums::CardId| data.house_cards.card(house, *card).id.as_str();
        let entry: Vec<Item<'py>> = vec![
            ("power", int(py, u64::from(house_state.power))?),
            (
                "supply_level",
                int(py, u64::from(house_state.supply_level))?,
            ),
            (
                "orders_available",
                text_list(
                    py,
                    house_state.orders_available.iter().map(|order| order.key()),
                )?,
            ),
            (
                "orders_placed",
                text_list(
                    py,
                    house_state.orders_placed.iter().map(|order| order.key()),
                )?,
            ),
            (
                "cards_available",
                text_list(py, house_state.house_cards_available.iter().map(card_id))?,
            ),
            (
                "cards_used",
                text_list(py, house_state.house_cards_used.iter().map(card_id))?,
            ),
            ("blade_used", flag(py, house_state.blade_used)?),
            ("passed", flag(py, house_state.passed)?),
            ("eliminated", flag(py, house_state.eliminated)?),
        ];
        houses.push((house.name(), sorted_dict(py, entry)?));
    }
    root.push(("houses", sorted_dict(py, houses)?));

    let mut regions: Vec<(&str, Bound<'py, PyAny>)> = Vec::new();
    for region_id in state.region_ids() {
        let region = state.region(region_id);
        let entry: Vec<Item<'py>> = vec![
            ("controller", house_name(py, region.controller)?),
            ("owner", house_name(py, region.owner)?),
            ("units", unit_counts(py, |unit| region.unit_count(unit))?),
            ("routed", unit_counts(py, |unit| region.routed_count(unit))?),
            ("neutral_force", int(py, u64::from(region.neutral_force))?),
            ("garrison", int(py, u64::from(region.garrison))?),
            (
                "mustering_points",
                int(py, u64::from(region.mustering_points))?,
            ),
            ("influence_token", house_name(py, region.influence_token)?),
            (
                "order",
                optional_text(py, region.order.map(|order| order.key()))?,
            ),
            ("order_house", house_name(py, region.order_house)?),
        ];
        regions.push((data.map.id_of(region_id), sorted_dict(py, entry)?));
    }
    root.push(("regions", sorted_dict(py, regions)?));

    root.push((
        "wildling_threat",
        int(py, u64::from(state.wildling_threat))?,
    ));
    root.push((
        "wildling_deck",
        sorted_text_list(
            py,
            state
                .wildling_deck
                .cards
                .iter()
                .map(|card| data.wildlings.card_id(*card)),
        )?,
    ));
    root.push((
        "wildling_discard",
        sorted_text_list(
            py,
            state
                .wildling_discard
                .iter()
                .map(|card| data.wildlings.card_id(*card)),
        )?,
    ));

    let mut decks: Vec<Item<'py>> = Vec::with_capacity(WesterosData::DECK_NAMES.len());
    for (position, name) in WesterosData::DECK_NAMES.iter().enumerate() {
        let deck = &state.westeros_decks[position];
        let entry: Vec<Item<'py>> = vec![
            (
                "cards",
                sorted_text_list(
                    py,
                    deck.cards.iter().map(|card| data.westeros.card_id(*card)),
                )?,
            ),
            ("position", int(py, deck.position as u64)?),
            (
                "drawn",
                optional_text(py, deck.drawn.map(|card| data.westeros.card_id(card)))?,
            ),
        ];
        decks.push((name, sorted_dict(py, entry)?));
    }
    root.push(("westeros_decks", sorted_dict(py, decks)?));

    root.push((
        "pending_westeros",
        match state.pending_westeros.as_ref() {
            Some(queue) => text_list(py, queue.iter().map(|card| data.westeros.card_id(*card)))?,
            None => py.None().into_bound(py),
        },
    ));
    root.push((
        "westeros_choice",
        match state.westeros_choice.as_ref() {
            Some(choice) => {
                let entry: Vec<Item<'py>> = vec![
                    ("card", text(py, data.westeros.card_id(choice.card))?),
                    ("holder", text(py, choice.holder.name())?),
                    (
                        "options",
                        // The "no effect" option has no card row; the
                        // fingerprint keeps the literal string.
                        text_list(
                            py,
                            choice.options.iter().map(|card| {
                                if *card == NONE_OPTION {
                                    "none"
                                } else {
                                    data.westeros.card_id(*card)
                                }
                            }),
                        )?,
                    ),
                ];
                sorted_dict(py, entry)?
            }
            None => py.None().into_bound(py),
        },
    ));
    root.push((
        "pending_march",
        match state.pending_march.as_ref() {
            Some(march) => sorted_dict(
                py,
                vec![
                    ("house", text(py, march.house.name())?),
                    ("origin", text(py, data.map.id_of(march.origin))?),
                ],
            )?,
            None => py.None().into_bound(py),
        },
    ));
    root.push((
        "pending_battle_present",
        flag(
            py,
            state.pending_battle.is_some() || state.deferred_battle.is_some(),
        )?,
    ));
    root.push((
        "wildling_attack_present",
        flag(py, state.wildling_attack.is_some())?,
    ));

    sorted_dict(py, root)
}
