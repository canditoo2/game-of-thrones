//! Loader and validator for the static board data.
//!
//! Port of `src/agot/engine/map.py`. The map lives in `src/agot/data/*.json`
//! so it can be corrected against the physical board without touching engine
//! code; this module loads those files from a directory the caller supplies
//! (the Python package hands over its own `agot/data` path at runtime).

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use smallvec::SmallVec;

use crate::enums::{CardId, HouseId, RegionId, Terrain, Track, N_HOUSES};

/// Raised when the bundled board data is structurally inconsistent.
#[derive(Debug)]
pub struct MapDataError(pub String);

impl fmt::Display for MapDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "map data error: {}", self.0)
    }
}

impl std::error::Error for MapDataError {}

impl From<std::io::Error> for MapDataError {
    fn from(error: std::io::Error) -> Self {
        MapDataError(error.to_string())
    }
}

impl From<serde_json::Error> for MapDataError {
    fn from(error: serde_json::Error) -> Self {
        MapDataError(error.to_string())
    }
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawMap {
    name: String,
    #[serde(default = "default_max_rounds")]
    max_rounds: u8,
    #[serde(default = "default_victory_castles")]
    victory_castles: u8,
    #[serde(default)]
    neutral_forces: HashMap<String, u8>,
    regions: Vec<RawRegion>,
    adjacency: Vec<(String, String)>,
}

fn default_max_rounds() -> u8 {
    10
}

fn default_victory_castles() -> u8 {
    7
}

#[derive(Deserialize)]
struct RawRegion {
    id: String,
    name: String,
    terrain: String,
    #[serde(default)]
    power: u8,
    #[serde(default)]
    supply: u8,
    #[serde(default)]
    mustering: u8,
    #[serde(default)]
    home: Option<String>,
    #[serde(default)]
    host: Option<String>,
    #[serde(default)]
    sea: Option<String>,
}

/// A single space on the board.
#[derive(Clone, Debug)]
pub struct RegionSpec {
    pub id: String,
    pub name: String,
    pub terrain: Terrain,
    pub power: u8,
    pub supply: u8,
    pub mustering: u8,
    pub home: Option<HouseId>,
    pub host: Option<RegionId>,
    pub sea: Option<RegionId>,
}

impl RegionSpec {
    #[inline]
    pub fn is_land(&self) -> bool {
        self.terrain == Terrain::Land
    }

    #[inline]
    pub fn is_sea(&self) -> bool {
        self.terrain == Terrain::Sea
    }

    #[inline]
    pub fn is_port(&self) -> bool {
        self.terrain == Terrain::Port
    }

    /// True for a castle (1 mustering point in the reference data).
    #[inline]
    pub fn is_castle(&self) -> bool {
        self.mustering == 1
    }

    /// True for a stronghold (2 mustering points in the reference data).
    #[inline]
    pub fn is_stronghold(&self) -> bool {
        self.mustering == 2
    }

    /// Castles and strongholds both count towards the victory condition.
    #[inline]
    pub fn is_fortification(&self) -> bool {
        self.mustering > 0
    }
}

/// Immutable board topology for the base 6-player game.
#[derive(Debug)]
pub struct GameMap {
    pub name: String,
    pub max_rounds: u8,
    pub victory_castles: u8,
    pub regions: Vec<RegionSpec>,
    index: HashMap<String, RegionId>,
    pub neighbours: Vec<SmallVec<[RegionId; 8]>>,
    pub neutral_forces: SmallVec<[(RegionId, u8); 4]>,
    pub fortifications: SmallVec<[RegionId; 32]>,
    pub land_regions: SmallVec<[RegionId; 32]>,
    pub sea_regions: SmallVec<[RegionId; 24]>,
    pub port_hosts: SmallVec<[(RegionId, RegionId); 8]>,
}

impl GameMap {
    /// Parses a map document. Used directly by tests; the engine goes through
    /// [`DataSet::load`].
    pub fn from_json(text: &str) -> Result<GameMap, MapDataError> {
        let raw: RawMap = serde_json::from_str(text)?;

        let mut regions: Vec<RegionSpec> = Vec::with_capacity(raw.regions.len());
        let mut index: HashMap<String, RegionId> = HashMap::with_capacity(raw.regions.len());
        for entry in &raw.regions {
            let terrain = Terrain::from_name(&entry.terrain)
                .ok_or_else(|| MapDataError(format!("unknown terrain {:?}", entry.terrain)))?;
            let home = match &entry.home {
                Some(name) => Some(
                    HouseId::from_name(name)
                        .ok_or_else(|| MapDataError(format!("unknown house {name:?}")))?,
                ),
                None => None,
            };
            if index.contains_key(&entry.id) {
                return Err(MapDataError(format!("duplicate region id: {}", entry.id)));
            }
            let id = RegionId(regions.len() as u16);
            index.insert(entry.id.clone(), id);
            regions.push(RegionSpec {
                id: entry.id.clone(),
                name: entry.name.clone(),
                terrain,
                power: entry.power,
                supply: entry.supply,
                mustering: entry.mustering,
                home,
                host: None,
                sea: None,
            });
        }

        // Ports name their host land region and their sea; resolve both now
        // that every region has an id.
        for (position, entry) in raw.regions.iter().enumerate() {
            if let Some(host) = &entry.host {
                let target = *index
                    .get(host)
                    .ok_or_else(|| MapDataError(format!("unknown port host {host:?}")))?;
                regions[position].host = Some(target);
            }
            if let Some(sea) = &entry.sea {
                let target = *index
                    .get(sea)
                    .ok_or_else(|| MapDataError(format!("unknown port sea {sea:?}")))?;
                regions[position].sea = Some(target);
            }
        }

        let mut neighbours: Vec<SmallVec<[RegionId; 8]>> = vec![SmallVec::new(); regions.len()];
        for (a, b) in &raw.adjacency {
            let left = *index
                .get(a)
                .ok_or_else(|| MapDataError(format!("unknown neighbour {a:?}")))?;
            let right = *index
                .get(b)
                .ok_or_else(|| MapDataError(format!("unknown neighbour {b:?}")))?;
            if !neighbours[left.index()].contains(&right) {
                neighbours[left.index()].push(right);
            }
            if !neighbours[right.index()].contains(&left) {
                neighbours[right.index()].push(left);
            }
        }
        // Sorted by region id, matching the Python engine's `neighbours_of`.
        // Python sorts because a set's iteration order follows the interpreter's
        // string-hash seed; without a fixed order the legal actions a Raid offers
        // differ from process to process and the differential corpus cannot be
        // compared.
        for list in neighbours.iter_mut() {
            list.sort_unstable_by(|left, right| {
                regions[left.index()].id.cmp(&regions[right.index()].id)
            });
        }

        let mut neutral_forces: SmallVec<[(RegionId, u8); 4]> = SmallVec::new();
        for (region, strength) in &raw.neutral_forces {
            let id = *index.get(region).ok_or_else(|| {
                MapDataError(format!("neutral force on unknown region {region:?}"))
            })?;
            neutral_forces.push((id, *strength));
        }

        let fortifications = regions
            .iter()
            .enumerate()
            .filter(|(_, region)| region.is_fortification())
            .map(|(position, _)| RegionId(position as u16))
            .collect();
        let land_regions = regions
            .iter()
            .enumerate()
            .filter(|(_, region)| region.is_land())
            .map(|(position, _)| RegionId(position as u16))
            .collect();
        let sea_regions = regions
            .iter()
            .enumerate()
            .filter(|(_, region)| region.is_sea())
            .map(|(position, _)| RegionId(position as u16))
            .collect();
        let port_hosts = regions
            .iter()
            .enumerate()
            .filter_map(|(position, region)| {
                region.host.map(|host| (RegionId(position as u16), host))
            })
            .collect();

        let map = GameMap {
            name: raw.name,
            max_rounds: raw.max_rounds,
            victory_castles: raw.victory_castles,
            regions,
            index,
            neighbours,
            neutral_forces,
            fortifications,
            land_regions,
            sea_regions,
            port_hosts,
        };
        map.validate()?;
        Ok(map)
    }

    // -- queries ---------------------------------------------------------

    #[inline]
    pub fn region(&self, id: RegionId) -> &RegionSpec {
        &self.regions[id.index()]
    }

    #[inline]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    #[inline]
    pub fn id_of(&self, id: RegionId) -> &str {
        &self.regions[id.index()].id
    }

    pub fn lookup(&self, id: &str) -> Option<RegionId> {
        self.index.get(id).copied()
    }

    #[inline]
    pub fn neighbours_of(&self, id: RegionId) -> &[RegionId] {
        &self.neighbours[id.index()]
    }

    pub fn land_neighbours_of(&self, id: RegionId) -> SmallVec<[RegionId; 8]> {
        self.neighbours_of(id)
            .iter()
            .copied()
            .filter(|other| self.region(*other).is_land())
            .collect()
    }

    pub fn sea_neighbours_of(&self, id: RegionId) -> SmallVec<[RegionId; 8]> {
        self.neighbours_of(id)
            .iter()
            .copied()
            .filter(|other| self.region(*other).is_sea())
            .collect()
    }

    /// The port that belongs to a coastal land region, if any.
    pub fn port_of(&self, land_region: RegionId) -> Option<RegionId> {
        self.port_hosts
            .iter()
            .find(|(_, host)| *host == land_region)
            .map(|(port, _)| *port)
    }

    // -- validation ------------------------------------------------------

    /// Checks the invariants the engine depends on.
    pub fn validate(&self) -> Result<(), MapDataError> {
        for (position, region) in self.regions.iter().enumerate() {
            let rid = RegionId(position as u16);
            if let Some(host) = region.host {
                if host.index() >= self.regions.len() {
                    return Err(MapDataError(format!("{}: unknown port host", region.id)));
                }
            }
            if let Some(sea) = region.sea {
                if sea.index() >= self.regions.len() {
                    return Err(MapDataError(format!("{}: unknown port sea", region.id)));
                }
            }
            if region.is_land() && region.mustering > 0 && !region.is_fortification() {
                return Err(MapDataError(format!(
                    "{}: inconsistent mustering value",
                    region.id
                )));
            }
            if self.neighbours[rid.index()].is_empty() {
                return Err(MapDataError(format!("{} has no neighbours", region.id)));
            }
            if self.neighbours[rid.index()].contains(&rid) {
                return Err(MapDataError(format!("{} is adjacent to itself", region.id)));
            }
        }

        for (position, list) in self.neighbours.iter().enumerate() {
            let rid = RegionId(position as u16);
            for other in list {
                if other.index() >= self.regions.len() {
                    return Err(MapDataError(format!(
                        "{}: unknown neighbour",
                        self.regions[position].id
                    )));
                }
                if !self.neighbours[other.index()].contains(&rid) {
                    return Err(MapDataError(format!(
                        "adjacency {}-{} is not symmetric",
                        self.regions[position].id,
                        self.regions[other.index()].id
                    )));
                }
            }
        }

        for (port, host) in &self.port_hosts {
            let spec = self.region(*port);
            if !self.neighbours_of(*port).contains(host) {
                return Err(MapDataError(format!(
                    "{} is not adjacent to its host",
                    spec.id
                )));
            }
            match spec.sea {
                Some(sea) if self.neighbours_of(*port).contains(&sea) => {}
                _ => {
                    return Err(MapDataError(format!(
                        "{} is not adjacent to its sea",
                        spec.id
                    )))
                }
            }
        }

        let mut homes: SmallVec<[HouseId; N_HOUSES]> = SmallVec::new();
        for region in &self.regions {
            if let Some(home) = region.home {
                if homes.contains(&home) {
                    return Err(MapDataError(
                        "two regions share the same home house".to_string(),
                    ));
                }
                homes.push(home);
            }
        }

        for (region, strength) in &self.neutral_forces {
            let spec = self.region(*region);
            if !spec.is_land() {
                return Err(MapDataError(format!(
                    "neutral force on non-land region {:?}",
                    spec.id
                )));
            }
            if *strength == 0 {
                return Err(MapDataError(format!(
                    "neutral force {:?} has non-positive strength",
                    spec.id
                )));
            }
            if spec.home.is_some() {
                return Err(MapDataError(format!(
                    "neutral force {:?} is a home area",
                    spec.id
                )));
            }
        }

        if self.sea_regions.is_empty() {
            return Err(MapDataError("no sea regions defined".to_string()));
        }
        if self.land_regions.is_empty() {
            return Err(MapDataError("no land regions defined".to_string()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Houses
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawHouses {
    #[serde(default = "default_player_count")]
    player_count: u8,
    houses: Vec<RawHouse>,
    initial_tracks: RawInitialTracks,
}

fn default_player_count() -> u8 {
    6
}

#[derive(Deserialize)]
struct RawHouse {
    id: String,
    name: String,
    home: String,
    #[serde(default = "default_initial_power")]
    initial_power: u8,
    #[serde(default = "default_initial_supply")]
    initial_supply: u8,
    #[serde(default)]
    starting_units: HashMap<String, RawUnitCounts>,
}

fn default_initial_power() -> u8 {
    5
}

fn default_initial_supply() -> u8 {
    2
}

#[derive(Deserialize, Default)]
struct RawUnitCounts {
    #[serde(default)]
    footmen: u8,
    #[serde(default)]
    knights: u8,
    #[serde(default)]
    siege: u8,
    #[serde(default)]
    ships: u8,
}

impl RawUnitCounts {
    fn as_array(&self) -> [u8; 4] {
        [self.footmen, self.knights, self.siege, self.ships]
    }
}

#[derive(Deserialize)]
struct RawInitialTracks {
    iron_throne: Vec<String>,
    fiefdoms: Vec<String>,
    kings_court: Vec<String>,
}

impl RawInitialTracks {
    fn by_track(&self, track: Track) -> &Vec<String> {
        match track {
            Track::IronThrone => &self.iron_throne,
            Track::Fiefdoms => &self.fiefdoms,
            Track::KingsCourt => &self.kings_court,
        }
    }
}

/// Static per-house data: home area, starting units and starting tracks.
#[derive(Debug)]
pub struct HouseSpec {
    pub id: HouseId,
    pub name: String,
    pub home: RegionId,
    pub initial_power: u8,
    pub initial_supply: u8,
    /// Starting units per region, as a count per `UnitType` index.
    pub starting_units: SmallVec<[(RegionId, [u8; 4]); 4]>,
}

#[derive(Debug)]
pub struct HouseData {
    pub player_count: u8,
    /// Indexed by [`HouseId`]; absent houses are `None`.
    pub houses: [Option<HouseSpec>; N_HOUSES],
    /// Starting track order per [`Track`].
    pub initial_tracks: [SmallVec<[HouseId; N_HOUSES]>; 3],
}

impl HouseData {
    pub fn house(&self, house: HouseId) -> Option<&HouseSpec> {
        self.houses[house.index()].as_ref()
    }
}

// ---------------------------------------------------------------------------
// House cards
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Deserialize)]
pub struct CardSpec {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub strength: u8,
    #[serde(default)]
    pub swords: u8,
    #[serde(default)]
    pub forts: u8,
    #[serde(default)]
    pub ability: String,
}

#[derive(Deserialize)]
struct RawHouseCards {
    houses: HashMap<String, RawHouseDeck>,
}

#[derive(Deserialize)]
struct RawHouseDeck {
    cards: Vec<CardSpec>,
}

/// The seven house cards each house brings to a battle.
#[derive(Debug)]
pub struct HouseCardData {
    /// Indexed by [`HouseId`]; a [`CardId`] is an index into this list, in the
    /// order the deck file lists them.
    pub decks: [Vec<CardSpec>; N_HOUSES],
}

impl HouseCardData {
    #[inline]
    pub fn deck(&self, house: HouseId) -> &[CardSpec] {
        &self.decks[house.index()]
    }

    #[inline]
    pub fn card(&self, house: HouseId, card: CardId) -> &CardSpec {
        &self.decks[house.index()][card.0 as usize]
    }

    pub fn card_index(&self, house: HouseId, id: &str) -> Option<CardId> {
        self.deck(house)
            .iter()
            .position(|spec| spec.id == id)
            .map(|index| CardId(index as u8))
    }

    pub fn deck_size(&self, house: HouseId) -> usize {
        self.deck(house).len()
    }
}

// ---------------------------------------------------------------------------
// Westeros and Wildling decks
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawWesteros {
    cards: serde_json::Map<String, serde_json::Value>,
    composition: HashMap<String, Vec<String>>,
}

/// The three Westeros decks and their effects.
///
/// Card payloads stay as raw JSON in this layer; `game.rs` interprets them.
#[derive(Debug)]
pub struct WesterosData {
    pub card_ids: Vec<String>,
    index: HashMap<String, u16>,
    pub cards: Vec<serde_json::Value>,
    /// `deck_i`, `deck_ii`, `deck_iii`, as card indices.
    pub decks: [Vec<u16>; 3],
}

impl WesterosData {
    pub fn card_index(&self, id: &str) -> Option<u16> {
        self.index.get(id).copied()
    }

    #[inline]
    pub fn card_id(&self, index: u16) -> &str {
        &self.card_ids[index as usize]
    }

    #[inline]
    pub fn card(&self, index: u16) -> &serde_json::Value {
        &self.cards[index as usize]
    }

    /// Deck names in the engine's resolution order.
    pub const DECK_NAMES: [&'static str; 3] = ["deck_i", "deck_ii", "deck_iii"];
}

#[derive(Deserialize)]
struct RawWildlings {
    track_max: u8,
    start_threat: u8,
    cards: serde_json::Map<String, serde_json::Value>,
    deck: Vec<String>,
}

/// The Wildling deck and track configuration.
#[derive(Debug)]
pub struct WildlingData {
    pub track_max: u8,
    pub start_threat: u8,
    pub card_ids: Vec<String>,
    index: HashMap<String, u16>,
    pub cards: Vec<serde_json::Value>,
    pub deck: Vec<u16>,
}

impl WildlingData {
    pub fn card_index(&self, id: &str) -> Option<u16> {
        self.index.get(id).copied()
    }

    #[inline]
    pub fn card_id(&self, index: u16) -> &str {
        &self.card_ids[index as usize]
    }

    #[inline]
    pub fn card(&self, index: u16) -> &serde_json::Value {
        &self.cards[index as usize]
    }
}

// ---------------------------------------------------------------------------
// Bundled data set
// ---------------------------------------------------------------------------

/// Every static table the engine needs, loaded from one directory.
#[derive(Debug)]
pub struct DataSet {
    pub map: GameMap,
    pub houses: HouseData,
    pub house_cards: HouseCardData,
    pub westeros: WesterosData,
    pub wildlings: WildlingData,
}

/// Directory the crate falls back to when no explicit path is given: the
/// Python package's data directory, reached relative to this crate.
pub fn default_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AGOT_DATA_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../src/agot/data"
    ))
}

impl DataSet {
    pub fn load(dir: &Path) -> Result<DataSet, MapDataError> {
        let read = |name: &str| -> Result<String, MapDataError> {
            let path = dir.join(name);
            std::fs::read_to_string(&path)
                .map_err(|error| MapDataError(format!("{}: {error}", path.display())))
        };

        let map = GameMap::from_json(&read("map_base_6p.json")?)?;

        // Houses.
        let raw: RawHouses = serde_json::from_str(&read("houses.json")?)?;
        let mut slots: [Option<HouseSpec>; N_HOUSES] = std::array::from_fn(|_| None);
        for entry in raw.houses {
            let id = HouseId::from_name(&entry.id)
                .ok_or_else(|| MapDataError(format!("unknown house id {:?}", entry.id)))?;
            let home = map
                .lookup(&entry.home)
                .ok_or_else(|| MapDataError(format!("unknown home region {:?}", entry.home)))?;
            let mut starting_units: SmallVec<[(RegionId, [u8; 4]); 4]> = SmallVec::new();
            for (region_id, counts) in &entry.starting_units {
                let region = map.lookup(region_id).ok_or_else(|| {
                    MapDataError(format!("starting unit on unknown region {region_id:?}"))
                })?;
                starting_units.push((region, counts.as_array()));
            }
            slots[id.index()] = Some(HouseSpec {
                id,
                name: entry.name,
                home,
                initial_power: entry.initial_power,
                initial_supply: entry.initial_supply,
                starting_units,
            });
        }
        let mut initial_tracks: [SmallVec<[HouseId; N_HOUSES]>; 3] =
            std::array::from_fn(|_| SmallVec::new());
        for track in Track::ALL {
            for name in raw.initial_tracks.by_track(track) {
                let house = HouseId::from_name(name)
                    .ok_or_else(|| MapDataError(format!("unknown house on track {name:?}")))?;
                initial_tracks[track.index()].push(house);
            }
        }
        let houses = HouseData {
            player_count: raw.player_count,
            houses: slots,
            initial_tracks,
        };

        // House cards.
        let raw: RawHouseCards = serde_json::from_str(&read("house_cards.json")?)?;
        let mut decks: [Vec<CardSpec>; N_HOUSES] = std::array::from_fn(|_| Vec::new());
        for (name, deck) in raw.houses {
            let id = HouseId::from_name(&name)
                .ok_or_else(|| MapDataError(format!("unknown house deck {name:?}")))?;
            decks[id.index()] = deck.cards;
        }
        let house_cards = HouseCardData { decks };

        // Westeros.
        let raw: RawWesteros = serde_json::from_str(&read("westeros.json")?)?;
        let mut card_ids: Vec<String> = Vec::with_capacity(raw.cards.len());
        let mut index: HashMap<String, u16> = HashMap::with_capacity(raw.cards.len());
        let mut cards: Vec<serde_json::Value> = Vec::with_capacity(raw.cards.len());
        for (id, payload) in raw.cards {
            index.insert(id.clone(), card_ids.len() as u16);
            card_ids.push(id);
            cards.push(payload);
        }
        let mut deck_lists: [Vec<u16>; 3] = std::array::from_fn(|_| Vec::new());
        for (position, name) in WesterosData::DECK_NAMES.iter().enumerate() {
            let ids = raw
                .composition
                .get(*name)
                .ok_or_else(|| MapDataError(format!("westeros composition is missing {name:?}")))?;
            deck_lists[position] = ids
                .iter()
                .map(|id| {
                    index
                        .get(id)
                        .copied()
                        .ok_or_else(|| MapDataError(format!("unknown westeros card {id:?}")))
                })
                .collect::<Result<Vec<u16>, MapDataError>>()?;
        }
        let westeros = WesterosData {
            card_ids,
            index,
            cards,
            decks: deck_lists,
        };

        // Wildlings.
        let raw: RawWildlings = serde_json::from_str(&read("wildlings.json")?)?;
        let mut card_ids: Vec<String> = Vec::with_capacity(raw.cards.len());
        let mut index: HashMap<String, u16> = HashMap::with_capacity(raw.cards.len());
        let mut cards: Vec<serde_json::Value> = Vec::with_capacity(raw.cards.len());
        for (id, payload) in raw.cards {
            index.insert(id.clone(), card_ids.len() as u16);
            card_ids.push(id);
            cards.push(payload);
        }
        let deck = raw
            .deck
            .iter()
            .map(|id| {
                index
                    .get(id)
                    .copied()
                    .ok_or_else(|| MapDataError(format!("unknown wildling card {id:?}")))
            })
            .collect::<Result<Vec<u16>, MapDataError>>()?;
        let wildlings = WildlingData {
            track_max: raw.track_max,
            start_threat: raw.start_threat,
            card_ids,
            index,
            cards,
            deck,
        };

        Ok(DataSet {
            map,
            houses,
            house_cards,
            westeros,
            wildlings,
        })
    }

    /// Loads the bundled data from [`default_data_dir`].
    pub fn load_default() -> Result<DataSet, MapDataError> {
        DataSet::load(&default_data_dir())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_map_loads_and_validates() {
        let data = DataSet::load_default().expect("bundled data must load");
        assert_eq!(data.map.region_count(), 58);
        assert_eq!(data.map.victory_castles, 7);
        assert_eq!(data.map.max_rounds, 10);
        assert_eq!(data.map.neutral_forces.len(), 2);
        // The base board has eight ports (Winterfell, White Harbor, Pyke, ...).
        assert_eq!(data.map.port_hosts.len(), 8);
        let ports = data
            .map
            .regions
            .iter()
            .filter(|region| region.is_port())
            .count();
        assert_eq!(ports, data.map.port_hosts.len());
        assert_eq!(
            data.map.land_regions.len() + data.map.sea_regions.len() + ports,
            data.map.region_count()
        );
    }

    #[test]
    fn house_decks_are_seven_cards() {
        let data = DataSet::load_default().unwrap();
        for house in HouseId::ALL {
            assert_eq!(
                data.house_cards.deck_size(house),
                7,
                "{house} must bring seven cards"
            );
            assert!(data.houses.house(house).is_some());
        }
    }

    #[test]
    fn westeros_and_wildling_decks_are_complete() {
        let data = DataSet::load_default().unwrap();
        assert_eq!(data.westeros.decks.len(), 3);
        for deck in &data.westeros.decks {
            assert_eq!(deck.len(), 10);
        }
        assert_eq!(data.wildlings.deck.len(), 9);
        assert_eq!(data.wildlings.track_max, 12);
        assert_eq!(data.wildlings.start_threat, 2);
    }

    #[test]
    fn adjacency_is_symmetric_and_ports_are_wired() {
        let data = DataSet::load_default().unwrap();
        let map = &data.map;
        for (position, list) in map.neighbours.iter().enumerate() {
            let rid = RegionId(position as u16);
            for other in list {
                assert!(
                    map.neighbours_of(*other).contains(&rid),
                    "{} -> {} must be symmetric",
                    map.id_of(rid),
                    map.id_of(*other)
                );
            }
        }
        // Every port knows its host and its sea.
        for (port, host) in &map.port_hosts {
            assert_eq!(map.port_of(*host), Some(*port));
            assert!(map.region(*port).is_port());
            assert!(map.region(*port).sea.is_some());
        }
    }
}
