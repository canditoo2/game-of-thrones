"""Structural tests for the bundled board data."""

from agot.engine.map import load_house_cards, load_houses, load_map


def test_region_counts() -> None:
    game_map = load_map()
    assert len(game_map.regions) == 58
    assert len(game_map.land_region_ids) == 38
    assert len(game_map.sea_region_ids) == 12
    assert len(game_map.port_region_ids) == 8


def test_adjacency_is_symmetric() -> None:
    game_map = load_map()
    for region_id, neighbours in game_map.neighbours.items():
        assert neighbours, f"{region_id} has no neighbours"
        for other in neighbours:
            assert region_id in game_map.neighbours[other]


def test_ports_are_well_formed() -> None:
    game_map = load_map()
    for port_id, host in game_map.port_hosts.items():
        assert host in game_map.neighbours[port_id]
        port = game_map.regions[port_id]
        assert port.sea is not None
        assert port.sea in game_map.neighbours[port_id]


def test_fortifications_have_mustering_values() -> None:
    game_map = load_map()
    for region in game_map.regions.values():
        if region.is_fortification:
            assert region.mustering in (1, 2)
    assert sum(1 for r in game_map.regions.values() if r.is_fortification) == 20


def test_houses_are_six_and_start_on_known_regions() -> None:
    houses = load_houses()
    game_map = load_map()
    assert len(houses.houses) == 6
    for house_id in houses.houses:
        home = houses.home(house_id)
        assert home in game_map.regions
        for region_id in houses.starting_units(house_id):
            assert region_id in game_map.regions


def test_every_house_has_seven_cards_with_stats() -> None:
    cards = load_house_cards()
    for house_id, deck in cards.cards.items():
        assert len(deck) == 7, house_id
        for card in deck:
            assert card["strength"] >= 0
            assert card["swords"] >= 0
            assert card["forts"] >= 0
