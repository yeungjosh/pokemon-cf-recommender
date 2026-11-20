"""Unit tests for the synthetic team generator."""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_generator import TeamDataGenerator  # noqa: E402

POKEDEX = ROOT / "data" / "pokedex.json"


@pytest.fixture
def generator():
    random.seed(1234)
    return TeamDataGenerator(POKEDEX)


def test_archetype_pokemon_exist_in_pokedex(generator):
    names = set(generator.pokemon_names)
    for arch, groups in generator.archetypes.items():
        for pokemon in groups["core"] + groups["flex"]:
            assert pokemon in names, f"{pokemon} ({arch}) missing from pokedex"


@pytest.mark.parametrize("archetype", [None, "balance", "offense", "stall",
                                       "hyper_offense", "bulky_offense"])
def test_generate_team_is_six_unique_known_pokemon(generator, archetype):
    names = set(generator.pokemon_names)
    for _ in range(200):
        team = generator.generate_team(archetype)
        assert len(team) == 6
        assert len(set(team)) == 6
        assert set(team) <= names


def test_generate_team_contains_archetype_core(generator):
    core = set(generator.archetypes["stall"]["core"])
    for _ in range(100):
        team = set(generator.generate_team("stall"))
        assert len(team & core) >= 2


def test_generate_team_unknown_archetype_raises(generator):
    with pytest.raises(KeyError):
        generator.generate_team("nonexistent")


def test_generate_dataset_size_and_seed_reproducibility():
    random.seed(7)
    first = TeamDataGenerator(POKEDEX).generate_dataset(n_teams=25)
    random.seed(7)
    second = TeamDataGenerator(POKEDEX).generate_dataset(n_teams=25)
    assert len(first) == 25
    assert first == second


def test_generate_dataset_respects_distribution(generator):
    teams = generator.generate_dataset(n_teams=50, archetype_distribution={"stall": 1.0})
    core = set(generator.archetypes["stall"]["core"])
    assert all(len(set(t) & core) >= 2 for t in teams)


def test_save_teams_roundtrip(generator, tmp_path, capsys):
    teams = generator.generate_dataset(n_teams=3)
    out = tmp_path / "teams.json"
    generator.save_teams(teams, out)
    assert json.loads(out.read_text()) == teams
    assert "Saved 3 teams" in capsys.readouterr().out
