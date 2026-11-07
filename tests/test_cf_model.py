"""Unit tests for the item-item CF model and trio recommender."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cf_model import CollaborativeFilteringModel  # noqa: E402
from src.recommender import CFTeamRecommender  # noqa: E402

TEAMS = [
    ["A", "B", "C", "D", "E", "F"],
    ["A", "B", "C", "G", "H", "I"],
    ["A", "B", "D", "G", "J", "K"],
    ["X", "Y", "Z", "W", "V", "U"],
]


@pytest.fixture
def model():
    m = CollaborativeFilteringModel()
    m.build_from_teams(TEAMS)
    return m


def test_build_matrices(model):
    assert model.n_pokemon == len({p for t in TEAMS for p in t})
    co = model.co_occurrence_matrix
    assert np.allclose(co, co.T)
    assert np.all(np.diag(co) == 0)
    a, b = model.pokemon_to_idx["A"], model.pokemon_to_idx["B"]
    assert co[a][b] == 3
    assert np.all(np.diag(model.similarity_matrix) == 0)


def test_recommendations_exclude_inputs_and_are_sorted(model):
    recs = model.get_recommendations(["A", "B"], top_k=5, exclude=["C"])
    names = [n for n, _ in recs]
    assert len(recs) == 5
    assert not {"A", "B", "C"} & set(names)
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommendations_unknown_input_returns_empty(model):
    assert model.get_recommendations(["Nope"]) == []


def test_recommendations_ignore_unknown_but_use_known(model):
    assert model.get_recommendations(["A", "Nope"], top_k=3)


def test_team_score(model):
    assert model.get_team_score(["A"]) == 0.0
    assert model.get_team_score(["A", "Nope"]) == 0.0
    assert model.get_team_score(["A", "B", "C"]) > model.get_team_score(["A", "X", "U"])


def test_similar_pokemon(model):
    assert model.get_similar_pokemon("Nope") == []
    sims = model.get_similar_pokemon("A", top_k=3)
    assert len(sims) == 3
    assert "A" not in [n for n, _ in sims[:1]]


def test_save_load_roundtrip(model, tmp_path):
    path = tmp_path / "m.json"
    model.save(path)
    loaded = CollaborativeFilteringModel()
    loaded.load(path)
    assert loaded.pokemon_to_idx == model.pokemon_to_idx
    assert loaded.idx_to_pokemon == model.idx_to_pokemon
    assert np.allclose(loaded.similarity_matrix, model.similarity_matrix)
    assert loaded.get_recommendations(["A", "B"]) == model.get_recommendations(["A", "B"])


def test_recommender_requires_three_inputs(model, tmp_path):
    dex = tmp_path / "dex.json"
    dex.write_text(json.dumps([{"name": "A", "sprite": "a.png"}, {"name": "B"}]))
    rec = CFTeamRecommender(model, dex)
    with pytest.raises(ValueError):
        rec.recommend(["A", "B"])
    assert rec.get_sprite("A") == "a.png"
    assert rec.get_sprite("B") == ""
    assert rec.get_sprite("zzz") == ""


def test_recommender_trios(model, tmp_path):
    dex = tmp_path / "dex.json"
    dex.write_text("[]")
    rec = CFTeamRecommender(model, dex)
    results = rec.recommend(["A", "B", "C"], top_k=3, candidate_pool_size=6)
    assert 0 < len(results) <= 3
    for r in results:
        assert len(r.trio) == 3 and len(set(r.trio)) == 3
        assert not {"A", "B", "C"} & set(r.trio)
        assert r.composite_score == pytest.approx(0.7 * r.cf_score + 0.3 * r.team_cohesion)
    comps = [r.composite_score for r in results]
    assert comps == sorted(comps, reverse=True)
