"""Team recommendation engine using collaborative filtering."""

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List

from src.cf_model import CollaborativeFilteringModel


@dataclass
class Recommendation:
    """A recommended trio with CF score."""

    trio: List[str]
    cf_score: float  # Collaborative filtering score
    team_cohesion: float  # How well the full 6-Pokemon team works together

    @property
    def composite_score(self) -> float:
        """Combined score for ranking."""
        return 0.7 * self.cf_score + 0.3 * self.team_cohesion


class CFTeamRecommender:
    """Recommend teams using collaborative filtering."""

    def __init__(self, cf_model: CollaborativeFilteringModel, pokedex_path: Path):
        self.cf_model = cf_model

        # Load Pokémon metadata for sprites
        with open(pokedex_path) as f:
            pokemon_data = json.load(f)
            self.pokemon_sprites = {p["name"]: p.get("sprite", "") for p in pokemon_data}

    def recommend(
        self,
        input_team: List[str],
        top_k: int = 5,
        candidate_pool_size: int = 50,
    ) -> List[Recommendation]:
        """Recommend trios to complete the input team.

        Args:
            input_team: List of 3 Pokémon names
            top_k: Number of recommendations to return
            candidate_pool_size: Number of candidates to consider

        Returns:
            List of top K recommendations
        """
        if len(input_team) != 3:
            raise ValueError(f"Input must be exactly 3 Pokémon (got {len(input_team)})")

        # Get candidate Pokémon based on CF similarity
        candidates = self.cf_model.get_recommendations(
            input_team,
            top_k=candidate_pool_size,
            exclude=input_team
        )

        candidate_names = [name for name, score in candidates]

        # Evaluate all possible trios
        recommendations = []

        for trio in combinations(candidate_names, 3):
            full_team = input_team + list(trio)

            # CF score: average similarity of trio Pokémon to input team
            trio_scores = []
            for mon in trio:
                # Get similarity to each input Pokémon
                similarities = []
                for input_mon in input_team:
                    if mon in self.cf_model.pokemon_to_idx and input_mon in self.cf_model.pokemon_to_idx:
                        idx1 = self.cf_model.pokemon_to_idx[mon]
                        idx2 = self.cf_model.pokemon_to_idx[input_mon]
                        sim = self.cf_model.similarity_matrix[idx1][idx2]
                        similarities.append(sim)

                if similarities:
                    trio_scores.append(sum(similarities) / len(similarities))

            cf_score = sum(trio_scores) / len(trio_scores) if trio_scores else 0.0

            # Team cohesion: how well all 6 Pokémon work together
            team_cohesion = self.cf_model.get_team_score(full_team)

            rec = Recommendation(
                trio=list(trio),
                cf_score=cf_score,
                team_cohesion=team_cohesion,
            )
            recommendations.append(rec)

        # Sort by composite score
        recommendations.sort(key=lambda x: x.composite_score, reverse=True)

        return recommendations[:top_k]

    def get_sprite(self, pokemon_name: str) -> str:
        """Get sprite URL for a Pokémon."""
        return self.pokemon_sprites.get(pokemon_name, "")
