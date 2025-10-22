"""Item-Item Collaborative Filtering for Pokémon recommendations.

Uses co-occurrence patterns in team data to recommend Pokémon that
frequently appear together.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringModel:
    """Item-item collaborative filtering model."""

    def __init__(self):
        self.pokemon_to_idx = {}
        self.idx_to_pokemon = {}
        self.co_occurrence_matrix = None
        self.similarity_matrix = None
        self.n_pokemon = 0

    def build_from_teams(self, teams: List[List[str]]):
        """Build CF model from team data.

        Args:
            teams: List of teams, where each team is a list of Pokémon names
        """
        # Get unique Pokémon
        all_pokemon = sorted(set(p for team in teams for p in team))
        self.n_pokemon = len(all_pokemon)

        # Create mappings
        self.pokemon_to_idx = {p: i for i, p in enumerate(all_pokemon)}
        self.idx_to_pokemon = {i: p for p, i in self.pokemon_to_idx.items()}

        # Build co-occurrence matrix
        self.co_occurrence_matrix = np.zeros((self.n_pokemon, self.n_pokemon))

        for team in teams:
            # For each pair of Pokémon in the team, increment co-occurrence
            indices = [self.pokemon_to_idx[p] for p in team]
            for i in indices:
                for j in indices:
                    if i != j:
                        self.co_occurrence_matrix[i][j] += 1

        # Build similarity matrix using cosine similarity
        # Normalize by row sums to get conditional probabilities
        row_sums = self.co_occurrence_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        normalized_matrix = self.co_occurrence_matrix / row_sums

        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(normalized_matrix)

        # Set diagonal to 0 (Pokémon shouldn't be similar to itself for recs)
        np.fill_diagonal(self.similarity_matrix, 0)

        print(f"Built CF model with {self.n_pokemon} Pokémon from {len(teams)} teams")

    def get_recommendations(
        self,
        input_pokemon: List[str],
        top_k: int = 10,
        exclude: List[str] = None
    ) -> List[Tuple[str, float]]:
        """Get recommendations based on input Pokémon.

        Args:
            input_pokemon: List of Pokémon names already on the team
            top_k: Number of recommendations to return
            exclude: Additional Pokémon to exclude from recommendations

        Returns:
            List of (pokemon_name, score) tuples, sorted by score descending
        """
        if exclude is None:
            exclude = []

        exclude_set = set(input_pokemon + exclude)

        # Get indices for input Pokémon
        input_indices = []
        for p in input_pokemon:
            if p in self.pokemon_to_idx:
                input_indices.append(self.pokemon_to_idx[p])
            else:
                print(f"Warning: {p} not found in training data")

        if not input_indices:
            return []

        # Aggregate similarity scores
        # For each candidate, sum similarities with input Pokémon
        aggregate_scores = np.zeros(self.n_pokemon)

        for idx in input_indices:
            aggregate_scores += self.similarity_matrix[idx]

        # Average across input Pokémon
        aggregate_scores /= len(input_indices)

        # Get recommendations
        recommendations = []
        for idx in np.argsort(-aggregate_scores):
            pokemon = self.idx_to_pokemon[idx]

            if pokemon not in exclude_set:
                score = float(aggregate_scores[idx])
                recommendations.append((pokemon, score))

                if len(recommendations) >= top_k:
                    break

        return recommendations

    def get_team_score(self, team: List[str]) -> float:
        """Score how well a team goes together (average pairwise similarity).

        Args:
            team: List of Pokémon names

        Returns:
            Average similarity score (0-1)
        """
        indices = [self.pokemon_to_idx[p] for p in team if p in self.pokemon_to_idx]

        if len(indices) < 2:
            return 0.0

        total_similarity = 0.0
        count = 0

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_similarity += self.similarity_matrix[indices[i]][indices[j]]
                count += 1

        return total_similarity / count if count > 0 else 0.0

    def get_similar_pokemon(self, pokemon: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar Pokémon to a given Pokémon.

        Args:
            pokemon: Pokémon name
            top_k: Number of similar Pokémon to return

        Returns:
            List of (pokemon_name, similarity) tuples
        """
        if pokemon not in self.pokemon_to_idx:
            return []

        idx = self.pokemon_to_idx[pokemon]
        similarities = self.similarity_matrix[idx]

        similar = []
        for sim_idx in np.argsort(-similarities)[:top_k]:
            similar_pokemon = self.idx_to_pokemon[sim_idx]
            score = float(similarities[sim_idx])
            similar.append((similar_pokemon, score))

        return similar

    def save(self, path: Path):
        """Save model to disk."""
        data = {
            "pokemon_to_idx": self.pokemon_to_idx,
            "idx_to_pokemon": self.idx_to_pokemon,
            "co_occurrence_matrix": self.co_occurrence_matrix.tolist(),
            "similarity_matrix": self.similarity_matrix.tolist(),
            "n_pokemon": self.n_pokemon,
        }

        with open(path, 'w') as f:
            json.dump(data, f)

        print(f"Model saved to {path}")

    def load(self, path: Path):
        """Load model from disk."""
        with open(path) as f:
            data = json.load(f)

        self.pokemon_to_idx = data["pokemon_to_idx"]
        self.idx_to_pokemon = {int(k): v for k, v in data["idx_to_pokemon"].items()}
        self.co_occurrence_matrix = np.array(data["co_occurrence_matrix"])
        self.similarity_matrix = np.array(data["similarity_matrix"])
        self.n_pokemon = data["n_pokemon"]

        print(f"Model loaded from {path}")
