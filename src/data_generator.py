"""Generate synthetic team data for collaborative filtering.

Since we don't have access to thousands of real Pokémon Showdown teams,
we generate synthetic teams based on usage statistics and common archetypes.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


class TeamDataGenerator:
    """Generate realistic synthetic team data."""

    def __init__(self, pokedex_path: Path):
        with open(pokedex_path) as f:
            self.pokemon_data = json.load(f)

        self.pokemon_names = [p["name"] for p in self.pokemon_data]

        # Define common team archetypes (based on competitive knowledge)
        self.archetypes = {
            "balance": {
                "core": ["Landorus-Therian", "Corviknight", "Kingambit"],
                "flex": ["Gholdengo", "Rillaboom", "Gliscor", "Garchomp"],
            },
            "offense": {
                "core": ["Dragapult", "Iron Valiant", "Kingambit"],
                "flex": ["Great Tusk", "Raging Bolt", "Zamazenta"],
            },
            "stall": {
                "core": ["Corviknight", "Gliscor", "Gholdengo"],
                "flex": ["Rillaboom", "Kingambit", "Landorus-Therian"],
            },
            "hyper_offense": {
                "core": ["Dragapult", "Zamazenta", "Iron Valiant"],
                "flex": ["Raging Bolt", "Kyurem", "Great Tusk"],
            },
            "bulky_offense": {
                "core": ["Garchomp", "Landorus-Therian", "Rillaboom"],
                "flex": ["Corviknight", "Kingambit", "Gholdengo"],
            },
        }

    def generate_team(self, archetype: str = None) -> List[str]:
        """Generate a single 6-Pokémon team.

        Args:
            archetype: Team archetype to use, or None for random

        Returns:
            List of 6 Pokémon names
        """
        if archetype is None:
            archetype = random.choice(list(self.archetypes.keys()))

        arch_data = self.archetypes[archetype]

        # Start with 2-3 core Pokémon
        num_core = random.randint(2, min(3, len(arch_data["core"])))
        team = random.sample(arch_data["core"], num_core)

        # Fill remaining slots with flex picks
        remaining = 6 - len(team)
        available_flex = [p for p in arch_data["flex"] if p not in team]

        # 80% from archetype flex, 20% completely random
        if random.random() < 0.8 and len(available_flex) >= remaining:
            team.extend(random.sample(available_flex, remaining))
        else:
            # Mix archetype flex with random choices
            num_flex = min(remaining, len(available_flex))
            if num_flex > 0:
                team.extend(random.sample(available_flex, num_flex))

            # Fill rest randomly
            still_remaining = 6 - len(team)
            if still_remaining > 0:
                available = [p for p in self.pokemon_names if p not in team]
                team.extend(random.sample(available, still_remaining))

        return team

    def generate_dataset(
        self,
        n_teams: int = 1000,
        archetype_distribution: Dict[str, float] = None
    ) -> List[List[str]]:
        """Generate a dataset of teams.

        Args:
            n_teams: Number of teams to generate
            archetype_distribution: Distribution of archetypes (defaults to uniform)

        Returns:
            List of teams, where each team is a list of 6 Pokémon names
        """
        if archetype_distribution is None:
            # Uniform distribution
            archetype_distribution = {
                arch: 1.0 / len(self.archetypes)
                for arch in self.archetypes
            }

        teams = []
        for _ in range(n_teams):
            # Sample archetype based on distribution
            archetype = random.choices(
                list(archetype_distribution.keys()),
                weights=list(archetype_distribution.values())
            )[0]

            team = self.generate_team(archetype)
            teams.append(team)

        return teams

    def save_teams(self, teams: List[List[str]], output_path: Path):
        """Save teams to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(teams, f, indent=2)
        print(f"Saved {len(teams)} teams to {output_path}")


if __name__ == "__main__":
    # Generate training data
    generator = TeamDataGenerator(Path("data/pokedex.json"))

    # Generate 1500 teams with realistic distribution
    teams = generator.generate_dataset(
        n_teams=1500,
        archetype_distribution={
            "balance": 0.35,      # Most common
            "bulky_offense": 0.25,
            "offense": 0.20,
            "hyper_offense": 0.15,
            "stall": 0.05,        # Least common
        }
    )

    generator.save_teams(teams, Path("data/teams.json"))

    # Print some stats
    from collections import Counter
    all_pokemon = [p for team in teams for p in team]
    usage = Counter(all_pokemon)

    print("\nTop 10 most used Pokémon:")
    for pokemon, count in usage.most_common(10):
        pct = count / len(teams) * 100
        print(f"  {pokemon:25s}: {count:4d} teams ({pct:.1f}%)")
