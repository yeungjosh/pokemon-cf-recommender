"""Add sprite URLs to pokedex.json for all Pokemon.

Uses Pokemon DB sprite URLs in format:
https://img.pokemondb.net/sprites/home/normal/{name}.png
"""

import json
from pathlib import Path


def pokemon_name_to_sprite_slug(name: str) -> str:
    """Convert Pokemon name to sprite URL slug.

    Examples:
        "Garchomp" -> "garchomp"
        "Landorus-Therian" -> "landorus-therian"
        "Ogerpon-Wellspring" -> "ogerpon-wellspring"
        "Samurott-Hisui" -> "samurott-hisuian"
        "Iron Valiant" -> "iron-valiant"
    """
    # Special case mappings
    special_cases = {
        "Samurott-Hisui": "samurott-hisuian",
        "Oinkologne-F": "oinkologne-female",
    }

    if name in special_cases:
        return special_cases[name]

    # Default: lowercase, replace spaces with hyphens
    slug = name.lower().replace(" ", "-")
    return slug


def add_sprites_to_pokedex():
    """Add sprite URLs to all Pokemon in pokedex.json."""
    pokedex_path = Path(__file__).parents[1] / "data" / "pokedex.json"

    with open(pokedex_path) as f:
        pokemon_data = json.load(f)

    print(f"Processing {len(pokemon_data)} Pokemon...")

    for entry in pokemon_data:
        name = entry["name"]
        slug = pokemon_name_to_sprite_slug(name)
        sprite_url = f"https://img.pokemondb.net/sprites/home/normal/{slug}.png"
        entry["sprite"] = sprite_url
        print(f"  {name} -> {sprite_url}")

    # Write back to file with nice formatting
    with open(pokedex_path, "w") as f:
        json.dump(pokemon_data, f, indent=2)

    print(f"\n✅ Added sprites to {len(pokemon_data)} Pokemon")


if __name__ == "__main__":
    add_sprites_to_pokedex()
