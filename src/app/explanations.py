"""
Explanation generation for collaborative filtering recommendations.

Provides layman-friendly explanations for why specific Pokémon trios
were recommended based on co-occurrence patterns.
"""

# Thresholds for team cohesion interpretation
TEAM_COHESION_HIGH = 0.5
TEAM_COHESION_MEDIUM = 0.3
CO_OCCURRENCE_THRESHOLD = 0.3  # Minimum similarity to mention

# Thresholds for recommendation strength tiers
RECOMMENDATION_STRENGTH_HIGH = 0.7
RECOMMENDATION_STRENGTH_MEDIUM = 0.5


def generate_cf_explanation(
    user_team: list[str],
    recommended_trio: list[str],
    cf_score: float,
    team_cohesion: float,
    cf_model,
) -> str:
    """Generate explanation for CF recommendation.

    Args:
        user_team: User's 3 Pokémon
        recommended_trio: Recommended 3 Pokémon
        cf_score: CF similarity score
        team_cohesion: Team cohesion score
        cf_model: CollaborativeFilteringModel instance

    Returns:
        Markdown-formatted explanation
    """
    explanation = "## Why these Pokémon?\n\n"

    # Explain co-occurrence patterns
    explanation += "### 📊 Team Pattern Analysis\n\n"
    explanation += "These Pokémon frequently appear together on successful teams:\n\n"

    # Find which recommended Pokémon have high similarity with user team
    high_similarity_pairs = []
    for user_mon in user_team:
        for rec_mon in recommended_trio:
            if user_mon in cf_model.pokemon_to_idx and rec_mon in cf_model.pokemon_to_idx:
                idx1 = cf_model.pokemon_to_idx[user_mon]
                idx2 = cf_model.pokemon_to_idx[rec_mon]
                sim = cf_model.similarity_matrix[idx1][idx2]
                if sim > CO_OCCURRENCE_THRESHOLD:
                    high_similarity_pairs.append((user_mon, rec_mon, sim))

    # Sort by similarity
    high_similarity_pairs.sort(key=lambda x: x[2], reverse=True)

    if high_similarity_pairs:
        for user_mon, rec_mon, sim in high_similarity_pairs[:3]:
            pct = sim * 100
            explanation += f"- **{rec_mon}** appears with **{user_mon}** in {pct:.0f}% of teams\n"
    else:
        explanation += "- This trio complements your team based on overall team patterns\n"

    explanation += "\n"

    # Explain team cohesion
    explanation += "### ⚔️ Team Synergy\n\n"
    if team_cohesion > TEAM_COHESION_HIGH:
        explanation += "This combination creates a **highly cohesive team** - these 6 Pokémon work well together based on historical patterns.\n"
    elif team_cohesion > TEAM_COHESION_MEDIUM:
        explanation += "This combination creates a **well-balanced team** with good synergy.\n"
    else:
        explanation += "This combination provides **diverse coverage** with complementary roles.\n"

    explanation += "\n"

    # Overall recommendation strength
    explanation += "### 💡 Recommendation Strength\n\n"
    composite_score = 0.7 * cf_score + 0.3 * team_cohesion

    if composite_score > RECOMMENDATION_STRENGTH_HIGH:
        explanation += "⭐⭐⭐ **Highly Recommended** - This trio has proven patterns on competitive teams.\n"
    elif composite_score > RECOMMENDATION_STRENGTH_MEDIUM:
        explanation += "⭐⭐ **Recommended** - This trio shows strong co-occurrence patterns.\n"
    else:
        explanation += "⭐ **Viable** - This trio is worth considering based on team data.\n"

    return explanation
