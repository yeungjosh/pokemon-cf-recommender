"""
Pokémon Team Recommender - Collaborative Filtering Approach

Uses team co-occurrence patterns to recommend Pokémon combinations.
"""

import json
import logging
from pathlib import Path

import gradio as gr

from src.app.explanations import generate_cf_explanation
from src.cf_model import CollaborativeFilteringModel
from src.recommender import CFTeamRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize model
print("Loading collaborative filtering model...")
cf_model = CollaborativeFilteringModel()

# Check if pre-trained model exists
model_path = Path("models/cf_model.json")
if model_path.exists():
    cf_model.load(model_path)
else:
    # Train model from scratch
    print("No pre-trained model found. Training new model...")
    from src.data_generator import TeamDataGenerator

    generator = TeamDataGenerator(Path("data/pokedex.json"))
    teams = generator.generate_dataset(n_teams=1500)

    # Save teams
    teams_path = Path("data/teams.json")
    generator.save_teams(teams, teams_path)

    # Train model
    cf_model.build_from_teams(teams)

    # Save model
    model_path.parent.mkdir(exist_ok=True, parents=True)
    cf_model.save(model_path)

# Initialize recommender
recommender = CFTeamRecommender(cf_model, Path("data/pokedex.json"))

# Get available Pokémon
with open("data/pokedex.json") as f:
    pokemon_data = json.load(f)
    AVAILABLE_POKEMON = sorted([p["name"] for p in pokemon_data])

print("Model loaded successfully!")


def get_pokemon_sprite(mon_name: str) -> str:
    """Get HTML for displaying a Pokemon sprite."""
    if not mon_name:
        return ""

    # Find pokemon in data
    for p in pokemon_data:
        if p["name"] == mon_name:
            sprite_url = p.get("sprite")
            if sprite_url:
                return f'<div style="text-align: center; margin: 10px 0;"><img src="{sprite_url}" width="96" height="96" alt="{mon_name}"></div>'
    return ""


def get_full_team_display(mon1: str, mon2: str, mon3: str) -> str:
    """Generate HTML display for the complete team."""
    if not all([mon1, mon2, mon3]):
        return ""

    team_html = '<div style="text-align: center; margin: 20px 0;">'
    team_html += '<h3 style="margin-bottom: 15px;">Your Complete Team</h3>'
    team_html += '<div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">'

    for mon_name in [mon1, mon2, mon3]:
        # Find pokemon in data
        for p in pokemon_data:
            if p["name"] == mon_name:
                sprite_url = p.get("sprite")
                if sprite_url:
                    team_html += f'''
                    <div style="text-align: center;">
                        <img src="{sprite_url}" width="96" height="96" alt="{mon_name}">
                        <p style="margin-top: 5px; font-weight: bold;">{mon_name}</p>
                    </div>
                    '''
                break

    team_html += '</div></div>'
    return team_html


def recommend_team(mon1: str, mon2: str, mon3: str) -> tuple[str, str]:
    """Generate team recommendations using collaborative filtering."""
    if not all([mon1, mon2, mon3]):
        return "❌ Please select all 3 Pokémon.", ""

    input_team = [mon1.strip(), mon2.strip(), mon3.strip()]

    # Validate Pokemon names against available list
    unknown = [m for m in input_team if m not in AVAILABLE_POKEMON]
    if unknown:
        suggestion = "❌ Unknown Pokémon: " + ", ".join(unknown) + "\n\n"
        suggestion += "**Try these instead:**\n"
        suggestion += ", ".join(AVAILABLE_POKEMON[:8]) + ", ..."
        suggestion += f"\n\n*({len(AVAILABLE_POKEMON)} total)*"
        return suggestion, ""

    try:
        recommendations = recommender.recommend(input_team, top_k=5, candidate_pool_size=12)
    except ValueError as e:
        error_msg = str(e)
        suggestion = f"❌ {error_msg}\n\n"
        suggestion += "**Available Pokémon in Gen 9 OU:**\n"
        suggestion += ", ".join(AVAILABLE_POKEMON[:8]) + ", ..."
        suggestion += f"\n\n*({len(AVAILABLE_POKEMON)} total)*"
        return suggestion, ""
    except Exception:
        logging.exception("Unexpected error in recommend_team")
        return "❌ Unexpected error.\n\nPlease check your selections and try again.", ""
    else:
        if not recommendations:
            return "No recommendations found. Try different Pokémon!", ""

        # Generate explanation for top recommendation
        top_rec = recommendations[0]
        explanation = generate_cf_explanation(
            user_team=input_team,
            recommended_trio=top_rec.trio,
            cf_score=top_rec.cf_score,
            team_cohesion=top_rec.team_cohesion,
            cf_model=cf_model,
        )

        # Format results with sprites
        result = f"## Top {len(recommendations)} Recommendations\n\n"

        for i, rec in enumerate(recommendations, 1):
            result += f"### #{i} - Score: {rec.composite_score:.3f}\n\n"

            # Add sprites for the trio
            sprite_row = ""
            for mon_name in rec.trio:
                sprite_url = recommender.get_sprite(mon_name)
                if sprite_url:
                    sprite_row += f'<img src="{sprite_url}" width="96" height="96" style="display:inline-block; vertical-align:middle;" alt="{mon_name}"> '

            if sprite_row:
                result += f'{sprite_row}\n\n'

            result += f"**Trio:** {', '.join(rec.trio)}\n\n"
            result += f"**Breakdown:**\n"
            result += f"- CF Similarity: {rec.cf_score:.3f}\n"
            result += f"- Team Cohesion: {rec.team_cohesion:.3f}\n\n"
            result += "---\n\n"

        return result, explanation


with gr.Blocks(title="Pokémon Team Recommender - Collaborative Filtering") as demo:
    gr.Markdown(
        """
        # ⚔️ Pokémon Team Recommender (Collaborative Filtering)

        Complete your competitive team with **collaborative filtering** recommendations.

        **How it works:** Analyzes 1,500+ teams to find Pokémon that frequently appear together → Recommends trios based on co-occurrence patterns.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Your Team")
            mon1 = gr.Dropdown(
                label="Pokémon 1",
                choices=AVAILABLE_POKEMON,
                value="Garchomp",
                allow_custom_value=True,
            )
            mon1_sprite = gr.HTML()

            mon2 = gr.Dropdown(
                label="Pokémon 2",
                choices=AVAILABLE_POKEMON,
                value="Raging Bolt",
                allow_custom_value=True,
            )
            mon2_sprite = gr.HTML()

            mon3 = gr.Dropdown(
                label="Pokémon 3",
                choices=AVAILABLE_POKEMON,
                value="Great Tusk",
                allow_custom_value=True,
            )
            mon3_sprite = gr.HTML()

            submit = gr.Button("Get Recommendations", variant="primary")

        with gr.Column():
            gr.Markdown("### Recommendations")
            output = gr.Markdown()

            # Add explanation accordion
            with gr.Accordion("❓ How did you choose these?", open=False):
                explanation_output = gr.Markdown()

    gr.Examples(
        examples=[
            ["Garchomp", "Raging Bolt", "Great Tusk"],
            ["Dragapult", "Kingambit", "Gholdengo"],
            ["Landorus-Therian", "Corviknight", "Rillaboom"],
        ],
        inputs=[mon1, mon2, mon3],
    )

    submit.click(
        fn=recommend_team,
        inputs=[mon1, mon2, mon3],
        outputs=[output, explanation_output]
    )

    # Wire up sprite updates when Pokemon selections change
    mon1.change(fn=get_pokemon_sprite, inputs=[mon1], outputs=[mon1_sprite])
    mon2.change(fn=get_pokemon_sprite, inputs=[mon2], outputs=[mon2_sprite])
    mon3.change(fn=get_pokemon_sprite, inputs=[mon3], outputs=[mon3_sprite])

    # Add "Show Available Pokémon" section
    with gr.Accordion("📋 Show Available Pokémon", open=False):
        available_list = "\n".join([f"- {mon}" for mon in AVAILABLE_POKEMON])
        gr.Markdown(
            f"""
            ### All {len(AVAILABLE_POKEMON)} Pokémon in dataset:

            {available_list}

            *Note: Only these Pokémon are available for team building and recommendations.*
            """
        )

    # Display complete team at bottom
    gr.Markdown("---")
    team_display = gr.HTML()

    # Update team display when any Pokemon selection changes
    for dropdown in [mon1, mon2, mon3]:
        dropdown.change(
            fn=get_full_team_display,
            inputs=[mon1, mon2, mon3],
            outputs=[team_display],
        )

    # Load initial sprites and team display on page load
    demo.load(
        fn=lambda: (
            get_pokemon_sprite("Garchomp"),
            get_pokemon_sprite("Raging Bolt"),
            get_pokemon_sprite("Great Tusk"),
            get_full_team_display("Garchomp", "Raging Bolt", "Great Tusk"),
        ),
        outputs=[mon1_sprite, mon2_sprite, mon3_sprite, team_display],
    )

    gr.Markdown(
        """
        ---
        **Algorithm:** Item-Item Collaborative Filtering with Cosine Similarity

        **Training Data:** 1,500 synthetic teams based on competitive archetypes

        **Score Formula:** `0.7×CF_Similarity + 0.3×Team_Cohesion`

        **Data Sources:** [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) • [Smogon Stats](https://www.smogon.com/stats/) (Oct 2024)
        """
    )


if __name__ == "__main__":
    demo.launch()
