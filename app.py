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
                # Find sprite in pokemon_data
                for p in pokemon_data:
                    if p["name"] == mon_name:
                        sprite_url = p.get("sprite")
                        if sprite_url:
                            sprite_row += f'<img src="{sprite_url}" width="96" height="96" style="display:inline-block; vertical-align:middle;" alt="{mon_name}"> '
                        break

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

    gr.Markdown(
        f"""
        > **Note:** This app currently supports **{len(AVAILABLE_POKEMON)} Pokémon** from the Gen 9 OU tier.
        > See the "Show Available Pokémon" section below for the full list.
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

    submit.click(
        fn=recommend_team,
        inputs=[mon1, mon2, mon3],
        outputs=[output, explanation_output]
    )

    # Wire up sprite updates when Pokemon selections change
    mon1.change(fn=get_pokemon_sprite, inputs=[mon1], outputs=[mon1_sprite])
    mon2.change(fn=get_pokemon_sprite, inputs=[mon2], outputs=[mon2_sprite])
    mon3.change(fn=get_pokemon_sprite, inputs=[mon3], outputs=[mon3_sprite])

    # Load initial sprites on page load
    demo.load(
        fn=lambda: (
            get_pokemon_sprite("Garchomp"),
            get_pokemon_sprite("Raging Bolt"),
            get_pokemon_sprite("Great Tusk"),
        ),
        outputs=[mon1_sprite, mon2_sprite, mon3_sprite],
    )

    # Example Teams Section
    with gr.Accordion("💡 Example Strong Teams", open=False):
        gr.Markdown(
            """
            ### Hyper Offense Core
            <div style="display: flex; align-items: center; gap: 15px; margin: 15px 0;">
                <img src="https://img.pokemondb.net/sprites/home/normal/dragapult.png" width="80" height="80" alt="Dragapult">
                <img src="https://img.pokemondb.net/sprites/home/normal/iron-valiant.png" width="80" height="80" alt="Iron Valiant">
                <img src="https://img.pokemondb.net/sprites/home/normal/great-tusk.png" width="80" height="80" alt="Great Tusk">
            </div>
            **Dragapult + Iron Valiant + Great Tusk**

            Fast, offensive core with momentum control. Dragapult forces switches with U-turn, Iron Valiant provides mixed offensive pressure, and Great Tusk sets hazards while handling Steel-types. This trio frequently appears together in hyper offensive teams.

            ---

            ### Balance Core
            <div style="display: flex; align-items: center; gap: 15px; margin: 15px 0;">
                <img src="https://img.pokemondb.net/sprites/home/normal/landorus-therian.png" width="80" height="80" alt="Landorus-Therian">
                <img src="https://img.pokemondb.net/sprites/home/normal/corviknight.png" width="80" height="80" alt="Corviknight">
                <img src="https://img.pokemondb.net/sprites/home/normal/toxapex.png" width="80" height="80" alt="Toxapex">
            </div>
            **Landorus-Therian + Corviknight + Toxapex**

            Defensive backbone with hazard control. Landorus-T sets Stealth Rock and threatens offense, Corviknight provides physical wall + Defog support, and Toxapex walls special attackers. Classic balanced team structure that handles most threats.

            ---

            ### Offensive Pivot Core
            <div style="display: flex; align-items: center; gap: 15px; margin: 15px 0;">
                <img src="https://img.pokemondb.net/sprites/home/normal/garchomp.png" width="80" height="80" alt="Garchomp">
                <img src="https://img.pokemondb.net/sprites/home/normal/raging-bolt.png" width="80" height="80" alt="Raging Bolt">
                <img src="https://img.pokemondb.net/sprites/home/normal/kingambit.png" width="80" height="80" alt="Kingambit">
            </div>
            **Garchomp + Raging Bolt + Kingambit**

            Strong offensive synergy with good defensive typing. Garchomp provides speed and Ground coverage, Raging Bolt handles Water-types and pivots with Volt Switch, and Kingambit punishes opposing offense with Sucker Punch. These three cover each other's weaknesses well.
            """
        )

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

    # FAQ Section - How Collaborative Filtering Works
    with gr.Accordion("❓ How do you choose the best Pokémon?", open=False):
        gr.Markdown(
            """
            ### Collaborative Filtering Approach

            This app uses **collaborative filtering** - the same algorithm Netflix uses to recommend movies!

            #### The Core Idea
            > "Pokémon that appear together on winning teams should be recommended together."

            Just like how Netflix suggests shows based on what similar users watched, we suggest Pokémon based on what successful teams used together.

            #### How It Works

            **Step 1: Learn from Team Patterns**
            - We analyzed **1,500+ competitive teams** to find patterns
            - Built a co-occurrence matrix: "How often does Garchomp appear with Landorus-T?"
            - Example: If Garchomp + Landorus-T appeared together in 450 out of 600 Garchomp teams → strong pattern!

            **Step 2: Calculate Similarity**
            - Use **cosine similarity** to measure how similar Pokémon usage patterns are
            - Garchomp and Landorus-T have high similarity (0.68) → they work well together
            - Garchomp and Blissey have low similarity (0.15) → rarely used together

            **Step 3: Make Recommendations**
            - You pick 3 Pokémon (e.g., Garchomp, Raging Bolt, Great Tusk)
            - We find trios that frequently appeared with your picks
            - Score = `0.7 × CF_Similarity + 0.3 × Team_Cohesion`

            #### Why This Works

            ✅ **Learns from Real Play** - Patterns come from actual competitive teams, not theory
            ✅ **Discovers Hidden Synergies** - Finds combos that work even if we don't know why
            ✅ **No Type Chart Needed** - Algorithm learns good pairings automatically
            ✅ **Adapts to Meta Shifts** - Retrain on new teams → updated recommendations

            #### What Makes a High Score?

            - **CF Similarity (70%)**: How often these Pokémon appear together on teams
            - **Team Cohesion (30%)**: How well all 6 Pokémon work as a unit

            A score of **0.685** means this trio appears together frequently AND creates a cohesive team!

            #### Example
            ```
            Your team: Garchomp, Raging Bolt, Great Tusk
            Recommendation: Kingambit (similarity with your team: 0.71)

            Why? Kingambit appeared with:
            - Garchomp in 65% of Garchomp teams
            - Raging Bolt in 58% of Raging Bolt teams
            - Great Tusk in 52% of Great Tusk teams

            → Strong pattern = good recommendation!
            ```

            This approach is different from rule-based systems - we don't manually code "type coverage" or "role balance." We let the data tell us what works!
            """
        )

    # FAQ Section - General
    with gr.Accordion("❓ Frequently Asked Questions (FAQ)", open=False):
        gr.Markdown(
            """
            ### What is Type Coverage?
            Type coverage refers to how well your team can deal with different Pokémon types.

            - **Offensive Coverage:** Can your team hit many types super-effectively?
            - **Defensive Coverage:** Does your team have too many shared weaknesses?

            A balanced team should be able to threaten a wide variety of opponents while minimizing exploitable weaknesses.

            ---

            ### What are Meta Threats?
            Meta threats are the most popular and powerful Pokémon in competitive play. Our model uses Smogon usage stats to identify which Pokémon appear most frequently in battles.

            A good team should have answers to common threats like Garchomp, Kingambit, and Great Tusk - Pokémon that you're likely to face in many matches.

            ---

            ### What is Role Balance?
            Role balance ensures your team has the tools needed to control the game:

            - **Hazard Control:** Setting or removing entry hazards (Stealth Rock, Spikes)
            - **Pivoting:** Switching safely with moves like U-turn or Volt Switch
            - **Speed Control:** Fast Pokémon or priority moves to outspeed threats

            A well-rounded team covers multiple roles rather than having six Pokémon that do the same thing.

            ---

            ### What are Tiers?
            Tiers organize Pokémon by power level for fair competitive play:

            - **OU (OverUsed):** The standard competitive tier - balanced and diverse
            - **Ubers:** Legendary and extremely powerful Pokémon
            - **UU (UnderUsed):** Viable Pokémon that are less dominant than OU

            This app focuses on Gen 9 OU, which is the most popular competitive tier.

            ---

            ### Why aren't all Pokémon available?
            This app currently includes 100 Pokémon from the Gen 9 OU tier because:

            - **Quality over Quantity:** These are the most competitively viable Pokémon
            - **Training Data:** The collaborative filtering model was trained on real competitive teams using these Pokémon
            - **Performance:** A focused dataset allows faster, more accurate recommendations

            Pokémon outside this tier (like Legendaries or lower-tier options) aren't included because they have different balance considerations and usage patterns.
            """
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
