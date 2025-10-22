# Pokémon Team Recommender - Collaborative Filtering

A **collaborative filtering** approach to Pokémon team recommendations.

## 🤔 How This Differs from the ML-Hybrid Version

**[Main Project](https://github.com/yeungjosh/pokemon-team-recommender)**: Hybrid ML + Rules
- Uses domain knowledge (type charts, meta threats, roles)
- ML learns optimal feature weights
- Feature-based scoring

**This Project**: Pure Collaborative Filtering
- "Pokémon that appear together on teams should be recommended together"
- Item-item similarity based on co-occurrence
- No domain knowledge required - learns from data

## 🧠 The Approach

**Collaborative Filtering** - Like Amazon's "Customers who bought X also bought Y"

1. **Data Source**: Analyze thousands of real competitive teams
2. **Co-occurrence Matrix**: Count how often Pokémon appear together
3. **Similarity Calculation**: Compute cosine similarity between Pokémon
4. **Recommendation**: Given 3 Pokémon, recommend others that frequently co-occur with them

## 📊 Example

```
Team Database:
- Team 1: [Garchomp, Landorus-T, Rillaboom, Corviknight, Gholdengo, Kingambit]
- Team 2: [Garchomp, Landorus-T, Rillaboom, Zamazenta, Dragapult, Great Tusk]
- Team 3: [Garchomp, Kingambit, Corviknight, Gholdengo, Iron Valiant, Gliscor]
...

Co-occurrence with Garchomp:
- Landorus-T: 67% of teams
- Rillaboom: 52% of teams
- Kingambit: 48% of teams
...

Your Input: [Garchomp, Raging Bolt, Great Tusk]
→ Recommend Pokémon that frequently appear with these 3
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## 📁 Project Structure

```
pokemon-cf-recommender/
├── app.py                 # Gradio UI
├── data/
│   ├── teams.json        # Real/synthetic team data
│   └── pokedex.json      # Pokémon metadata
├── src/
│   ├── data_generator.py # Generate synthetic team data
│   ├── cf_model.py       # Collaborative filtering implementation
│   └── recommender.py    # Recommendation engine
└── models/
    └── similarity_matrix.npy  # Precomputed Pokémon similarities
```

## 🔬 Technical Details

**Algorithm**: Item-Item Collaborative Filtering
- **Similarity Metric**: Cosine Similarity
- **Data**: 1000+ synthetic competitive teams (or real Showdown data)
- **Matrix**: 15×15 Pokémon similarity matrix

**Advantages:**
- ✅ No domain knowledge needed
- ✅ Captures implicit synergies
- ✅ Learns from real team patterns

**Disadvantages:**
- ❌ Needs lots of team data
- ❌ Cold start problem (new Pokémon)
- ❌ Can't explain WHY recommendations are good

## 📚 Comparison

| Feature | Hybrid ML+Rules | Collaborative Filtering |
|---------|----------------|------------------------|
| **Data Needed** | Pokédex + Type Chart | Team Database |
| **Domain Knowledge** | Required | Not Required |
| **Explainability** | High | Low |
| **Data Efficiency** | Works with 15 Pokémon | Needs 100s of teams |
| **Novel Combos** | Can find them | Only suggests known patterns |

## 🎯 Deployed At

- **Hugging Face**: https://huggingface.co/spaces/joshuajoshy/pokemon-cf-recommender
- **GitHub**: https://github.com/yeungjosh/pokemon-cf-recommender

---

Built with ⚔️ as a comparison to the [ML-Hybrid approach](https://github.com/yeungjosh/pokemon-team-recommender)
