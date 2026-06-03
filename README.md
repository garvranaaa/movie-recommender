# 📼 RetroReel — Nostalgic Movie Recommender

An interactive movie recommendation web app built with Python and Streamlit. Powered by the **MovieLens Latest Small** dataset — 610 users, 9,742 movies, 100,836 ratings spanning classic cinema through 2018.

🔗 **Live Demo:** [movie-recommender-garvrana.streamlit.app](https://movie-recommender-garvrana.streamlit.app/)

---

## Demo

![App Demo](app_demo.png)

---

## Features

**Similar Movies Engine**  
Search any title across 9,700+ films. Hybrid similarity scoring computes results on-the-fly — no pre-built matrices sitting in memory.

**Interactive Taste Profiler**  
Rate popular classics (1–5 stars) to build a custom profile and get personalized recommendations. No user ID required — you define your own taste.

**Historic Profile Explorer**  
Browse anonymous MovieLens user profiles (1–610) and see what the original dataset participants would be recommended.

**TMDB Poster Integration**  
Movie posters pulled via TMDB API. Managed through Streamlit Secrets on deployment — visitors don't need their own key.

**Memory-Optimized for Streamlit Cloud**  
Pre-computing a full 9700×9700 similarity matrix would require ~1.5 GB RAM, which exceeds Streamlit Community Cloud's 1 GB limit. Instead, cosine similarities are computed dynamically on only the requested items using vectorized NumPy operations — keeping idle RAM under 80 MB with sub-50ms response times.

---

## How the Recommender Works

Two approaches, blended into one:

**Item-Based Collaborative Filtering (60% weight)**  
Builds a user-item rating matrix and computes cosine similarity between movies based on shared rating patterns. If users who rated *Star Wars* highly also rated *The Empire Strikes Back* highly, those films are considered similar.

**Content-Based Filtering (40% weight)**  
Each movie is represented as a 20-dimensional binary genre vector. Cosine similarity on these vectors finds thematically related films — useful for movies with few ratings where collaborative signals are weak.

**Hybrid Scoring**  
Both scores are min-max normalized then blended (60/40). Final scores are weighted by a log-scaled popularity factor and the movie's mean rating, so obscure films with a single rating don't surface above well-reviewed classics.

$$\text{score} = (0.6 \cdot \text{collab} + 0.4 \cdot \text{content}) \times \left(1 + \ln(1 + \text{count}) \cdot 0.05\right) \times \frac{\mu}{5}$$

---

## Run Locally

```bash
git clone https://github.com/garvranaaa/movie-recommender
cd movie-recommender

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate          # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

The dataset downloads automatically on first run (~6 MB).

**To enable movie posters**, create a `.streamlit/secrets.toml` file:
```toml
TMDB_API_KEY = "your_key_here"
```
Get a free key at [themoviedb.org](https://www.themoviedb.org/settings/api).

---

## Deployment Notes (Streamlit Cloud)

Add your TMDB key in the Streamlit Cloud dashboard under **Settings → Secrets**:
```toml
TMDB_API_KEY = "your_key_here"
```

No other configuration needed — the dataset downloads automatically at runtime.

---

## Stack

| Library | Purpose |
|---|---|
| Streamlit | Web app framework |
| Pandas | Data loading and manipulation |
| Scikit-learn | Cosine similarity |
| NumPy | Vectorized matrix operations |
| Requests | TMDB API calls |

**Dataset:** MovieLens Latest Small — GroupLens Research, University of Minnesota

---

## Project Structure

```
movie-recommender/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml        # TMDB key (local only, not committed)
```

---

Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) · [GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
