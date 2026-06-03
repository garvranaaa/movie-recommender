import os
import re
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import urllib.request
import zipfile

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Updated genre list to support ml-latest-small (including IMAX and updated 'Children')
GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
    '(no genres listed)'
]

# ── Data ──────────────────────────────────────────────────────────────────────

def download_data():
    if not os.path.exists("ml-latest-small"):
        with st.spinner("Downloading MovieLens Latest Small dataset (first run only)..."):
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            urllib.request.urlretrieve(url, "ml-latest-small.zip")
            with zipfile.ZipFile("ml-latest-small.zip", "r") as z:
                z.extractall(".")
            os.remove("ml-latest-small.zip")


@st.cache_data
def load_data():
    # Load and rename columns to match the code's expected snake_case format
    ratings = pd.read_csv(
        "ml-latest-small/ratings.csv"
    ).rename(columns={"userId": "user_id", "movieId": "movie_id"})

    movies_raw = pd.read_csv(
        "ml-latest-small/movies.csv"
    ).rename(columns={"movieId": "movie_id"})

    # Dynamically one-hot encode the pipe-separated genres column
    for g in GENRES:
        movies_raw[g] = movies_raw["genres"].apply(lambda x: 1 if g in str(x).split("|") else 0)

    # Clean the dataset to use only essential columns
    movies = movies_raw[["movie_id", "title"] + GENRES]

    # Build User-Item sparse ratings matrix
    ratings_matrix = ratings.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)

    # Compute item similarities
    item_sim = cosine_similarity(ratings_matrix.T)
    item_sim_df = pd.DataFrame(
        item_sim,
        index=ratings_matrix.columns,
        columns=ratings_matrix.columns
    )

    # Compute content/genre similarities
    genre_matrix = movies.set_index("movie_id")[GENRES].values
    genre_sim = cosine_similarity(genre_matrix)
    genre_sim_df = pd.DataFrame(
        genre_sim,
        index=movies["movie_id"],
        columns=movies["movie_id"]
    )

    rating_stats = ratings.groupby("movie_id")["rating"].agg(["mean", "count"])

    return ratings, movies, ratings_matrix, item_sim_df, genre_sim_df, rating_stats

# ── Recommenders ──────────────────────────────────────────────────────────────

def get_hybrid_recs(movie_id, item_sim_df, genre_sim_df, movies, rating_stats, n=10):
    if movie_id not in genre_sim_df.index:
        return pd.DataFrame()

    content = genre_sim_df[movie_id].drop(movie_id, errors='ignore')
    
    # Fallback structure if the movie exists but hasn't received any ratings
    if movie_id in item_sim_df.columns:
        collab = item_sim_df[movie_id].drop(movie_id, errors='ignore')
    else:
        collab = pd.Series(0, index=content.index)

    def norm(s):
        mn, mx = s.min(), s.max()
        if mn == mx:
            return pd.Series(0, index=s.index)
        return (s - mn) / (mx - mn + 1e-8)

    collab_n = norm(collab)
    content_n = norm(content).reindex(collab_n.index, fill_value=0)
    
    # If the item has no collaborative data, weigh 100% on content similarities
    if movie_id in item_sim_df.columns:
        hybrid = (0.6 * collab_n + 0.4 * content_n).sort_values(ascending=False).head(n * 2)
    else:
        hybrid = content_n.sort_values(ascending=False).head(n * 2)

    recs = movies[movies["movie_id"].isin(hybrid.index)].copy()
    recs["score"] = recs["movie_id"].map(hybrid)

    recs = recs.merge(rating_stats, left_on="movie_id", right_index=True, how="left")
    recs["count"] = recs["count"].fillna(0)
    recs["mean"] = recs["mean"].fillna(3.0)

    recs["score"] *= (
        (1 + np.log1p(recs["count"]) * 0.05)
        * (recs["mean"] / 5)
    )

    return recs.sort_values("score", ascending=False).head(n)


def get_user_recs(user_id, ratings, ratings_matrix, item_sim_df, movies, rating_stats, n=10):
    if user_id not in ratings_matrix.index:
        return pd.DataFrame()

    user_ratings = ratings_matrix.loc[user_id]
    rated = user_ratings[user_ratings > 0]
    unrated = user_ratings[user_ratings == 0].index

    scores = {}
    sim_sums = {}

    for movie_id, rating in rated.items():
        if movie_id not in item_sim_df.columns:
            continue
        sims = item_sim_df[movie_id].reindex(unrated, fill_value=0)
        for mid, sim in sims.items():
            scores[mid] = scores.get(mid, 0) + sim * rating
            sim_sums[mid] = sim_sums.get(mid, 0) + abs(sim)

    if not scores:
        return pd.DataFrame()

    normalized = {mid: scores[mid] / (sim_sums[mid] + 1e-8) for mid in scores}

    scores_s = pd.Series(normalized).sort_values(ascending=False).head(n * 2)
    recs = movies[movies["movie_id"].isin(scores_s.index)].copy()
    recs["score"] = recs["movie_id"].map(scores_s)
    recs = recs.merge(rating_stats[["count"]], left_on="movie_id", right_index=True, how="left")

    return recs.sort_values("score", ascending=False).head(n)

# ── TMDB ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_poster(title, api_key):
    if not api_key:
        return None
    try:
        clean = title.split("(")[0].strip()
        
        # Clean up titles with trailing articles (e.g. "Avengers, The" -> "The Avengers")
        if ", The" in clean:
            clean = "The " + clean.replace(", The", "").strip()
        elif ", A" in clean:
            clean = "A " + clean.replace(", A", "").strip()
        elif ", An" in clean:
            clean = "An " + clean.replace(", An", "").strip()

        # Parse release year from dataset title format
        year_match = re.search(r'\((\d{4})\)', title)
        params = {"api_key": api_key, "query": clean}
        if year_match:
            params["year"] = year_match.group(1)

        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=5
        )
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w300{results[0]['poster_path']}"
    except Exception:
        pass
    return None


def movie_cards(recs, api_key):
    if recs.empty:
        st.warning("No recommendations found.")
        return

    cols = st.columns(5)
    for i, (_, row) in enumerate(recs.head(10).iterrows()):
        with cols[i % 5]:
            poster = get_poster(row["title"], api_key)
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(
                    "<div style='background:#2d2d2d;height:160px;border-radius:8px;"
                    "display:flex;align-items:center;justify-content:center;font-size:36px'>🎬</div>",
                    unsafe_allow_html=True
                )
            genres = [g for g in GENRES if row.get(g, 0) == 1]
            st.markdown(f"**{row['title']}**")
            if genres:
                st.caption(", ".join(genres[:3]))

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 Movie Recommender")
    st.divider()
    api_key = st.text_input(
        "TMDB API Key (optional)",
        type="password",
        help="Free key at themoviedb.org — enables movie posters"
    )
    if not api_key:
        st.info("Add a TMDB key to load movie posters")
    st.divider()
    st.caption("📊 MovieLens Latest Small\n610 users · 9,742 movies · 100,836 ratings")

# ── Load ──────────────────────────────────────────────────────────────────────

download_data()

with st.spinner("Loading data and computing similarity matrices..."):
    ratings, movies, ratings_matrix, item_sim_df, genre_sim_df, rating_stats = load_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🎯 Similar Movies", "👤 For You", "ℹ️ About"])

# Tab 1 — Movie search
with tab1:
    st.header("Find Similar Movies")
    query = st.text_input("Type a movie title", placeholder="Toy Story, Avengers, Iron Man, Titanic...")

    selected_id = None
    if query:
        matches = movies[movies["title"].str.contains(query, case=False, na=False)]

        if matches.empty:
            st.error(f"No movie found matching '{query}'")
        elif len(matches) == 1:
            selected_id = int(matches["movie_id"].values[0])
            st.success(f"**{matches['title'].values[0]}**")
        else:
            choice = st.selectbox(f"{len(matches)} matches — pick one:", matches["title"].tolist())
            selected_id = int(matches[matches["title"] == choice]["movie_id"].values[0])

    if selected_id is not None:
        with st.spinner("Computing recommendations..."):
            recs = get_hybrid_recs(selected_id, item_sim_df, genre_sim_df, movies, rating_stats)
        st.subheader("Top 10 similar movies")
        movie_cards(recs, api_key)

# Tab 2 — User recommendations
with tab2:
    st.header("Personalized Recommendations")
    st.caption("Movies you haven't seen, predicted from your rating history.")

    user_id = st.number_input("User ID (1–610)", min_value=1, max_value=610, value=1, step=1)

    if st.button("Get recommendations", type="primary"):
        rated_df = (
            ratings[ratings["user_id"] == user_id]
            .merge(movies[["movie_id", "title"]], on="movie_id")
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**User {user_id}** rated **{len(rated_df)} movies**")
            for _, r in rated_df.sort_values("rating", ascending=False).head(5).iterrows():
                st.markdown(f"{'⭐' * int(r['rating'])} {r['title']}")

        with col2:
            with st.spinner("Finding recommendations..."):
                user_recs = get_user_recs(
                    user_id, ratings, ratings_matrix, item_sim_df, movies, rating_stats
                )
            movie_cards(user_recs, api_key)

# Tab 3 — About
with tab3:
    st.header("How this works")
    st.markdown("""
    **Item-Based Collaborative Filtering**  
    Builds a user-item rating matrix. Computes cosine similarity between items — if users who liked Movie A also liked Movie B, they're marked similar.

    **Content-Based Filtering**  
    Each movie maps to a 20-dimensional binary genre vector. Cosine similarity on these vectors finds movies in the same genre space. Helps handle newer or less-rated items.

    **Hybrid (default)**  
    60% collaborative + 40% content-based. Collaborative captures user behavior trends; content ensures structural alignment.

    ---
    **Stack:** Pandas · Scikit-learn · Streamlit · TMDB API · NumPy  
    **Dataset:** MovieLens Latest (Small) — GroupLens Research, University of Minnesota

    Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) · [GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
    """)