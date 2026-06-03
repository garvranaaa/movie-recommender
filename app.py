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

GENRES = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]

# ── Data ──────────────────────────────────────────────────────────────────────

# FIX 1: download_data() is now outside @st.cache_data so the spinner
# actually renders. Previously it was buried inside load_data() where
# Streamlit can't show UI elements during a cached call.
def download_data():
    if not os.path.exists("ml-100k"):
        with st.spinner("Downloading MovieLens 100K dataset (first run only)..."):
            url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            urllib.request.urlretrieve(url, "ml-100k.zip")
            with zipfile.ZipFile("ml-100k.zip", "r") as z:
                z.extractall(".")
            os.remove("ml-100k.zip")


@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "ml-100k/u.data", sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movie_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    movies = pd.read_csv(
        "ml-100k/u.item", sep="|",
        names=movie_cols, encoding="latin-1"
    )[["movie_id", "title"] + GENRES]

    ratings_matrix = ratings.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)

    item_sim = cosine_similarity(ratings_matrix.T)
    item_sim_df = pd.DataFrame(
        item_sim,
        index=ratings_matrix.columns,
        columns=ratings_matrix.columns
    )

    genre_matrix = movies.set_index("movie_id")[GENRES].values
    genre_sim = cosine_similarity(genre_matrix)
    genre_sim_df = pd.DataFrame(
        genre_sim,
        index=movies["movie_id"],
        columns=movies["movie_id"]
    )

    # FIX 2: Keep both mean and count — mean is now actually used in scoring
    # (previously mean was computed but silently dropped in the merge)
    rating_stats = ratings.groupby("movie_id")["rating"].agg(["mean", "count"])

    return ratings, movies, ratings_matrix, item_sim_df, genre_sim_df, rating_stats

# ── Recommenders ──────────────────────────────────────────────────────────────

def get_hybrid_recs(movie_id, item_sim_df, genre_sim_df, movies, rating_stats, n=10):
    if movie_id not in item_sim_df.columns:
        return pd.DataFrame()

    collab = item_sim_df[movie_id].drop(movie_id)
    content = (
        genre_sim_df[movie_id].drop(movie_id)
        if movie_id in genre_sim_df.columns
        else pd.Series(dtype=float)
    )

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-8)

    collab_n = norm(collab)
    content_n = norm(content).reindex(collab_n.index, fill_value=0)
    hybrid = (0.6 * collab_n + 0.4 * content_n).sort_values(ascending=False).head(n * 2)

    recs = movies[movies["movie_id"].isin(hybrid.index)].copy()
    recs["score"] = recs["movie_id"].map(hybrid)

    # FIX 2 (continued): merge both columns and apply mean to scoring
    # A highly similar movie with a 4.5 avg rating beats one with 2.1
    recs = recs.merge(rating_stats, left_on="movie_id", right_index=True, how="left")
    recs["score"] *= (
        (1 + np.log1p(recs["count"].fillna(0)) * 0.05)
        * (recs["mean"].fillna(3.5) / 5)
    )

    return recs.sort_values("score", ascending=False).head(n)


def get_user_recs(user_id, ratings, ratings_matrix, item_sim_df, movies, rating_stats, n=10):
    if user_id not in ratings_matrix.index:
        return pd.DataFrame()

    user_ratings = ratings_matrix.loc[user_id]
    rated = user_ratings[user_ratings > 0]
    unrated = user_ratings[user_ratings == 0].index

    scores = {}
    sim_sums = {}  # FIX 3: track total similarity weight per candidate movie

    for movie_id, rating in rated.items():
        if movie_id not in item_sim_df.columns:
            continue
        sims = item_sim_df[movie_id].reindex(unrated, fill_value=0)
        for mid, sim in sims.items():
            scores[mid] = scores.get(mid, 0) + sim * rating
            sim_sums[mid] = sim_sums.get(mid, 0) + abs(sim)

    if not scores:
        return pd.DataFrame()

    # FIX 3: divide by sum-of-similarities so a user with 300 rated movies
    # doesn't automatically outscore a user with 20 rated movies
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

        # FIX 4: extract the year from the title string (e.g. "Toy Story (1995)")
        # and pass it as a param so TMDB returns the right film, not a remake
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
    st.caption("📊 MovieLens 100K\n943 users · 1,682 movies · 100,000 ratings")

# ── Load ──────────────────────────────────────────────────────────────────────

# FIX 1 (continued): download runs before the cached function,
# so the spinner is visible on first launch
download_data()

with st.spinner("Loading data and computing similarity matrices..."):
    ratings, movies, ratings_matrix, item_sim_df, genre_sim_df, rating_stats = load_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🎯 Similar Movies", "👤 For You", "ℹ️ About"])

# Tab 1 — Movie search
with tab1:
    st.header("Find Similar Movies")
    query = st.text_input("Type a movie title", placeholder="Toy Story, Star Wars, Titanic...")

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

    user_id = st.number_input("User ID (1–943)", min_value=1, max_value=943, value=1, step=1)

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
    Builds a 943×1682 user-item rating matrix. Computes cosine similarity between items — if users who liked Movie A also liked Movie B, they're similar.

    **Content-Based Filtering**  
    Each movie has a 19-dimensional binary genre vector. Cosine similarity on these vectors finds movies in the same genre space. Handles cases where a movie has few ratings.

    **Hybrid (default)**  
    60% collaborative + 40% content-based. Collaborative captures taste patterns; content adds genre diversity. The blend outperforms either alone.

    ---
    **Stack:** Pandas · Scikit-learn · Streamlit · TMDB API · NumPy  
    **Dataset:** MovieLens 100K — GroupLens Research, University of Minnesota

    Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) · [GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
    """)