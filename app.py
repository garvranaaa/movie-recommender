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

# Retro / Nostalgia Page Configuration
st.set_page_config(page_title="RetroReel Archive", page_icon="📼", layout="wide")

GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
    '(no genres listed)'
]

# ── Data ──────────────────────────────────────────────────────────────────────

def download_data():
    if not os.path.exists("ml-latest-small"):
        with st.spinner("Rewinding the tape... Downloading the RetroReel Database..."):
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            urllib.request.urlretrieve(url, "ml-latest-small.zip")
            with zipfile.ZipFile("ml-latest-small.zip", "r") as z:
                z.extractall(".")
            os.remove("ml-latest-small.zip")


@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "ml-latest-small/ratings.csv"
    ).rename(columns={"userId": "user_id", "movieId": "movie_id"})

    movies_raw = pd.read_csv(
        "ml-latest-small/movies.csv"
    ).rename(columns={"movieId": "movie_id"})

    # One-hot encode genres dynamically
    for g in GENRES:
        movies_raw[g] = movies_raw["genres"].apply(lambda x: 1 if g in str(x).split("|") else 0)

    movies = movies_raw[["movie_id", "title"] + GENRES]

    # Sparse user-movie matrix
    ratings_matrix = ratings.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)

    # Genre lookup index
    movies_genres_set = movies.set_index("movie_id")[GENRES]

    rating_stats = ratings.groupby("movie_id")["rating"].agg(["mean", "count"])

    return ratings, movies, ratings_matrix, movies_genres_set, rating_stats

# ── Recommenders (Optimized Dynamic Engine to prevent crashes) ───────────────

def get_hybrid_recs(movie_id, ratings_matrix, movies_genres_set, movies, rating_stats, n=10):
    if movie_id not in movies_genres_set.index:
        return pd.DataFrame()

    movie_genre_vec = movies_genres_set.loc[movie_id].values.reshape(1, -1)
    all_genre_vecs = movies_genres_set.values
    content_sims = cosine_similarity(movie_genre_vec, all_genre_vecs).flatten()
    content = pd.Series(content_sims, index=movies_genres_set.index).drop(movie_id, errors='ignore')
    
    if movie_id in ratings_matrix.columns:
        movie_collab_vec = ratings_matrix[movie_id].values.reshape(1, -1)
        all_collab_vecs = ratings_matrix.values.T
        collab_sims = cosine_similarity(movie_collab_vec, all_collab_vecs).flatten()
        collab = pd.Series(collab_sims, index=ratings_matrix.columns).drop(movie_id, errors='ignore')
    else:
        collab = pd.Series(0, index=content.index)

    def norm(s):
        mn, mx = s.min(), s.max()
        if mn == mx:
            return pd.Series(0, index=s.index)
        return (s - mn) / (mx - mn + 1e-8)

    collab_n = norm(collab)
    content_n = norm(content).reindex(collab_n.index, fill_value=0)
    
    if movie_id in ratings_matrix.columns:
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


def get_user_recs(ratings_dict, ratings_matrix, movies, rating_stats, n=10):
    if not ratings_dict:
        return pd.DataFrame()

    rated_ids = list(ratings_dict.keys())
    # Ensure selected IDs are strictly inside our ratings matrix columns
    valid_rated_ids = [mid for mid in rated_ids if mid in ratings_matrix.columns]
    
    if not valid_rated_ids:
        return pd.DataFrame()

    rated_series = pd.Series({mid: ratings_dict[mid] for mid in valid_rated_ids})
    all_movie_ids = ratings_matrix.columns
    unrated = [mid for mid in all_movie_ids if mid not in valid_rated_ids]

    if not unrated:
        return pd.DataFrame()

    # Optimized dynamic item-item calculations based on new user profile
    rated_vectors = ratings_matrix[valid_rated_ids].values.T
    unrated_vectors = ratings_matrix[unrated].values.T
    
    sim_matrix = cosine_similarity(rated_vectors, unrated_vectors)
    ratings_arr = rated_series.values.reshape(-1, 1)
    
    scores_arr = (sim_matrix * ratings_arr).sum(axis=0)
    sim_sums_arr = np.abs(sim_matrix).sum(axis=0) + 1e-8
    norm_scores = scores_arr / sim_sums_arr
    
    scores_s = pd.Series(norm_scores, index=unrated).sort_values(ascending=False).head(n * 2)
    
    recs = movies[movies["movie_id"].isin(scores_s.index)].copy()
    recs["score"] = recs["movie_id"].map(scores_s)
    recs = recs.merge(rating_stats[["count"]], left_on="movie_id", right_index=True, how="left")

    return recs.sort_values("score", ascending=False).head(n)

# ── TMDB Poster Downloader ───────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_poster(title, api_key):
    if not api_key:
        return None
    try:
        clean = title.split("(")[0].strip()
        
        if ", The" in clean:
            clean = "The " + clean.replace(", The", "").strip()
        elif ", A" in clean:
            clean = "A " + clean.replace(", A", "").strip()
        elif ", An" in clean:
            clean = "An " + clean.replace(", An", "").strip()

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
        st.warning("The projectionist couldn't find any recommendations in the archive.")
        return

    cols = st.columns(5)
    for i, (_, row) in enumerate(recs.head(10).iterrows()):
        with cols[i % 5]:
            poster = get_poster(row["title"], api_key)
            if poster:
                st.image(poster, width="stretch")
            else:
                st.markdown(
                    "<div style='background:#1e1e1e;height:240px;border-radius:8px;"
                    "display:flex;align-items:center;justify-content:center;font-size:48px;"
                    "border: 2px dashed #444;'>📼</div>",
                    unsafe_allow_html=True
                )
            genres = [g for g in GENRES if row.get(g, 0) == 1 and g != '(no genres listed)']
            st.markdown(f"**{row['title']}**")
            if genres:
                st.caption(", ".join(genres[:3]))

# ── Sidebar & Secrets ─────────────────────────────────────────────────────────

# Checks for Streamlit Community Cloud secret keys
api_key = None
try:
    if "TMDB_API_KEY" in st.secrets:
        api_key = st.secrets["TMDB_API_KEY"]
except Exception:
    pass

with st.sidebar:
    st.title("📼 RetroReel")
    st.subheader("Nostalgic Movie Archive")
    st.divider()
    
    if api_key:
        st.success("🔑 Poster Database Connected")
    else:
        st.warning("⚠️ Poster Database Disconnected (Configure TMDB_API_KEY in Secrets)")
            
    st.divider()
    st.caption("📻 **Nostalgia Archive Statistics**\n610 classic users\n9,742 nostalgic movies\n100,836 rating records (up to 2018)")

# ── Load ──────────────────────────────────────────────────────────────────────

download_data()

with st.spinner("Dusting off the archives and loading data..."):
    ratings, movies, ratings_matrix, movies_genres_set, rating_stats = load_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🎯 Similar Classics", "👤 Your Nostalgic Taste", "ℹ️ Archive Blueprint"])

# Tab 1 — Movie search
with tab1:
    st.header("Find Similar Classics")
    st.caption("Search across legendary cinema milestones, 90s hits, and nostalgic favorites released up to 2018.")
    
    query = st.text_input(
        "Type a nostalgic film title", 
        placeholder="Toy Story (1995), Avengers (2012), Interstellar (2014), Iron Man (2008)..."
    )

    selected_id = None
    if query:
        matches = movies[movies["title"].str.contains(query, case=False, na=False)]

        if matches.empty:
            st.error(f"No movie found matching '{query}' in our pre-2018 archive.")
        elif len(matches) == 1:
            selected_id = int(matches["movie_id"].values[0])
            st.success(f"**📼 Now Playing: {matches['title'].values[0]}**")
        else:
            choice = st.selectbox(f"{len(matches)} retro matches — select yours:", matches["title"].tolist())
            selected_id = int(matches[matches["title"] == choice]["movie_id"].values[0])

    if selected_id is not None:
        with st.spinner("Spooling identical recommendation tape..."):
            recs = get_hybrid_recs(selected_id, ratings_matrix, movies_genres_set, movies, rating_stats)
        st.subheader("Top 10 similar favorites recommended for you")
        movie_cards(recs, api_key)

# Tab 2 — Interactive User Profile Builder (Fixes the UX Flaw)
with tab2:
    st.header("Your Personal Retro Reel Profile")
    st.caption("Build your own cinematic spool! Rate a few classic movies you have seen to get custom recommendations tailored specifically to your taste.")

    # Select mode: Live Interactive Profile vs Offline demo
    profile_mode = st.radio(
        "Choose profile setup method:",
        ["🎨 Rate Popular Classics (Build New Profile)", "🔎 Explore Historic Database Profiles (Demo)"],
        horizontal=True
    )

    if profile_mode == "🎨 Rate Popular Classics (Build New Profile)":
        # Pull 30 of the most rated movies in our archive to present as choices
        popular_ids = rating_stats.sort_values("count", ascending=False).head(30).index
        popular_movies = movies[movies["movie_id"].isin(popular_ids)]

        selected_classics = st.multiselect(
            "Which of these classic movies have you watched? (Choose at least 3 for best results)",
            options=popular_movies["title"].tolist(),
            default=["Toy Story (1995)", "Matrix, The (1999)", "Forrest Gump (1994)"]
        )

        if selected_classics:
            st.markdown("### Rate Your Choices (1 to 5 Stars):")
            custom_ratings = {}
            
            # Render a grid / rating options
            for title in selected_classics:
                mid = int(movies[movies["title"] == title]["movie_id"].values[0])
                col_title, col_stars = st.columns([3, 1])
                with col_title:
                    st.markdown(f"🎞️ **{title}**")
                with col_stars:
                    rating = st.selectbox(
                        "Stars", [5, 4, 3, 2, 1], 
                        key=f"user_rate_{mid}", 
                        help=f"Rate {title}"
                    )
                    custom_ratings[mid] = rating
            
            if st.button("Generate My Custom Recommendation Spool", type="primary"):
                with st.spinner("Mapping your unique taste to historic rating profiles..."):
                    user_recs = get_user_recs(custom_ratings, ratings_matrix, movies, rating_stats)
                if not user_recs.empty:
                    st.success("🎉 Your customized recommendation spool is ready!")
                    st.subheader("Top 10 nostalgic recommendations for you")
                    movie_cards(user_recs, api_key)
                else:
                    st.error("Something went wrong processing your profile ratings. Please try again.")
        else:
            st.info("Select classic movies from the box above to begin building your custom reel profile.")

    else:
        # Demo / Explorer Mode: Browse anonymous MovieLens user profiles (1–610)
        st.subheader("Explore Historical Profiles")
        st.caption("This mode lets you browse the raw anonymous movie choices made by the original MovieLens research participants.")
        
        user_id = st.number_input(
            "Select Anonymous Database Profile ID (1–610)", 
            min_value=1, max_value=610, value=1, step=1
        )

        if st.button("Load Archival Profile Recommendation Spool"):
            rated_df = (
                ratings[ratings["user_id"] == user_id]
                .merge(movies[["movie_id", "title"]], on="movie_id")
            )

            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"📂 **Archival Profile ID {user_id}**")
                st.caption(f"Rated **{len(rated_df)} historic movies**")
                st.markdown("### Top Ratings:")
                for _, r in rated_df.sort_values("rating", ascending=False).head(5).iterrows():
                    st.markdown(f"{'⭐' * int(r['rating'])} {r['title']}")

            with col2:
                with st.spinner("Rewinding recommendations tape..."):
                    # Map profile ratings to dynamic recommendations
                    db_user_ratings = ratings_matrix.loc[user_id].to_dict()
                    db_user_ratings = {mid: r for mid, r in db_user_ratings.items() if r > 0}
                    user_recs = get_user_recs(db_user_ratings, ratings_matrix, movies, rating_stats)
                movie_cards(user_recs, api_key)

# Tab 3 — About
with tab3:
    st.header("The Vintage Engine Blueprint")
    st.markdown("""
    Welcome to **RetroReel**! This application is purposely tuned to provide nostalgic recommendations using standard collaborative filtering math on the historic GroupLens data.
    
    ### 📼 Why only pre-2018 movies?
    This site runs on the celebrated **MovieLens Latest Small** dataset, compiled in **September 2018**. Instead of clogging memory with modern databases, this platform celebrates the **Golden Age of Hollywood, 90s Blockbusters, and Nostalgic 2000s/2010s Hits**. 

    ### 🎞️ Recommendation Mathematics:
    *   **Item-Based Collaborative Filtering**: Maps similar movies by analyzing how users jointly rated older releases. If vintage archivists who loved *Star Wars* also loved *The Empire Strikes Back*, a mathematical correlation is formed.
    *   **Content-Based Modeling**: Maps similarity vectors over 20 distinct classic genres (like *Film-Noir*, *Musical*, *Sci-Fi*, and *IMAX*) to balance out ratings and find hidden thematic gems.
    *   **Hybrid Blend**: Combined weighting (60% Collaborative + 40% Content) tuned for vintage accuracy.

    ---
    **Engine Stack:** Python · Pandas · Scikit-learn · Streamlit · TMDB API (Optional)  
    **Archive Source:** MovieLens Latest (Small) — GroupLens Research, University of Minnesota

    Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) · [GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
    """)