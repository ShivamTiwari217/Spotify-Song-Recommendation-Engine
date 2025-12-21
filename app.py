import streamlit as st
import pandas as pd
import joblib
import time

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🎧 Spotify Song Recommender",
    layout="wide"
)

st.title("🎧 Spotify Song Recommendation Engine")
st.caption("Content-Based Filtering using ANN (Cosine Similarity)")

# --------------------------------------------------
# Load trained models (cached)
# --------------------------------------------------
@st.cache_resource
def load_models():
    nn_model = joblib.load("models/song_ann_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    df = joblib.load("models/song_metadata.pkl")
    return nn_model, scaler, df

nn_model, scaler, df = load_models()

# --------------------------------------------------
# Recreate feature matrix (CRITICAL)
# --------------------------------------------------
audio_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo"
]

X = scaler.transform(df[audio_features]).astype("float32")

# --------------------------------------------------
# Fallback: random popular songs
# --------------------------------------------------
def random_popular_fallback(top_n=10):
    threshold = df["track_popularity"].quantile(0.80)
    pool = df[df["track_popularity"] >= threshold]

    return (
        pool.sample(n=min(top_n, len(pool)))
        [["track_name", "track_artist", "playlist_genre", "track_popularity"]]
        .reset_index(drop=True)
    )

# --------------------------------------------------
# Recommendation logic
# --------------------------------------------------
def recommend_songs(song_name, artist=None, top_n=10, alpha=0.75):

    if artist:
        matches = df[
            (df["track_name"].str.lower() == song_name.lower()) &
            (df["track_artist"].str.lower() == artist.lower())
        ]
    else:
        matches = df[df["track_name"].str.lower() == song_name.lower()]

    # ---- Fallback if song not found ----
    if matches.empty:
        return random_popular_fallback(top_n), True

    idx = matches.index[0]
    query_vector = X[idx].reshape(1, -1)

    distances, indices = nn_model.kneighbors(
        query_vector,
        n_neighbors=min(30, len(df))
    )

    results = []
    seen = set()

    for i, dist in zip(indices[0], distances[0]):
        if i == idx:
            continue

        key = (df.iloc[i]["track_name"], df.iloc[i]["track_artist"])
        if key in seen:
            continue
        seen.add(key)

        similarity = 1 - dist
        popularity = df.iloc[i]["popularity_norm"]
        final_score = alpha * similarity + (1 - alpha) * popularity

        results.append({
            "Track": df.iloc[i]["track_name"],
            "Artist": df.iloc[i]["track_artist"],
            "Genre": df.iloc[i]["playlist_genre"],
            "Similarity": round(similarity, 3),
            "Popularity": round(popularity, 3),
            "Final Score": round(final_score, 3)
        })

        if len(results) == top_n:
            break

    return pd.DataFrame(results), False

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("🎛 Recommendation Controls")

song_name = st.sidebar.text_input("Song name")
artist_name = st.sidebar.text_input("Artist (optional)")
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
alpha = st.sidebar.slider(
    "Similarity vs Popularity Weight",
    0.0, 1.0, 0.75
)

# --------------------------------------------------
# Run recommendation
# --------------------------------------------------
if st.sidebar.button("Recommend 🎶"):
    start_time = time.time()

    results, used_fallback = recommend_songs(
        song_name,
        artist_name if artist_name else None,
        top_n,
        alpha
    )

    latency = (time.time() - start_time) * 1000

    if used_fallback:
        st.warning("Song not found — showing popular songs instead 🎵")
    else:
        st.success("Here are your recommendations 🎧")

    st.dataframe(results, use_container_width=True)
    st.caption(f"⏱ Response time: {latency:.2f} ms")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit • Scikit-learn • ANN-based Recommendation")
