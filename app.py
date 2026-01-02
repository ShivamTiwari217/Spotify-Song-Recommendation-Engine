import streamlit as st
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🎧 Spotify Song Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #0E1117;
    }
    .song-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #1DB954;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .song-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 5px;
    }
    .song-artist {
        font-size: 1rem;
        color: #b3b3b3;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #b3b3b3;
    }
    .metric-value {
        font-size: 1rem;
        font-weight: bold;
        color: #1DB954;
    }
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
    h1 {
        color: #1DB954;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.write("") # Spacer
    st.markdown("# 🎧")
with col2:
    st.title("Spotify Song Recommender")
    st.markdown("**Discover new music based on what you love.** Powered by Machine Learning.")

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
    
    selected = pool.sample(n=min(top_n, len(pool)))
    
    results = []
    for _, row in selected.iterrows():
        results.append({
            "Track": row["track_name"],
            "Artist": row["track_artist"],
            "Genre": row["playlist_genre"],
            "Similarity": 0.0,
            "Popularity": row["track_popularity"], # Keep as 0-100 to match display logic
            "Final Score": row["track_popularity"] # Simple score for fallback
        })
        
    return pd.DataFrame(results)

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

def plot_audio_radar(song_row):
    features = [
        "danceability", "energy", "speechiness",
        "acousticness", "instrumentalness",
        "liveness", "valence"
    ]

    values = song_row[features].values.tolist()
    values.append(values[0])  # close radar

    features.append(features[0])

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=features,
                fill="toself",
                name="Audio Profile",
                line_color='#1DB954'
            )
        ]
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        title="🎵 Audio Feature Profile"
    )

    return fig


def plot_feature_comparison(input_song, recommended_songs_df):
    """
    Compares the input song's features with the average of the recommended songs.
    """
    comparison_features = [
        "danceability", "energy", "acousticness", 
        "instrumentalness", "valence"
    ]
    
    # Get values
    input_values = input_song[comparison_features].values.flatten().tolist()
    avg_values = recommended_songs_df[comparison_features].mean().tolist()
    
    fig = go.Figure(data=[
        go.Bar(name='Selected Song', x=comparison_features, y=input_values, marker_color='#1DB954'),
        go.Bar(name='Avg. Recommendation', x=comparison_features, y=avg_values, marker_color='#535353')
    ])
    
    fig.update_layout(
        barmode='group',
        title="📊 Feature Comparison",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(gridcolor='#333'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

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
    
    st.caption(f"⏱ Response time: {latency:.2f} ms")

    # Display results
    for index, row in results.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="song-card">
                <div class="song-title">{row['Track']}</div>
                <div class="song-artist">{row['Artist']}</div>
                <div style="display: flex; justify-content: space-between;">
                    <span class="metric-label">Genre: {row['Genre']}</span>
                    <span class="metric-value">{row['Final Score']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show progress bars for details
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Similarity")
                st.progress(float(row['Similarity']))
            with col_b:
                st.caption("Popularity")
                st.progress(float(row['Popularity']) if row['Popularity'] <= 1.0 else float(row['Popularity']) / 100)

    # --- Analytics Section ---
    st.markdown("---")
    st.header("📊 Analysis")
    
    song_row = df[
        df["track_name"].str.lower() == song_name.lower()
    ]
    
    if not song_row.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Audio Profile")
            st.plotly_chart(
                plot_audio_radar(song_row.iloc[0]),
                use_container_width=True
            )
            
        with col2:
            st.subheader("Feature Comparison")
            # Get features for recommended songs
            recommended_features = df[df['track_name'].isin(results['Track'])]
            if not recommended_features.empty:
                st.plotly_chart(
                    plot_feature_comparison(song_row.iloc[0], recommended_features),
                    use_container_width=True
                )
    
    elif used_fallback:
        st.info("Select a specific song to see detailed analytics.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit • Scikit-learn • ANN-based Recommendation")
