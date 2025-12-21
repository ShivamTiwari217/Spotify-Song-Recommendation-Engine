import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Song Recommender", layout="wide")

@st.cache_resource
def load_models():
    nn = joblib.load("models/song_ann_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    df = joblib.load("models/song_metadata.pkl")
    return nn, scaler, df

nn_model, scaler, df = load_models()

st.title("🎧 Song Recommendation Dashboard")
st.success("Models loaded successfully!")

audio_features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo"
]

X = scaler.transform(df[audio_features]).astype("float32")


st.sidebar.header("🎛 Controls")

song_name = st.sidebar.text_input("Song name")
artist_name = st.sidebar.text_input("Artist (optional)")
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
alpha = st.sidebar.slider("Similarity vs Popularity", 0.0, 1.0, 0.75)

def recommend_songs(song_name, artist=None, top_n=10, alpha=0.75):
    # (same logic you already built)

if st.sidebar.button("Recommend Songs"):
    results = recommend_songs(
        song_name,
        artist_name if artist_name else None,
        top_n,
        alpha
    )

    st.subheader("Recommended Songs")
    st.dataframe(results, use_container_width=True)

