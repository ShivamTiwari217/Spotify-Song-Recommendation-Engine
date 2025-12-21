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

