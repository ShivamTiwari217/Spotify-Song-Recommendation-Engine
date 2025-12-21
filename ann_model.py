import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Paths
DATA_PATH = "spotify_songs.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

audio_features = [
    "danceability","energy","loudness","speechiness",
    "acousticness","instrumentalness","liveness",
    "valence","tempo"
]

df = df[audio_features + [
    "track_name","track_artist","playlist_genre","track_popularity"
]].dropna()

# Deduplicate
df = (
    df.sort_values("track_popularity", ascending=False)
      .drop_duplicates(subset=["track_name","track_artist"])
      .sample(n=10001, random_state=42)
      .reset_index(drop=True)
)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(df[audio_features]).astype("float32")

# Popularity normalization
df["popularity_norm"] = (
    df["track_popularity"] - df["track_popularity"].min()
) / (
    df["track_popularity"].max() - df["track_popularity"].min() + 1e-6
)

# ANN model
nn_model = NearestNeighbors(n_neighbors=30, metric="cosine")
nn_model.fit(X)

# Save artifacts
joblib.dump(nn_model, f"{MODEL_DIR}/song_ann_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/feature_scaler.pkl")
joblib.dump(df, f"{MODEL_DIR}/song_metadata.pkl")

print("✅ Model training complete")
