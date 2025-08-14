import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


FEATURE_COLS: List[str] = [
    "danceability",
    "energy",
    "tempo",
    "valence",
    "acousticness",
    "loudness",
]


class Recommender:
    """Content-based recommender using standardized audio features.

    - Loads `spotify_songs.csv`
    - Standardizes 6 numeric features with a shared `StandardScaler`
    - Uses cosine similarity in standardized feature space
    """

    def __init__(self, csv_path: str = "spotify_songs.csv") -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")

        self.df: pd.DataFrame = pd.read_csv(csv_path)
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")

        # Keep minimal columns for downstream display
        keep_cols = ["track_id", "track_name", "track_artist"] + FEATURE_COLS
        self.df = self.df[keep_cols].copy()

        # Standardize features
        self.scaler = StandardScaler()
        self.X_all: np.ndarray = self.scaler.fit_transform(self.df[FEATURE_COLS])

    def get_features_for_track_ids(self, track_ids: List[str]) -> np.ndarray:
        seed_df = self.df[self.df["track_id"].isin(track_ids)]
        if seed_df.empty:
            return np.empty((0, len(FEATURE_COLS)))
        return self.scaler.transform(seed_df[FEATURE_COLS].values)

    def make_taste_vector(self, track_ids: List[str]) -> np.ndarray:
        features = self.get_features_for_track_ids(track_ids)
        if features.size == 0:
            return np.empty((0, len(FEATURE_COLS)))
        return np.mean(features, axis=0, keepdims=True)

    def recommend_by_track_ids(
        self,
        example_track_ids: List[str],
        top_n: int = 10,
    ) -> List[Dict[str, object]]:
        """Recommend songs most similar to user's taste vector.

        - Uses cosine similarity in standardized feature space (no training)
        - Excludes seed tracks from results
        """

        taste_vector = self.make_taste_vector(example_track_ids)
        if taste_vector.size == 0:
            return []

        sims = cosine_similarity(taste_vector, self.X_all).ravel()

        seed_set = set(example_track_ids)
        indexed = [
            (idx, score)
            for idx, score in enumerate(sims)
            if self.df.iloc[idx]["track_id"] not in seed_set
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:top_n]

        results: List[Dict[str, object]] = []
        for idx, score in top:
            row = self.df.iloc[idx]
            results.append(
                {
                    "track_id": row["track_id"],
                    "track_name": row["track_name"],
                    "track_artist": row["track_artist"],
                    "similarity": float(score),
                }
            )
        return results


