"""
Train pipeline for Unsupervised Discovery.
Trains KMeans clustering models for both datasets and saves fitted pipelines.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = {
    "smartwatch": {
        "file": "smartwatch_activity.csv",
        "model": "kmeans_smartwatch.pkl",
        "n_clusters": 4,
        "drop_cols": ["true_activity"],
    },
    "wafer": {
        "file": "wafer_defect_patterns.csv",
        "model": "kmeans_wafer.pkl",
        "n_clusters": 5,
        "drop_cols": ["true_pattern"],
    },
}


def train(name: str, cfg: dict):
    """Train KMeans clustering pipeline and save to models/."""
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    df = pd.read_csv(DATA_DIR / cfg["file"])
    X = df.drop(columns=cfg["drop_cols"], errors="ignore")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=cfg["n_clusters"], random_state=42, n_init=10)),
    ])
    pipe.fit(X)

    labels = pipe.predict(X)
    sil = silhouette_score(pipe.named_steps["scaler"].transform(X), labels)
    print(f"Clusters: {cfg['n_clusters']}")
    print(f"Inertia: {pipe.named_steps['kmeans'].inertia_:.2f}")
    print(f"Silhouette: {sil:.4f}")

    model_path = MODEL_DIR / cfg["model"]
    joblib.dump(pipe, model_path)
    print(f"Saved → {model_path}")
    return pipe


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        train(name, cfg)
