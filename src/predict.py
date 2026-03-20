"""
Inference for Unsupervised Discovery.
Load trained clustering model and assign clusters to new data.
"""
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def predict(data: pd.DataFrame, model_path: str = None) -> list:
    """Load clustering model and predict cluster assignments."""
    if model_path is None:
        model_path = str(MODEL_DIR / "kmeans_smartwatch.pkl")
    pipe = joblib.load(model_path)
    preds = pipe.predict(data)
    return preds.tolist()


if __name__ == "__main__":
    for label, csv, model, drop_cols in [
        ("Smartwatch", "smartwatch_activity.csv", "kmeans_smartwatch.pkl", ["true_activity"]),
        ("Wafer", "wafer_defect_patterns.csv", "kmeans_wafer.pkl", ["true_pattern"]),
    ]:
        df = pd.read_csv(ROOT / "data" / csv)
        features = df.drop(columns=drop_cols, errors="ignore").head(5)
        preds = predict(features, str(MODEL_DIR / model))
        print(f"{label} cluster assignments: {preds}")
