"""Tests for Unsupervised Discovery models."""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_smartwatch_model_exists():
    assert (ROOT / "models" / "kmeans_smartwatch.pkl").exists(), "Smartwatch model not found. Run src/train.py first."


def test_wafer_model_exists():
    assert (ROOT / "models" / "kmeans_wafer.pkl").exists(), "Wafer model not found. Run src/train.py first."


def test_smartwatch_prediction():
    from predict import predict
    df = pd.read_csv(ROOT / "data" / "smartwatch_activity.csv")
    features = df.drop(columns=["true_activity"]).head(3)
    preds = predict(features, str(ROOT / "models" / "kmeans_smartwatch.pkl"))
    assert len(preds) == 3


def test_wafer_prediction():
    from predict import predict
    df = pd.read_csv(ROOT / "data" / "wafer_defect_patterns.csv")
    features = df.drop(columns=["true_pattern"]).head(3)
    preds = predict(features, str(ROOT / "models" / "kmeans_wafer.pkl"))
    assert len(preds) == 3


if __name__ == "__main__":
    test_smartwatch_model_exists()
    test_wafer_model_exists()
    test_smartwatch_prediction()
    test_wafer_prediction()
    print("All tests passed.")
