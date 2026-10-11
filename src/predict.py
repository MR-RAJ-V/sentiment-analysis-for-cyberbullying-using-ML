"""Load the saved artifact and make predictions with its training features."""
from __future__ import annotations

from pathlib import Path

import joblib

from src.features import combine_features
from src.preprocessing import preprocess_text
from src.sentiment_analysis import sentiment_label, sentiment_scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "cyberbullying_model.joblib"


def load_bundle(path: Path = DEFAULT_MODEL) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def predict_text(text: str, bundle: dict) -> dict:
    clean = preprocess_text(text)
    vector = bundle["vectorizer"].transform([clean])
    features = combine_features(vector, [clean])
    model = bundle["model"]
    prediction = str(model.predict(features)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(features)[0]))
    scores = sentiment_scores(clean)
    return {"category": prediction, "sentiment": sentiment_label(scores["compound"]),
            "sentiment_scores": scores, "confidence": confidence,
            "confidence_note": "Classifier probability; not independently calibrated." if confidence is not None else None}
