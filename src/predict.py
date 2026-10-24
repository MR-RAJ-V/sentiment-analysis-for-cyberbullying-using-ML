"""Load the saved artifact and make predictions with its training features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.preprocessing import preprocess_text
from src.sentiment_analysis import sentiment_label, sentiment_scores

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "cyberbullying_model.joblib"


def validate_bundle(bundle: object) -> dict:
    """Validate the single supported production artifact layout."""
    if not isinstance(bundle, dict):
        raise ValueError("Invalid model artifact: expected a mapping")
    labels = bundle.get("labels")
    if not isinstance(labels, list) or not labels or not all(
            isinstance(label, str) and label for label in labels):
        raise ValueError("Invalid model artifact: labels must be a non-empty string list")
    if bundle.get("schema_version") != 2:
        raise ValueError("Invalid model artifact: unsupported schema version")
    pipeline = bundle.get("pipeline")
    if pipeline is None or not callable(getattr(pipeline, "predict", None)):
        raise ValueError("Invalid model artifact: pipeline is missing predict()")
    if not isinstance(bundle.get("model_name"), str) or not bundle["model_name"].strip():
        raise ValueError("Invalid model artifact: model_name is missing")
    if bundle.get("feature_mode") != "pipeline_text_features":
        raise ValueError("Invalid model artifact: unsupported feature mode")
    for key in ("training_data", "split", "package_versions", "evaluation_metrics"):
        if not isinstance(bundle.get(key), dict):
            raise ValueError(f"Invalid model artifact: {key} metadata is missing")
    if not isinstance(bundle["training_data"].get("name"), str) or not isinstance(
            bundle["training_data"].get("sha256"), str):
        raise ValueError("Invalid model artifact: training data metadata is incomplete")
    if not isinstance(bundle["split"].get("strategy"), str) or not isinstance(
            bundle["split"].get("random_seed"), int):
        raise ValueError("Invalid model artifact: split metadata is incomplete")
    if not {"selected_model", "current_configuration_baseline"}.issubset(
            bundle["evaluation_metrics"]):
        raise ValueError("Invalid model artifact: evaluation metrics are incomplete")
    if not isinstance(bundle.get("seed"), int):
        raise ValueError("Invalid model artifact: seed is missing")
    model_steps = getattr(pipeline, "named_steps", {})
    if not {"normalize", "features", "classifier"}.issubset(model_steps):
        raise ValueError("Invalid model artifact: production pipeline steps are incomplete")
    classifier = model_steps.get("classifier") if hasattr(model_steps, "get") else None
    model_labels = getattr(classifier, "classes_", None)
    if model_labels is None or sorted(map(str, model_labels)) != sorted(labels):
        raise ValueError("Invalid model artifact: model classes do not match labels")
    return bundle


def load_bundle(path: Path = DEFAULT_MODEL) -> dict:
    """Load a trusted artifact written by this project and validate its shape.

    Joblib uses pickle internally and can execute code while loading. Never load
    model files supplied by an untrusted user.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return validate_bundle(joblib.load(path))


def predict_text(text: str, bundle: dict) -> dict:
    bundle = validate_bundle(bundle)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must contain at least one non-whitespace character")
    max_chars = int(bundle.get("text_max_chars", 5000))
    if len(text) > max_chars:
        raise ValueError(f"Text exceeds the {max_chars}-character limit")
    clean = preprocess_text(text)
    model = bundle["pipeline"]
    prediction = str(model.predict([text])[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([text])[0]))
    scores = sentiment_scores(clean)
    return {"category": prediction, "sentiment": sentiment_label(scores["compound"]),
            "sentiment_scores": scores, "confidence": confidence,
            "confidence_note": "Classifier probability; not independently calibrated." if confidence is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one social media post.")
    parser.add_argument("text", help="Post text to classify (maximum 5,000 characters)")
    args = parser.parse_args()
    try:
        result = predict_text(args.text, load_bundle())
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
