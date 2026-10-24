from pathlib import Path

import joblib
import pytest

from src.predict import load_bundle, predict_text, validate_bundle


def test_production_pipeline_predicts_with_expected_output(lightweight_bundle):
    result = predict_text("You are a kind and supportive friend.", lightweight_bundle)
    assert result["category"] == "not_cyberbullying"
    assert set(result["sentiment_scores"]) == {"negative", "neutral", "positive", "compound"}
    assert result["confidence"] == 1.0


def test_fixture_classifier_uses_production_feature_dimensions(lightweight_bundle):
    pipeline = lightweight_bundle["pipeline"]
    raw_texts = ["A common kind supportive sample"]
    normalized = pipeline.named_steps["normalize"].transform(raw_texts)
    features = pipeline.named_steps["features"].transform(normalized)
    classifier = pipeline.named_steps["classifier"]
    assert features.shape[1] == classifier.n_features_in_
    assert predict_text(raw_texts[0], lightweight_bundle)["category"] == \
        "not_cyberbullying"


@pytest.mark.parametrize("value", ["", "  ", 5, None])
def test_prediction_rejects_invalid_text(value, lightweight_bundle):
    with pytest.raises(ValueError):
        predict_text(value, lightweight_bundle)


def test_bundle_validation_rejects_unknown_or_malformed_bundles():
    with pytest.raises(ValueError, match="mapping"):
        validate_bundle(None)
    with pytest.raises(ValueError, match="labels"):
        validate_bundle({"labels": [], "model": object(), "vectorizer": object()})
    with pytest.raises(ValueError, match="schema version"):
        validate_bundle({"schema_version": 1, "labels": ["age"]})


def test_model_loading_error_is_clear(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_bundle(tmp_path / "missing.joblib")


def test_model_loading_validates_bundle_after_deserialization(tmp_path, lightweight_bundle):
    path = tmp_path / "model.joblib"
    joblib.dump(lightweight_bundle, path)
    assert load_bundle(path)["model_name"] == "Test dummy"
    broken = tmp_path / "broken.joblib"
    joblib.dump({"labels": []}, broken)
    with pytest.raises(ValueError, match="labels"):
        load_bundle(broken)
