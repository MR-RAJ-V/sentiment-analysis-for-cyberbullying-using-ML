from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.predict import load_bundle, predict_text, validate_bundle
from src.preprocessing import preprocess_text


def test_legacy_bundle_predicts_with_expected_output(lightweight_bundle):
    result = predict_text("You are a kind and supportive friend.", lightweight_bundle)
    assert result["category"] == "not_cyberbullying"
    assert set(result["sentiment_scores"]) == {"negative", "neutral", "positive", "compound"}
    assert result["confidence"] == 1.0


def test_pipeline_bundle_predicts_after_shared_preprocessing():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", DummyClassifier(strategy="constant", constant="age")),
    ])
    pipeline.fit([preprocess_text("you are kind"),
                  preprocess_text("you are cruel")], ["age", "age"])
    bundle = {"schema_version": 2, "pipeline": pipeline, "labels": ["age"],
              "model_name": "Test pipeline", "feature_mode": "pipeline_text_features",
              "seed": 42, "training_data": {}, "split": {},
              "package_versions": {}, "evaluation_metrics": {}}
    assert predict_text("You are KIND!", bundle)["category"] == "age"


@pytest.mark.parametrize("value", ["", "  ", 5, None])
def test_prediction_rejects_invalid_text(value, lightweight_bundle):
    with pytest.raises(ValueError):
        predict_text(value, lightweight_bundle)


def test_bundle_validation_rejects_unknown_or_malformed_bundles():
    with pytest.raises(ValueError, match="mapping"):
        validate_bundle(None)
    with pytest.raises(ValueError, match="labels"):
        validate_bundle({"labels": [], "model": object(), "vectorizer": object()})


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
