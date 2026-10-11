from pathlib import Path

import pytest

from src.predict import DEFAULT_MODEL, load_bundle, predict_text


def test_saved_model_reloads_and_predicts():
    if not DEFAULT_MODEL.exists():
        pytest.skip("Run src.train_models before testing saved model inference.")
    bundle = load_bundle()
    result = predict_text("You are a kind and supportive friend.", bundle)
    assert result["category"] in bundle["labels"]
    assert set(result["sentiment_scores"]) == {"negative", "neutral", "positive", "compound"}
    assert result["confidence"] is None or 0 <= result["confidence"] <= 1


def test_model_loading_error_is_clear(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_bundle(tmp_path / "missing.joblib")
