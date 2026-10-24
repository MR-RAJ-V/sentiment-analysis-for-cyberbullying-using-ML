"""Tests that require the project's generated local model artifact."""
import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app
from src.predict import DEFAULT_MODEL, load_bundle, predict_text


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="Train the model for integration coverage")
def test_saved_model_reloads_and_predicts():
    bundle = load_bundle()
    result = predict_text("You are a kind and supportive friend.", bundle)
    assert result["category"] in bundle["labels"]
    assert set(result["sentiment_scores"]) == {"negative", "neutral", "positive", "compound"}
    assert result["confidence"] is None or 0 <= result["confidence"] <= 1


@pytest.mark.anyio
async def test_final_model_serves_a_real_api_prediction():
    if not DEFAULT_MODEL.exists():
        pytest.skip("Train the model for integration coverage")
    previous = getattr(app.state, "bundle", None)
    app.state.bundle = load_bundle()
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://testserver") as client:
            response = await client.post("/predict", json={
                "text": "People should support and care for one another."})
        assert response.status_code == 200
        body = response.json()
        assert body["category"] in app.state.bundle["labels"]
        assert set(body["sentiment_scores"]) == {"negative", "neutral", "positive", "compound"}
    finally:
        app.state.bundle = previous
