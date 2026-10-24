"""Tests that require the project's generated local model artifact."""
import pytest
from httpx import ASGITransport, AsyncClient

from backend import main as backend_main
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
    previous_error = getattr(app.state, "model_error", None)
    try:
        async with backend_main.lifespan(app):
            async with AsyncClient(transport=ASGITransport(app=app),
                                   base_url="http://testserver") as client:
                health = await client.get("/health")
                response = await client.post("/predict", json={
                    "text": "People should support and care for one another."})
                info = await client.get("/model-info")
            assert health.status_code == 200
            assert health.json() == {"status": "ok", "model_loaded": True}
            assert response.status_code == 200
            body = response.json()
            assert body["category"] in app.state.bundle["labels"]
            assert set(body["sentiment_scores"]) == {
                "negative", "neutral", "positive", "compound"}
            assert info.status_code == 200
            assert info.json()["available"] is True
    finally:
        app.state.bundle = previous
        app.state.model_error = previous_error
