import json
import logging

from httpx import ASGITransport, AsyncClient
import pytest

from backend import main as backend_main
from backend.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(lightweight_bundle):
    previous = getattr(app.state, "bundle", None)
    previous_error = getattr(app.state, "model_error", None)
    app.state.bundle = lightweight_bundle
    app.state.model_error = None
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://testserver") as test_client:
            yield test_client
    finally:
        app.state.bundle = previous
        app.state.model_error = previous_error


@pytest.mark.anyio
async def test_health_reports_loaded_model(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


@pytest.mark.anyio
async def test_prediction_returns_expected_response_shape(client):
    response = await client.post("/predict", json={
        "text": "People should support and care for one another."})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "not_cyberbullying"
    assert body["sentiment"] in {"Positive", "Negative", "Neutral"}
    assert set(body["sentiment_scores"]) == {"negative", "neutral", "positive", "compound"}
    assert 0 <= body["confidence"] <= 1


@pytest.mark.parametrize("payload", [
    {"text": ""}, {"text": "   "}, {"text": 42}, {"text": None},
    {"text": ["not", "a", "string"]}, {},
])
@pytest.mark.anyio
async def test_invalid_prediction_input_is_rejected(client, payload):
    assert (await client.post("/predict", json=payload)).status_code == 422


@pytest.mark.anyio
async def test_overlong_input_is_rejected(client):
    assert (await client.post("/predict", json={"text": "a" * 5001})).status_code == 422


@pytest.mark.anyio
async def test_missing_model_returns_consistent_health_and_prediction_errors(client):
    app.state.bundle = None
    app.state.model_error = "FileNotFoundError: private path"
    assert (await client.get("/health")).json() == {
        "status": "model_unavailable", "model_loaded": False}
    response = await client.post("/predict", json={"text": "example"})
    assert response.status_code == 503
    assert "private path" not in response.text


@pytest.mark.anyio
async def test_prediction_error_is_safe_and_does_not_log_user_text(client, monkeypatch,
                                                                   caplog):
    secret_text = "private user content that must not be logged"
    def fail_prediction(text, bundle):
        raise RuntimeError("internal failure containing " + text)
    monkeypatch.setattr(backend_main, "predict_text", fail_prediction)
    with caplog.at_level(logging.ERROR):
        response = await client.post("/predict", json={"text": secret_text})
    assert response.status_code == 500
    assert "internal" not in response.text
    assert secret_text not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.anyio
async def test_model_info_is_available_without_exposing_paths(client, tmp_path, monkeypatch):
    metrics = {
        "dataset": {"source": "/private/dataset.csv", "labels": ["age"]},
        "model_artifact": "/private/models/model.joblib",
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(backend_main, "METRICS_PATH", path)
    body = (await client.get("/model-info")).json()
    assert body["available"] is True
    assert "source" not in body["dataset"]
    assert body["model_artifact"] == "model.joblib"


@pytest.mark.anyio
async def test_model_info_reports_missing_or_invalid_metrics(client, tmp_path, monkeypatch):
    path = tmp_path / "missing.json"
    monkeypatch.setattr(backend_main, "METRICS_PATH", path)
    assert (await client.get("/model-info")).json()["available"] is False
    path.write_text("{bad", encoding="utf-8")
    assert (await client.get("/model-info")).json()["available"] is False


@pytest.mark.anyio
async def test_startup_model_error_is_logged_and_remains_internal(monkeypatch, caplog):
    # load_bundle is synchronous; inject a failure while exercising startup.
    def fail_load(path):
        raise FileNotFoundError("/private/path/model.joblib")
    monkeypatch.setattr(backend_main, "load_bundle", fail_load)
    with caplog.at_level(logging.ERROR):
        async with backend_main.lifespan(app):
            assert app.state.bundle is None
            assert "FileNotFoundError" in app.state.model_error
    assert "Could not load the local prediction model" in caplog.text
