from httpx import ASGITransport, AsyncClient
import pytest

from backend.main import app
from src.predict import DEFAULT_MODEL, load_bundle


@pytest.fixture(scope="module")
def model_bundle():
    if not DEFAULT_MODEL.exists():
        pytest.skip("Run src.train_models before API tests.")
    bundle = load_bundle()
    app.state.bundle = bundle
    return bundle


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(anyio_backend, model_bundle):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as test_client:
        yield test_client


@pytest.mark.anyio
async def test_health_reports_loaded_model(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


@pytest.mark.anyio
async def test_prediction_endpoint_returns_real_model_output(client):
    response = await client.post("/predict", json={"text": "People like you should leave this school."})
    assert response.status_code == 200
    body = response.json()
    assert body["category"]
    assert set(body["sentiment_scores"]) == {"negative", "neutral", "positive", "compound"}


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": 42}])
@pytest.mark.anyio
async def test_invalid_prediction_input_is_rejected(client, payload):
    assert (await client.post("/predict", json=payload)).status_code == 422


@pytest.mark.anyio
async def test_overlong_input_is_rejected(client):
    assert (await client.post("/predict", json={"text": "a" * 5001})).status_code == 422


@pytest.mark.anyio
async def test_model_info_is_available(client):
    body = (await client.get("/model-info")).json()
    assert body["available"] is True
    assert body["dataset"]["labels"]


@pytest.mark.anyio
async def test_prediction_reports_model_loading_failure(client):
    loaded = app.state.bundle
    app.state.bundle = None
    try:
        assert (await client.get("/health")).json()["model_loaded"] is False
        response = await client.post("/predict", json={"text": "example"})
        assert response.status_code == 503
    finally:
        app.state.bundle = loaded
