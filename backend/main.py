"""FastAPI application exposing health, prediction, and model metadata."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictionRequest, PredictionResponse
from src.predict import DEFAULT_MODEL, load_bundle, predict_text

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.bundle = load_bundle(DEFAULT_MODEL)
        app.state.model_error = None
    except Exception as exc:
        app.state.bundle = None
        app.state.model_error = str(exc)
    yield


app = FastAPI(title="Cyberbullying Detection API", version="1.0.0", lifespan=lifespan,
              description="TF-IDF and VADER features with a locally trained classifier.")
app.add_middleware(CORSMiddleware, allow_origins=[
    "http://localhost:5500", "http://127.0.0.1:5500",
    "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/health")
async def health(request: Request):
    loaded = request.app.state.bundle is not None
    return {"status": "ok" if loaded else "model_unavailable", "model_loaded": loaded}


@app.post("/predict", response_model=PredictionResponse)
async def prediction(payload: PredictionRequest, request: Request):
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Prediction model is unavailable. Train the model first.")
    try:
        return predict_text(payload.text, bundle)
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed. Check the submitted text and try again.")


@app.get("/model-info")
async def model_info():
    if not METRICS_PATH.is_file():
        return {"available": False, "message": "Training metrics are not available yet."}
    try:
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "message": "Training metrics could not be loaded."}
    return {"available": True, **data}
