from __future__ import annotations

from fastapi import FastAPI

from app.bert_predict import predict
from app.config import get_config
from app.model_store import clear_model_cache, load_model_bundle
from app.schemas import (
    HealthResponse,
    MetadataResponse,
)

config = get_config()
app = FastAPI(title=config.app_title, version="0.1.0")


@app.get("/")
def root() -> dict:
    bundle = load_model_bundle()
    return {
        "service": config.app_title,
        "model_version": bundle.metadata["model_version"],
        "classes": bundle.metadata["classes"],
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = load_model_bundle()
    return HealthResponse(
        status="ok",
        model_version=bundle.metadata["model_version"],
        classes=bundle.metadata["classes"],
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    bundle = load_model_bundle()
    return MetadataResponse(metadata=bundle.metadata)


@app.post("/predict")
def predict_api(req: dict):
    return predict(req["text"])


@app.post("/reload")
def reload_model() -> dict:
    clear_model_cache()
    bundle = load_model_bundle()
    return {
        "status": "reloaded",
        "model_version": bundle.metadata["model_version"],
    }
