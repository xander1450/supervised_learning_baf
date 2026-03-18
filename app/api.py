from __future__ import annotations

import os

import pandas as pd
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


def log_unlabeled(text: str, confidence: float) -> None:
    file_path = "data/feedback/unlabeled.csv"
    row = pd.DataFrame([{"text": text, "confidence": confidence}])
    if os.path.exists(file_path):
        row.to_csv(file_path, mode="a", header=False, index=False)
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        row.to_csv(file_path, index=False)


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
    result = predict(req["text"])
    if result["confidence"] < 0.5:
        log_unlabeled(req["text"], result["confidence"])
        result["label"] = "uncertain"
    return result


@app.post("/reload")
def reload_model() -> dict:
    clear_model_cache()
    bundle = load_model_bundle()
    return {
        "status": "reloaded",
        "model_version": bundle.metadata["model_version"],
    }
