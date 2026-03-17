from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRecord(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1, description="Raw text to classify")


class PredictRequest(BaseModel):
    records: List[PredictRecord] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    id: Optional[str] = None
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str


class PredictResponse(BaseModel):
    results: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    classes: List[str]


class MetadataResponse(BaseModel):
    metadata: dict
