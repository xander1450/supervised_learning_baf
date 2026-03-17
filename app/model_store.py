from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from app.config import get_config


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Any
    metadata: dict



def _load_bundle_from_dir(model_dir: Path) -> ModelBundle:
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Train a model first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata artifact not found at {metadata_path}. Train a model first."
        )

    pipeline = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return ModelBundle(pipeline=pipeline, metadata=metadata)


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    config = get_config()
    return _load_bundle_from_dir(config.model_dir)



def load_model_bundle_from_path(model_dir: str | Path) -> ModelBundle:
    return _load_bundle_from_dir(Path(model_dir))



def clear_model_cache() -> None:
    load_model_bundle.cache_clear()
