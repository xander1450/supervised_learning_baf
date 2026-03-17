from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    model_dir: Path
    model_path: Path
    metadata_path: Path
    max_batch_size: int
    app_title: str



def get_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = Path(os.getenv("MODEL_DIR", base_dir / "models" / "latest"))
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "1000"))
    app_title = os.getenv("APP_TITLE", "Text Classification Service")

    return AppConfig(
        base_dir=base_dir,
        model_dir=model_dir,
        model_path=model_dir / "model.joblib",
        metadata_path=model_dir / "metadata.json",
        max_batch_size=max_batch_size,
        app_title=app_title,
    )
