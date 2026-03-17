from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"id", "text", "label"}



def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_text}")



def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["id"] = cleaned["id"].astype(str).str.strip()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].astype(str).str.strip()

    cleaned = cleaned[(cleaned["text"] != "") & (cleaned["label"] != "")]
    cleaned = cleaned.drop_duplicates(subset=["id"], keep="last")
    cleaned = cleaned.reset_index(drop=True)
    return cleaned



def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    validate_schema(df)
    return clean_dataset(df)



def compute_dataset_hash(path: str | Path) -> str:
    dataset_path = Path(path)
    digest = hashlib.sha256()
    with dataset_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
