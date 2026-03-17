from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from training.data import compute_dataset_hash, load_dataset
from training.evaluate import evaluate_classifier


@dataclass(frozen=True)
class TrainConfig:
    train_path: Path
    model_root: Path
    test_size: float
    random_state: int
    min_df: int
    max_df: float
    max_features: int
    calibration_method: str



def infer_cv_splits(labels: list[str]) -> int:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    min_count = min(counts.values())
    if min_count < 2:
        raise ValueError(
            "Each class needs at least 2 examples in the training split for calibration."
        )
    return min(5, min_count)



def build_pipeline(*, min_df: int, max_df: float, max_features: int, cv_splits: int, calibration_method: str) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )

    base_classifier = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced"
    )

    classifier = CalibratedClassifierCV(
        estimator=base_classifier,
        method=calibration_method,
        cv=cv_splits,
    )

    return Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )



def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



def train_and_save(config: TrainConfig) -> dict[str, Any]:
    df = load_dataset(config.train_path)
    classes = sorted(df["label"].unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=config.test_size,
        stratify=df["label"].tolist(),
        random_state=config.random_state,
    )

    cv_splits = infer_cv_splits(y_train)
    pipeline = build_pipeline(
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        cv_splits=cv_splits,
        calibration_method=config.calibration_method,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test).tolist()
    y_proba = pipeline.predict_proba(X_test).tolist()
    metrics = evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=classes,
    )

    trained_at = datetime.now(timezone.utc)
    version = f"clf_{trained_at.strftime('%Y%m%d_%H%M%S')}"
    version_dir = config.model_root / "archive" / version
    latest_dir = config.model_root / "latest"
    version_dir.mkdir(parents=True, exist_ok=False)
    latest_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.joblib"
    metadata_path = version_dir / "metadata.json"
    metrics_path = version_dir / "metrics.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_version": version,
        "trained_at_utc": trained_at.isoformat(),
        "classes": classes,
        "text_column": "text",
        "label_column": "label",
        "training_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_sha256": compute_dataset_hash(config.train_path),
        "train_path": str(config.train_path),
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "min_df": config.min_df,
            "max_df": config.max_df,
            "max_features": config.max_features,
            "sublinear_tf": True,
        },
        "classifier": {
            "base_estimator": "LogisticRegression",
            "calibration": {
                "type": "CalibratedClassifierCV",
                "method": config.calibration_method,
                "cv": cv_splits,
            },
        },
        "summary_metrics": {
            "accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
            "weighted_f1": metrics["classification_report"]["weighted avg"]["f1-score"],
        },
    }

    _write_json(metadata_path, metadata)
    _write_json(metrics_path, metrics)

    shutil.copy2(model_path, latest_dir / "model.joblib")
    shutil.copy2(metadata_path, latest_dir / "metadata.json")
    shutil.copy2(metrics_path, latest_dir / "metrics.json")

    return {
        "model_version": version,
        "model_dir": str(version_dir),
        "latest_dir": str(latest_dir),
        "metrics": metadata["summary_metrics"],
    }



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a text classification model.")
    parser.add_argument("--train-path", default="data/raw/train.csv")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
    )
    args = parser.parse_args()

    return TrainConfig(
        train_path=Path(args.train_path),
        model_root=Path(args.model_root),
        test_size=args.test_size,
        random_state=args.random_state,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        calibration_method=args.calibration_method,
    )


if __name__ == "__main__":
    output = train_and_save(parse_args())
    print(json.dumps(output, indent=2))
