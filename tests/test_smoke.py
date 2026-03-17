from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model_store import load_model_bundle_from_path
from app.predictor import TextClassifierService
from training.train import TrainConfig, train_and_save



def _build_sample_df() -> pd.DataFrame:
    rows = [
        {"id": "1", "text": "refund the second charge", "label": "billing"},
        {"id": "2", "text": "invoice missing from payment", "label": "billing"},
        {"id": "3", "text": "charged twice on my card", "label": "billing"},
        {"id": "4", "text": "refund has not arrived yet", "label": "billing"},
        {"id": "5", "text": "wrong tax amount on receipt", "label": "billing"},
        {"id": "6", "text": "subscription billed after cancel", "label": "billing"},
        {"id": "7", "text": "app crashes on launch", "label": "bug"},
        {"id": "8", "text": "dashboard keeps loading forever", "label": "bug"},
        {"id": "9", "text": "export button does nothing", "label": "bug"},
        {"id": "10", "text": "search results disappear after filter", "label": "bug"},
        {"id": "11", "text": "500 error on save", "label": "bug"},
        {"id": "12", "text": "page freezes on settings", "label": "bug"},
        {"id": "13", "text": "please add dark mode", "label": "feature_request"},
        {"id": "14", "text": "need export to pdf", "label": "feature_request"},
        {"id": "15", "text": "support scheduled reports", "label": "feature_request"},
        {"id": "16", "text": "add slack notifications", "label": "feature_request"},
        {"id": "17", "text": "want bulk edit for tags", "label": "feature_request"},
        {"id": "18", "text": "need custom fields on ticket form", "label": "feature_request"},
    ]
    return pd.DataFrame(rows)



def test_training_and_prediction(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    model_root = tmp_path / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"

    _build_sample_df().to_csv(train_path, index=False)

    output = train_and_save(
        TrainConfig(
            train_path=train_path,
            model_root=model_root,
            test_size=0.33,
            random_state=42,
            min_df=1,
            max_df=1.0,
            max_features=5000,
            calibration_method="sigmoid",
        )
    )

    assert output["model_version"].startswith("clf_")

    bundle = load_model_bundle_from_path(model_root / "latest")
    service = TextClassifierService(bundle)
    results = service.predict_records([
        {"id": "a", "text": "please refund the duplicate payment"},
        {"id": "b", "text": "the app crashes when i open reports"},
        {"id": "c", "text": "please add dark mode and export"},
    ])

    assert len(results) == 3
    assert all("confidence" in result for result in results)
    assert all(result["model_version"] == output["model_version"] for result in results)
