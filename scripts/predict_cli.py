from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model_store import load_model_bundle
from app.predictor import TextClassifierService


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a text classification prediction.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--id", default="cli-1")
    args = parser.parse_args()

    bundle = load_model_bundle()
    service = TextClassifierService(bundle)
    results = service.predict_records([
        {
            "id": args.id,
            "text": args.text,
        }
    ])
    print(json.dumps(results[0], indent=2, ensure_ascii=False))
