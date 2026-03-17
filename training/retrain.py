from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from training.data import load_dataset
from training.train import TrainConfig, train_and_save



def merge_datasets(base_path: Path, new_data_path: Path) -> pd.DataFrame:
    base_df = load_dataset(base_path)
    new_df = load_dataset(new_data_path)

    merged = pd.concat([base_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["id"], keep="last")
    merged = merged.reset_index(drop=True)
    return merged



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain with new labeled data.")
    parser.add_argument("--base-train-path", default="data/raw/train.csv")
    parser.add_argument("--new-data-path", default="data/raw/new_labeled.csv")
    parser.add_argument("--merged-output-path", default="data/processed/merged_train.csv")
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--promote-data", action="store_true")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_path = Path(args.base_train_path)
    new_data_path = Path(args.new_data_path)
    merged_output_path = Path(args.merged_output_path)

    merged_df = merge_datasets(base_path, new_data_path)
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(merged_output_path, index=False)

    output = train_and_save(
        TrainConfig(
            train_path=merged_output_path,
            model_root=Path(args.model_root),
            test_size=args.test_size,
            random_state=args.random_state,
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            calibration_method=args.calibration_method,
        )
    )

    if args.promote_data:
        merged_df.to_csv(base_path, index=False)

    print(
        json.dumps(
            {
                **output,
                "merged_rows": int(len(merged_df)),
                "merged_output_path": str(merged_output_path),
                "promote_data": bool(args.promote_data),
            },
            indent=2,
        )
    )
