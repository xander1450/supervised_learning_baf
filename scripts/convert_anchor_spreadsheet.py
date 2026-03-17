
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the uploaded issue-anchor spreadsheet into trainable text-classification CSV files."
    )
    parser.add_argument("--input-path", default="data/source/uploaded_issue_anchor_dataset.xlsx")
    parser.add_argument("--train-output-path", default="data/raw/train.csv")
    parser.add_argument("--raw-output-path", default="data/raw/train_raw_converted.csv")
    parser.add_argument("--mapping-output-path", default="data/processed/label_mapping.csv")
    parser.add_argument("--profile-output-path", default="data/processed/dataset_profile.json")
    parser.add_argument("--min-label-count", type=int, default=3)
    return parser.parse_args()


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).dropna(how="all").copy()
    for column in df.columns:
        df[column] = df[column].fillna("").astype(str).str.strip()
    return df


def expand_rows(df: pd.DataFrame) -> pd.DataFrame:
    examples: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        source_row_id = idx + 2
        issue_type = row.get("Issue Type", "") or "unknown_issue"
        sub_issue = row.get("Sub Issue", "")
        raw_label = f"{issue_type} > {sub_issue}" if sub_issue else issue_type

        parent = row.get("Parent Anchor", "")
        baby = row.get("Baby Anchor", "")
        variants: list[tuple[str, str]] = []

        if parent:
            variants.append(("parent", parent))
        if baby:
            variants.append(("baby", baby))
        if parent and baby:
            variants.append(("combined", f"{parent} || {baby}"))

        for anchor_source, text in variants:
            examples.append(
                {
                    "id": f"sheet1_r{source_row_id:03d}_{anchor_source}",
                    "text": text,
                    "issue_type": issue_type,
                    "sub_issue": sub_issue,
                    "raw_label": raw_label,
                    "anchor_source": anchor_source,
                    "source_row_id": source_row_id,
                }
            )
    return pd.DataFrame(examples)


def build_outputs(expanded: pd.DataFrame, *, min_label_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_export = expanded.copy()
    raw_export.insert(2, "label", raw_export["raw_label"])

    raw_label_counts = expanded["raw_label"].value_counts()
    issue_type_counts = expanded["issue_type"].value_counts()

    def resolve_label(row: pd.Series) -> tuple[str, str]:
        if raw_label_counts.get(row["raw_label"], 0) >= min_label_count:
            return row["raw_label"], "raw_label"
        if issue_type_counts.get(row["issue_type"], 0) >= min_label_count:
            return row["issue_type"], "issue_type_fallback"
        return "other", "other_fallback"

    resolved_pairs = expanded.apply(resolve_label, axis=1)
    trainable = expanded.copy()
    trainable["label"] = [pair[0] for pair in resolved_pairs]
    trainable["label_strategy"] = [pair[1] for pair in resolved_pairs]

    label_counts = trainable["label"].value_counts()
    mask_small = trainable["label"].map(label_counts) < min_label_count
    trainable.loc[mask_small, "label"] = "other"
    trainable.loc[mask_small, "label_strategy"] = "other_fallback"

    trainable = trainable[
        ["id", "text", "label", "issue_type", "sub_issue", "raw_label", "anchor_source", "label_strategy", "source_row_id"]
    ].copy()
    raw_export = raw_export[
        ["id", "text", "label", "issue_type", "sub_issue", "raw_label", "anchor_source", "source_row_id"]
    ].copy()

    mapping = (
        trainable.groupby(["issue_type", "sub_issue", "raw_label", "label", "label_strategy"], dropna=False)
        .size()
        .reset_index(name="training_example_count")
        .sort_values(["training_example_count", "issue_type", "sub_issue"], ascending=[False, True, True])
    )

    profile = {
        "usable_source_rows": int(len(expanded["source_row_id"].unique())),
        "expanded_training_examples": int(len(trainable)),
        "raw_labels_before_fallback": int(raw_export["label"].nunique()),
        "resolved_train_labels": int(trainable["label"].nunique()),
        "min_label_count": int(min_label_count),
        "conversion_rules": {
            "text_variants": ["parent", "baby", "combined when both anchors exist"],
            "default_label_resolution": "use raw Issue Type > Sub Issue label when it has enough examples; otherwise fallback to Issue Type; if still too sparse, map to other",
        },
    }
    return trainable, raw_export, mapping, profile


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    train_output_path = Path(args.train_output_path)
    raw_output_path = Path(args.raw_output_path)
    mapping_output_path = Path(args.mapping_output_path)
    profile_output_path = Path(args.profile_output_path)

    source_df = load_source(input_path)
    expanded = expand_rows(source_df)
    trainable, raw_export, mapping, profile = build_outputs(
        expanded,
        min_label_count=args.min_label_count,
    )

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_output_path.parent.mkdir(parents=True, exist_ok=True)

    trainable.to_csv(train_output_path, index=False)
    raw_export.to_csv(raw_output_path, index=False)
    mapping.to_csv(mapping_output_path, index=False)

    with profile_output_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "train_output_path": str(train_output_path),
                "raw_output_path": str(raw_output_path),
                "mapping_output_path": str(mapping_output_path),
                "profile_output_path": str(profile_output_path),
                "train_rows": int(len(trainable)),
                "train_labels": int(trainable["label"].nunique()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
