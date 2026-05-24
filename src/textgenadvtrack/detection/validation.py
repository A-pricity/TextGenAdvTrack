from __future__ import annotations

from pathlib import Path

import pandas as pd

from textgenadvtrack.data.official_splits import OfficialSplitInputs, build_official_detection_splits
from textgenadvtrack.detection.ensemble import read_prediction_file
from textgenadvtrack.detection.evaluate import detection_metrics


def build_repeated_detection_splits(
    val_with_label_csv: Path,
    output_dir: Path,
    seeds: list[int],
    dev_fraction: float = 0.2,
    language: str = "zh",
) -> dict:
    if not seeds:
        raise ValueError("At least one seed is required")
    folds = []
    for seed in seeds:
        fold_dir = output_dir / f"seed_{seed}"
        result = build_official_detection_splits(
            OfficialSplitInputs(
                val_with_label_csv=val_with_label_csv,
                output_dir=fold_dir,
                dev_fraction=dev_fraction,
                language=language,
                seed=seed,
            )
        )
        result["seed"] = seed
        folds.append(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(output_dir / "repeated_split_inventory.csv", index=False)
    return {"output_dir": str(output_dir), "folds": len(folds), "seeds": seeds}


def evaluate_prediction_slices(
    labels_csv: Path,
    predictions_path: Path,
    group_columns: list[str] | None = None,
) -> dict:
    labels = pd.read_csv(labels_csv).reset_index(drop=True)
    predictions = read_prediction_file(predictions_path).reset_index(drop=True)
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same row count")
    if "label" not in labels.columns:
        raise ValueError(f"{labels_csv} missing label column")

    merged = labels.copy()
    merged["text_prediction"] = predictions["text_prediction"].astype(float)
    report = {
        "overall": detection_metrics(merged["label"].astype(int).tolist(), merged["text_prediction"].tolist()),
        "groups": {},
    }
    for column in group_columns or []:
        if column not in merged.columns:
            continue
        for value, group in merged.groupby(column, sort=True):
            if group["label"].nunique() < 2:
                continue
            key = f"{column}={value}"
            report["groups"][key] = detection_metrics(
                group["label"].astype(int).tolist(),
                group["text_prediction"].tolist(),
            )
    return report
