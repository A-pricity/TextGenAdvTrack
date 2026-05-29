from __future__ import annotations

from pathlib import Path

import pandas as pd

from textgenadvtrack.data.official_splits import OfficialSplitInputs, build_official_detection_splits
from textgenadvtrack.data.official_splits import _to_detection_rows
from textgenadvtrack.detection.ensemble import read_prediction_file
from textgenadvtrack.detection.evaluate import detection_metrics
from textgenadvtrack.io import save_csv


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


def build_kfold_detection_splits(
    val_with_label_csv: Path,
    output_dir: Path,
    folds: int = 10,
    seed: int = 42,
    language: str = "zh",
) -> dict:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if language not in {"zh", "en", "ru"}:
        raise ValueError("language must be one of: zh, en, ru")

    frame = pd.read_csv(val_with_label_csv).fillna("")
    missing = [column for column in ["prompt", "text", "label"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{val_with_label_csv} missing required columns: {missing}")
    frame = frame[["prompt", "text", "label"]].copy()
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"]).issubset({0, 1}):
        raise ValueError("label values must be 0 or 1")

    min_class_count = int(frame["label"].value_counts().min())
    if folds > min_class_count:
        raise ValueError(f"folds={folds} is larger than the smallest class count {min_class_count}")

    parts = []
    for _, label_frame in frame.groupby("label", sort=True):
        shuffled = label_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True).copy()
        shuffled["_fold"] = [index % folds for index in range(len(shuffled))]
        parts.append(shuffled)
    folded = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = []
    all_rows = _to_detection_rows(frame.sample(frac=1.0, random_state=seed).reset_index(drop=True), "train", language)
    for fold_index in range(folds):
        fold_dir = output_dir / f"fold_{fold_index + 1:02d}"
        dev = folded[folded["_fold"] == fold_index].drop(columns=["_fold"]).reset_index(drop=True)
        train = folded[folded["_fold"] != fold_index].drop(columns=["_fold"]).reset_index(drop=True)
        train_rows = _to_detection_rows(train, "train", language)
        dev_rows = _to_detection_rows(dev, "dev", language)
        save_csv(train_rows, fold_dir / "official_train.csv")
        save_csv(dev_rows, fold_dir / "official_dev.csv")
        save_csv(all_rows, fold_dir / "official_all_train.csv")
        inventory.append(
            {
                "fold": fold_index + 1,
                "train_rows": len(train_rows),
                "dev_rows": len(dev_rows),
                "all_train_rows": len(all_rows),
            }
        )

    pd.DataFrame(inventory).to_csv(output_dir / "kfold_split_inventory.csv", index=False)
    return {"output_dir": str(output_dir), "folds": folds, "seed": seed, "rows": len(frame)}


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
