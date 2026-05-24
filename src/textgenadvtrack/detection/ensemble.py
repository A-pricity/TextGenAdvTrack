from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from textgenadvtrack.detection.evaluate import detection_metrics


def read_prediction_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name="predictions")
    else:
        frame = pd.read_csv(path)
    if "text_prediction" not in frame.columns:
        raise ValueError(f"{path} missing text_prediction column")
    if "prompt" not in frame.columns:
        frame.insert(0, "prompt", "")
    return frame


def _normalized_weights(raw_weights: list[float]) -> list[float] | None:
    total = sum(raw_weights)
    if total <= 0:
        return None
    return [weight / total for weight in raw_weights]


def _candidate_weights(model_count: int, step: float) -> list[list[float]]:
    if model_count < 2:
        return [[1.0]]
    grid = np.arange(0.0, 1.0 + step / 2.0, step)
    candidates = []
    for raw in product(grid, repeat=model_count):
        weights = _normalized_weights([float(value) for value in raw])
        if weights is not None:
            candidates.append(weights)
    return candidates


def search_blend_weights(labels_csv: Path, prediction_paths: list[Path], step: float = 0.1) -> dict:
    if not prediction_paths:
        raise ValueError("At least one prediction file is required")
    labels = pd.read_csv(labels_csv)["label"].astype(int).tolist()
    predictions = [read_prediction_file(path)["text_prediction"].astype(float).to_numpy() for path in prediction_paths]
    row_counts = {len(values) for values in predictions}
    row_counts.add(len(labels))
    if len(row_counts) != 1:
        raise ValueError("Labels and predictions must have the same row count")

    best = None
    for weights in _candidate_weights(len(predictions), step):
        blended = sum(weight * values for weight, values in zip(weights, predictions, strict=True))
        metrics = detection_metrics(labels, blended)
        candidate = {"weights": weights, "metrics": metrics}
        if best is None or (
            metrics["Final_Score"],
            metrics["AUC"],
            metrics["ACC"],
            metrics["F1"],
        ) > (
            best["metrics"]["Final_Score"],
            best["metrics"]["AUC"],
            best["metrics"]["ACC"],
            best["metrics"]["F1"],
        ):
            best = candidate
    if best is None:
        raise ValueError("No blend weights generated")
    return best


def blend_prediction_files(prediction_paths: list[Path], weights: list[float], output_xlsx: Path) -> dict:
    if len(prediction_paths) != len(weights):
        raise ValueError("prediction_paths and weights must have the same length")
    normalized = _normalized_weights([float(weight) for weight in weights])
    if normalized is None:
        raise ValueError("weights must sum to a positive value")
    frames = [read_prediction_file(path) for path in prediction_paths]
    row_counts = {len(frame) for frame in frames}
    if len(row_counts) != 1:
        raise ValueError("All prediction files must have the same row count")

    blended = sum(
        weight * frame["text_prediction"].astype(float).to_numpy()
        for weight, frame in zip(normalized, frames, strict=True)
    )
    output = pd.DataFrame(
        {
            "prompt": frames[0]["prompt"].astype(str).tolist(),
            "text_prediction": np.clip(blended, 0.0, 1.0),
        }
    )
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="predictions", index=False)
        pd.DataFrame([{"Data Volume": len(output), "Time": 0.0}]).to_excel(writer, sheet_name="time", index=False)
    return {"rows": len(output), "output_xlsx": str(output_xlsx), "weights": normalized}
