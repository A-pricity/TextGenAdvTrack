from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from textgenadvtrack.detection.evaluate import detection_metrics


def apply_score_transform(scores: Iterable[float], scale: float = 1.0, bias: float = 0.0) -> list[float]:
    values = np.asarray(list(scores), dtype=float)
    transformed = (values - 0.5) * scale + 0.5 + bias
    return np.clip(transformed, 0.0, 1.0).astype(float).tolist()


def tune_scores(
    labels: Iterable[int],
    scores: Iterable[float],
    scales: Iterable[float] = (0.7, 0.85, 1.0, 1.15, 1.3),
    biases: Iterable[float] = (-0.05, -0.025, 0.0, 0.025, 0.05),
) -> dict:
    labels_list = [int(label) for label in labels]
    scores_list = [float(score) for score in scores]
    best = None
    for scale in scales:
        for bias in biases:
            adjusted = apply_score_transform(scores_list, scale=float(scale), bias=float(bias))
            metrics = detection_metrics(labels_list, adjusted)
            candidate = {
                "best_scale": float(scale),
                "best_bias": float(bias),
                "best_metrics": metrics,
            }
            if best is None or (
                metrics["Final_Score"],
                metrics["AUC"],
                metrics["ACC"],
                metrics["F1"],
            ) > (
                best["best_metrics"]["Final_Score"],
                best["best_metrics"]["AUC"],
                best["best_metrics"]["ACC"],
                best["best_metrics"]["F1"],
            ):
                best = candidate
    if best is None:
        raise ValueError("No scale/bias candidates provided")
    return best


def tune_submission_scores(
    input_path: Path,
    output_path: Path,
    scale: float,
    bias: float,
) -> dict:
    predictions = _read_predictions(input_path)
    predictions["text_prediction"] = apply_score_transform(predictions["text_prediction"], scale=scale, bias=bias)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_frame = pd.DataFrame([{"Data Volume": len(predictions), "Time": 0.0}])
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        predictions[["prompt", "text_prediction"]].to_excel(writer, sheet_name="predictions", index=False)
        time_frame.to_excel(writer, sheet_name="time", index=False)
    return {"rows": len(predictions), "output_xlsx": str(output_path), "scale": scale, "bias": bias}


def _read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name="predictions")
    else:
        frame = pd.read_csv(path)
    if "text_prediction" not in frame.columns:
        raise ValueError(f"{path} missing text_prediction column")
    if "prompt" not in frame.columns:
        frame.insert(0, "prompt", "")
    return frame.copy()
