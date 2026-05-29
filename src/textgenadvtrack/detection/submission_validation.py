from __future__ import annotations

from pathlib import Path

import pandas as pd


def _first_mismatch(left: list[str], right: list[str]) -> dict:
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if left_value != right_value:
            return {
                "row": index + 2,
                "input_prompt": left_value[:120],
                "submission_prompt": right_value[:120],
            }
    return {"row": None, "input_prompt": "", "submission_prompt": ""}


def validate_detection_submission(
    input_csv: Path,
    submission_xlsx: Path,
    min_unique_scores: int = 10,
) -> dict:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")
    if not submission_xlsx.exists():
        raise FileNotFoundError(f"Missing submission XLSX: {submission_xlsx}")

    input_frame = pd.read_csv(input_csv).fillna("")
    missing_input_columns = [column for column in ["prompt", "text"] if column not in input_frame.columns]
    if missing_input_columns:
        raise ValueError(f"{input_csv} missing required columns: {missing_input_columns}")

    try:
        prediction_frame = pd.read_excel(submission_xlsx, sheet_name="predictions").fillna("")
    except ValueError as exc:
        raise ValueError(f"{submission_xlsx} must contain a 'predictions' sheet") from exc

    expected_columns = ["prompt", "text_prediction"]
    if list(prediction_frame.columns) != expected_columns:
        raise ValueError(
            f"{submission_xlsx} predictions sheet columns must be exactly {expected_columns}, "
            f"got {list(prediction_frame.columns)}"
        )

    if len(prediction_frame) != len(input_frame):
        raise ValueError(
            f"Row count mismatch: input has {len(input_frame)} rows, submission has {len(prediction_frame)} rows"
        )

    input_prompts = input_frame["prompt"].astype(str).tolist()
    submission_prompts = prediction_frame["prompt"].astype(str).tolist()
    if input_prompts != submission_prompts:
        mismatch = _first_mismatch(input_prompts, submission_prompts)
        raise ValueError(f"Prompt alignment mismatch at Excel row {mismatch['row']}: {mismatch}")

    scores = pd.to_numeric(prediction_frame["text_prediction"], errors="coerce")
    if scores.isna().any():
        bad_rows = (scores[scores.isna()].index + 2).tolist()[:5]
        raise ValueError(f"text_prediction contains non-numeric values at Excel rows: {bad_rows}")
    if not scores.between(0.0, 1.0).all():
        bad_rows = (scores[~scores.between(0.0, 1.0)].index + 2).tolist()[:5]
        raise ValueError(f"text_prediction values must be in [0, 1], bad Excel rows: {bad_rows}")

    unique_scores = int(scores.round(12).nunique())
    required_unique_scores = min(min_unique_scores, len(scores))
    if unique_scores < required_unique_scores:
        raise ValueError(
            f"text_prediction appears collapsed: {unique_scores} unique values, "
            f"expected at least {required_unique_scores}"
        )

    sheet_names = pd.ExcelFile(submission_xlsx).sheet_names
    if "time" not in sheet_names:
        raise ValueError(f"{submission_xlsx} must contain a 'time' sheet")

    return {
        "status": "ok",
        "input_csv": str(input_csv),
        "submission_xlsx": str(submission_xlsx),
        "rows": len(prediction_frame),
        "columns": expected_columns,
        "score_min": float(scores.min()) if len(scores) else 0.0,
        "score_max": float(scores.max()) if len(scores) else 0.0,
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "unique_scores": unique_scores,
        "sheets": sheet_names,
    }
