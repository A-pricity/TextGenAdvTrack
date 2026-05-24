from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_training_batch(
    existing_csv: Path,
    batch_csv: Path,
    output_csv: Path | None = None,
    dedupe_on: list[str] | None = None,
) -> dict:
    target_csv = output_csv or existing_csv
    existing = pd.read_csv(existing_csv).fillna("")
    batch = pd.read_csv(batch_csv).fillna("")

    if list(existing.columns) != list(batch.columns):
        raise ValueError("Batch columns do not match existing dataset columns")

    combined = pd.concat([existing, batch], ignore_index=True)
    dedupe_columns = dedupe_on or ["sample_id"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedupe_columns, keep="first")
    removed = before - len(combined)

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target_csv, index=False)
    return {
        "existing_rows": int(len(existing)),
        "batch_rows": int(len(batch)),
        "final_rows": int(len(combined)),
        "duplicates_removed": int(removed),
        "output_csv": str(target_csv),
    }
