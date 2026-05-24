from __future__ import annotations

from pathlib import Path

import pandas as pd

from textgenadvtrack.io import save_csv


def collect_human_data(
    input_csv: Path,
    output_csv: Path,
    source_name: str,
    language: str,
    domain: str,
    text_column: str = "text",
    start_index: int = 1,
) -> dict:
    frame = pd.read_csv(input_csv).fillna("")
    if text_column not in frame.columns:
        raise ValueError(f"Missing text column: {text_column}")

    rows = []
    for offset, text in enumerate(frame[text_column].astype(str).tolist(), start=start_index):
        text = text.strip()
        if not text:
            continue
        rows.append(
            {
                "sample_id": f"{language}-human-{offset:04d}",
                "language": language,
                "domain": domain,
                "label": 1,
                "text_type": "human",
                "source_name": source_name,
                "source_model": "",
                "prompt_type": "",
                "prompt_id": "",
                "rewrite_type": "",
                "parent_id": "",
                "text": text,
            }
        )

    save_csv(rows, output_csv)
    return {"rows": len(rows), "output_csv": str(output_csv)}
