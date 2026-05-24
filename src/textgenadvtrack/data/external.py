from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from textgenadvtrack.io import save_csv


@dataclass(frozen=True)
class ExternalDetectionDatasetInputs:
    input_path: Path
    output_csv: Path
    source_name: str
    language: str
    domain: str
    split: str = "train"
    text_column: str = "text"
    label_column: str = "label"
    source_model: str = ""
    prompt_type: str = ""
    prompt_id_prefix: str = "external"
    sample_prefix: str | None = None


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).fillna("")
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl").fillna("")
    if suffix == ".parquet":
        return pd.read_parquet(path).fillna("")
    raise ValueError(f"Unsupported external dataset format: {path.suffix}")


def _coerce_label(value) -> int:
    if isinstance(value, bool):
        return 1 if value is False else 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) == 1:
            return 1
        if int(value) == 0:
            return 0
    normalized = str(value).strip().lower()
    if normalized in {"human", "human-written", "human_written", "humanauthored", "authored", "1", "true"}:
        return 1
    if normalized in {"machine", "ai", "ai-generated", "ai_generated", "generated", "0", "false"}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def _infer_text_type(label: int) -> str:
    return "human" if label == 1 else "ai_original"


def ingest_external_detection_dataset(inputs: ExternalDetectionDatasetInputs) -> dict:
    if inputs.split not in {"train", "dev", "rewrite_dev"}:
        raise ValueError("split must be one of train, dev, rewrite_dev")

    frame = _read_tabular(inputs.input_path)
    if inputs.text_column not in frame.columns:
        raise ValueError(f"Missing text column: {inputs.text_column}")
    if inputs.label_column not in frame.columns:
        raise ValueError(f"Missing label column: {inputs.label_column}")

    prefix = inputs.sample_prefix or f"{inputs.language}-{inputs.source_name}"
    rows = []
    for offset, record in enumerate(frame.to_dict(orient="records"), start=1):
        text = str(record[inputs.text_column]).strip()
        if not text:
            continue
        label = _coerce_label(record[inputs.label_column])
        rows.append(
            {
                "sample_id": f"{prefix}-{offset:05d}",
                "split": inputs.split,
                "language": inputs.language,
                "domain": inputs.domain,
                "label": label,
                "text_type": _infer_text_type(label),
                "source_name": inputs.source_name,
                "source_model": inputs.source_model,
                "prompt_type": inputs.prompt_type,
                "prompt_id": f"{inputs.prompt_id_prefix}_{offset:05d}" if inputs.prompt_type else "",
                "rewrite_type": "",
                "parent_id": "",
                "text": text,
            }
        )

    save_csv(rows, inputs.output_csv)
    return {
        "rows": len(rows),
        "output_csv": str(inputs.output_csv),
        "source_name": inputs.source_name,
        "language": inputs.language,
        "domain": inputs.domain,
        "split": inputs.split,
    }
