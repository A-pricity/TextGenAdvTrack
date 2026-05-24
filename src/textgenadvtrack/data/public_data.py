from __future__ import annotations

from pathlib import Path

import pandas as pd

from textgenadvtrack.io import save_csv

REQUIRED_COLUMNS = [
    "sample_id",
    "language",
    "domain",
    "label",
    "text_type",
    "source_name",
    "source_model",
    "prompt_type",
    "prompt_id",
    "rewrite_type",
    "parent_id",
    "text",
]


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def ingest_public_data(
    input_csv: Path,
    output_csv: Path,
    source_name: str,
    language: str,
    domain: str,
    text_column: str = "text",
    text_type: str = "human",
    label: int = 1,
    source_model: str = "",
    prompt_type: str = "",
    prompt_id_prefix: str = "public",
    sample_prefix: str | None = None,
    start_index: int = 1,
) -> dict:
    frame = pd.read_csv(input_csv).fillna("")
    if text_column not in frame.columns:
        raise ValueError(f"Missing text column: {text_column}")
    if text_type not in {"human", "ai_original", "ai_rewritten"}:
        raise ValueError(f"Unsupported text_type: {text_type}")
    if text_type == "human" and label != 1:
        raise ValueError("human rows must use label=1")
    if text_type != "human" and label != 0:
        raise ValueError("machine rows must use label=0")

    prefix = sample_prefix or f"{language}-{_slugify(text_type)}-{_slugify(source_name)}"
    rows = []
    for offset, text in enumerate(frame[text_column].astype(str).tolist(), start=start_index):
        text = text.strip()
        if not text:
            continue
        rows.append(
            {
                "sample_id": f"{prefix}-{offset:04d}",
                "language": language,
                "domain": domain,
                "label": label,
                "text_type": text_type,
                "source_name": source_name,
                "source_model": source_model,
                "prompt_type": prompt_type,
                "prompt_id": f"{prompt_id_prefix}_{offset:04d}" if prompt_type else "",
                "rewrite_type": "",
                "parent_id": "",
                "text": text,
            }
        )

    save_csv(rows, output_csv)
    return {
        "rows": len(rows),
        "output_csv": str(output_csv),
        "source_name": source_name,
        "text_type": text_type,
    }


def build_merged_training_pool(
    human_csvs: list[Path],
    ai_original_csvs: list[Path],
    ai_rewritten_csvs: list[Path],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _merge(csvs: list[Path], expected_text_type: str, output_name: str) -> dict:
        frames = []
        sources = []
        for csv_path in csvs:
            frame = pd.read_csv(csv_path).fillna("")
            missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
            if missing:
                raise ValueError(f"{csv_path} missing required columns: {missing}")
            if not frame.empty and set(frame["text_type"]) != {expected_text_type}:
                raise ValueError(f"{csv_path} contains unexpected text_type values")
            frames.append(frame[REQUIRED_COLUMNS].copy())
            sources.append({"split": output_name, "source_csv": str(csv_path), "rows": int(len(frame))})

        if frames:
            merged = pd.concat(frames, ignore_index=True)
            before = len(merged)
            merged["_normalized_text"] = (
                merged["text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
            )
            merged = merged.drop_duplicates(subset=["sample_id"], keep="first")
            merged = merged.drop_duplicates(subset=["language", "text_type", "_normalized_text"], keep="first")
            removed = before - len(merged)
            merged = merged.drop(columns=["_normalized_text"])
        else:
            merged = pd.DataFrame(columns=REQUIRED_COLUMNS)
            removed = 0

        output_csv = output_dir / output_name
        merged.to_csv(output_csv, index=False)
        return {
            "output_csv": str(output_csv),
            "rows": int(len(merged)),
            "duplicates_removed": int(removed),
            "sources": sources,
        }

    human_result = _merge(human_csvs, "human", "merged_human.csv")
    ai_original_result = _merge(ai_original_csvs, "ai_original", "merged_ai_original.csv")
    ai_rewritten_result = _merge(ai_rewritten_csvs, "ai_rewritten", "merged_ai_rewritten.csv")

    inventory_rows = []
    for result in [human_result, ai_original_result, ai_rewritten_result]:
        inventory_rows.extend(result["sources"])
    pd.DataFrame(inventory_rows).to_csv(output_dir / "merged_pool_inventory.csv", index=False)

    return {
        "output_dir": str(output_dir),
        "merged_human_rows": human_result["rows"],
        "merged_ai_original_rows": ai_original_result["rows"],
        "merged_ai_rewritten_rows": ai_rewritten_result["rows"],
        "duplicates_removed": int(
            human_result["duplicates_removed"]
            + ai_original_result["duplicates_removed"]
            + ai_rewritten_result["duplicates_removed"]
        ),
    }
