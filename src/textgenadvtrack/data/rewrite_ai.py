from __future__ import annotations

from pathlib import Path

import pandas as pd

from textgenadvtrack.io import save_csv
from textgenadvtrack.llm.providers import generate_completion


def mock_rewrite_text(text: str, rewrite_type: str, model_name: str) -> str:
    if rewrite_type == "paraphrase":
        return f"[{model_name}] Restated version: {text}"
    if rewrite_type == "reorder":
        parts = text.split()
        if len(parts) > 6:
            midpoint = len(parts) // 2
            return " ".join(parts[midpoint:] + parts[:midpoint])
    if rewrite_type == "compress":
        parts = text.split()
        return " ".join(parts[: max(1, len(parts) // 2)])
    if rewrite_type == "expand":
        return f"{text} Additional natural elaboration is added here for a more human-like style."
    if rewrite_type == "synonym":
        return text.replace("generated", "produced").replace("placeholder", "draft")
    return text


def build_rewrite_instruction(rewrite_type: str, language: str) -> str:
    return (
        "Rewrite the user text to reduce obvious machine-writing patterns while preserving meaning. "
        f"Language={language}. Rewrite strategy={rewrite_type}. "
        "Do not add unsupported facts. Keep the output fluent and human-like."
    )


def generate_ai_rewritten_data(
    ai_original_csv: Path,
    rewrite_registry_csv: Path,
    output_csv: Path,
    model_name: str,
    provider: str = "mock",
    max_rows: int | None = None,
) -> dict:
    original = pd.read_csv(ai_original_csv).fillna("")
    rewrite_registry = pd.read_csv(rewrite_registry_csv).fillna("")
    rewrite_types = rewrite_registry["rewrite_type"].astype(str).tolist()
    if max_rows is not None:
        original = original.head(max_rows)

    rows = []
    counter = 1
    for record in original.to_dict(orient="records"):
        for rewrite_type in rewrite_types:
            rewritten_text = (
                mock_rewrite_text(record["text"], rewrite_type, model_name)
                if provider == "mock"
                else generate_completion(
                    provider=provider,
                    model_name=model_name,
                    system_prompt=build_rewrite_instruction(rewrite_type, record["language"]),
                    prompt=record["text"],
                )
            )
            rows.append(
                {
                    "sample_id": f"{record['language']}-re-{counter:04d}",
                    "language": record["language"],
                    "domain": record["domain"],
                    "label": 0,
                    "text_type": "ai_rewritten",
                    "source_name": "self_generated",
                    "source_model": model_name,
                    "prompt_type": record["prompt_type"],
                    "prompt_id": record["prompt_id"],
                    "rewrite_type": rewrite_type,
                    "parent_id": record["sample_id"],
                    "text": rewritten_text,
                }
            )
            counter += 1

    save_csv(rows, output_csv)
    return {
        "rows": len(rows),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "provider": provider,
    }
