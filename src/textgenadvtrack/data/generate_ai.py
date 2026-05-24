from __future__ import annotations

from pathlib import Path

import pandas as pd

from textgenadvtrack.io import save_csv
from textgenadvtrack.llm.providers import generate_completion


def build_generation_instruction(language: str, domain: str) -> str:
    return (
        "You are generating benchmark training data for AI-generated text detection. "
        f"Language={language}. Domain={domain}. "
        "Write a natural, coherent answer with enough detail to look realistic."
    )


def generate_ai_original_data(
    prompt_registry_csv: Path,
    output_csv: Path,
    model_name: str,
    provider: str = "mock",
    language: str | None = None,
    max_prompts: int | None = None,
) -> dict:
    frame = pd.read_csv(prompt_registry_csv).fillna("")
    if language:
        frame = frame[frame["language"] == language]
    if max_prompts is not None:
        frame = frame.head(max_prompts)

    rows = []
    for index, record in enumerate(frame.to_dict(orient="records"), start=1):
        rows.append(
            {
                "sample_id": f"{record['language']}-ai-{index:04d}",
                "language": record["language"],
                "domain": record["prompt_type"],
                "label": 0,
                "text_type": "ai_original",
                "source_name": "self_generated",
                "source_model": model_name,
                "prompt_type": record["prompt_type"],
                "prompt_id": record["prompt_id"],
                "rewrite_type": "",
                "parent_id": "",
                "text": generate_completion(
                    provider=provider,
                    model_name=model_name,
                    system_prompt=build_generation_instruction(
                        language=record["language"],
                        domain=record["prompt_type"],
                    ),
                    prompt=record["prompt_text"],
                ),
            }
        )

    save_csv(rows, output_csv)
    return {
        "rows": len(rows),
        "output_csv": str(output_csv),
        "model_name": model_name,
        "provider": provider,
    }
