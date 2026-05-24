from __future__ import annotations

from pathlib import Path

from textgenadvtrack.io import load_detection_rows, save_csv


def _rewrite_variants(text: str) -> list[tuple[str, str]]:
    words = text.split()
    variants = [
        ("prefix_trim", text.replace("Please consider this restated version:", "").replace("Restated:", "").strip()),
        ("casual_expand", f"{text} 这句话也可以换成更自然的表达，但核心意思保持不变。"),
    ]
    if len(words) >= 6:
        midpoint = len(words) // 2
        variants.append(("clause_reorder", " ".join(words[midpoint:] + words[:midpoint])))
        variants.append(("shorten", " ".join(words[:midpoint])))
    else:
        variants.append(("punctuation_shift", text.replace("。", "，").replace(".", ",")))
    return [(rewrite_type, rewritten.strip()) for rewrite_type, rewritten in variants if rewritten.strip()]


def build_adversarial_training_rows(
    detection_csv: Path,
    output_csv: Path,
    max_rows: int | None = None,
) -> dict:
    rows = load_detection_rows(detection_csv)
    machine_rows = [row for row in rows if row.label == 0]
    if max_rows is not None:
        machine_rows = machine_rows[:max_rows]

    augmented = []
    counter = 1
    for row in machine_rows:
        for rewrite_type, rewritten in _rewrite_variants(row.text):
            augmented.append(
                {
                    "sample_id": f"adv-{counter:06d}",
                    "language": row.language,
                    "domain": row.domain,
                    "label": 0,
                    "text_type": "ai_rewritten",
                    "source_name": f"{row.source_name}_adversarial",
                    "source_model": row.source_model or "unknown",
                    "prompt_type": row.prompt_type or "",
                    "prompt_id": row.prompt_id or "",
                    "rewrite_type": rewrite_type,
                    "parent_id": row.sample_id,
                    "text": rewritten,
                }
            )
            counter += 1

    save_csv(augmented, output_csv)
    return {"rows": len(augmented), "output_csv": str(output_csv)}
