from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from textgenadvtrack.io import save_csv


@dataclass(frozen=True)
class CompletedDatasetInputs:
    official_val_with_label_csv: Path
    output_dir: Path
    dev_fraction: float = 0.2
    seed: int = 42
    language: str = "zh"


def _rewrite_machine_text(text: str) -> tuple[str, str]:
    parts = str(text).split()
    if len(parts) >= 10:
        midpoint = len(parts) // 2
        return "reorder", " ".join(parts[midpoint:] + parts[:midpoint])
    if len(parts) >= 4:
        return "compress", " ".join(parts[: max(1, len(parts) // 2)])
    return "expand", f"{text} In short, the same answer is expressed in a more direct way."


def _row(
    sample_id: str,
    split: str,
    label: int,
    text_type: str,
    text: str,
    prompt: str,
    language: str,
    parent_id: str = "",
    rewrite_type: str = "",
) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "language": language,
        "domain": "official_val",
        "label": label,
        "text_type": text_type,
        "source_name": "official_val_completed",
        "source_model": "" if label == 1 else "unknown",
        "prompt_type": "official_prompt",
        "prompt_id": sample_id.replace("completed-", "prompt-"),
        "rewrite_type": rewrite_type,
        "parent_id": parent_id,
        "text": str(text),
        "_prompt": str(prompt),
    }


def _stratified_dev_indices(frame: pd.DataFrame, dev_fraction: float, seed: int) -> set[int]:
    dev_indices: set[int] = set()
    for _, group in frame.groupby("label", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed)
        dev_count = max(1, int(round(len(shuffled) * dev_fraction)))
        dev_indices.update(shuffled.head(dev_count).index.tolist())
    return dev_indices


def _write_rows(rows: list[dict], output_csv: Path) -> None:
    cleaned = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    save_csv(cleaned, output_csv)


def build_completed_detection_dataset(inputs: CompletedDatasetInputs) -> dict:
    if not 0.0 < inputs.dev_fraction < 0.5:
        raise ValueError("dev_fraction must be between 0 and 0.5")

    frame = pd.read_csv(inputs.official_val_with_label_csv).fillna("")
    missing = [column for column in ["prompt", "text", "label"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{inputs.official_val_with_label_csv} missing required columns: {missing}")
    frame = frame[["prompt", "text", "label"]].copy()
    frame["label"] = frame["label"].astype(int)

    dev_indices = _stratified_dev_indices(frame, inputs.dev_fraction, inputs.seed)
    human_counter = 1
    machine_counter = 1
    rows = []
    for index, record in frame.iterrows():
        split = "dev" if index in dev_indices else "train"
        label = int(record["label"])
        if label == 1:
            sample_id = f"completed-human-{human_counter:05d}"
            rows.append(
                _row(
                    sample_id=sample_id,
                    split=split,
                    label=1,
                    text_type="human",
                    text=record["text"],
                    prompt=record["prompt"],
                    language=inputs.language,
                )
            )
            human_counter += 1
        elif label == 0:
            sample_id = f"completed-ai-{machine_counter:05d}"
            rows.append(
                _row(
                    sample_id=sample_id,
                    split=split,
                    label=0,
                    text_type="ai_original",
                    text=record["text"],
                    prompt=record["prompt"],
                    language=inputs.language,
                )
            )
            rewrite_type, rewritten = _rewrite_machine_text(record["text"])
            rows.append(
                _row(
                    sample_id=f"completed-rewrite-{machine_counter:05d}",
                    split="rewrite_dev" if split == "dev" else "train",
                    label=0,
                    text_type="ai_rewritten",
                    text=rewritten,
                    prompt=record["prompt"],
                    language=inputs.language,
                    parent_id=sample_id,
                    rewrite_type=rewrite_type,
                )
            )
            machine_counter += 1
        else:
            raise ValueError("label values must be 0 or 1")

    available_train_rows = [row for row in rows if row["split"] == "train"]
    human_train = [row for row in available_train_rows if row["label"] == 1]
    original_train = [row for row in available_train_rows if row["text_type"] == "ai_original"]
    rewritten_train = [row for row in available_train_rows if row["text_type"] == "ai_rewritten"]
    machine_target = len(human_train)
    original_target = machine_target // 2
    rewritten_target = machine_target - original_target
    train_rows = human_train + original_train[:original_target] + rewritten_train[:rewritten_target]

    dev_rows = [row for row in rows if row["split"] == "dev"]
    dev_human_by_order = [row for row in dev_rows if row["label"] == 1]
    rewritten_dev = [row for row in rows if row["split"] == "rewrite_dev"]
    rewrite_dev_human = [
        {**row, "sample_id": row["sample_id"].replace("completed-human", "completed-rewrite-human"), "split": "rewrite_dev"}
        for row in dev_human_by_order[: len(rewritten_dev)]
    ]
    rewrite_dev_rows = rewrite_dev_human + rewritten_dev
    all_rows = rows

    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(all_rows, inputs.output_dir / "completed_all.csv")
    _write_rows(train_rows, inputs.output_dir / "completed_train.csv")
    _write_rows(dev_rows, inputs.output_dir / "completed_dev.csv")
    _write_rows(rewrite_dev_rows, inputs.output_dir / "completed_rewrite_dev.csv")

    inventory = pd.DataFrame(
        [
            {"file": "completed_all.csv", "rows": len(all_rows), "purpose": "all_completed_detection_rows"},
            {"file": "completed_train.csv", "rows": len(train_rows), "purpose": "main_train"},
            {"file": "completed_dev.csv", "rows": len(dev_rows), "purpose": "original_dev"},
            {"file": "completed_rewrite_dev.csv", "rows": len(rewrite_dev_rows), "purpose": "rewrite_dev"},
        ]
    )
    inventory.to_csv(inputs.output_dir / "completed_inventory.csv", index=False)

    def counts(subset: list[dict], key: str) -> dict:
        return pd.Series([row[key] for row in subset]).value_counts().to_dict() if subset else {}

    report = {
        "output_dir": str(inputs.output_dir),
        "all_rows": len(all_rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "rewrite_dev_rows": len(rewrite_dev_rows),
        "train_label_counts": counts(train_rows, "label"),
        "train_text_type_counts": counts(train_rows, "text_type"),
    }
    pd.DataFrame([report]).to_json(inputs.output_dir / "completed_report.json", orient="records", force_ascii=False, indent=2)
    return report
