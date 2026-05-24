from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from textgenadvtrack.io import save_csv


@dataclass(frozen=True)
class OfficialSplitInputs:
    val_with_label_csv: Path
    output_dir: Path
    dev_fraction: float = 0.2
    language: str = "zh"
    seed: int = 42


def _to_detection_rows(frame: pd.DataFrame, split: str, language: str) -> list[dict]:
    rows = []
    for index, record in enumerate(frame.to_dict(orient="records"), start=1):
        label = int(record["label"])
        rows.append(
            {
                "sample_id": f"official-{split}-{index:05d}",
                "split": split,
                "language": language,
                "domain": "official_val",
                "label": label,
                "text_type": "human" if label == 1 else "ai_original",
                "source_name": "official_val",
                "source_model": "" if label == 1 else "unknown",
                "prompt_type": "official_prompt",
                "prompt_id": f"official-{split}-prompt-{index:05d}",
                "rewrite_type": "",
                "parent_id": "",
                "text": str(record["text"]),
            }
        )
    return rows


def build_official_detection_splits(inputs: OfficialSplitInputs) -> dict:
    if not 0.0 < inputs.dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    if inputs.language not in {"zh", "en", "ru"}:
        raise ValueError("language must be one of: zh, en, ru")

    frame = pd.read_csv(inputs.val_with_label_csv).fillna("")
    missing = [column for column in ["prompt", "text", "label"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{inputs.val_with_label_csv} missing required columns: {missing}")
    frame = frame[["prompt", "text", "label"]].copy()
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"]).issubset({0, 1}):
        raise ValueError("label values must be 0 or 1")

    dev_parts = []
    train_parts = []
    for _, label_frame in frame.groupby("label", sort=True):
        shuffled = label_frame.sample(frac=1.0, random_state=inputs.seed).reset_index(drop=True)
        dev_count = max(1, int(round(len(shuffled) * inputs.dev_fraction)))
        dev_parts.append(shuffled.head(dev_count))
        train_parts.append(shuffled.iloc[dev_count:])

    dev = pd.concat(dev_parts, ignore_index=True).sample(frac=1.0, random_state=inputs.seed).reset_index(drop=True)
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=inputs.seed).reset_index(drop=True)

    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _to_detection_rows(train, "train", inputs.language)
    dev_rows = _to_detection_rows(dev, "dev", inputs.language)
    all_rows = _to_detection_rows(frame.sample(frac=1.0, random_state=inputs.seed).reset_index(drop=True), "train", inputs.language)

    save_csv(train_rows, inputs.output_dir / "official_train.csv")
    save_csv(dev_rows, inputs.output_dir / "official_dev.csv")
    save_csv(all_rows, inputs.output_dir / "official_all_train.csv")
    pd.DataFrame(
        [
            {"file": "official_train.csv", "rows": len(train_rows)},
            {"file": "official_dev.csv", "rows": len(dev_rows)},
            {"file": "official_all_train.csv", "rows": len(all_rows)},
        ]
    ).to_csv(inputs.output_dir / "official_split_inventory.csv", index=False)

    return {
        "output_dir": str(inputs.output_dir),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "all_train_rows": len(all_rows),
        "dev_fraction": inputs.dev_fraction,
    }
