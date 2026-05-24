from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

SPLIT_PROFILES = {
    "small": {
        "train_human": {"zh": 1200, "en": 1050, "ru": 750},
        "train_ai_original": {"zh": 1200, "en": 1050, "ru": 750},
        "train_ai_rewritten": {"zh": 480, "en": 420, "ru": 300},
        "dev_human": {"zh": 200, "en": 175, "ru": 125},
        "dev_ai_original": {"zh": 200, "en": 175, "ru": 125},
        "rewrite_dev_human": {"zh": 120, "en": 105, "ru": 75},
        "rewrite_dev_ai_rewritten": {"zh": 120, "en": 105, "ru": 75},
    },
    "seed": {
        "dev_human": {"zh": 4, "en": 4, "ru": 4},
        "dev_ai_original": {"zh": 4, "en": 4, "ru": 4},
        "rewrite_dev_human": {"zh": 4, "en": 4, "ru": 4},
        "rewrite_dev_ai_rewritten": {"zh": 4, "en": 4, "ru": 4},
    },
}


@dataclass(frozen=True)
class SplitInputs:
    human_csv: Path
    ai_original_csv: Path
    ai_rewritten_csv: Path
    output_dir: Path
    profile: str = "small"


def _load_rows(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path).fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    return frame[REQUIRED_COLUMNS].copy()


def _with_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(1, "split", split_name)
    return frame


def _stratified_take(frame: pd.DataFrame, per_language: dict[str, int]) -> pd.DataFrame:
    parts = []
    for language, count in per_language.items():
        lang_frame = frame[frame["language"] == language]
        if len(lang_frame) < count:
            raise ValueError(f"Not enough rows for language={language}. Need {count}, found {len(lang_frame)}")
        parts.append(lang_frame.head(count))
    return pd.concat(parts, ignore_index=True)


def _shuffled(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_small_validation_splits(inputs: SplitInputs) -> dict:
    human = _shuffled(_load_rows(inputs.human_csv))
    ai_original = _shuffled(_load_rows(inputs.ai_original_csv))
    ai_rewritten = _shuffled(_load_rows(inputs.ai_rewritten_csv))
    if inputs.profile not in SPLIT_PROFILES:
        raise ValueError(f"Unsupported split profile: {inputs.profile}")
    profile = SPLIT_PROFILES[inputs.profile]

    if inputs.profile == "seed":
        human_dev = _with_split(_stratified_take(human, profile["dev_human"]), "dev")
        human_after_dev = human[~human["sample_id"].isin(human_dev["sample_id"])]
        human_rewrite = _with_split(_stratified_take(human_after_dev, profile["rewrite_dev_human"]), "rewrite_dev")
        human_train = _with_split(
            human_after_dev[~human_after_dev["sample_id"].isin(human_rewrite["sample_id"])],
            "train",
        )

        ai_original_dev = _with_split(_stratified_take(ai_original, profile["dev_ai_original"]), "dev")
        ai_original_train = _with_split(
            ai_original[~ai_original["sample_id"].isin(ai_original_dev["sample_id"])],
            "train",
        )

        ai_rewritten_rewrite = _with_split(
            _stratified_take(ai_rewritten, profile["rewrite_dev_ai_rewritten"]),
            "rewrite_dev",
        )
        ai_rewritten_train = _with_split(
            ai_rewritten[~ai_rewritten["sample_id"].isin(ai_rewritten_rewrite["sample_id"])],
            "train",
        )

        det_train = pd.concat([human_train, ai_original_train, ai_rewritten_train], ignore_index=True)
        det_dev = pd.concat([human_dev, ai_original_dev], ignore_index=True)
        det_rewrite_dev = pd.concat([human_rewrite, ai_rewritten_rewrite], ignore_index=True)
    else:
        det_train = pd.concat(
            [
                _with_split(_stratified_take(human, profile["train_human"]), "train"),
                _with_split(_stratified_take(ai_original, profile["train_ai_original"]), "train"),
                _with_split(_stratified_take(ai_rewritten, profile["train_ai_rewritten"]), "train"),
            ],
            ignore_index=True,
        )

        human_remaining = human[~human["sample_id"].isin(det_train["sample_id"])]
        ai_original_remaining = ai_original[~ai_original["sample_id"].isin(det_train["sample_id"])]
        ai_rewritten_remaining = ai_rewritten[~ai_rewritten["sample_id"].isin(det_train["sample_id"])]

        det_dev = pd.concat(
            [
                _with_split(_stratified_take(human_remaining, profile["dev_human"]), "dev"),
                _with_split(_stratified_take(ai_original_remaining, profile["dev_ai_original"]), "dev"),
            ],
            ignore_index=True,
        )

        human_remaining = human_remaining[~human_remaining["sample_id"].isin(det_dev["sample_id"])]
        ai_rewritten_remaining = ai_rewritten_remaining[
            ~ai_rewritten_remaining["sample_id"].isin(det_dev["sample_id"])
        ]

        det_rewrite_dev = pd.concat(
            [
                _with_split(_stratified_take(human_remaining, profile["rewrite_dev_human"]), "rewrite_dev"),
                _with_split(
                    _stratified_take(ai_rewritten_remaining, profile["rewrite_dev_ai_rewritten"]),
                    "rewrite_dev",
                ),
            ],
            ignore_index=True,
        )

    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(det_train.to_dict(orient="records"), inputs.output_dir / "det_train.csv")
    save_csv(det_dev.to_dict(orient="records"), inputs.output_dir / "det_dev.csv")
    save_csv(det_rewrite_dev.to_dict(orient="records"), inputs.output_dir / "det_rewrite_dev.csv")

    inventory = pd.DataFrame(
        [
            {"file": "det_train.csv", "rows": len(det_train)},
            {"file": "det_dev.csv", "rows": len(det_dev)},
            {"file": "det_rewrite_dev.csv", "rows": len(det_rewrite_dev)},
        ]
    )
    inventory.to_csv(inputs.output_dir / "split_inventory.csv", index=False)

    return {
        "output_dir": str(inputs.output_dir),
        "profile": inputs.profile,
        "det_train_rows": int(len(det_train)),
        "det_dev_rows": int(len(det_dev)),
        "det_rewrite_dev_rows": int(len(det_rewrite_dev)),
    }
