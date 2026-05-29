from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_evasion_submission(official_input_csv: Path, submission_csv: Path) -> dict:
    if not official_input_csv.exists():
        raise FileNotFoundError(f"Missing official input CSV: {official_input_csv}")
    if not submission_csv.exists():
        raise FileNotFoundError(f"Missing submission CSV: {submission_csv}")

    official = pd.read_csv(official_input_csv).fillna("")
    submission = pd.read_csv(submission_csv).fillna("")

    missing_official = [column for column in ["prompt", "text", "label"] if column not in official.columns]
    if missing_official:
        raise ValueError(f"{official_input_csv} missing required columns: {missing_official}")

    expected_columns = ["prompt", "text"]
    if list(submission.columns) != expected_columns:
        raise ValueError(
            f"{submission_csv} columns must be exactly {expected_columns}, got {list(submission.columns)}"
        )
    if len(submission) != len(official):
        raise ValueError(f"Row count mismatch: official has {len(official)}, submission has {len(submission)}")

    official_prompts = official["prompt"].astype(str).tolist()
    submission_prompts = submission["prompt"].astype(str).tolist()
    if official_prompts != submission_prompts:
        for index, (official_prompt, submission_prompt) in enumerate(
            zip(official_prompts, submission_prompts, strict=False),
            start=2,
        ):
            if official_prompt != submission_prompt:
                raise ValueError(
                    "Prompt alignment mismatch at CSV row "
                    f"{index}: official={official_prompt[:120]!r}, submission={submission_prompt[:120]!r}"
                )

    texts = submission["text"].astype(str)
    if (texts.str.len() == 0).any():
        bad_rows = (texts[texts.str.len() == 0].index + 2).tolist()[:5]
        raise ValueError(f"Submission contains empty text at CSV rows: {bad_rows}")

    labels = official["label"].astype(int)
    human_mask = labels == 1
    if not (official.loc[human_mask, "text"].astype(str).reset_index(drop=True) == texts[human_mask].reset_index(drop=True)).all():
        raise ValueError("Human rows must be preserved unchanged in the evasion submission")

    machine_mask = labels == 0
    changed_machine_rows = int(
        (
            official.loc[machine_mask, "text"].astype(str).reset_index(drop=True)
            != texts[machine_mask].reset_index(drop=True)
        ).sum()
    )

    return {
        "status": "ok",
        "official_input_csv": str(official_input_csv),
        "submission_csv": str(submission_csv),
        "rows": len(submission),
        "human_rows": int(human_mask.sum()),
        "machine_rows": int(machine_mask.sum()),
        "changed_machine_rows": changed_machine_rows,
    }
