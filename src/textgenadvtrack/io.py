from pathlib import Path

import pandas as pd

from textgenadvtrack.schemas import (
    DetectionRow,
    DetectionSubmitInputRow,
    EvasionCandidateRow,
    EvasionOfficialInputRow,
    EvasionSelectedRow,
    EvasionSourceRow,
)


def load_detection_rows(csv_path: Path) -> list[DetectionRow]:
    frame = pd.read_csv(csv_path).fillna("")
    return [DetectionRow(**record) for record in frame.to_dict(orient="records")]


def load_evasion_source_rows(csv_path: Path) -> list[EvasionSourceRow]:
    frame = pd.read_csv(csv_path).fillna("")
    return [EvasionSourceRow(**record) for record in frame.to_dict(orient="records")]


def load_evasion_candidate_rows(csv_path: Path) -> list[EvasionCandidateRow]:
    frame = pd.read_csv(csv_path).fillna("")
    return [EvasionCandidateRow(**record) for record in frame.to_dict(orient="records")]


def save_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def save_selected_rows(rows: list[EvasionSelectedRow], csv_path: Path) -> None:
    save_csv([row.model_dump() for row in rows], csv_path)


def load_detection_submit_rows(csv_path: Path) -> list[DetectionSubmitInputRow]:
    frame = pd.read_csv(csv_path).fillna("")
    return [DetectionSubmitInputRow(**record) for record in frame.to_dict(orient="records")]


def load_evasion_official_rows(csv_path: Path) -> list[EvasionOfficialInputRow]:
    frame = pd.read_csv(csv_path).fillna("")
    return [EvasionOfficialInputRow(**record) for record in frame.to_dict(orient="records")]
