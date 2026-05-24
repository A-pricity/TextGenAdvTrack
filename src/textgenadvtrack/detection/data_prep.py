from collections import Counter
from pathlib import Path

from textgenadvtrack.io import load_detection_rows


def summarize_detection_split(csv_path: Path) -> dict:
    rows = load_detection_rows(csv_path)
    return {
        "rows": len(rows),
        "label_counts": dict(Counter(row.label for row in rows)),
        "text_type_counts": dict(Counter(row.text_type for row in rows)),
        "language_counts": dict(Counter(row.language for row in rows)),
    }


def detection_examples(csv_path: Path) -> tuple[list[str], list[int]]:
    rows = load_detection_rows(csv_path)
    return [row.text for row in rows], [row.label for row in rows]
