from collections import defaultdict
from pathlib import Path

from textgenadvtrack.detection.predict import predict_human_scores
from textgenadvtrack.io import load_evasion_candidate_rows, save_selected_rows
from textgenadvtrack.schemas import EvasionSelectedRow


def choose_best_candidate(candidates: list[dict]) -> dict:
    def rank_key(candidate: dict):
        joint = candidate["proxy_score_1"] + candidate["proxy_score_2"]
        return (joint, candidate["semantic_score"])

    return max(candidates, key=rank_key)


def heuristic_semantic_score(parent_id: str, text: str) -> float:
    return min(1.0, 0.5 + (len(text) / max(len(parent_id), 1)) * 0.01)


def heuristic_humanity_score(text: str) -> float:
    unique_ratio = len(set(text)) / max(len(text), 1)
    punctuation_bonus = 0.1 if any(char in text for char in ",.;:!?，。！？") else 0.0
    return min(1.0, 0.4 + unique_ratio + punctuation_bonus)


def score_and_select_candidates(candidates_csv: Path, output_csv: Path, model_dir: Path | None = None) -> list[EvasionSelectedRow]:
    candidates = load_evasion_candidate_rows(candidates_csv)
    candidate_dicts = [row.model_dump() for row in candidates]

    if model_dir is not None:
        scores = predict_human_scores([row["text"] for row in candidate_dicts], model_dir)
        for row, score in zip(candidate_dicts, scores, strict=True):
            row["proxy_score_1"] = score
    else:
        for row in candidate_dicts:
            row["proxy_score_1"] = heuristic_humanity_score(row["text"])

    for row in candidate_dicts:
        row["proxy_score_2"] = heuristic_humanity_score(row["text"])
        row["semantic_score"] = heuristic_semantic_score(row["parent_id"], row["text"])

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_dicts:
        grouped[row["parent_id"]].append(row)

    selected_rows: list[EvasionSelectedRow] = []
    for parent_id, rows in grouped.items():
        best = choose_best_candidate(rows)
        selected_rows.append(
            EvasionSelectedRow(
                sample_id=best["candidate_id"],
                parent_id=parent_id,
                language=best["language"],
                final_text=best["text"],
                proxy_score_1=float(best["proxy_score_1"]),
                proxy_score_2=float(best["proxy_score_2"]),
                selection_reason=f"best_{best['rewrite_type']}_joint_proxy",
            )
        )

    save_selected_rows(selected_rows, output_csv)
    return selected_rows
