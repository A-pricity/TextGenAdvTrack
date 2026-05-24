from pathlib import Path

import pandas as pd

from textgenadvtrack.evasion.export import build_evasion_submission
from textgenadvtrack.evasion.generate_candidates import generate_candidates
from textgenadvtrack.evasion.select_candidates import choose_best_candidate
from textgenadvtrack.evasion.select_candidates import score_and_select_candidates


def test_choose_best_candidate_prefers_joint_proxy_gain():
    candidates = [
        {"candidate_id": "c1", "proxy_score_1": 0.80, "proxy_score_2": 0.78, "semantic_score": 0.85},
        {"candidate_id": "c2", "proxy_score_1": 0.82, "proxy_score_2": 0.40, "semantic_score": 0.90},
    ]
    best = choose_best_candidate(candidates)
    assert best["candidate_id"] == "c1"


def test_choose_best_candidate_breaks_ties_with_semantics():
    candidates = [
        {"candidate_id": "c1", "proxy_score_1": 0.75, "proxy_score_2": 0.75, "semantic_score": 0.80},
        {"candidate_id": "c2", "proxy_score_1": 0.75, "proxy_score_2": 0.75, "semantic_score": 0.90},
    ]
    best = choose_best_candidate(candidates)
    assert best["candidate_id"] == "c2"


def test_generate_candidates_creates_six_candidates_per_source(tmp_path):
    output_csv = tmp_path / "candidates.csv"
    rows = generate_candidates(Path("tests/fixtures/evasion_source.csv"), output_csv)
    assert len(rows) == 12
    frame = pd.read_csv(output_csv)
    assert len(frame) == 12


def test_generate_candidates_accepts_official_unknown_language(tmp_path):
    source_csv = tmp_path / "official_source.csv"
    source_csv.write_text(
        "sample_id,language,domain,source_model,prompt_type,prompt_id,prompt,source_text\n"
        "eva-0001,unknown,official_val,unknown,unknown,p1,prompt text,机器文本一\n"
    )
    output_csv = tmp_path / "candidates.csv"

    rows = generate_candidates(source_csv, output_csv)

    assert len(rows) == 6
    assert pd.read_csv(output_csv).iloc[0]["language"] == "unknown"


def test_score_and_select_candidates_writes_one_row_per_parent(tmp_path):
    output_csv = tmp_path / "selected.csv"
    rows = score_and_select_candidates(Path("tests/fixtures/evasion_candidates.csv"), output_csv)
    assert len(rows) == 2
    frame = pd.read_csv(output_csv)
    assert len(frame) == 2


def test_build_evasion_submission_preserves_human_rows(tmp_path):
    selected_csv = tmp_path / "selected.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "c1",
                "parent_id": "eva-0001",
                "language": "zh",
                "final_text": "改写后的机器文本",
                "proxy_score_1": 0.9,
                "proxy_score_2": 0.8,
                "selection_reason": "best_paraphrase_joint_proxy",
            }
        ]
    ).to_csv(selected_csv, index=False)
    output_csv = tmp_path / "submission.csv"
    result = build_evasion_submission(
        Path("tests/fixtures/evasion_official_input.csv"),
        selected_csv,
        output_csv,
    )
    assert result["rows"] == 2
    frame = pd.read_csv(output_csv)
    assert frame.iloc[0]["text"] == "改写后的机器文本"
    assert frame.iloc[1]["text"] == "Human text stays unchanged."


def test_build_evasion_submission_maps_machine_only_ids_without_human_offset(tmp_path):
    official_csv = tmp_path / "official.csv"
    official_csv.write_text(
        "prompt,text,label\n"
        "human prompt,Human first,1\n"
        "machine prompt,Machine second,0\n"
    )
    selected_csv = tmp_path / "selected.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "c1",
                "parent_id": "eva-0001",
                "language": "unknown",
                "final_text": "Rewritten machine second",
                "proxy_score_1": 0.9,
                "proxy_score_2": 0.8,
                "selection_reason": "best_expand_joint_proxy",
            }
        ]
    ).to_csv(selected_csv, index=False)
    output_csv = tmp_path / "submission.csv"

    build_evasion_submission(official_csv, selected_csv, output_csv)

    frame = pd.read_csv(output_csv)
    assert frame.iloc[0]["text"] == "Human first"
    assert frame.iloc[1]["text"] == "Rewritten machine second"
