from pydantic import ValidationError

from textgenadvtrack.cli import build_parser
from textgenadvtrack.schemas import DetectionRow, EvasionCandidateRow


def test_detection_row_accepts_valid_ai_rewritten_row():
    row = DetectionRow(
        sample_id="det-0001",
        split="train",
        language="zh",
        domain="qa",
        label=0,
        text_type="ai_rewritten",
        source_name="self_generated",
        source_model="Model-C1",
        prompt_type="question_answering",
        prompt_id="p-001",
        rewrite_type="paraphrase",
        parent_id="det-parent-1",
        text="这是一个改写后的机器文本。",
    )
    assert row.text_type == "ai_rewritten"


def test_detection_row_rejects_rewritten_without_parent_id():
    try:
        DetectionRow(
            sample_id="det-0002",
            split="train",
            language="en",
            domain="news",
            label=0,
            text_type="ai_rewritten",
            source_name="self_generated",
            source_model="Model-O1",
            prompt_type="summary",
            prompt_id="p-002",
            rewrite_type="compress",
            parent_id="",
            text="Rewritten machine text",
        )
    except ValidationError:
        assert True
        return
    assert False, "Expected ValidationError"


def test_evasion_candidate_accepts_proxy_scores():
    row = EvasionCandidateRow(
        candidate_id="cand-001",
        parent_id="eva-001",
        language="ru",
        domain="daily",
        source_model="Model-C2",
        prompt_type="daily_writing",
        rewrite_model="Model-C1",
        rewrite_type="paraphrase",
        semantic_score=0.91,
        proxy_score_1=0.77,
        proxy_score_2=0.72,
        selected=False,
        text="Переписанный текст",
    )
    assert row.proxy_score_1 > 0.7


def test_cli_exposes_detection_and_evasion_commands():
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices.keys()
    assert "validate-detection" in subcommands
    assert "train-detector" in subcommands
    assert "select-evasion" in subcommands
    assert "ingest-external-detection-dataset" in subcommands
