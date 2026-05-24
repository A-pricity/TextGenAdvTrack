from pathlib import Path

import pandas as pd

from textgenadvtrack.data.collect_human import collect_human_data
from textgenadvtrack.data.generate_ai import generate_ai_original_data
from textgenadvtrack.data.rewrite_ai import generate_ai_rewritten_data


def test_collect_human_data_normalizes_rows(tmp_path):
    output_csv = tmp_path / "human.csv"
    result = collect_human_data(
        Path("tests/fixtures/raw_human_input.csv"),
        output_csv,
        source_name="human_zh_qa",
        language="zh",
        domain="qa",
    )
    assert result["rows"] == 2
    frame = pd.read_csv(output_csv)
    assert frame.iloc[0]["label"] == 1
    assert frame.iloc[0]["text_type"] == "human"


def test_generate_ai_original_data_uses_prompt_registry(tmp_path):
    output_csv = tmp_path / "ai_original.csv"
    result = generate_ai_original_data(
        Path("data/manifests/prompt_registry.csv"),
        output_csv,
        model_name="Model-C1",
        provider="mock",
        language="zh",
        max_prompts=2,
    )
    assert result["rows"] == 2
    frame = pd.read_csv(output_csv)
    assert set(frame["text_type"]) == {"ai_original"}


def test_generate_ai_rewritten_data_expands_rewrite_types(tmp_path):
    ai_original_csv = tmp_path / "ai_original.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "zh-ai-0001",
                "language": "zh",
                "domain": "qa",
                "label": 0,
                "text_type": "ai_original",
                "source_name": "self_generated",
                "source_model": "Model-C1",
                "prompt_type": "question_answering",
                "prompt_id": "zh_qa_001",
                "rewrite_type": "",
                "parent_id": "",
                "text": "placeholder generated text",
            }
        ]
    ).to_csv(ai_original_csv, index=False)
    output_csv = tmp_path / "ai_rewritten.csv"
    result = generate_ai_rewritten_data(
        ai_original_csv,
        Path("data/manifests/rewrite_strategy_registry.csv"),
        output_csv,
        model_name="Model-C2",
        provider="mock",
        max_rows=1,
    )
    assert result["rows"] >= 4
    frame = pd.read_csv(output_csv)
    assert set(frame["text_type"]) == {"ai_rewritten"}
