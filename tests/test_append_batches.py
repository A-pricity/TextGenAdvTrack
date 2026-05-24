from pathlib import Path

import pandas as pd

from textgenadvtrack.data.append_batches import append_training_batch


def test_append_training_batch_adds_new_rows(tmp_path):
    existing_csv = tmp_path / "human_existing.csv"
    pd.read_csv(Path("datasets/training/human.csv")).head(2).to_csv(existing_csv, index=False)
    batch_csv = Path("tests/fixtures/human_batch.csv")
    output_csv = tmp_path / "human_merged.csv"
    result = append_training_batch(existing_csv, batch_csv, output_csv=output_csv)
    assert result["batch_rows"] == 2
    assert result["final_rows"] == result["existing_rows"] + 2
    frame = pd.read_csv(output_csv)
    assert "zh-human-0101" in set(frame["sample_id"])


def test_append_training_batch_dedupes_on_sample_id(tmp_path):
    existing_csv = Path("datasets/training/human.csv")
    batch_csv = tmp_path / "dup_batch.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "zh-human-0001",
                "language": "zh",
                "domain": "qa",
                "label": 1,
                "text_type": "human",
                "source_name": "human_zh_qa",
                "source_model": "",
                "prompt_type": "",
                "prompt_id": "",
                "rewrite_type": "",
                "parent_id": "",
                "text": "重复文本",
            }
        ]
    ).to_csv(batch_csv, index=False)
    output_csv = tmp_path / "human_deduped.csv"
    result = append_training_batch(existing_csv, batch_csv, output_csv=output_csv)
    assert result["duplicates_removed"] == 1
