from pathlib import Path

import pandas as pd

from textgenadvtrack.data.public_data import build_merged_training_pool, ingest_public_data


def test_ingest_public_data_normalizes_rows(tmp_path):
    output_csv = tmp_path / "public_human.csv"
    result = ingest_public_data(
        Path("tests/fixtures/public_human_input.csv"),
        output_csv,
        source_name="public_zh_news",
        language="zh",
        domain="news",
    )
    assert result["rows"] == 2
    frame = pd.read_csv(output_csv)
    assert set(frame["text_type"]) == {"human"}
    assert set(frame["label"]) == {1}


def test_build_merged_training_pool_merges_and_dedupes(tmp_path):
    public_csv = tmp_path / "public_human.csv"
    ingest_public_data(
        Path("tests/fixtures/public_human_input.csv"),
        public_csv,
        source_name="public_zh_news",
        language="zh",
        domain="news",
    )
    duplicate_public_csv = tmp_path / "public_human_dup.csv"
    pd.read_csv(public_csv).to_csv(duplicate_public_csv, index=False)

    output_dir = tmp_path / "merged"
    result = build_merged_training_pool(
        human_csvs=[Path("datasets/training/human.csv"), public_csv, duplicate_public_csv],
        ai_original_csvs=[Path("datasets/training/ai_original.csv")],
        ai_rewritten_csvs=[Path("datasets/training/ai_rewritten.csv")],
        output_dir=output_dir,
    )
    assert result["merged_human_rows"] >= 74
    assert result["duplicates_removed"] >= 2
    assert (output_dir / "merged_human.csv").exists()
    assert (output_dir / "merged_pool_inventory.csv").exists()
