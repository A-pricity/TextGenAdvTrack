from pathlib import Path

import pytest

from textgenadvtrack.data.build_training_splits import SplitInputs, build_small_validation_splits
from textgenadvtrack.data.official_splits import OfficialSplitInputs, build_official_detection_splits
from textgenadvtrack.data.prepare_official import prepare_official_data


def test_prepare_official_data_builds_expected_outputs(tmp_path):
    raw_dir = Path("datasets/official/raw")
    result = prepare_official_data(raw_dir, tmp_path)
    assert result["val_rows"] == 12000
    assert (tmp_path / "detection" / "official" / "val_submit_input.csv").exists()
    assert (tmp_path / "evasion" / "official" / "val_machine_only.csv").exists()


def test_build_small_validation_splits_requires_enough_rows(tmp_path):
    with pytest.raises(ValueError):
        build_small_validation_splits(
            SplitInputs(
                human_csv=Path("tests/fixtures/training_human.csv"),
                ai_original_csv=Path("tests/fixtures/training_ai_original.csv"),
                ai_rewritten_csv=Path("tests/fixtures/training_ai_rewritten.csv"),
                output_dir=tmp_path / "splits",
            )
        )


def test_build_seed_validation_splits_succeeds_for_seed_scale_data(tmp_path):
    result = build_small_validation_splits(
            SplitInputs(
            human_csv=Path("datasets/training/human.csv"),
            ai_original_csv=Path("datasets/training/ai_original.csv"),
            ai_rewritten_csv=Path("datasets/training/ai_rewritten.csv"),
            output_dir=tmp_path / "seed_splits",
            profile="seed",
        )
    )
    assert result["profile"] == "seed"
    assert result["det_train_rows"] == 222
    assert result["det_dev_rows"] == 24
    assert result["det_rewrite_dev_rows"] == 24
    assert (tmp_path / "seed_splits" / "det_train.csv").exists()


def test_build_official_detection_splits_creates_detection_schema_rows(tmp_path):
    source = tmp_path / "val_with_label.csv"
    source.write_text(
        "prompt,text,label\n"
        "p1,human zh,1\n"
        "p2,machine zh,0\n"
        "p3,human en,1\n"
        "p4,machine en,0\n"
        "p5,human ru,1\n"
        "p6,machine ru,0\n"
    )

    result = build_official_detection_splits(
        OfficialSplitInputs(
            val_with_label_csv=source,
            output_dir=tmp_path / "official_splits",
            dev_fraction=0.5,
            language="zh",
        )
    )

    assert result["train_rows"] == 2
    assert result["dev_rows"] == 4
    assert (tmp_path / "official_splits" / "official_train.csv").exists()
    assert (tmp_path / "official_splits" / "official_dev.csv").exists()
