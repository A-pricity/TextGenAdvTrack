from pathlib import Path

import pandas as pd

from textgenadvtrack.data.complete import CompletedDatasetInputs, build_completed_detection_dataset


def test_build_completed_detection_dataset_adds_balanced_rewrite_split(tmp_path):
    source = tmp_path / "val_with_label.csv"
    source.write_text(
        "prompt,text,label\n"
        "p1,human one,1\n"
        "p2,machine one,0\n"
        "p3,human two,1\n"
        "p4,machine two,0\n"
        "p5,human three,1\n"
        "p6,machine three,0\n"
        "p7,human four,1\n"
        "p8,machine four,0\n"
    )

    result = build_completed_detection_dataset(
        CompletedDatasetInputs(
            official_val_with_label_csv=source,
            output_dir=tmp_path / "completed",
            dev_fraction=0.25,
            seed=7,
        )
    )

    assert result["all_rows"] == 12
    assert result["train_rows"] == 6
    assert result["dev_rows"] == 2
    assert result["rewrite_dev_rows"] == 2

    train = pd.read_csv(tmp_path / "completed" / "completed_train.csv")
    rewrite_dev = pd.read_csv(tmp_path / "completed" / "completed_rewrite_dev.csv")
    assert set(train["text_type"]) == {"human", "ai_original", "ai_rewritten"}
    assert set(rewrite_dev["text_type"]) == {"human", "ai_rewritten"}
    assert train["label"].value_counts().to_dict() == {1: 3, 0: 3}
