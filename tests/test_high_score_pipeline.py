from pathlib import Path

import pandas as pd

from textgenadvtrack.data.augment import build_adversarial_training_rows
from textgenadvtrack.detection.calibration import tune_scores
from textgenadvtrack.detection.ensemble import blend_prediction_files, search_blend_weights
from textgenadvtrack.detection.predict import export_detection_scores
from textgenadvtrack.detection.train import train_detector
from textgenadvtrack.detection.validation import build_repeated_detection_splits, evaluate_prediction_slices


def test_build_repeated_detection_splits_writes_seeded_splits(tmp_path):
    source = tmp_path / "val.csv"
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

    result = build_repeated_detection_splits(source, tmp_path / "splits", seeds=[1, 2], dev_fraction=0.25)

    assert result["folds"] == 2
    assert (tmp_path / "splits" / "seed_1" / "official_train.csv").exists()
    assert (tmp_path / "splits" / "seed_2" / "official_dev.csv").exists()


def test_evaluate_prediction_slices_reports_overall_and_groups(tmp_path):
    labels = tmp_path / "labels.csv"
    preds = tmp_path / "preds.xlsx"
    labels.write_text("label,language\n1,zh\n0,zh\n1,en\n0,en\n")
    pd.DataFrame(
        {
            "prompt": ["p1", "p2", "p3", "p4"],
            "text_prediction": [0.9, 0.1, 0.8, 0.2],
        }
    ).to_excel(preds, sheet_name="predictions", index=False)

    report = evaluate_prediction_slices(labels, preds, group_columns=["language"])

    assert report["overall"]["AUC"] == 1.0
    assert "language=zh" in report["groups"]
    assert report["groups"]["language=en"]["ACC"] == 1.0


def test_search_blend_weights_finds_best_weighted_average(tmp_path):
    labels = tmp_path / "labels.csv"
    model_a = tmp_path / "a.csv"
    model_b = tmp_path / "b.csv"
    labels.write_text("label\n1\n0\n1\n0\n")
    pd.DataFrame({"text_prediction": [0.9, 0.1, 0.7, 0.3]}).to_csv(model_a, index=False)
    pd.DataFrame({"text_prediction": [0.2, 0.8, 0.3, 0.7]}).to_csv(model_b, index=False)

    result = search_blend_weights(labels, [model_a, model_b], step=0.5)

    assert result["metrics"]["AUC"] == 1.0
    assert result["weights"][0] > result["weights"][1]


def test_blend_prediction_files_writes_submission_shape(tmp_path):
    out = tmp_path / "blend.xlsx"
    p1 = tmp_path / "p1.csv"
    p2 = tmp_path / "p2.csv"
    pd.DataFrame({"prompt": ["a", "b"], "text_prediction": [0.8, 0.2]}).to_csv(p1, index=False)
    pd.DataFrame({"prompt": ["a", "b"], "text_prediction": [0.6, 0.4]}).to_csv(p2, index=False)

    result = blend_prediction_files([p1, p2], [0.75, 0.25], out)

    assert result["rows"] == 2
    assert list(pd.read_excel(out, sheet_name="predictions").columns) == ["prompt", "text_prediction"]


def test_tune_scores_can_improve_fixed_threshold_metrics():
    labels = [1, 0, 1, 0]
    scores = [0.49, 0.48, 0.7, 0.2]

    result = tune_scores(labels, scores, biases=[-0.1, 0.0, 0.015], scales=[1.0])

    assert result["best_metrics"]["ACC"] == 1.0
    assert result["best_bias"] > 0.0


def test_build_adversarial_training_rows_adds_machine_rewrites(tmp_path):
    source = Path("tests/fixtures/detection_rows.csv")
    output = tmp_path / "augmented.csv"

    result = build_adversarial_training_rows(source, output, max_rows=2)

    frame = pd.read_csv(output)
    assert result["rows"] == len(frame)
    assert set(frame["label"]) == {0}
    assert set(frame["text_type"]) == {"ai_rewritten"}


def test_export_detection_scores_writes_labels_and_predictions(tmp_path):
    model_dir = tmp_path / "model"
    output_csv = tmp_path / "scores.csv"
    train_detector(
        Path("tests/fixtures/detection_rows.csv"),
        Path("tests/fixtures/detection_rows.csv"),
        "microsoft/deberta-v3-base",
        model_dir,
        backend="classic_plus",
    )

    result = export_detection_scores(Path("tests/fixtures/detection_rows.csv"), model_dir, output_csv)

    frame = pd.read_csv(output_csv)
    assert result["rows"] == 4
    assert {"sample_id", "label", "text_prediction"}.issubset(frame.columns)
