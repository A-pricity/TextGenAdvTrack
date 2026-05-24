import pandas as pd
from pathlib import Path

from textgenadvtrack.detection.predict import export_detection_submission
from textgenadvtrack.detection.predict import predict_human_scores
from textgenadvtrack.detection.data_prep import summarize_detection_split
from textgenadvtrack.detection.train import supported_backbones, supported_training_backends, train_detector


def test_summarize_detection_split_counts_labels_and_types():
    csv_path = Path("tests/fixtures/detection_rows.csv")
    summary = summarize_detection_split(csv_path)
    assert summary["rows"] == 4
    assert summary["label_counts"][1] == 2
    assert summary["text_type_counts"]["ai_rewritten"] == 1


def test_summarize_detection_split_tracks_languages():
    csv_path = Path("tests/fixtures/detection_rows.csv")
    summary = summarize_detection_split(csv_path)
    assert summary["language_counts"]["zh"] == 2
    assert summary["language_counts"]["en"] == 1


def test_supported_backbones_contains_expected_candidates():
    names = supported_backbones()
    assert "microsoft/deberta-v3-base" in names
    assert "xlm-roberta-base" in names


def test_supported_training_backends_contains_classic_and_transformer():
    names = supported_training_backends()
    assert "classic" in names
    assert "classic_plus" in names
    assert "transformer" in names


def test_train_detector_saves_artifacts(tmp_path):
    train_csv = Path("tests/fixtures/detection_rows.csv")
    dev_csv = Path("tests/fixtures/detection_rows.csv")
    output_dir = tmp_path / "detector"
    result = train_detector(train_csv, dev_csv, "microsoft/deberta-v3-base", output_dir)
    assert result["status"] == "trained"
    assert (output_dir / "vectorizer.joblib").exists()
    assert (output_dir / "classifier.joblib").exists()
    scores = predict_human_scores(["人工撰写文本一", "机器文本一"], output_dir)
    assert len(scores) == 2


def test_export_detection_submission_writes_required_sheets(tmp_path):
    train_csv = Path("tests/fixtures/detection_rows.csv")
    output_dir = tmp_path / "detector"
    train_detector(train_csv, train_csv, "microsoft/deberta-v3-base", output_dir, backend="classic")
    output_xlsx = tmp_path / "submission.xlsx"
    result = export_detection_submission(Path("tests/fixtures/detection_submit.csv"), output_dir, output_xlsx)
    assert result["rows"] == 2
    predictions = pd.read_excel(output_xlsx, sheet_name="predictions")
    timing = pd.read_excel(output_xlsx, sheet_name="time")
    assert list(predictions.columns) == ["prompt", "text_prediction"]
    assert list(timing.columns) == ["Data Volume", "Time"]


def test_classic_plus_backend_trains_and_predicts(tmp_path):
    train_csv = Path("tests/fixtures/detection_rows.csv")
    output_dir = tmp_path / "detector_plus"
    result = train_detector(train_csv, train_csv, "microsoft/deberta-v3-base", output_dir, backend="classic_plus")
    assert result["backend"] == "classic_plus"
    scores = predict_human_scores(["人工撰写文本一", "机器文本一"], output_dir)
    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_legacy_classic_metadata_can_still_export(tmp_path):
    train_csv = Path("tests/fixtures/detection_rows.csv")
    output_dir = tmp_path / "legacy_detector"
    train_detector(train_csv, train_csv, "microsoft/deberta-v3-base", output_dir, backend="classic")
    (output_dir / "metadata.json").write_text(
        '{"implementation": "tfidf-logreg-baseline", "model_name": "microsoft/deberta-v3-base"}'
    )

    output_xlsx = tmp_path / "legacy_submission.xlsx"
    result = export_detection_submission(Path("tests/fixtures/detection_submit.csv"), output_dir, output_xlsx)

    assert result["backend"] == "classic"
    assert output_xlsx.exists()


def test_transformer_options_are_saved_for_gpu_runs(monkeypatch, tmp_path):
    import textgenadvtrack.detection.train as train_module

    captured = {}

    def fake_train_transformer(train_csv, dev_csv, model_name, output_dir, options):
        captured["options"] = options
        output_dir.mkdir(parents=True)
        train_module._save_metadata(
            output_dir,
            {
                "backend": "transformer",
                "model_name": model_name,
                "max_length": options.max_length,
                "training_options": options.model_dump(),
            },
        )
        return {"AUC": 0.5, "ACC": 0.5, "F1": 0.5}

    monkeypatch.setattr(train_module, "_train_transformer", fake_train_transformer)

    result = train_detector(
        Path("tests/fixtures/detection_rows.csv"),
        Path("tests/fixtures/detection_rows.csv"),
        "xlm-roberta-base",
        tmp_path / "transformer",
        backend="transformer",
        epochs=3,
        batch_size=16,
        eval_batch_size=32,
        learning_rate=1e-5,
        max_length=384,
        fp16=True,
        gradient_accumulation_steps=2,
    )

    assert result["backend"] == "transformer"
    assert captured["options"].epochs == 3
    assert captured["options"].batch_size == 16
    assert captured["options"].max_length == 384
    assert captured["options"].fp16 is True
