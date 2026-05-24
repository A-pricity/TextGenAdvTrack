import pandas as pd

from textgenadvtrack.data.external import ExternalDetectionDatasetInputs, ingest_external_detection_dataset


def test_ingest_external_detection_dataset_normalizes_csv_rows(tmp_path):
    input_path = tmp_path / "external.csv"
    pd.DataFrame(
        [
            {"text": "human sample one", "label": "human"},
            {"text": "machine sample one", "label": 0},
            {"text": " ", "label": "ai"},
        ]
    ).to_csv(input_path, index=False)

    output_csv = tmp_path / "external_detection.csv"
    result = ingest_external_detection_dataset(
        ExternalDetectionDatasetInputs(
            input_path=input_path,
            output_csv=output_csv,
            source_name="public_external_demo",
            language="en",
            domain="news",
            split="dev",
            source_model="demo-model",
            prompt_type="news_summary",
        )
    )

    assert result["rows"] == 2
    frame = pd.read_csv(output_csv)
    assert set(frame["split"]) == {"dev"}
    assert set(frame["label"]) == {0, 1}
    assert set(frame["text_type"]) == {"human", "ai_original"}
    assert list(frame["sample_id"]) == ["en-public_external_demo-00001", "en-public_external_demo-00002"]
    assert list(frame["prompt_id"]) == ["external_00001", "external_00002"]


def test_ingest_external_detection_dataset_supports_jsonl(tmp_path):
    input_path = tmp_path / "external.jsonl"
    input_path.write_text(
        '{"text":"human sample","label":1}\n'
        '{"text":"machine sample","label":0}\n'
    )

    output_csv = tmp_path / "external_detection.csv"
    result = ingest_external_detection_dataset(
        ExternalDetectionDatasetInputs(
            input_path=input_path,
            output_csv=output_csv,
            source_name="demo",
            language="zh",
            domain="qa",
            split="rewrite_dev",
        )
    )

    assert result["rows"] == 2
    frame = pd.read_csv(output_csv)
    assert set(frame["split"]) == {"rewrite_dev"}
    assert set(frame["source_name"]) == {"demo"}
