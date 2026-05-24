from pathlib import Path

import pandas as pd


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    return frame


def prepare_official_data(raw_dir: Path, output_root: Path) -> dict:
    val_path = raw_dir / "UCAS_AISAD_TEXT-val.csv"
    val_label_path = raw_dir / "UCAS_AISAD_TEXT-val_label.csv"
    test1_path = raw_dir / "UCAS_AISAD_TEXT-test1.csv"

    detection_official_dir = output_root / "detection" / "official"
    evasion_official_dir = output_root / "evasion" / "official"
    manifests_dir = output_root / "manifests"
    detection_official_dir.mkdir(parents=True, exist_ok=True)
    evasion_official_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    val_df = _normalize_columns(pd.read_csv(val_path))
    val_label_df = _normalize_columns(pd.read_csv(val_label_path))
    test1_df = _normalize_columns(pd.read_csv(test1_path))

    if "label" not in val_df.columns and "label" in val_label_df.columns:
        val_df = val_df.copy()
        val_df["label"] = val_label_df["label"].astype(int)

    val_with_label_df = val_df[["prompt", "text", "label"]].copy()
    val_submit_input_df = val_df[["prompt", "text"]].copy()
    val_label_only_df = val_with_label_df[["label"]].copy()
    test1_input_df = test1_df[["prompt", "text"]].copy()

    val_machine_df = val_with_label_df[val_with_label_df["label"].astype(int) == 0].copy().reset_index(drop=True)
    val_machine_df.insert(0, "sample_id", [f"eva-{idx + 1:04d}" for idx in range(len(val_machine_df))])
    val_machine_df["language"] = "unknown"
    val_machine_df["domain"] = "official_val"
    val_machine_df["source_model"] = "unknown"
    val_machine_df["prompt_type"] = "unknown"
    val_machine_df["prompt_id"] = [f"official-val-{idx + 1:04d}" for idx in range(len(val_machine_df))]
    val_machine_df = val_machine_df[
        ["sample_id", "language", "domain", "source_model", "prompt_type", "prompt_id", "prompt", "text"]
    ].rename(columns={"text": "source_text"})

    val_with_label_out = detection_official_dir / "val_with_label.csv"
    val_submit_out = detection_official_dir / "val_submit_input.csv"
    val_label_out = detection_official_dir / "val_label.csv"
    test1_out = detection_official_dir / "test1_input.csv"
    eva_val_machine_out = evasion_official_dir / "val_machine_only.csv"

    val_with_label_df.to_csv(val_with_label_out, index=False)
    val_submit_input_df.to_csv(val_submit_out, index=False)
    val_label_only_df.to_csv(val_label_out, index=False)
    test1_input_df.to_csv(test1_out, index=False)
    val_machine_df.to_csv(eva_val_machine_out, index=False)

    summary_df = pd.DataFrame(
        [
            {"file": "val_with_label.csv", "rows": len(val_with_label_df), "purpose": "detection_eval"},
            {"file": "val_submit_input.csv", "rows": len(val_submit_input_df), "purpose": "detection_submit_input"},
            {"file": "val_label.csv", "rows": len(val_label_only_df), "purpose": "detection_labels"},
            {"file": "test1_input.csv", "rows": len(test1_input_df), "purpose": "detection_test1_input"},
            {"file": "val_machine_only.csv", "rows": len(val_machine_df), "purpose": "evasion_source_seed"},
        ]
    )
    summary_df.to_csv(manifests_dir / "official_data_inventory.csv", index=False)

    return {
        "raw_dir": str(raw_dir),
        "output_root": str(output_root),
        "val_rows": int(len(val_with_label_df)),
        "test1_rows": int(len(test1_input_df)),
        "val_machine_rows": int(len(val_machine_df)),
    }
