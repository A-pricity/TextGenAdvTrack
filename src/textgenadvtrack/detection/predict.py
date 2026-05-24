from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from textgenadvtrack.io import load_detection_submit_rows
from textgenadvtrack.io import load_detection_rows


def _load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {model_dir}")
    metadata = json.loads(metadata_path.read_text())
    if "backend" not in metadata and metadata.get("implementation") == "tfidf-logreg-baseline":
        metadata["backend"] = "classic"
    return metadata


def _predict_classic(texts: list[str], model_dir: Path) -> np.ndarray:
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    classifier = joblib.load(model_dir / "classifier.joblib")
    features = vectorizer.transform(texts)
    return classifier.predict_proba(features)[:, 1]


def _predict_transformer(texts: list[str], model_dir: Path, max_length: int) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Transformer inference requires torch and transformers with PyTorch support.") from exc

    model_path = model_dir / "transformer_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    outputs = []
    batch_size = 8
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        outputs.append(probs)
    return np.concatenate(outputs) if outputs else np.array([], dtype=float)


def export_detection_submission(input_csv: Path, model_dir: Path, output_xlsx: Path) -> dict:
    rows = load_detection_submit_rows(input_csv)
    prompts = [row.prompt for row in rows]
    texts = [row.text for row in rows]
    metadata = _load_metadata(model_dir)

    started = time.time()
    backend = metadata["backend"]
    if backend in {"classic", "classic_plus"}:
        human_probs = _predict_classic(texts, model_dir)
    elif backend == "transformer":
        human_probs = _predict_transformer(texts, model_dir, int(metadata.get("max_length", 512)))
    else:
        raise ValueError(f"Unsupported prediction backend: {backend}")

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "prompt": prompts,
            "text_prediction": human_probs,
        }
    )
    time_frame = pd.DataFrame(
        [
            {
                "Data Volume": len(rows),
                "Time": round(time.time() - started, 6),
            }
        ]
    )
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
        time_frame.to_excel(writer, sheet_name="time", index=False)

    return {
        "input_csv": str(input_csv),
        "model_dir": str(model_dir),
        "output_xlsx": str(output_xlsx),
        "rows": len(rows),
        "backend": backend,
    }


def predict_human_scores(texts: list[str], model_dir: Path) -> list[float]:
    metadata = _load_metadata(model_dir)
    backend = metadata["backend"]
    if backend in {"classic", "classic_plus"}:
        scores = _predict_classic(texts, model_dir)
    elif backend == "transformer":
        scores = _predict_transformer(texts, model_dir, int(metadata.get("max_length", 512)))
    else:
        raise ValueError(f"Unsupported prediction backend: {backend}")
    return [float(score) for score in scores.tolist()]


def export_detection_scores(input_csv: Path, model_dir: Path, output_csv: Path) -> dict:
    rows = load_detection_rows(input_csv)
    texts = [row.text for row in rows]
    scores = predict_human_scores(texts, model_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_id": [row.sample_id for row in rows],
            "split": [row.split for row in rows],
            "language": [row.language for row in rows],
            "domain": [row.domain for row in rows],
            "text_type": [row.text_type for row in rows],
            "label": [row.label for row in rows],
            "text_prediction": scores,
        }
    )
    frame.to_csv(output_csv, index=False)
    return {"input_csv": str(input_csv), "model_dir": str(model_dir), "output_csv": str(output_csv), "rows": len(rows)}
