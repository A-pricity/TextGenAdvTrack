from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion

from textgenadvtrack.io import load_detection_rows


class TransformerTrainingOptions(BaseModel):
    epochs: float = Field(default=2.0, gt=0)
    batch_size: int = Field(default=8, gt=0)
    eval_batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0)
    max_length: int = Field(default=512, gt=0)
    max_steps: int = -1
    fp16: bool = True
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    seed: int = 42


def supported_backbones() -> list[str]:
    return ["microsoft/deberta-v3-base", "xlm-roberta-base"]


def supported_training_backends() -> list[str]:
    return ["classic", "classic_plus", "transformer"]


def _compute_binary_metrics(labels: np.ndarray, human_probs: np.ndarray) -> dict[str, float]:
    predictions = (human_probs >= 0.5).astype(int)
    metrics = {
        "ACC": float(accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
    }
    try:
        metrics["AUC"] = float(roc_auc_score(labels, human_probs))
    except ValueError:
        metrics["AUC"] = 0.0
    return metrics


def _load_detection_frame(csv_path: Path) -> pd.DataFrame:
    rows = [row.model_dump() for row in load_detection_rows(csv_path)]
    return pd.DataFrame(rows)


def _save_metadata(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _feature_extractor(backend: str, train_rows: int):
    min_df = 1 if train_rows < 100 else 2
    if backend == "classic":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
    if backend == "classic_plus":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=min_df,
                        max_features=120000,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=min_df,
                        max_features=180000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported classic backend: {backend}")


def _train_classic(train_csv: Path, dev_csv: Path, model_name: str, output_dir: Path, backend: str = "classic") -> dict:
    train_frame = _load_detection_frame(train_csv)
    dev_frame = _load_detection_frame(dev_csv)

    vectorizer = _feature_extractor(backend, len(train_frame))
    train_x = vectorizer.fit_transform(train_frame["text"].astype(str).tolist())
    dev_x = vectorizer.transform(dev_frame["text"].astype(str).tolist())

    classifier = LogisticRegression(
        C=4.0 if backend == "classic_plus" else 1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
        solver="liblinear",
    )
    classifier.fit(train_x, train_frame["label"].astype(int).to_numpy())

    human_probs = classifier.predict_proba(dev_x)[:, 1]
    metrics = _compute_binary_metrics(dev_frame["label"].astype(int).to_numpy(), human_probs)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")
    joblib.dump(classifier, output_dir / "classifier.joblib")
    _save_metadata(
        output_dir,
        {
            "backend": backend,
            "model_name": model_name,
            "metrics": metrics,
            "label_semantics": {"0": "machine", "1": "human"},
            "artifacts": {
                "vectorizer": "vectorizer.joblib",
                "classifier": "classifier.joblib",
            },
        },
    )
    return metrics


def _require_transformer_dependencies():
    try:
        import torch  # noqa: F401
        from datasets import Dataset  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Transformer backend requires torch, datasets, and transformers with PyTorch support installed."
        ) from exc


def _train_transformer(
    train_csv: Path,
    dev_csv: Path,
    model_name: str,
    output_dir: Path,
    options: TransformerTrainingOptions,
) -> dict:
    _require_transformer_dependencies()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    train_frame = _load_detection_frame(train_csv)
    dev_frame = _load_detection_frame(dev_csv)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    use_cpu = not torch.cuda.is_available()
    max_length = min(options.max_length, 256) if use_cpu else options.max_length
    max_steps = 12 if use_cpu and options.max_steps < 0 else options.max_steps

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_frame[["text", "label"]], preserve_index=False).map(
        _tokenize,
        batched=True,
    )
    dev_ds = Dataset.from_pandas(dev_frame[["text", "label"]], preserve_index=False).map(
        _tokenize,
        batched=True,
    )
    train_ds = train_ds.rename_column("label", "labels")
    dev_ds = dev_ds.rename_column("label", "labels")
    keep_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        keep_columns.append("token_type_ids")
    train_ds = train_ds.with_format("torch", columns=keep_columns)
    dev_ds = dev_ds.with_format("torch", columns=keep_columns)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _compute(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        return _compute_binary_metrics(np.asarray(labels), probs)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "hf_training"),
        per_device_train_batch_size=4 if use_cpu else options.batch_size,
        per_device_eval_batch_size=4 if use_cpu else options.eval_batch_size,
        learning_rate=options.learning_rate,
        num_train_epochs=1 if use_cpu else options.epochs,
        max_steps=max_steps,
        weight_decay=options.weight_decay,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch" if not use_cpu else "no",
        load_best_model_at_end=not use_cpu,
        metric_for_best_model="AUC",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        disable_tqdm=True,
        seed=options.seed,
        do_train=True,
        do_eval=True,
        use_cpu=use_cpu,
        fp16=bool(options.fp16 and not use_cpu),
        gradient_accumulation_steps=options.gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    metrics = {
        "AUC": float(eval_metrics.get("eval_AUC", 0.0)),
        "ACC": float(eval_metrics.get("eval_ACC", 0.0)),
        "F1": float(eval_metrics.get("eval_F1", 0.0)),
    }

    model_output_dir = output_dir / "transformer_model"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))
    _save_metadata(
        output_dir,
        {
            "backend": "transformer",
            "model_name": model_name,
            "metrics": metrics,
            "label_semantics": {"0": "machine", "1": "human"},
            "artifacts": {
                "model_dir": "transformer_model",
            },
            "max_length": max_length,
            "training_mode": "cpu_smoke" if use_cpu else "full",
            "training_options": options.model_dump(),
        },
    )
    return metrics


def train_detector(
    train_csv: Path,
    dev_csv: Path,
    model_name: str,
    output_dir: Path,
    backend: str = "classic",
    epochs: float = 2.0,
    batch_size: int = 8,
    eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    max_steps: int = -1,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
) -> dict:
    if backend not in supported_training_backends():
        raise ValueError(f"Unsupported backend: {backend}")
    if model_name not in supported_backbones():
        raise ValueError(f"Unsupported model name: {model_name}")

    started = time.time()
    if backend in {"classic", "classic_plus"}:
        metrics = _train_classic(train_csv, dev_csv, model_name, output_dir, backend)
    else:
        options = TransformerTrainingOptions(
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            max_steps=max_steps,
            fp16=fp16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=weight_decay,
        )
        metrics = _train_transformer(train_csv, dev_csv, model_name, output_dir, options)

    return {
        "train_csv": str(train_csv),
        "dev_csv": str(dev_csv),
        "model_name": model_name,
        "output_dir": str(output_dir),
        "status": "trained",
        "backend": backend,
        "metrics": metrics,
        "elapsed_seconds": round(time.time() - started, 4),
    }
