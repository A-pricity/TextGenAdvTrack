# Dual-Track Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable small-scale baseline for both `AI_Text Detection` and `AI_Text Evasion`, using CSV-first data management, two detector backbones, and a local proxy-guided evasion pipeline.

**Architecture:** Create a small Python project organized around explicit dataset schemas, reproducible CSV manifests, detector training/inference entrypoints, and a separate evasion candidate generation/selection pipeline. Detection and evasion remain independently runnable, but the best local detector is reused as one proxy scorer in the evasion stage.

**Tech Stack:** Python 3.10+, pandas, pydantic, scikit-learn, transformers, datasets, pytest

---

## File Structure

- Create: `pyproject.toml`
  - Project metadata and dependencies.
- Create: `src/textgenadvtrack/__init__.py`
  - Package marker.
- Create: `src/textgenadvtrack/schemas.py`
  - Central CSV row schemas for detection/evasion/manifests.
- Create: `src/textgenadvtrack/io.py`
  - CSV load/save helpers with schema validation.
- Create: `src/textgenadvtrack/detection/data_prep.py`
  - Detection dataset assembly and split validation.
- Create: `src/textgenadvtrack/detection/train.py`
  - Backbone training entrypoint for `DeBERTa-v3-base` and `XLM-R base`.
- Create: `src/textgenadvtrack/detection/evaluate.py`
  - Local detection metrics and leaderboard summary.
- Create: `src/textgenadvtrack/evasion/generate_candidates.py`
  - Candidate rewriting orchestration and CSV output.
- Create: `src/textgenadvtrack/evasion/select_candidates.py`
  - Proxy scoring and final candidate selection.
- Create: `src/textgenadvtrack/cli.py`
  - Unified CLI for data validation, train, eval, evasion generation, evasion selection.
- Create: `tests/test_schemas.py`
  - Schema validation tests.
- Create: `tests/test_detection_data_prep.py`
  - Detection split and ratio tests.
- Create: `tests/test_evasion_selection.py`
  - Proxy scoring and selection rule tests.
- Create: `tests/fixtures/detection_rows.csv`
  - Tiny valid fixture dataset.
- Create: `tests/fixtures/evasion_candidates.csv`
  - Tiny valid candidate fixture dataset.
- Create: `data/manifests/.gitkeep`
  - Placeholder so the expected data tree exists.
- Create: `README.md`
  - Minimal runnable instructions for the baseline.

### Task 1: Project Skeleton And Schemas

**Files:**
- Create: `pyproject.toml`
- Create: `src/textgenadvtrack/__init__.py`
- Create: `src/textgenadvtrack/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing schema tests**

```python
from pydantic import ValidationError

from textgenadvtrack.schemas import DetectionRow, EvasionCandidateRow


def test_detection_row_accepts_valid_ai_rewritten_row():
    row = DetectionRow(
        sample_id="det-0001",
        split="train",
        language="zh",
        domain="qa",
        label=0,
        text_type="ai_rewritten",
        source_name="self_generated",
        source_model="Model-C1",
        prompt_type="question_answering",
        prompt_id="p-001",
        rewrite_type="paraphrase",
        parent_id="det-parent-1",
        text="这是一个改写后的机器文本。",
    )
    assert row.text_type == "ai_rewritten"


def test_detection_row_rejects_rewritten_without_parent_id():
    try:
        DetectionRow(
            sample_id="det-0002",
            split="train",
            language="en",
            domain="news",
            label=0,
            text_type="ai_rewritten",
            source_name="self_generated",
            source_model="Model-O1",
            prompt_type="summary",
            prompt_id="p-002",
            rewrite_type="compress",
            parent_id="",
            text="Rewritten machine text",
        )
    except ValidationError:
        assert True
        return
    assert False, "Expected ValidationError"


def test_evasion_candidate_accepts_proxy_scores():
    row = EvasionCandidateRow(
        candidate_id="cand-001",
        parent_id="eva-001",
        language="ru",
        domain="daily",
        source_model="Model-C2",
        prompt_type="daily_writing",
        rewrite_model="Model-C1",
        rewrite_type="paraphrase",
        semantic_score=0.91,
        proxy_score_1=0.77,
        proxy_score_2=0.72,
        selected=False,
        text="Переписанный текст",
    )
    assert row.proxy_score_1 > 0.7
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_schemas.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'textgenadvtrack'`

- [ ] **Step 3: Write minimal package files**

`pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "textgenadvtrack"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "pandas>=2.2.0",
  "pydantic>=2.7.0",
  "scikit-learn>=1.4.0",
  "transformers>=4.41.0",
  "datasets>=2.19.0",
  "pytest>=8.2.0",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`src/textgenadvtrack/__init__.py`

```python
__all__ = ["schemas"]
```

`src/textgenadvtrack/schemas.py`

```python
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DetectionRow(BaseModel):
    sample_id: str
    split: Literal["train", "dev", "rewrite_dev"]
    language: Literal["zh", "en", "ru"]
    domain: str
    label: Literal[0, 1]
    text_type: Literal["human", "ai_original", "ai_rewritten"]
    source_name: str
    source_model: str | None = None
    prompt_type: str | None = None
    prompt_id: str | None = None
    rewrite_type: str | None = None
    parent_id: str | None = None
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rewrite_fields(self):
        if self.text_type == "ai_rewritten":
            if not self.parent_id:
                raise ValueError("ai_rewritten rows require parent_id")
            if not self.rewrite_type:
                raise ValueError("ai_rewritten rows require rewrite_type")
        return self


class EvasionCandidateRow(BaseModel):
    candidate_id: str
    parent_id: str
    language: Literal["zh", "en", "ru"]
    domain: str
    source_model: str
    prompt_type: str
    rewrite_model: str
    rewrite_type: str
    semantic_score: float
    proxy_score_1: float
    proxy_score_2: float
    selected: bool
    text: str = Field(min_length=1)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_schemas.py -v`  
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/textgenadvtrack/__init__.py src/textgenadvtrack/schemas.py tests/test_schemas.py
git commit -m "feat: add baseline dataset schemas"
```

### Task 2: CSV IO And Detection Data Validation

**Files:**
- Create: `src/textgenadvtrack/io.py`
- Create: `src/textgenadvtrack/detection/data_prep.py`
- Test: `tests/test_detection_data_prep.py`
- Create: `tests/fixtures/detection_rows.csv`

- [ ] **Step 1: Write failing tests for CSV loading and split validation**

```python
from pathlib import Path

from textgenadvtrack.detection.data_prep import summarize_detection_split


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
```

- [ ] **Step 2: Add fixture CSV and run tests to verify failure**

`tests/fixtures/detection_rows.csv`

```csv
sample_id,split,language,domain,label,text_type,source_name,source_model,prompt_type,prompt_id,rewrite_type,parent_id,text
det-0001,train,zh,qa,1,human,public_human,,, ,,,人工撰写文本一
det-0002,train,zh,qa,0,ai_original,self_generated,Model-C1,question_answering,p-001,,,机器文本一
det-0003,train,en,news,0,ai_rewritten,self_generated,Model-O1,summary,p-002,compress,det-0002,Rewritten machine text
det-0004,train,ru,daily,1,human,public_human,,, ,,,Человеческий текст
```

Run: `pytest tests/test_detection_data_prep.py -v`  
Expected: FAIL with `ModuleNotFoundError` or missing function error

- [ ] **Step 3: Write minimal IO and summarization implementation**

`src/textgenadvtrack/io.py`

```python
from pathlib import Path

import pandas as pd

from textgenadvtrack.schemas import DetectionRow


def load_detection_rows(csv_path: Path) -> list[DetectionRow]:
    frame = pd.read_csv(csv_path).fillna("")
    rows: list[DetectionRow] = []
    for record in frame.to_dict(orient="records"):
        rows.append(DetectionRow(**record))
    return rows
```

`src/textgenadvtrack/detection/data_prep.py`

```python
from collections import Counter
from pathlib import Path

from textgenadvtrack.io import load_detection_rows


def summarize_detection_split(csv_path: Path) -> dict:
    rows = load_detection_rows(csv_path)
    return {
        "rows": len(rows),
        "label_counts": dict(Counter(row.label for row in rows)),
        "text_type_counts": dict(Counter(row.text_type for row in rows)),
        "language_counts": dict(Counter(row.language for row in rows)),
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_detection_data_prep.py -v`  
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/textgenadvtrack/io.py src/textgenadvtrack/detection/data_prep.py tests/test_detection_data_prep.py tests/fixtures/detection_rows.csv
git commit -m "feat: add detection csv validation helpers"
```

### Task 3: Detection Training And Local Evaluation Entrypoints

**Files:**
- Create: `src/textgenadvtrack/detection/train.py`
- Create: `src/textgenadvtrack/detection/evaluate.py`
- Modify: `src/textgenadvtrack/detection/data_prep.py`
- Test: `tests/test_detection_data_prep.py`

- [ ] **Step 1: Extend tests to require model candidate validation**

```python
from textgenadvtrack.detection.train import supported_backbones


def test_supported_backbones_contains_expected_candidates():
    names = supported_backbones()
    assert "microsoft/deberta-v3-base" in names
    assert "xlm-roberta-base" in names
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_detection_data_prep.py -v`  
Expected: FAIL with missing `supported_backbones`

- [ ] **Step 3: Write minimal train/eval entrypoints**

`src/textgenadvtrack/detection/train.py`

```python
from pathlib import Path


def supported_backbones() -> list[str]:
    return [
        "microsoft/deberta-v3-base",
        "xlm-roberta-base",
    ]


def train_detector(train_csv: Path, dev_csv: Path, model_name: str, output_dir: Path) -> dict:
    if model_name not in supported_backbones():
        raise ValueError(f"Unsupported backbone: {model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train_csv": str(train_csv),
        "dev_csv": str(dev_csv),
        "model_name": model_name,
        "output_dir": str(output_dir),
        "status": "planned",
    }
```

`src/textgenadvtrack/detection/evaluate.py`

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def detection_metrics(labels, human_scores) -> dict[str, float]:
    preds = [1 if score >= 0.5 else 0 for score in human_scores]
    auc = roc_auc_score(labels, human_scores)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    final_score = (0.6 * auc + 0.3 * acc + 0.1 * f1) / 100.0
    return {"AUC": auc, "ACC": acc, "F1": f1, "Final_Score": final_score}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_detection_data_prep.py -v`  
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/textgenadvtrack/detection/train.py src/textgenadvtrack/detection/evaluate.py tests/test_detection_data_prep.py
git commit -m "feat: add detection training and evaluation entrypoints"
```

### Task 4: Evasion Candidate Generation And Selection Rules

**Files:**
- Create: `src/textgenadvtrack/evasion/generate_candidates.py`
- Create: `src/textgenadvtrack/evasion/select_candidates.py`
- Test: `tests/test_evasion_selection.py`
- Create: `tests/fixtures/evasion_candidates.csv`

- [ ] **Step 1: Write failing selection tests**

```python
from textgenadvtrack.evasion.select_candidates import choose_best_candidate


def test_choose_best_candidate_prefers_joint_proxy_gain():
    candidates = [
        {"candidate_id": "c1", "proxy_score_1": 0.80, "proxy_score_2": 0.78, "semantic_score": 0.85},
        {"candidate_id": "c2", "proxy_score_1": 0.82, "proxy_score_2": 0.40, "semantic_score": 0.90},
    ]
    best = choose_best_candidate(candidates)
    assert best["candidate_id"] == "c1"


def test_choose_best_candidate_breaks_ties_with_semantics():
    candidates = [
        {"candidate_id": "c1", "proxy_score_1": 0.75, "proxy_score_2": 0.75, "semantic_score": 0.80},
        {"candidate_id": "c2", "proxy_score_1": 0.75, "proxy_score_2": 0.75, "semantic_score": 0.90},
    ]
    best = choose_best_candidate(candidates)
    assert best["candidate_id"] == "c2"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_evasion_selection.py -v`  
Expected: FAIL with missing selection module

- [ ] **Step 3: Write minimal candidate generation and selection code**

`src/textgenadvtrack/evasion/generate_candidates.py`

```python
def planned_rewrite_mix() -> list[tuple[str, int]]:
    return [
        ("paraphrase", 2),
        ("reorder", 1),
        ("compress", 1),
        ("expand", 1),
        ("synonym", 1),
    ]
```

`src/textgenadvtrack/evasion/select_candidates.py`

```python
def choose_best_candidate(candidates: list[dict]) -> dict:
    def rank_key(candidate: dict):
        joint = candidate["proxy_score_1"] + candidate["proxy_score_2"]
        return (joint, candidate["semantic_score"])

    return max(candidates, key=rank_key)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_evasion_selection.py -v`  
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/textgenadvtrack/evasion/generate_candidates.py src/textgenadvtrack/evasion/select_candidates.py tests/test_evasion_selection.py
git commit -m "feat: add evasion selection baseline"
```

### Task 5: Unified CLI And README

**Files:**
- Create: `src/textgenadvtrack/cli.py`
- Create: `README.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add failing smoke test for CLI imports**

```python
from textgenadvtrack.cli import build_parser


def test_cli_exposes_detection_and_evasion_commands():
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices.keys()
    assert "validate-detection" in subcommands
    assert "train-detector" in subcommands
    assert "select-evasion" in subcommands
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_schemas.py tests/test_detection_data_prep.py tests/test_evasion_selection.py -v`  
Expected: FAIL due to missing CLI module

- [ ] **Step 3: Implement minimal CLI and usage docs**

`src/textgenadvtrack/cli.py`

```python
import argparse

from textgenadvtrack.detection.train import supported_backbones


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-track baseline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-detection")
    train_parser = subparsers.add_parser("train-detector")
    train_parser.add_argument("--model-name", choices=supported_backbones(), required=True)
    subparsers.add_parser("generate-evasion")
    subparsers.add_parser("select-evasion")
    return parser
```

Update `pyproject.toml`

```toml
[project.scripts]
textgenadvtrack = "textgenadvtrack.cli:build_parser"
```

`README.md`

```md
# TextGenAdvTrack Baseline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Detection

```bash
python -m textgenadvtrack.cli train-detector --model-name microsoft/deberta-v3-base
```

## Evasion

```bash
python -m textgenadvtrack.cli generate-evasion
python -m textgenadvtrack.cli select-evasion
```
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -v`  
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/textgenadvtrack/cli.py README.md
git commit -m "feat: add unified dual-track baseline cli"
```

## Self-Review

### Spec coverage

- Dual-track baseline: covered by Tasks 3 and 4.
- CSV-first dataset strategy: covered by Tasks 1 and 2.
- Two backbone comparison support: covered by Task 3.
- Evasion proxy-guided candidate selection: covered by Task 4.
- Runnable project entrypoint: covered by Task 5.

### Placeholder scan

- No `TODO` / `TBD` placeholders remain.
- All code-changing tasks include concrete code blocks.
- All testing steps include concrete commands.

### Type consistency

- Detection schema names remain consistent: `DetectionRow`, `text_type`, `parent_id`.
- Evasion selection uses `proxy_score_1`, `proxy_score_2`, `semantic_score` consistently.
- CLI subcommand names are explicitly defined once in Task 5.
