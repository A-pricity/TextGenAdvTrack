# High Score 4090 Runbook

This runbook maps the high-score work items to reproducible commands for a single RTX 4090 server.

## 0. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install a CUDA-enabled PyTorch build on the server before transformer training.

## 1. Reliable Validation

Create repeated official validation splits:

```bash
.venv/bin/python -m textgenadvtrack.cli build-repeated-detection-splits \
  --val-with-label-csv datasets/official/raw/UCAS_AISAD_TEXT-val.csv \
  --output-dir outputs/detection/repeated_splits \
  --seeds 11 22 33 44 55 \
  --dev-fraction 0.2 \
  --language zh
```

After scoring each fold dev CSV, run grouped metrics:

```bash
.venv/bin/python -m textgenadvtrack.cli evaluate-detection-slices \
  --labels-csv outputs/detection/repeated_splits/seed_11/official_dev.csv \
  --predictions outputs/detection/scores/classic_plus_seed_11.csv \
  --group-columns language domain text_type
```

## 2. Transformer Main Model

Recommended first 4090 run:

```bash
.venv/bin/python -m textgenadvtrack.cli train-detector \
  --backend transformer \
  --model-name xlm-roberta-base \
  --train-csv data/detection/train.csv \
  --dev-csv data/detection/val.csv \
  --output-dir models/detector_xlmr_official \
  --epochs 3 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 \
  --max-length 512
```

If VRAM is tight, use `--batch-size 8 --gradient-accumulation-steps 2`.

Score the dev split for ensemble search:

```bash
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv data/detection/val.csv \
  --model-dir models/detector_xlmr_official \
  --output-csv outputs/detection/scores/xlmr_dev.csv
```

## 3. Ensemble

Score `classic_plus` on the same dev split:

```bash
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv data/detection/val.csv \
  --model-dir models/detector_official_classic_plus_eval \
  --output-csv outputs/detection/scores/classic_plus_dev.csv
```

Search blend weights:

```bash
.venv/bin/python -m textgenadvtrack.cli search-detection-blend \
  --labels-csv data/detection/val.csv \
  --prediction outputs/detection/scores/classic_plus_dev.csv \
  --prediction outputs/detection/scores/xlmr_dev.csv \
  --step 0.05
```

Blend final test submissions with the chosen weights:

```bash
.venv/bin/python -m textgenadvtrack.cli blend-detection-submit \
  --prediction outputs/detection/submissions/textgenadvtrack_test1_classic_plus.xlsx \
  --prediction outputs/detection/submissions/textgenadvtrack_test1_xlmr.xlsx \
  --weight 0.4 \
  --weight 0.6 \
  --output-xlsx outputs/detection/submissions/textgenadvtrack_test1_ensemble.xlsx
```

## 4. Training Data Boost

Build the completed detection dataset:

```bash
.venv/bin/python -m textgenadvtrack.cli build-completed-detection-dataset \
  --official-val-with-label-csv data/detection/official/val_with_label.csv \
  --output-dir /tmp/textgenadvtrack_completed_detection \
  --dev-fraction 0.2 \
  --seed 42 \
  --language zh
```

Copy the generated files into the standard locations:

```bash
cp /tmp/textgenadvtrack_completed_detection/completed_train.csv data/detection/train.csv
cp /tmp/textgenadvtrack_completed_detection/completed_dev.csv data/detection/val.csv
cp /tmp/textgenadvtrack_completed_detection/completed_rewrite_dev.csv data/detection/rewrite_val.csv
```

If you want extra adversarial rows, build them after that:

```bash
.venv/bin/python -m textgenadvtrack.cli build-adversarial-training-rows \
  --detection-csv data/detection/train.csv \
  --output-csv data/training/adversarial_official_train.csv
```

Merge these CSVs into a larger training pool before retraining. Keep a data-source note for course compliance.

If you add a public or external corpus, normalize it first and keep the raw source in `datasets/`:

```bash
.venv/bin/python -m textgenadvtrack.cli ingest-external-detection-dataset \
  --input-path datasets/public/your_source.csv \
  --output-csv data/detection/your_source_normalized.csv \
  --source-name your_source \
  --language zh \
  --domain news \
  --split train
```

## 5. Evasion Risk

Use `data/training/adversarial_official_train.csv` as machine rewritten data and evaluate slices by `text_type`.
The weak slice to watch is usually:

```text
text_type=ai_rewritten
```

Do not trust aggregate AUC if rewritten/evasion slices are much lower.

## 6. Score Tuning

Search a simple monotonic score transform on dev:

```bash
.venv/bin/python -m textgenadvtrack.cli tune-detection-scores \
  --labels-csv data/detection/val.csv \
  --predictions outputs/detection/scores/ensemble_dev.csv \
  --scale 0.8 0.9 1.0 1.1 1.2 \
  --bias -0.03 -0.015 0 0.015 0.03
```

Apply the chosen transform to the test submission:

```bash
.venv/bin/python -m textgenadvtrack.cli apply-detection-score-tuning \
  --input-xlsx outputs/detection/submissions/textgenadvtrack_test1_ensemble.xlsx \
  --output-xlsx outputs/detection/submissions/textgenadvtrack_test1_ensemble_tuned.xlsx \
  --scale 1.0 \
  --bias 0.0
```

Use tuning only when it improves held-out final score consistently across repeated splits.
