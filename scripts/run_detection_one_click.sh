#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

MODE="${MODE:-single}"
BACKEND="${BACKEND:-transformer}"
MODEL_NAME="${MODEL_NAME:-xlm-roberta-base}"
MODEL_DIR="${MODEL_DIR:-models/detector_xlmr_local}"
TRAIN_CSV="${TRAIN_CSV:-data/detection/train.csv}"
DEV_CSV="${DEV_CSV:-data/detection/val.csv}"
OFFICIAL_INPUT_CSV="${OFFICIAL_INPUT_CSV:-data/detection/official/test1_input.csv}"
OFFICIAL_VAL_WITH_LABEL_CSV="${OFFICIAL_VAL_WITH_LABEL_CSV:-data/detection/official/val_with_label.csv}"
OUTPUT_XLSX="${OUTPUT_XLSX:-outputs/detection/submissions/textgenadvtrack_test1_xlmr.xlsx}"
DEV_SCORE_CSV="${DEV_SCORE_CSV:-outputs/detection/scores/xlmr_dev_scores.csv}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_DEV_SCORE="${RUN_DEV_SCORE:-1}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
CV_FOLDS="${CV_FOLDS:-10}"
CV_SEED="${CV_SEED:-42}"
CV_OUTPUT_DIR="${CV_OUTPUT_DIR:-outputs/detection/kfold_splits}"
CV_SUBMISSION_DIR="${CV_SUBMISSION_DIR:-outputs/detection/submissions/cv}"
CV_MODEL_DIR="${CV_MODEL_DIR:-models/cv_xlmr}"
CV_ENSEMBLE_XLSX="${CV_ENSEMBLE_XLSX:-outputs/detection/submissions/textgenadvtrack_test1_xlmr_cv_ensemble.xlsx}"

mkdir -p logs models outputs/detection/scores outputs/detection/submissions

run_cmd() {
  printf '\n[run] %s\n' "$*"
  "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf '[error] missing required file: %s\n' "$1" >&2
    exit 1
  fi
}

require_file "$OFFICIAL_INPUT_CSV"

train_model() {
  local train_csv="$1"
  local dev_csv="$2"
  local model_dir="$3"
  if [[ "$RUN_TRAIN" == "1" ]]; then
    run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli train-detector \
      --backend "$BACKEND" \
      --model-name "$MODEL_NAME" \
      --train-csv "$train_csv" \
      --dev-csv "$dev_csv" \
      --output-dir "$model_dir" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --eval-batch-size "$EVAL_BATCH_SIZE" \
      --gradient-accumulation-steps "$GRAD_ACCUM" \
      --learning-rate "$LEARNING_RATE" \
      --max-length "$MAX_LENGTH" \
      --weight-decay "$WEIGHT_DECAY"
  else
    require_file "$model_dir/metadata.json"
    printf '[skip] RUN_TRAIN=0, using existing model: %s\n' "$model_dir"
  fi
}

export_and_validate() {
  local model_dir="$1"
  local output_xlsx="$2"
  run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli export-detection-submit \
    --input-csv "$OFFICIAL_INPUT_CSV" \
    --model-dir "$model_dir" \
    --output-xlsx "$output_xlsx"
  run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli validate-detection-submit \
    --input-csv "$OFFICIAL_INPUT_CSV" \
    --submission-xlsx "$output_xlsx"
}

if [[ "$MODE" == "single" ]]; then
  require_file "$TRAIN_CSV"
  require_file "$DEV_CSV"
  train_model "$TRAIN_CSV" "$DEV_CSV" "$MODEL_DIR"
  if [[ "$RUN_DEV_SCORE" == "1" ]]; then
    run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli score-detection-csv \
      --input-csv "$DEV_CSV" \
      --model-dir "$MODEL_DIR" \
      --output-csv "$DEV_SCORE_CSV"
  fi
  export_and_validate "$MODEL_DIR" "$OUTPUT_XLSX"
  printf '\n[done] submission: %s\n' "$OUTPUT_XLSX"
elif [[ "$MODE" == "cv" ]]; then
  require_file "$OFFICIAL_VAL_WITH_LABEL_CSV"
  run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli build-kfold-detection-splits \
    --val-with-label-csv "$OFFICIAL_VAL_WITH_LABEL_CSV" \
    --output-dir "$CV_OUTPUT_DIR" \
    --folds "$CV_FOLDS" \
    --seed "$CV_SEED" \
    --language zh

  mkdir -p "$CV_SUBMISSION_DIR" "$CV_MODEL_DIR"
  blend_args=()
  for fold in $(seq 1 "$CV_FOLDS"); do
    fold_name="$(printf 'fold_%02d' "$fold")"
    fold_dir="$CV_OUTPUT_DIR/$fold_name"
    fold_model_dir="$CV_MODEL_DIR/$fold_name"
    fold_submission="$CV_SUBMISSION_DIR/textgenadvtrack_test1_xlmr_${fold_name}.xlsx"
    train_model "$fold_dir/official_train.csv" "$fold_dir/official_dev.csv" "$fold_model_dir"
    export_and_validate "$fold_model_dir" "$fold_submission"
    blend_args+=(--prediction "$fold_submission" --weight 1)
  done

  run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli blend-detection-submit \
    "${blend_args[@]}" \
    --output-xlsx "$CV_ENSEMBLE_XLSX"
  run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli validate-detection-submit \
    --input-csv "$OFFICIAL_INPUT_CSV" \
    --submission-xlsx "$CV_ENSEMBLE_XLSX"
  printf '\n[done] cv ensemble submission: %s\n' "$CV_ENSEMBLE_XLSX"
else
  printf '[error] MODE must be single or cv, got: %s\n' "$MODE" >&2
  exit 1
fi
