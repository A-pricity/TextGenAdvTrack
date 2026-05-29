#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

RAW_OFFICIAL_DIR="${RAW_OFFICIAL_DIR:-datasets/official/raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data}"
MODEL_NAME="${MODEL_NAME:-./xlm-roberta-base}"
DETECTION_MODE="${DETECTION_MODE:-cv}"
CV_FOLDS="${CV_FOLDS:-10}"
RUN_DETECTION="${RUN_DETECTION:-1}"
RUN_EVASION="${RUN_EVASION:-1}"
USE_DETECTOR="${USE_DETECTOR:-1}"

OFFICIAL_TEST_CSV="${OFFICIAL_TEST_CSV:-data/detection/official/test1_input.csv}"
OFFICIAL_VAL_CSV="${OFFICIAL_VAL_CSV:-data/detection/official/val_with_label.csv}"
EVASION_SOURCE_CSV="${EVASION_SOURCE_CSV:-data/evasion/official/val_machine_only.csv}"
DETECTION_CV_XLSX="${DETECTION_CV_XLSX:-outputs/detection/submissions/textgenadvtrack_test1_xlmr_cv_ensemble.xlsx}"
EVASION_OUTPUT_CSV="${EVASION_OUTPUT_CSV:-outputs/evasion/submissions/textgenadvtrack_evasion_val.csv}"

mkdir -p logs outputs/detection/submissions outputs/evasion/submissions

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

prepare_data_if_needed() {
  if [[ -f "$OFFICIAL_TEST_CSV" && -f "$OFFICIAL_VAL_CSV" && -f "$EVASION_SOURCE_CSV" ]]; then
    printf '[ok] standardized data already exists\n'
    return
  fi

  if [[ -f "$RAW_OFFICIAL_DIR/UCAS_AISAD_TEXT-test1.csv" \
     && -f "$RAW_OFFICIAL_DIR/UCAS_AISAD_TEXT-val.csv" \
     && -f "$RAW_OFFICIAL_DIR/UCAS_AISAD_TEXT-val_label.csv" ]]; then
    run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli prepare-official-data \
      --raw-dir "$RAW_OFFICIAL_DIR" \
      --output-root "$OUTPUT_ROOT"
  else
    printf '[info] raw official files not found under %s; using existing standardized data\n' "$RAW_OFFICIAL_DIR"
  fi

  require_file "$OFFICIAL_TEST_CSV"
  require_file "$OFFICIAL_VAL_CSV"
  require_file "$EVASION_SOURCE_CSV"
}

prepare_data_if_needed

if [[ "$RUN_DETECTION" == "1" ]]; then
  run_cmd env \
    MODEL_NAME="$MODEL_NAME" \
    MODE="$DETECTION_MODE" \
    CV_FOLDS="$CV_FOLDS" \
    OFFICIAL_INPUT_CSV="$OFFICIAL_TEST_CSV" \
    OFFICIAL_VAL_WITH_LABEL_CSV="$OFFICIAL_VAL_CSV" \
    CV_ENSEMBLE_XLSX="$DETECTION_CV_XLSX" \
    bash scripts/run_detection_one_click.sh
else
  printf '[skip] RUN_DETECTION=0\n'
fi

DETECTOR_MODEL_DIR="${DETECTOR_MODEL_DIR:-}"
if [[ -z "$DETECTOR_MODEL_DIR" ]]; then
  if [[ -f models/cv_xlmr/fold_01/metadata.json ]]; then
    DETECTOR_MODEL_DIR="models/cv_xlmr/fold_01"
  elif [[ -f models/detector_xlmr_local/metadata.json ]]; then
    DETECTOR_MODEL_DIR="models/detector_xlmr_local"
  fi
fi

if [[ "$RUN_EVASION" == "1" ]]; then
  run_cmd env \
    SOURCE_CSV="$EVASION_SOURCE_CSV" \
    OFFICIAL_INPUT_CSV="$OFFICIAL_VAL_CSV" \
    OUTPUT_CSV="$EVASION_OUTPUT_CSV" \
    USE_DETECTOR="$USE_DETECTOR" \
    DETECTOR_MODEL_DIR="$DETECTOR_MODEL_DIR" \
    bash scripts/run_evasion_one_click.sh
else
  printf '[skip] RUN_EVASION=0\n'
fi

printf '\n[done] detection: %s\n' "$DETECTION_CV_XLSX"
printf '[done] evasion: %s\n' "$EVASION_OUTPUT_CSV"
