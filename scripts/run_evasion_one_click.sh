#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

SOURCE_CSV="${SOURCE_CSV:-data/evasion/source.csv}"
OFFICIAL_INPUT_CSV="${OFFICIAL_INPUT_CSV:-data/detection/official/val_with_label.csv}"
CANDIDATES_CSV="${CANDIDATES_CSV:-outputs/evasion/candidates/official_val_candidates.csv}"
SELECTED_CSV="${SELECTED_CSV:-outputs/evasion/selected/official_val_selected.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/evasion/submissions/textgenadvtrack_evasion_val.csv}"
REWRITE_MODEL="${REWRITE_MODEL:-rule_rewriter}"
USE_DETECTOR="${USE_DETECTOR:-1}"
DETECTOR_MODEL_DIR="${DETECTOR_MODEL_DIR:-}"

mkdir -p logs outputs/evasion/candidates outputs/evasion/selected outputs/evasion/submissions

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

require_file "$SOURCE_CSV"
require_file "$OFFICIAL_INPUT_CSV"

if [[ -z "$DETECTOR_MODEL_DIR" ]]; then
  if [[ -f models/detector_xlmr_local/metadata.json ]]; then
    DETECTOR_MODEL_DIR="models/detector_xlmr_local"
  elif [[ -f models/cv_xlmr/fold_01/metadata.json ]]; then
    DETECTOR_MODEL_DIR="models/cv_xlmr/fold_01"
  fi
fi

run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli generate-evasion \
  --source-csv "$SOURCE_CSV" \
  --output-csv "$CANDIDATES_CSV" \
  --rewrite-model "$REWRITE_MODEL"

select_args=(
  "$PYTHON_BIN" -m textgenadvtrack.cli select-evasion
  --candidates-csv "$CANDIDATES_CSV"
  --output-csv "$SELECTED_CSV"
)

if [[ "$USE_DETECTOR" == "1" && -n "$DETECTOR_MODEL_DIR" ]]; then
  require_file "$DETECTOR_MODEL_DIR/metadata.json"
  select_args+=(--model-dir "$DETECTOR_MODEL_DIR")
  printf '[info] using detector model for evasion proxy scoring: %s\n' "$DETECTOR_MODEL_DIR"
else
  printf '[info] using heuristic evasion proxy scoring\n'
fi

run_cmd "${select_args[@]}"

run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli build-evasion-submit \
  --official-input-csv "$OFFICIAL_INPUT_CSV" \
  --selected-csv "$SELECTED_CSV" \
  --output-csv "$OUTPUT_CSV"

run_cmd "$PYTHON_BIN" -m textgenadvtrack.cli validate-evasion-submit \
  --official-input-csv "$OFFICIAL_INPUT_CSV" \
  --submission-csv "$OUTPUT_CSV"

printf '\n[done] evasion submission: %s\n' "$OUTPUT_CSV"
