#!/usr/bin/env bash
# ====================================================
# Detection test2 high-score runner.
#
# Uses the strongest already-trained local models found in this repo:
#   - classic_plus_full: strong calibration / threshold behavior
#   - detector_xlmr_local: 3 epoch XLM-R
#   - xlmr_full_8ep: best single-model validation AUC
#
# Validation search on data/detection/val.csv favored weights:
#   classic_plus_full=0.65, detector_xlmr_local=0.15, xlmr_full_8ep=0.20
# plus a light score transform:
#   scale=0.70, bias=-0.005
# ====================================================
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

TEST2_INPUT="${TEST2_INPUT:-data/detection/official/test2_input.csv}"
WORK_DIR="${WORK_DIR:-outputs/detection/submissions/test2_highscore_parts}"
OUTPUT_RAW_XLSX="${OUTPUT_RAW_XLSX:-outputs/detection/submissions/Perplexity-Hunters_detection_test2_blend_raw.xlsx}"
OUTPUT_XLSX="${OUTPUT_XLSX:-outputs/detection/submissions/Perplexity-Hunters_detection_test2_highscore.xlsx}"

CLASSIC_MODEL_DIR="${CLASSIC_MODEL_DIR:-models/classic_plus_full}"
XLMR3_MODEL_DIR="${XLMR3_MODEL_DIR:-models/detector_xlmr_local}"
XLMR8_MODEL_DIR="${XLMR8_MODEL_DIR:-models/xlmr_full_8ep}"

CLASSIC_WEIGHT="${CLASSIC_WEIGHT:-0.65}"
XLMR3_WEIGHT="${XLMR3_WEIGHT:-0.15}"
XLMR8_WEIGHT="${XLMR8_WEIGHT:-0.20}"
SCALE="${SCALE:-0.70}"
BIAS="${BIAS:--0.005}"

require_file() {
  if [[ ! -f "$1" ]]; then
    printf '[error] missing required file: %s\n' "$1" >&2
    exit 1
  fi
}

require_model() {
  require_file "$1/metadata.json"
}

require_file "$TEST2_INPUT"
require_model "$CLASSIC_MODEL_DIR"
require_model "$XLMR3_MODEL_DIR"
require_model "$XLMR8_MODEL_DIR"

mkdir -p "$WORK_DIR" "$(dirname "$OUTPUT_XLSX")"

printf '\n====================================================\n'
printf '  Detection Test2 High-Score Pipeline\n'
printf '====================================================\n'
printf '  Input: %s\n' "$TEST2_INPUT"
printf '  classic_plus: %s weight=%s\n' "$CLASSIC_MODEL_DIR" "$CLASSIC_WEIGHT"
printf '  xlmr3:        %s weight=%s\n' "$XLMR3_MODEL_DIR" "$XLMR3_WEIGHT"
printf '  xlmr8:        %s weight=%s\n' "$XLMR8_MODEL_DIR" "$XLMR8_WEIGHT"
printf '  transform:    scale=%s bias=%s\n' "$SCALE" "$BIAS"
printf '  Output:       %s\n' "$OUTPUT_XLSX"
printf '====================================================\n'

classic_xlsx="$WORK_DIR/test2_classic_plus.xlsx"
xlmr3_xlsx="$WORK_DIR/test2_xlmr3.xlsx"
xlmr8_xlsx="$WORK_DIR/test2_xlmr8.xlsx"

printf '\n[1/5] Export classic_plus predictions...\n'
"$PYTHON_BIN" -m textgenadvtrack.cli export-detection-submit \
  --input-csv "$TEST2_INPUT" \
  --model-dir "$CLASSIC_MODEL_DIR" \
  --output-xlsx "$classic_xlsx"

printf '\n[2/5] Export XLM-R 3ep predictions...\n'
"$PYTHON_BIN" -m textgenadvtrack.cli export-detection-submit \
  --input-csv "$TEST2_INPUT" \
  --model-dir "$XLMR3_MODEL_DIR" \
  --output-xlsx "$xlmr3_xlsx"

printf '\n[3/5] Export XLM-R 8ep predictions...\n'
"$PYTHON_BIN" -m textgenadvtrack.cli export-detection-submit \
  --input-csv "$TEST2_INPUT" \
  --model-dir "$XLMR8_MODEL_DIR" \
  --output-xlsx "$xlmr8_xlsx"

printf '\n[4/5] Blend predictions...\n'
"$PYTHON_BIN" -m textgenadvtrack.cli blend-detection-submit \
  --prediction "$classic_xlsx" --weight "$CLASSIC_WEIGHT" \
  --prediction "$xlmr3_xlsx" --weight "$XLMR3_WEIGHT" \
  --prediction "$xlmr8_xlsx" --weight "$XLMR8_WEIGHT" \
  --output-xlsx "$OUTPUT_RAW_XLSX"

printf '\n[5/5] Apply score transform and validate...\n'
"$PYTHON_BIN" -m textgenadvtrack.cli apply-detection-score-tuning \
  --input-xlsx "$OUTPUT_RAW_XLSX" \
  --output-xlsx "$OUTPUT_XLSX" \
  --scale "$SCALE" \
  --bias "$BIAS"

"$PYTHON_BIN" -m textgenadvtrack.cli validate-detection-submit \
  --input-csv "$TEST2_INPUT" \
  --submission-xlsx "$OUTPUT_XLSX"

printf '\n====================================================\n'
printf '  DONE\n'
printf '  Detection test2 submission: %s\n' "$OUTPUT_XLSX"
printf '====================================================\n'
