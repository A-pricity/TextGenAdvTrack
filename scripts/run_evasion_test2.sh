#!/usr/bin/env bash
# ====================================================
# Evasion test2: 与 test1 相同的处理方式
# 检测器识别机器文本 -> 回译改写 -> 合并输出
# ====================================================
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

TEST2_INPUT="${TEST2_INPUT:-data/detection/official/test2_input.csv}"
DETECTOR_MODEL="${DETECTOR_MODEL:-models/cv_xlmr/fold_01}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/evasion/submissions/Perplexity-Hunters_evasion_test2.csv}"
THRESHOLD="${THRESHOLD:-0.5}"
EVASION_METHOD="${EVASION_METHOD:-backtranslate}"

mkdir -p outputs/evasion/submissions

require_file() {
  if [[ ! -f "$1" ]]; then
    printf '[error] missing required file: %s\n' "$1" >&2
    exit 1
  fi
}

require_file "$TEST2_INPUT"
require_file "$DETECTOR_MODEL/metadata.json"

printf '\n====================================================\n'
printf '  Evasion Test2 Pipeline\n'
printf '====================================================\n'
printf '  Input: %s\n' "$TEST2_INPUT"
printf '  Detector: %s\n' "$DETECTOR_MODEL"
printf '  Threshold: %s\n' "$THRESHOLD"
printf '  Method: %s\n' "$EVASION_METHOD"
printf '  Output: %s\n' "$OUTPUT_CSV"
printf '====================================================\n'

"$PYTHON_BIN" -u scripts/generate_evasion_test2.py \
  --test-input "$TEST2_INPUT" \
  --model-dir "$DETECTOR_MODEL" \
  --output-csv "$OUTPUT_CSV" \
  --threshold "$THRESHOLD" \
  --evasion-method "$EVASION_METHOD"

printf '\n====================================================\n'
printf '  DONE\n'
printf '  Evasion test2 submission: %s\n' "$OUTPUT_CSV"
printf '====================================================\n'
