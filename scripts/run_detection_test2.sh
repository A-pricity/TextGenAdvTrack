#!/usr/bin/env bash
# ====================================================
# Detection test2: 用已训练的 10-fold CV 模型对 test2 做预测
# 不重新训练，只做 inference + blend
# ====================================================
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

TEST2_INPUT="${TEST2_INPUT:-data/detection/official/test2_input.csv}"
CV_FOLDS="${CV_FOLDS:-10}"
CV_MODEL_DIR="${CV_MODEL_DIR:-models/cv_xlmr}"
CV_SUBMISSION_DIR="${CV_SUBMISSION_DIR:-outputs/detection/submissions/cv_test2}"
CV_ENSEMBLE_XLSX="${CV_ENSEMBLE_XLSX:-outputs/detection/submissions/Perplexity-Hunters_detection_test2.xlsx}"

mkdir -p "$CV_SUBMISSION_DIR" outputs/detection/submissions

require_file() {
  if [[ ! -f "$1" ]]; then
    printf '[error] missing required file: %s\n' "$1" >&2
    exit 1
  fi
}

require_file "$TEST2_INPUT"

printf '\n====================================================\n'
printf '  Detection Test2 Pipeline\n'
printf '====================================================\n'
printf '  Input: %s\n' "$TEST2_INPUT"
printf '  Folds: %s\n' "$CV_FOLDS"
printf '  Models: %s\n' "$CV_MODEL_DIR"
printf '====================================================\n'

# Step 1: 每个 fold 做预测
blend_args=()
for fold in $(seq 1 "$CV_FOLDS"); do
  fold_name="$(printf 'fold_%02d' "$fold")"
  fold_model_dir="$CV_MODEL_DIR/$fold_name"
  fold_submission="$CV_SUBMISSION_DIR/test2_xlmr_${fold_name}.xlsx"

  require_file "$fold_model_dir/metadata.json"

  printf '\n[Step 1/%d] Fold %s -> %s\n' "$CV_FOLDS" "$fold_name" "$fold_submission"
  "$PYTHON_BIN" -m textgenadvtrack.cli export-detection-submit \
    --input-csv "$TEST2_INPUT" \
    --model-dir "$fold_model_dir" \
    --output-xlsx "$fold_submission"

  blend_args+=(--prediction "$fold_submission" --weight 1)
done

# Step 2: Blend 所有 fold
printf '\n[Step 2] Blending %d folds -> %s\n' "$CV_FOLDS" "$CV_ENSEMBLE_XLSX"
"$PYTHON_BIN" -m textgenadvtrack.cli blend-detection-submit \
  "${blend_args[@]}" \
  --output-xlsx "$CV_ENSEMBLE_XLSX"

# Step 3: 验证
printf '\n[Step 3] Validating...\n'
"$PYTHON_BIN" -m textgenadvtrack.cli validate-detection-submit \
  --input-csv "$TEST2_INPUT" \
  --submission-xlsx "$CV_ENSEMBLE_XLSX"

printf '\n====================================================\n'
printf '  DONE\n'
printf '  Detection test2 submission: %s\n' "$CV_ENSEMBLE_XLSX"
printf '====================================================\n'
