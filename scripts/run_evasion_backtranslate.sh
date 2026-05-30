#!/usr/bin/env bash
# ====================================================
# Evasion 回译法 (不需要 LLM API)
# ====================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SOURCE_CSV="data/evasion/official/val_machine_only.csv"
OFFICIAL_INPUT="data/detection/official/val_with_label.csv"
DETECTOR_MODEL="models/cv_xlmr/fold_01"
CANDIDATES_CSV="outputs/evasion/candidates/bt_candidates.csv"
SELECTED_CSV="outputs/evasion/selected/bt_selected.csv"
SUBMISSION_CSV="outputs/evasion/submissions/textgenadvtrack_evasion_val_bt.csv"

CANDIDATES_PER_TEXT="${CANDIDATES_PER_TEXT:-3}"
CONCURRENCY="${CONCURRENCY:-5}"

mkdir -p outputs/evasion/candidates outputs/evasion/selected outputs/evasion/submissions

echo ""
echo "===================================================="
echo "  Evasion 回译法 Pipeline"
echo "===================================================="
echo "  每条文本生成: ${CANDIDATES_PER_TEXT} 个候选"
echo "  并发数: ${CONCURRENCY}"
echo "===================================================="

# Step 1: 回译生成候选
echo ""
echo "===================================================="
echo "  Step 1/4: 回译生成候选"
echo "===================================================="
python -u scripts/backtranslate_evasion.py \
  --source-csv "$SOURCE_CSV" \
  --output-csv "$CANDIDATES_CSV" \
  --candidates-per-text "$CANDIDATES_PER_TEXT" \
  --concurrency "$CONCURRENCY"

# Step 2: 检测器打分 + 选择
echo ""
echo "===================================================="
echo "  Step 2/4: 检测器打分 + 选择最优"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli select-evasion \
  --candidates-csv "$CANDIDATES_CSV" \
  --output-csv "$SELECTED_CSV" \
  --model-dir "$DETECTOR_MODEL"

# Step 3: 构建 submission
echo ""
echo "===================================================="
echo "  Step 3/4: 构建 submission"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli build-evasion-submit \
  --official-input-csv "$OFFICIAL_INPUT" \
  --selected-csv "$SELECTED_CSV" \
  --output-csv "$SUBMISSION_CSV"

# Step 4: 验证
echo ""
echo "===================================================="
echo "  Step 4/4: 验证"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli validate-evasion-submit \
  --official-input-csv "$OFFICIAL_INPUT" \
  --submission-csv "$SUBMISSION_CSV"

echo ""
echo "===================================================="
echo "  DONE"
echo "===================================================="
echo "  最终 submission: $SUBMISSION_CSV"
echo "===================================================="
