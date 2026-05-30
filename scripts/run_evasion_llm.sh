#!/usr/bin/env bash
# ====================================================
# Evasion 提分一键脚本
# 策略: LLM改写 + 检测器proxy选择
# ====================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- 配置 (通过环境变量或默认值) ---
LLM_BASE_URL="${LLM_BASE_URL:?ERROR: 请设置 LLM_BASE_URL 环境变量}"
LLM_MODEL="${LLM_MODEL:?ERROR: 请设置 LLM_MODEL 环境变量}"
LLM_API_KEY="${LLM_API_KEY:?ERROR: 请设置 LLM_API_KEY 环境变量}"

SOURCE_CSV="data/evasion/official/val_machine_only.csv"
OFFICIAL_INPUT="data/detection/official/val_with_label.csv"
DETECTOR_MODEL="models/cv_xlmr/fold_01"
CANDIDATES_CSV="outputs/evasion/candidates/llm_candidates.csv"
SELECTED_CSV="outputs/evasion/selected/llm_selected.csv"
SUBMISSION_CSV="outputs/evasion/submissions/textgenadvtrack_evasion_val_llm.csv"

CANDIDATES_PER_TEXT="${CANDIDATES_PER_TEXT:-3}"
CONCURRENCY="${CONCURRENCY:-10}"

mkdir -p outputs/evasion/candidates outputs/evasion/selected outputs/evasion/submissions

echo ""
echo "===================================================="
echo "  Evasion LLM 改写 Pipeline"
echo "===================================================="
echo "  LLM: $LLM_MODEL"
echo "  每条文本生成: ${CANDIDATES_PER_TEXT} 个候选"
echo "  并发数: ${CONCURRENCY}"
echo "  检测器: $DETECTOR_MODEL"
echo "===================================================="

# --- Step 1: LLM 生成候选 ---
echo ""
echo "===================================================="
echo "  Step 1/4: LLM 改写生成候选"
echo "===================================================="
python scripts/llm_evasion_generate.py \
  --source-csv "$SOURCE_CSV" \
  --output-csv "$CANDIDATES_CSV" \
  --candidates-per-text "$CANDIDATES_PER_TEXT" \
  --concurrency "$CONCURRENCY" \
  --base-url "$LLM_BASE_URL" \
  --model "$LLM_MODEL" \
  --api-key "$LLM_API_KEY"

# --- Step 2: 检测器打分 + 选择最优 ---
echo ""
echo "===================================================="
echo "  Step 2/4: 检测器打分 + 选择最优候选"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli select-evasion \
  --candidates-csv "$CANDIDATES_CSV" \
  --output-csv "$SELECTED_CSV" \
  --model-dir "$DETECTOR_MODEL"

# --- Step 3: 构建提交文件 ---
echo ""
echo "===================================================="
echo "  Step 3/4: 构建 evasion submission"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli build-evasion-submit \
  --official-input-csv "$OFFICIAL_INPUT" \
  --selected-csv "$SELECTED_CSV" \
  --output-csv "$SUBMISSION_CSV"

# --- Step 4: 验证 ---
echo ""
echo "===================================================="
echo "  Step 4/4: 验证 submission"
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
