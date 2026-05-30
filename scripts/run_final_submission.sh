#!/usr/bin/env bash
# ====================================================
# 最终提交一键脚本
# 策略: classic_plus + XLM-R 10-fold ensemble 融合
# ====================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- 配置 ---
VAL_CSV="data/detection/official/val_with_label.csv"
TEST1_CSV="data/detection/official/test1_input.csv"
CLASSIC_MODEL="models/classic_plus_full"
XLMR_MODEL="models/cv_xlmr/fold_01"
OUT_DIR="outputs/detection"
SUB_DIR="$OUT_DIR/submissions"
SCORE_DIR="$OUT_DIR/scores"

mkdir -p "$SCORE_DIR" "$SUB_DIR"

# --- Step 1: 训练 classic_plus ---
echo ""
echo "===================================================="
echo "  Step 1/6: 训练 classic_plus 模型"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli train-detector \
  --backend classic_plus \
  --model-name xlm-roberta-base \
  --train-csv "$VAL_CSV" \
  --dev-csv "$VAL_CSV" \
  --output-dir "$CLASSIC_MODEL"

# --- Step 2: classic_plus 对 val 打分 ---
echo ""
echo "===================================================="
echo "  Step 2/6: classic_plus 对 val 打分"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv "$VAL_CSV" \
  --model-dir "$CLASSIC_MODEL" \
  --output-csv "$SCORE_DIR/classic_plus_val.csv"

# --- Step 3: XLM-R 对 val 打分 ---
echo ""
echo "===================================================="
echo "  Step 3/6: XLM-R fold_01 对 val 打分"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv "$VAL_CSV" \
  --model-dir "$XLMR_MODEL" \
  --output-csv "$SCORE_DIR/xlmr_val.csv"

# --- Step 4: 搜索最优融合权重 ---
echo ""
echo "===================================================="
echo "  Step 4/6: 搜索最优融合权重"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli search-detection-blend \
  --labels-csv "$VAL_CSV" \
  --prediction "$SCORE_DIR/classic_plus_val.csv" \
  --prediction "$SCORE_DIR/xlmr_val.csv" \
  --step 0.05

# --- Step 5: 生成 classic_plus test1 submission ---
echo ""
echo "===================================================="
echo "  Step 5/6: 生成 classic_plus test1 submission"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli export-detection-submit \
  --input-csv "$TEST1_CSV" \
  --model-dir "$CLASSIC_MODEL" \
  --output-xlsx "$SUB_DIR/textgenadvtrack_test1_classic_plus.xlsx"

# --- Step 6: 融合两个 submission ---
echo ""
echo "===================================================="
echo "  Step 6/6: 融合 submission (等权 0.5/0.5)"
echo "===================================================="
echo "NOTE: 请根据 Step 4 的搜索结果调整下面的权重!"
echo "      当前使用默认 0.5/0.5，最优权重可能不同。"

# 默认等权融合，根据 Step 4 结果手动调整
W_CLASSIC="${W_CLASSIC:-0.5}"
W_XLMR="${W_XLMR:-0.5}"

FINAL_XLSX="$SUB_DIR/textgenadvtrack_test1_final_blend.xlsx"

.venv/bin/python -m textgenadvtrack.cli blend-detection-submit \
  --prediction "$SUB_DIR/textgenadvtrack_test1_classic_plus.xlsx" \
  --weight "$W_CLASSIC" \
  --prediction "$SUB_DIR/textgenadvtrack_test1_xlmr_cv_ensemble.xlsx" \
  --weight "$W_XLMR" \
  --output-xlsx "$FINAL_XLSX"

# --- 验证 ---
echo ""
echo "===================================================="
echo "  验证最终 submission"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli validate-detection-submit \
  --input-csv "$TEST1_CSV" \
  --submission-xlsx "$FINAL_XLSX"

echo ""
echo "===================================================="
echo "  DONE"
echo "===================================================="
echo "  最终 submission: $FINAL_XLSX"
echo ""
echo "  如果 Step 4 搜索出的最优权重不是 0.5/0.5,"
echo "  重新运行:"
echo "    W_CLASSIC=<最优权重> W_XLMR=<1-最优权重> bash scripts/run_final_submission.sh"
echo "===================================================="
