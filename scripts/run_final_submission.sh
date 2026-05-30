#!/usr/bin/env bash
# ====================================================
# Detection 基础版一键脚本
# 策略: classic_plus + XLM-R 10-fold ensemble 融合
# ====================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- 配置 ---
VAL_LABEL_CSV="data/detection/official/val_with_label.csv"
TEST1_CSV="data/detection/official/test1_input.csv"
PREPARED_DIR="/tmp/textgenadvtrack_completed_detection"
CLASSIC_MODEL="models/classic_plus_full"
XLMR_MODEL="models/cv_xlmr/fold_01"
OUT_DIR="outputs/detection"
SUB_DIR="$OUT_DIR/submissions"
SCORE_DIR="$OUT_DIR/scores"
mkdir -p "$SCORE_DIR" "$SUB_DIR"

# ====================================================
# Step 1: 准备训练数据
# ====================================================
echo ""
echo "===================================================="
echo "  Step 1/6: 准备训练数据"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli build-completed-detection-dataset \
  --official-val-with-label-csv "$VAL_LABEL_CSV" \
  --output-dir "$PREPARED_DIR" \
  --dev-fraction 0.2 \
  --seed 42 \
  --language en

TRAIN_CSV="$PREPARED_DIR/completed_train.csv"
DEV_CSV="$PREPARED_DIR/completed_dev.csv"

# ====================================================
# Step 2: 训练 classic_plus
# ====================================================
echo ""
echo "===================================================="
echo "  Step 2/6: 训练 classic_plus 模型"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli train-detector \
  --backend classic_plus \
  --model-name xlm-roberta-base \
  --train-csv "$TRAIN_CSV" \
  --dev-csv "$DEV_CSV" \
  --output-dir "$CLASSIC_MODEL"

# ====================================================
# Step 3: 对 val 打分
# ====================================================
echo ""
echo "===================================================="
echo "  Step 3/6: 对 val 打分"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv "$DEV_CSV" \
  --model-dir "$CLASSIC_MODEL" \
  --output-csv "$SCORE_DIR/classic_plus_val.csv"

.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv "$DEV_CSV" \
  --model-dir "$XLMR_MODEL" \
  --output-csv "$SCORE_DIR/xlmr_val.csv"

# ====================================================
# Step 4: 搜索最优融合权重
# ====================================================
echo ""
echo "===================================================="
echo "  Step 4/6: 搜索最优融合权重"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli search-detection-blend \
  --labels-csv "$DEV_CSV" \
  --prediction "$SCORE_DIR/classic_plus_val.csv" \
  --prediction "$SCORE_DIR/xlmr_val.csv" \
  --step 0.05

# ====================================================
# Step 5: 生成 test1 submission + 融合
# ====================================================
echo ""
echo "===================================================="
echo "  Step 5/6: 生成 classic_plus test1 submission"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli export-detection-submit \
  --input-csv "$TEST1_CSV" \
  --model-dir "$CLASSIC_MODEL" \
  --output-xlsx "$SUB_DIR/textgenadvtrack_test1_classic_plus.xlsx"

echo ""
echo "===================================================="
echo "  Step 5/6: 融合 submission"
echo "===================================================="
echo "NOTE: 请根据 Step 4 的搜索结果调整权重!"

W_CLASSIC="${W_CLASSIC:-0.5}"
W_XLMR="${W_XLMR:-0.5}"

FINAL_XLSX="$SUB_DIR/textgenadvtrack_test1_final_blend.xlsx"

.venv/bin/python -m textgenadvtrack.cli blend-detection-submit \
  --prediction "$SUB_DIR/textgenadvtrack_test1_classic_plus.xlsx" \
  --weight "$W_CLASSIC" \
  --prediction "$SUB_DIR/textgenadvtrack_test1_xlmr_cv_ensemble.xlsx" \
  --weight "$W_XLMR" \
  --output-xlsx "$FINAL_XLSX"

# ====================================================
# Step 6: 验证
# ====================================================
echo ""
echo "===================================================="
echo "  Step 6/6: 验证最终 submission"
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
echo "  如需调整权重:"
echo "    W_CLASSIC=0.6 W_XLMR=0.4 bash scripts/run_final_submission.sh"
echo "===================================================="
