#!/usr/bin/env bash
# ====================================================
# Detection 增强版脚本
# 策略: classic_plus + XLM-R + XLM-R-8ep + DeBERTa 四方融合
# 预计耗时: ~90 分钟 (4090)
# ====================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- 配置 ---
VAL_CSV="data/detection/official/val_with_label.csv"
TEST1_CSV="data/detection/official/test1_input.csv"
OUT_DIR="outputs/detection"
SUB_DIR="$OUT_DIR/submissions"
SCORE_DIR="$OUT_DIR/scores"
mkdir -p "$SCORE_DIR" "$SUB_DIR"

# 模型目录
CLASSIC_MODEL="models/classic_plus_full"
XLMR3_MODEL="models/cv_xlmr/fold_01"           # 已有的 3-epoch 模型
XLMR8_MODEL="models/xlmr_full_8ep"
DEBERTA_MODEL="models/deberta_full"

# ====================================================
# Step 1: 训练 classic_plus
# ====================================================
echo ""
echo "===================================================="
echo "  Step 1/8: 训练 classic_plus"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli train-detector \
  --backend classic_plus \
  --model-name xlm-roberta-base \
  --train-csv "$VAL_CSV" \
  --dev-csv "$VAL_CSV" \
  --output-dir "$CLASSIC_MODEL"

# ====================================================
# Step 2: 训练 XLM-R 8-epoch
# ====================================================
echo ""
echo "===================================================="
echo "  Step 2/8: 训练 XLM-R (8 epochs)"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli train-detector \
  --backend transformer \
  --model-name ./xlm-roberta-base \
  --train-csv "$VAL_CSV" \
  --dev-csv "$VAL_CSV" \
  --output-dir "$XLMR8_MODEL" \
  --epochs 8 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-5 \
  --max-length 512 \
  --weight-decay 0.01

# ====================================================
# Step 3: 训练 DeBERTa-v3-base
# ====================================================
echo ""
echo "===================================================="
echo "  Step 3/8: 训练 DeBERTa-v3-base"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli train-detector \
  --backend transformer \
  --model-name microsoft/deberta-v3-base \
  --train-csv "$VAL_CSV" \
  --dev-csv "$VAL_CSV" \
  --output-dir "$DEBERTA_MODEL" \
  --epochs 5 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --max-length 512 \
  --weight-decay 0.01

# ====================================================
# Step 4: 四个模型对 val 打分
# ====================================================
echo ""
echo "===================================================="
echo "  Step 4/8: 四个模型对 val 打分"
echo "===================================================="

# classic_plus
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv "$VAL_CSV" \
  --model-dir "$CLASSIC_MODEL" \
  --output-csv "$SCORE_DIR/classic_plus_val.csv"

# XLM-R 3-epoch (已有)
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv "$VAL_CSV" \
  --model-dir "$XLMR3_MODEL" \
  --output-csv "$SCORE_DIR/xlmr3_val.csv"

# XLM-R 8-epoch
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv "$VAL_CSV" \
  --model-dir "$XLMR8_MODEL" \
  --output-csv "$SCORE_DIR/xlmr8_val.csv"

# DeBERTa
.venv/bin/python -m textgenadvtrack.cli score-detection-csv \
  --input-csv "$VAL_CSV" \
  --model-dir "$DEBERTA_MODEL" \
  --output-csv "$SCORE_DIR/deberta_val.csv"

# ====================================================
# Step 5: 搜索最优融合权重 (两两搜索)
# ====================================================
echo ""
echo "===================================================="
echo "  Step 5/8: 搜索融合权重 (classic + xlmr3)"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli search-detection-blend \
  --labels-csv "$VAL_CSV" \
  --prediction "$SCORE_DIR/classic_plus_val.csv" \
  --prediction "$SCORE_DIR/xlmr3_val.csv" \
  --step 0.05

echo ""
echo "===================================================="
echo "  Step 5/8: 搜索融合权重 (classic + xlmr8)"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli search-detection-blend \
  --labels-csv "$VAL_CSV" \
  --prediction "$SCORE_DIR/classic_plus_val.csv" \
  --prediction "$SCORE_DIR/xlmr8_val.csv" \
  --step 0.05

echo ""
echo "===================================================="
echo "  Step 5/8: 搜索融合权重 (classic + deberta)"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli search-detection-blend \
  --labels-csv "$VAL_CSV" \
  --prediction "$SCORE_DIR/classic_plus_val.csv" \
  --prediction "$SCORE_DIR/deberta_val.csv" \
  --step 0.05

# ====================================================
# Step 6: 生成四个模型的 test1 submission
# ====================================================
echo ""
echo "===================================================="
echo "  Step 6/8: 生成 test1 submission (4个模型)"
echo "===================================================="

.venv/bin/python -m textgenadvtrack.cli export-detection-submit \
  --input-csv "$TEST1_CSV" \
  --model-dir "$CLASSIC_MODEL" \
  --output-xlsx "$SUB_DIR/textgenadvtrack_test1_classic_plus.xlsx"

# XLM-R 3-epoch 已有，不需要重新生成
# XLM-R 10-fold ensemble 已有

.venv/bin/python -m textgenadvtrack.cli export-detection-submit \
  --input-csv "$TEST1_CSV" \
  --model-dir "$XLMR8_MODEL" \
  --output-xlsx "$SUB_DIR/textgenadvtrack_test1_xlmr8ep.xlsx"

.venv/bin/python -m textgenadvtrack.cli export-detection-submit \
  --input-csv "$TEST1_CSV" \
  --model-dir "$DEBERTA_MODEL" \
  --output-xlsx "$SUB_DIR/textgenadvtrack_test1_deberta.xlsx"

# ====================================================
# Step 7: 四方融合
# ====================================================
echo ""
echo "===================================================="
echo "  Step 7/8: 四方融合 (等权)"
echo "===================================================="
echo "NOTE: 请根据 Step 5 搜索结果调整权重!"
echo "      当前默认等权 0.25/0.25/0.25/0.25"

W_CLASSIC="${W_CLASSIC:-0.25}"
W_XLMR3="${W_XLMR3:-0.25}"
W_XLMR8="${W_XLMR8:-0.25}"
W_DEBERTA="${W_DEBERTA:-0.25}"

FINAL_XLSX="$SUB_DIR/textgenadvtrack_test1_final_4way_blend.xlsx"

.venv/bin/python -m textgenadvtrack.cli blend-detection-submit \
  --prediction "$SUB_DIR/textgenadvtrack_test1_classic_plus.xlsx" \
  --weight "$W_CLASSIC" \
  --prediction "$SUB_DIR/textgenadvtrack_test1_xlmr_cv_ensemble.xlsx" \
  --weight "$W_XLMR3" \
  --prediction "$SUB_DIR/textgenadvtrack_test1_xlmr8ep.xlsx" \
  --weight "$W_XLMR8" \
  --prediction "$SUB_DIR/textgenadvtrack_test1_deberta.xlsx" \
  --weight "$W_DEBERTA" \
  --output-xlsx "$FINAL_XLSX"

# ====================================================
# Step 8: 验证
# ====================================================
echo ""
echo "===================================================="
echo "  Step 8/8: 验证最终 submission"
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
echo "  如果 Step 5 搜索出的最优权重不是等权，"
echo "  重新运行:"
echo "    W_CLASSIC=0.4 W_XLMR3=0.2 W_XLMR8=0.2 W_DEBERTA=0.2 \\"
echo "    bash scripts/run_detection_enhanced.sh"
echo "===================================================="
