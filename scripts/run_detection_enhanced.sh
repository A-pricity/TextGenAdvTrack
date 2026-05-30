#!/usr/bin/env bash
# ====================================================
# Detection 增强版脚本 (支持断点续跑)
# 策略: classic_plus + XLM-R + XLM-R-8ep + DeBERTa 四方融合
# 预计耗时: ~90 分钟 (4090)
# ====================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- 配置 ---
VAL_LABEL_CSV="data/detection/official/val_with_label.csv"
TEST1_CSV="data/detection/official/test1_input.csv"
OUT_DIR="outputs/detection"
SUB_DIR="$OUT_DIR/submissions"
SCORE_DIR="$OUT_DIR/scores"
PREPARED_DIR="/tmp/textgenadvtrack_completed_detection"
mkdir -p "$SCORE_DIR" "$SUB_DIR"

# 模型目录
CLASSIC_MODEL="models/classic_plus_full"
XLMR3_MODEL="models/cv_xlmr/fold_01"
XLMR8_MODEL="models/xlmr_full_8ep"
DEBERTA_MODEL="models/deberta_full"
DEBERTA_LOCAL="pretrained/deberta-v3-base"

# --- 断点续跑辅助函数 ---
skip_if_done() {
  local marker="$1"
  local desc="$2"
  if [ -f "$marker" ]; then
    echo ""
    echo "  [SKIP] $desc (已完成: $marker)"
    return 0
  fi
  return 1
}

# ====================================================
# Step 1: 准备训练数据
# ====================================================
if skip_if_done "$PREPARED_DIR/completed_train.csv" "Step 1/8: 准备训练数据"; then
  true
else
  echo ""
  echo "===================================================="
  echo "  Step 1/8: 准备训练数据"
  echo "===================================================="
  .venv/bin/python -m textgenadvtrack.cli build-completed-detection-dataset \
    --official-val-with-label-csv "$VAL_LABEL_CSV" \
    --output-dir "$PREPARED_DIR" \
    --dev-fraction 0.2 \
    --seed 42 \
    --language en
fi

TRAIN_CSV="$PREPARED_DIR/completed_train.csv"
DEV_CSV="$PREPARED_DIR/completed_dev.csv"

# ====================================================
# Step 2: 训练 classic_plus
# ====================================================
if skip_if_done "$CLASSIC_MODEL/model.joblib" "Step 2/8: 训练 classic_plus"; then
  true
else
  echo ""
  echo "===================================================="
  echo "  Step 2/8: 训练 classic_plus"
  echo "===================================================="
  .venv/bin/python -m textgenadvtrack.cli train-detector \
    --backend classic_plus \
    --model-name xlm-roberta-base \
    --train-csv "$TRAIN_CSV" \
    --dev-csv "$DEV_CSV" \
    --output-dir "$CLASSIC_MODEL"
fi

# ====================================================
# Step 3: 训练 XLM-R 8-epoch
# ====================================================
if skip_if_done "$XLMR8_MODEL/transformer_model/model.safetensors" "Step 3/8: 训练 XLM-R (8 epochs)"; then
  true
else
  echo ""
  echo "===================================================="
  echo "  Step 3/8: 训练 XLM-R (8 epochs)"
  echo "===================================================="
  .venv/bin/python -m textgenadvtrack.cli train-detector \
    --backend transformer \
    --model-name ./xlm-roberta-base \
    --train-csv "$TRAIN_CSV" \
    --dev-csv "$DEV_CSV" \
    --output-dir "$XLMR8_MODEL" \
    --epochs 8 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-5 \
    --max-length 512 \
    --weight-decay 0.01
fi

# ====================================================
# Step 3.5: 下载 DeBERTa-v3-base (ModelScope)
# ====================================================
if skip_if_done "$DEBERTA_LOCAL/config.json" "Step 3.5/8: 下载 DeBERTa"; then
  true
else
  echo ""
  echo "===================================================="
  echo "  Step 3.5/8: 下载 DeBERTa-v3-base (ModelScope)"
  echo "===================================================="
  pip install modelscope -q
  modelscope download --model iic/nlp_debertav3-base_chinese --local_dir "$DEBERTA_LOCAL"
  echo "  DeBERTa 模型路径: $DEBERTA_LOCAL"
fi

# ====================================================
# Step 4: 训练 DeBERTa-v3-base
# ====================================================
if skip_if_done "$DEBERTA_MODEL/transformer_model/model.safetensors" "Step 4/8: 训练 DeBERTa-v3-base"; then
  true
else
  echo ""
  echo "===================================================="
  echo "  Step 4/8: 训练 DeBERTa-v3-base"
  echo "===================================================="
  .venv/bin/python -m textgenadvtrack.cli train-detector \
    --backend transformer \
    --model-name "$DEBERTA_LOCAL" \
    --train-csv "$TRAIN_CSV" \
    --dev-csv "$DEV_CSV" \
    --output-dir "$DEBERTA_MODEL" \
    --epochs 5 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-5 \
    --max-length 512 \
    --weight-decay 0.01
fi

# ====================================================
# Step 5: 四个模型对 val 打分
# ====================================================
echo ""
echo "===================================================="
echo "  Step 5/8: 四个模型对 val 打分"
echo "===================================================="

for pair in \
  "$CLASSIC_MODEL|$SCORE_DIR/classic_plus_val.csv|classic_plus" \
  "$XLMR3_MODEL|$SCORE_DIR/xlmr3_val.csv|xlmr3" \
  "$XLMR8_MODEL|$SCORE_DIR/xlmr8_val.csv|xlmr8" \
  "$DEBERTA_MODEL|$SCORE_DIR/deberta_val.csv|deberta"
do
  IFS='|' read -r MODEL_DIR SCORE_FILE NAME <<< "$pair"
  if [ -f "$SCORE_FILE" ]; then
    echo "  [SKIP] 打分 $NAME (已完成)"
  else
    echo "  打分 $NAME ..."
    .venv/bin/python -m textgenadvtrack.cli score-detection-csv \
      --input-csv "$DEV_CSV" \
      --model-dir "$MODEL_DIR" \
      --output-csv "$SCORE_FILE"
  fi
done

# ====================================================
# Step 6: 搜索最优融合权重
# ====================================================
echo ""
echo "===================================================="
echo "  Step 6/8: 搜索融合权重"
echo "===================================================="
.venv/bin/python -m textgenadvtrack.cli search-detection-blend \
  --labels-csv "$DEV_CSV" \
  --prediction "$SCORE_DIR/classic_plus_val.csv" \
  --prediction "$SCORE_DIR/xlmr3_val.csv" \
  --prediction "$SCORE_DIR/xlmr8_val.csv" \
  --prediction "$SCORE_DIR/deberta_val.csv" \
  --step 0.05

# ====================================================
# Step 7: 生成 test1 submission + 融合
# ====================================================
echo ""
echo "===================================================="
echo "  Step 7/8: 生成 test1 submission (4个模型)"
echo "===================================================="

for pair in \
  "$CLASSIC_MODEL|$SUB_DIR/textgenadvtrack_test1_classic_plus.xlsx|classic_plus" \
  "$XLMR8_MODEL|$SUB_DIR/textgenadvtrack_test1_xlmr8ep.xlsx|xlmr8ep" \
  "$DEBERTA_MODEL|$SUB_DIR/textgenadvtrack_test1_deberta.xlsx|deberta"
do
  IFS='|' read -r MODEL_DIR XLSX_FILE NAME <<< "$pair"
  if [ -f "$XLSX_FILE" ]; then
    echo "  [SKIP] 导出 $NAME (已完成)"
  else
    echo "  导出 $NAME ..."
    .venv/bin/python -m textgenadvtrack.cli export-detection-submit \
      --input-csv "$TEST1_CSV" \
      --model-dir "$MODEL_DIR" \
      --output-xlsx "$XLSX_FILE"
  fi
done

echo ""
echo "===================================================="
echo "  Step 7/8: 四方融合 (等权)"
echo "===================================================="
echo "NOTE: 请根据 Step 6 搜索结果调整权重!"

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
echo "  如需调整权重，重新运行:"
echo "    W_CLASSIC=0.4 W_XLMR3=0.2 W_XLMR8=0.2 W_DEBERTA=0.2 \\"
echo "    bash scripts/run_detection_enhanced.sh"
echo "===================================================="
