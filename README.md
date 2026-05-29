# TextGenAdvTrack Baseline

双子赛道 baseline 工程骨架：

- `AI_Text Detection`
- `AI_Text Evasion`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

在 `.env` 里至少填写：

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
TEXTGENADVTRACK_DEFAULT_PROVIDER=openai
TEXTGENADVTRACK_DEFAULT_MODEL=your-model-id
```

## Detection

数据目录先看：

- 原始数据布局：`datasets/README.md`
- 原始数据索引：`datasets/catalog.csv`
- 派生数据布局：`data/README.md`
- 当前 Detection 训练集：`data/detection/train.csv`
- 当前 Detection 验证集：`data/detection/val.csv`
- 当前 Detection 测试输入：`data/detection/test_input.csv`

新增公开/外部检测数据时，先放入 `datasets/`，再用统一命令归一化：

```bash
.venv/bin/python -m textgenadvtrack.cli ingest-external-detection-dataset \
  --input-path datasets/public/your_source.csv \
  --output-csv data/detection/your_source_normalized.csv \
  --source-name your_source \
  --language zh \
  --domain news \
  --split train
```

服务器一键训练、导出、校验 Detection 提交：

```bash
nohup bash scripts/run_detection_one_click.sh > logs/one_click_xlmr.log 2>&1 &
tail -f logs/one_click_xlmr.log
```

常用覆盖项：

```bash
MODEL_NAME=pretrained/xlm-roberta-base BATCH_SIZE=8 GRAD_ACCUM=2 \
nohup bash scripts/run_detection_one_click.sh > logs/one_click_xlmr.log 2>&1 &
```

脚本默认产出并校验：

- `outputs/detection/scores/xlmr_dev_scores.csv`
- `outputs/detection/submissions/textgenadvtrack_test1_xlmr.xlsx`

可选重复划分/多模型平均：

```bash
MODE=cv CV_FOLDS=10 \
nohup bash scripts/run_detection_one_click.sh > logs/one_click_xlmr_cv.log 2>&1 &
```

```bash
.venv/bin/python -m textgenadvtrack.cli train-detector \
  --backend classic_plus \
  --model-name microsoft/deberta-v3-base \
  --train-csv tests/fixtures/detection_rows.csv \
  --dev-csv tests/fixtures/detection_rows.csv \
  --output-dir tmp_smoke_v2/detector

.venv/bin/python -m textgenadvtrack.cli export-detection-submit \
  --input-csv tests/fixtures/detection_submit.csv \
  --model-dir tmp_smoke_v2/detector \
  --output-xlsx tmp_smoke_v2/detection_submission.xlsx
```

当前有效检测提交产物：

- 持出验证报告：`reports/detection_classic_plus_report.md`
- 最终模型：`models/detector_official_classic_plus_final`
- Test1 提交文件：`outputs/detection/submissions/textgenadvtrack_test1_classic_plus.xlsx`
- 标准训练集：`data/detection/train.csv`

## Evasion

服务器一键生成、选择、导出、校验 Evasion 提交：

```bash
nohup bash scripts/run_evasion_one_click.sh > logs/one_click_evasion.log 2>&1 &
tail -f logs/one_click_evasion.log
```

如果 Detection 模型已经训练好，脚本会优先用它给 Evasion 候选打 proxy 分：

- `models/detector_xlmr_local`
- `models/cv_xlmr/fold_01`

也可以手动指定：

```bash
DETECTOR_MODEL_DIR=models/cv_xlmr/fold_01 \
nohup bash scripts/run_evasion_one_click.sh > logs/one_click_evasion.log 2>&1 &
```

默认产出：

- `outputs/evasion/candidates/official_val_candidates.csv`
- `outputs/evasion/selected/official_val_selected.csv`
- `outputs/evasion/submissions/textgenadvtrack_evasion_val.csv`

```bash
.venv/bin/python -m textgenadvtrack.cli generate-evasion \
  --source-csv tests/fixtures/evasion_source.csv \
  --output-csv tmp_smoke_v2/candidates.csv

.venv/bin/python -m textgenadvtrack.cli select-evasion \
  --candidates-csv tests/fixtures/evasion_candidates.csv \
  --output-csv tmp_smoke_v2/selected.csv

.venv/bin/python -m textgenadvtrack.cli build-evasion-submit \
  --official-input-csv tests/fixtures/evasion_official_input.csv \
  --selected-csv tmp_smoke_v2/selected.csv \
  --output-csv tmp_smoke_v2/evasion_submission.csv
```

当前 Evasion baseline 产物：

- 候选文件：`outputs/evasion/candidates/official_val_candidates.csv`
- 选择结果：`outputs/evasion/selected/official_val_selected.csv`
- 提交 CSV：`outputs/evasion/submissions/textgenadvtrack_evasion_val_baseline.csv`
- 校验报告：`reports/evasion_baseline_report.md`
- 标准 Evasion 输入：`data/evasion/source.csv`
