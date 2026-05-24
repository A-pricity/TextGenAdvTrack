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
