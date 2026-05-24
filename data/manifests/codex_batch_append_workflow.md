# Codex Batch Append Workflow

目标：让后续每次用 Codex 生成的新数据批次，都能稳定追加到现有训练原料里。

## 推荐流程

### 1. 复制对应批次模板

- `data/templates/human_batch_template.csv`
- `data/templates/ai_original_batch_template.csv`
- `data/templates/ai_rewritten_batch_template.csv`

填入一批新的 CSV 行。

### 2. 追加到主训练原料

```bash
.venv/bin/python -m textgenadvtrack.cli append-training-batch \
  --existing-csv datasets/training/human.csv \
  --batch-csv data/batches/human_batch_001.csv
```

```bash
.venv/bin/python -m textgenadvtrack.cli append-training-batch \
  --existing-csv datasets/training/ai_original.csv \
  --batch-csv data/batches/ai_original_batch_001.csv
```

```bash
.venv/bin/python -m textgenadvtrack.cli append-training-batch \
  --existing-csv datasets/training/ai_rewritten.csv \
  --batch-csv data/batches/ai_rewritten_batch_001.csv
```

默认按 `sample_id` 去重。

## Codex 提示要求

每次让 Codex 生成批次时，必须明确：

- 只输出 CSV 行
- 不要输出解释
- 字段顺序不能变化
- `sample_id` 不能与历史重复
- `ai_rewritten` 必须带 `parent_id`

## 推荐命名

- `data/batches/human_batch_001.csv`
- `data/batches/ai_original_batch_001.csv`
- `data/batches/ai_rewritten_batch_001.csv`

这样后续可以持续追溯每一批数据来源。
