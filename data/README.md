# Derived Data

This tree holds processed, split, and model-ready datasets.

## Canonical outputs

- `data/detection/train.csv` - detection training set
- `data/detection/val.csv` - detection validation set
- `data/detection/rewrite_val.csv` - rewritten robustness validation set
- `data/detection/test_input.csv` - official detection test input
- `data/evasion/source.csv` - evasion source texts
- `data/evasion/candidates.csv` - evasion generated candidates
- `data/evasion/selected.csv` - evasion selected rewrites
- `data/manifests/` - inventories and derived metadata

## Legacy or intermediate

- `data/public/` - historical processed public-human copies
- `data/batches/` - batched generation outputs
- `data/detection/official/` - legacy prepared official detection inputs
- `data/evasion/official/` - legacy evasion source seed copy
- `data/training/completed_detection/` - legacy completed detection dataset location
- `data/training/merged_seed/` - legacy merged seed training set
- `data/detection/train_seed/` - legacy small split
- `data/detection/train_merged_seed/` - legacy merged split

## Templates

- `data/templates/` - CSV templates for new batches

## Ingestion

External detection source files can be normalized into this tree with:

```bash
.venv/bin/python -m textgenadvtrack.cli ingest-external-detection-dataset \
  --input-path datasets/public/your_source.csv \
  --output-csv data/detection/your_source_normalized.csv \
  --source-name your_source \
  --language zh \
  --domain news \
  --split train
```

## Rule

- Raw inputs belong in `datasets/`.
- Cleaned or split artifacts belong in `data/`.
- Prefer `data/detection/train.csv` and `data/detection/val.csv` for new detection experiments.
