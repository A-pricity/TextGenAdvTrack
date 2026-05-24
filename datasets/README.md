# Source Datasets

This tree holds raw and source-side inputs.

## Canonical layout

- `datasets/official/raw/` - official raw inputs
- `datasets/public/` - source human corpora and extras
- `datasets/training/` - seed training sources
- `datasets/evasion/official/` - evasion seed source
- `datasets/manifests/` - registries used to generate or classify source data

## Rule

- Put new raw inputs here.
- Put cleaned, split, or model-ready outputs under `data/`.
- Treat everything under `data/` as derived unless a file is explicitly a source registry.

## Adding new data

1. Copy the raw source into the matching `datasets/` subfolder.
2. Update the manifest entry in `datasets/catalog.csv`.
3. Run the relevant build command to generate processed files into `data/`.

## External detection sources

For any new public or external detection corpus, keep the raw file under `datasets/` and normalize it with:

```bash
.venv/bin/python -m textgenadvtrack.cli ingest-external-detection-dataset \
  --input-path datasets/public/your_source.csv \
  --output-csv data/detection/your_source_normalized.csv \
  --source-name your_source \
  --language zh \
  --domain news \
  --split train
```

Supported raw formats:

- CSV
- JSONL
- Parquet
