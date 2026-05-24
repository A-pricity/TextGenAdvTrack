# Current Dataset Summary

## Use These First

| Purpose | Path | Rows |
| --- | --- | ---: |
| Detection train | `data/detection/train.csv` | 9600 |
| Detection val | `data/detection/val.csv` | 2400 |
| Detection rewrite val | `data/detection/rewrite_val.csv` | 2400 |
| Detection test input | `data/detection/test_input.csv` | 24000 |
| Evasion source | `data/evasion/source.csv` | 6000 |
| Evasion candidates | `data/evasion/candidates.csv` | 36000 |
| Evasion selected | `data/evasion/selected.csv` | 6000 |

## Completed Detection Split

### `data/detection/train.csv`

- Rows: `9600`
- Labels: `human=4800`, `machine=4800`
- Text types: `human=4800`, `ai_original=2400`, `ai_rewritten=2400`

### `data/detection/val.csv`

- Rows: `2400`
- Labels: `human=1200`, `machine=1200`
- Text types: `human=1200`, `ai_original=1200`

### `data/detection/rewrite_val.csv`

- Rows: `2400`
- Labels: `human=1200`, `machine=1200`
- Text types: `human=1200`, `ai_rewritten=1200`

## Legacy Data

The following folders are kept for traceability but should not be the default choice:

- `data/training/merged_seed/`
- `data/detection/train_seed/`
- `data/detection/train_merged_seed/`
- `data/batches/`

## Recommended Server Training Inputs

For first transformer run:

```bash
--train-csv data/detection/train.csv
--dev-csv data/detection/val.csv
```

For rewritten robustness evaluation:

```bash
--input-csv data/detection/rewrite_val.csv
```
