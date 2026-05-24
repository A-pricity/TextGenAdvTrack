# Detection Classic Plus Report

## Model

- Backend: `classic_plus`
- Features: word TF-IDF `(1, 2)` plus character TF-IDF `(3, 5)`
- Classifier: balanced logistic regression
- Label semantics: `0 = machine`, `1 = human`

## Validation

Official validation data was split with seed `42`:

- Train: `outputs/detection/official_splits/official_train.csv` (`9600` rows)
- Dev: `outputs/detection/official_splits/official_dev.csv` (`2400` rows)
- Eval model: `models/detector_official_classic_plus_eval`

Held-out dev metrics:

| Metric | Value |
| --- | ---: |
| AUC | 0.98201875 |
| ACC | 0.9308333333333333 |
| F1 | 0.9326845093268451 |

## Final Artifacts

- Final model trained on all official validation rows: `models/detector_official_classic_plus_final`
- Test1 submission: `outputs/detection/submissions/textgenadvtrack_test1_classic_plus.xlsx`
- Submission rows: `24000`
- Submission sheets: `predictions`, `time`

## Notes

The final model metadata reports metrics against `official_dev.csv`, but that split is included in the final all-train model. Use the eval model metrics above as the independent held-out estimate.
