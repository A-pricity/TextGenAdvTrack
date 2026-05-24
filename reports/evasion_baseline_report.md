# Evasion Baseline Report

## Artifacts

- Candidates: `outputs/evasion/candidates/official_val_candidates.csv`
- Selected rewrites: `outputs/evasion/selected/official_val_selected.csv`
- Submission CSV: `outputs/evasion/submissions/textgenadvtrack_evasion_val_baseline.csv`

## Validation

The submission was built from `data/detection/official/val_with_label.csv`.

| Check | Result |
| --- | ---: |
| Candidate rows | 36000 |
| Selected rows | 6000 |
| Submission rows | 12000 |
| Columns | `prompt`, `text` |
| Prompt order unchanged | true |
| Human rows | 6000 |
| Machine rows | 6000 |
| Human text unchanged | true |
| Machine text changed | 6000 |
| Null cells | 0 |

## Notes

This is a rule-based baseline. It completes the official CSV formatting and preservation constraints, but it is not a high-score evasion strategy. For higher scores, replace candidate generation with stronger paraphrase models and use the detector ensemble as a proxy scorer.
