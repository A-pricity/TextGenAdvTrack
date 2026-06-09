#!/usr/bin/env python3
"""
Split test input into machine-only texts using a detector model.
Used for evasion pipeline: identify which texts need humanization.

Output format matches what backtranslate_evasion.py expects:
  sample_id, language, domain, source_model, prompt_type, prompt_id, prompt, source_text
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from textgenadvtrack.detection.predict import predict_human_scores


def main():
    parser = argparse.ArgumentParser(
        description="Split machine texts from test input using detector"
    )
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Test input CSV (prompt, text)")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Detection model directory")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output: machine-only CSV for evasion")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold: below = machine (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for scoring")
    args = parser.parse_args()

    print(f"Loading test input from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows")

    texts = df["text"].tolist()

    print(f"Scoring with detector model from {args.model_dir}...")
    scores = predict_human_scores(texts, args.model_dir)
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")

    # Low score = machine, high score = human
    machine_mask = pd.Series(scores) < args.threshold
    machine_df = df[machine_mask].copy().reset_index(drop=True)
    human_count = len(df) - len(machine_df)

    print(f"Detected: {len(machine_df)} machine texts, {human_count} human texts "
          f"(threshold={args.threshold})")

    # Add columns expected by backtranslate_evasion.py
    machine_df.insert(0, "sample_id", [f"test2-{i+1:05d}" for i in range(len(machine_df))])
    machine_df["language"] = "unknown"
    machine_df["domain"] = "official_test2"
    machine_df["source_model"] = "unknown"
    machine_df["prompt_type"] = "unknown"
    machine_df["prompt_id"] = [f"official-test2-{i+1:05d}" for i in range(len(machine_df))]
    machine_df = machine_df.rename(columns={"text": "source_text"})

    # Reorder columns
    cols = ["sample_id", "language", "domain", "source_model",
            "prompt_type", "prompt_id", "prompt", "source_text"]
    machine_df = machine_df[cols]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    machine_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(machine_df)} machine texts to {args.output_csv}")


if __name__ == "__main__":
    main()
