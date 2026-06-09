#!/usr/bin/env python3
"""
Generate evasion submission for test1 dataset.
Steps:
1. Load test1 input (prompt, text)
2. Score with detector to identify machine-generated texts
3. Apply evasion (back-translation) to machine texts
4. Keep human texts unchanged
5. Output final submission CSV
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from textgenadvtrack.detection.predict import predict_human_scores


def load_test_input(csv_path: Path) -> list[dict]:
    """Load test input with prompt, text columns."""
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    assert "prompt" in df.columns, f"Missing 'prompt' column. Found: {df.columns.tolist()}"
    assert "text" in df.columns, f"Missing 'text' column. Found: {df.columns.tolist()}"
    return df.to_dict(orient="records")


def apply_evasion_simple(text: str) -> str:
    """Simple rule-based evasion (back-translation substitute)."""
    # Simple paraphrase: reorder sentences, add filler
    sentences = text.split(". ")
    if len(sentences) > 2:
        # Reorder middle sentences
        mid = len(sentences) // 2
        reordered = sentences[:1] + sentences[mid:] + sentences[1:mid]
        return ". ".join(reordered)
    return text


def apply_evasion_backtranslate(text: str) -> str:
    """Apply back-translation style evasion."""
    # Simulate back-translation by rephrasing
    # In practice, this would use translation APIs
    words = text.split()
    if len(words) > 20:
        # Swap some phrases
        mid = len(words) // 3
        return " ".join(words[mid:] + words[:mid])
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate evasion submission for test1")
    parser.add_argument("--test-input", type=Path, required=True,
                        help="Path to test1 input CSV (prompt, text)")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to detection model directory")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Path to output submission CSV")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection score threshold for machine text (default: 0.5)")
    parser.add_argument("--evasion-method", choices=["simple", "backtranslate"],
                        default="simple", help="Evasion method to use")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for scoring")
    args = parser.parse_args()

    print(f"Loading test input from {args.test_input}...")
    rows = load_test_input(args.test_input)
    print(f"Loaded {len(rows)} rows")

    # Extract texts for scoring
    texts = [row["text"] for row in rows]

    # Score with detector
    print(f"Scoring with detector model from {args.model_dir}...")
    start_time = time.time()
    scores = predict_human_scores(texts, args.model_dir)
    elapsed = time.time() - start_time
    print(f"Scoring completed in {elapsed:.1f}s")

    # Identify machine texts (low score = machine, high score = human)
    # Note: predict_human_scores returns P(human), so low score = machine
    machine_indices = [i for i, s in enumerate(scores) if s < args.threshold]
    human_indices = [i for i, s in enumerate(scores) if s >= args.threshold]
    print(f"Identified {len(machine_indices)} machine texts (score < {args.threshold})")
    print(f"Identified {len(human_indices)} human texts (score >= {args.threshold})")

    # Apply evasion to machine texts
    print(f"Applying evasion method: {args.evasion_method}...")
    evasion_fn = apply_evasion_simple if args.evasion_method == "simple" else apply_evasion_backtranslate

    output_rows = []
    for i, row in enumerate(rows):
        if i in machine_indices:
            # Apply evasion to machine text
            evaded_text = evasion_fn(row["text"])
            output_rows.append({"prompt": row["prompt"], "text": evaded_text})
        else:
            # Keep human text unchanged
            output_rows.append({"prompt": row["prompt"], "text": row["text"]})

    # Save output
    print(f"Saving submission to {args.output_csv}...")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(args.output_csv, index=False)

    print(f"Done! Submission saved with {len(output_rows)} rows")
    print(f"  - Machine texts (evaded): {len(machine_indices)}")
    print(f"  - Human texts (unchanged): {len(human_indices)}")


if __name__ == "__main__":
    main()
