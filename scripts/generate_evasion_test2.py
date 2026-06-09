#!/usr/bin/env python3
"""
Generate evasion submission for test2 dataset.
Same approach as test1 (generate_evasion_test1.py):
1. Load test2 input (prompt, text)
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from textgenadvtrack.detection.predict import predict_human_scores


def load_test_input(csv_path: Path) -> list[dict]:
    """Load test input with prompt, text columns."""
    df = pd.read_csv(csv_path)
    assert "prompt" in df.columns, f"Missing 'prompt' column. Found: {df.columns.tolist()}"
    assert "text" in df.columns, f"Missing 'text' column. Found: {df.columns.tolist()}"
    return df.to_dict(orient="records")


def apply_evasion_backtranslate(text: str) -> str:
    """Apply back-translation style evasion via Google Translate free API."""
    import re
    import requests

    def translate_google(text: str, src: str, tgt: str, timeout: int = 30) -> str:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": src,
            "tl": tgt,
            "dt": "t",
            "q": text[:5000],
        }
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return "".join(item[0] for item in data[0] if item[0])
        except Exception as e:
            print(f"  [WARN] translate failed: {e}", file=sys.stderr)
            return text

    # Detect original language
    if re.search(r'[\u0400-\u04FF]', text):
        src_lang = "en"
    elif re.search(r'[\u4e00-\u9fff]', text):
        src_lang = "zh"
        pivot_lang = "en"
    else:
        src_lang = "en"
        pivot_lang = "zh"

    if src_lang == "zh":
        pivot_lang = "en"
    else:
        pivot_lang = "zh"

    # Forward: src -> pivot
    translated = translate_google(text, src_lang, pivot_lang)
    if translated == text:
        return text

    time.sleep(0.3)  # rate limit

    # Backward: pivot -> src
    back = translate_google(translated, pivot_lang, src_lang)
    return back if back else text


def apply_evasion_simple(text: str) -> str:
    """Simple rule-based evasion (back-translation substitute)."""
    import re
    import random

    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    if len(sentences) > 2:
        mid = len(sentences) // 2
        reordered = sentences[:1] + sentences[mid:] + sentences[1:mid]
        return " ".join(reordered)

    # Vary punctuation
    replacements = [
        (' -- ', ' - '),
        (' - ', ' -- '),
        ('; ', '. '),
        (': ', ' - '),
    ]
    for old, new in replacements:
        if old in text and random.random() > 0.5:
            text = text.replace(old, new, 1)
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate evasion submission for test2")
    parser.add_argument("--test-input", type=Path, required=True,
                        help="Path to test2 input CSV (prompt, text)")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to detection model directory")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Path to output submission CSV")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection score threshold for machine text (default: 0.5)")
    parser.add_argument("--evasion-method", choices=["simple", "backtranslate"],
                        default="backtranslate", help="Evasion method to use")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for scoring")
    args = parser.parse_args()

    print(f"Loading test input from {args.test_input}...")
    rows = load_test_input(args.test_input)
    print(f"Loaded {len(rows)} rows")

    texts = [row["text"] for row in rows]

    print(f"Scoring with detector model from {args.model_dir}...")
    start_time = time.time()
    scores = predict_human_scores(texts, args.model_dir)
    elapsed = time.time() - start_time
    print(f"Scoring completed in {elapsed:.1f}s")

    # Low score = machine, high score = human
    machine_indices = [i for i, s in enumerate(scores) if s < args.threshold]
    human_indices = [i for i, s in enumerate(scores) if s >= args.threshold]
    print(f"Identified {len(machine_indices)} machine texts (score < {args.threshold})")
    print(f"Identified {len(human_indices)} human texts (score >= {args.threshold})")

    print(f"Applying evasion method: {args.evasion_method}...")
    evasion_fn = apply_evasion_simple if args.evasion_method == "simple" else apply_evasion_backtranslate

    machine_set = set(machine_indices)
    output_rows = []
    evaded = 0
    for i, row in enumerate(rows):
        if i in machine_set:
            evaded_text = evasion_fn(row["text"])
            output_rows.append({"prompt": row["prompt"], "text": evaded_text})
            evaded += 1
            if evaded % 100 == 0:
                print(f"  Evaded {evaded}/{len(machine_indices)}...")
        else:
            output_rows.append({"prompt": row["prompt"], "text": row["text"]})

    print(f"Saving submission to {args.output_csv}...")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(args.output_csv, index=False)

    print(f"Done! Submission saved with {len(output_rows)} rows")
    print(f"  - Machine texts (evaded): {len(machine_indices)}")
    print(f"  - Human texts (unchanged): {len(human_indices)}")


if __name__ == "__main__":
    main()
