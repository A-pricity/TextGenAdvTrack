#!/usr/bin/env python3
"""
Build evasion submission for test2 (no labels).
Merges detector-identified machine texts (evaded) with human texts (unchanged).

Logic:
  1. Load test2 input (all 24k rows)
  2. Load selected evaded texts (for machine rows only)
  3. For rows with evaded version -> use evaded text
  4. For rows without evaded version -> keep original (human)
  5. Output prompt, text CSV
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build evasion submission for test2")
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Original test2 input (prompt, text)")
    parser.add_argument("--machine-only-csv", type=Path, required=True,
                        help="Machine-only CSV with sample_id (from split_machine_texts.py)")
    parser.add_argument("--selected-csv", type=Path, required=True,
                        help="Selected evaded texts (from select-evasion)")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Final submission CSV (prompt, text)")
    args = parser.parse_args()

    # Load original test2
    original_df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(original_df)} original rows")

    # Load machine-only to get the mapping: original_index -> sample_id
    machine_df = pd.read_csv(args.machine_only_csv)
    print(f"Loaded {len(machine_df)} machine texts")

    # Load selected evaded texts
    selected_df = pd.read_csv(args.selected_csv).fillna("")
    print(f"Loaded {len(selected_df)} selected evaded texts")

    # Build lookup: parent_id -> final_text
    evaded_lookup = {}
    for _, row in selected_df.iterrows():
        evaded_lookup[row["parent_id"]] = row["final_text"]
    print(f"Evasion lookup has {len(evaded_lookup)} entries")

    # Build index mapping: sample_id -> original row index
    # machine_df was created from original_df[machine_mask], so indices match
    # But we need to recover the original indices
    # Re-detect: machine_df has sample_id like "test2-00001"
    # The order matches the filtered rows

    # Simple approach: iterate original, check if machine, apply evasion
    # We need the machine indices. Re-read the machine csv to get the count
    # and match by position in the filtered subset

    # Actually, let's just match by text content for safety
    # Or better: use the sample_id numbering to reconstruct

    # The machine_df was created by filtering original_df where score < threshold
    # The sample_ids are sequential: test2-00001, test2-00002, ...
    # So test2-00001 = first machine text in original order

    # Build a set of machine texts for matching
    machine_texts_set = set(machine_df["source_text"].tolist())

    output_rows = []
    evaded_count = 0
    for idx, row in original_df.iterrows():
        text = row["text"]
        # Check if this text was in the machine set
        if text in machine_texts_set:
            # Find the sample_id for this machine text
            # Use position-based matching: iterate machine_df in order
            pass
        output_rows.append({"prompt": row["prompt"], "text": text})

    # Better approach: use the machine_df index directly
    # machine_df was created from original_df[machine_mask].reset_index(drop=True)
    # So machine_df row i corresponds to the i-th machine text in original order

    # Reconstruct: find which original rows are machine
    # by matching text (works because texts are unique enough)
    machine_text_to_sample = {}
    for _, mrow in machine_df.iterrows():
        machine_text_to_sample[mrow["source_text"]] = mrow["sample_id"]

    output_rows = []
    evaded_count = 0
    for idx, row in original_df.iterrows():
        text = row["text"]
        sample_id = machine_text_to_sample.get(text)
        if sample_id and sample_id in evaded_lookup:
            output_rows.append({"prompt": row["prompt"], "text": evaded_lookup[sample_id]})
            evaded_count += 1
        else:
            output_rows.append({"prompt": row["prompt"], "text": text})

    output_df = pd.DataFrame(output_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    human_count = len(output_rows) - evaded_count
    print(f"\nDone! Saved {len(output_rows)} rows to {args.output_csv}")
    print(f"  Evaded (machine): {evaded_count}")
    print(f"  Unchanged (human): {human_count}")


if __name__ == "__main__":
    main()
