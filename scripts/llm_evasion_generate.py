"""
LLM-based evasion candidate generator.
Reads machine-generated texts and rewrites them via LLM API
to produce human-like candidates that evade detection.

Usage:
    python scripts/llm_evasion_generate.py \
        --source-csv data/evasion/official/val_machine_only.csv \
        --output-csv outputs/evasion/candidates/llm_candidates.csv \
        --candidates-per-text 3 \
        --concurrency 10
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


# ============================================================
# LLM API call (OpenAI-compatible)
# ============================================================

def call_llm(text: str, base_url: str, model: str, api_key: str, timeout: int = 120, max_retries: int = 5) -> str:
    """Call LLM API to rewrite a single text, with retry on failure."""
    import re
    if re.search(r'[\u0400-\u04FF]', text):
        lang_hint = "Russian"
    elif re.search(r'[\u4e00-\u9fff]', text):
        lang_hint = "Chinese"
    else:
        lang_hint = "English"

    system_prompt = (
        "You are a skilled human writer. Your task is to rewrite the given text "
        "so that it reads naturally as if written by a real human. "
        "Rules:\n"
        "- Keep the original meaning and key information intact\n"
        "- Change sentence structure, word choice, and phrasing\n"
        "- Add natural human touches: slight imperfections, personal voice, varied rhythm\n"
        "- Avoid the polished, uniform style typical of AI-generated text\n"
        "- Do NOT add any prefix like 'Here is the rewrite:' — just output the rewritten text\n"
        f"- Write in {lang_hint}, matching the original language\n"
        "- Keep roughly similar length to the original"
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Rewrite this text:\n\n{text}"},
                    ],
                    "temperature": 0.9,
                    "max_tokens": 4096,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_error = e
            wait = min(2 ** attempt, 30)  # exponential backoff, max 30s
            time.sleep(wait)
    raise last_error


# ============================================================
# Main logic
# ============================================================

def load_progress(output_csv: str) -> set:
    """Load already-processed parent_ids from existing output."""
    done = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["parent_id"])
    return done


def write_rows(rows: list[dict], output_csv: str, write_header: bool):
    """Append rows to output CSV."""
    fieldnames = [
        "candidate_id", "parent_id", "language", "domain",
        "source_model", "prompt_type", "rewrite_model",
        "rewrite_type", "semantic_score", "proxy_score_1",
        "proxy_score_2", "selected", "text",
    ]
    mode = "a" if not write_header else "w"
    with open(output_csv, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def process_one_text(
    source_row: dict,
    cand_index: int,
    base_url: str,
    model: str,
    api_key: str,
) -> dict | None:
    """Generate one candidate for a single text."""
    sample_id = source_row["sample_id"]
    source_text = source_row["source_text"]
    candidate_id = f"{sample_id}-llm-{cand_index:02d}"

    try:
        rewritten = call_llm(source_text, base_url, model, api_key)
        if not rewritten or len(rewritten) < 10:
            return None
        return {
            "candidate_id": candidate_id,
            "parent_id": sample_id,
            "language": source_row.get("language", "unknown"),
            "domain": source_row.get("domain", "official_val"),
            "source_model": source_row.get("source_model", "unknown"),
            "prompt_type": source_row.get("prompt_type", "unknown"),
            "rewrite_model": f"llm_{model}",
            "rewrite_type": "llm_paraphrase",
            "semantic_score": 0.0,
            "proxy_score_1": 0.0,
            "proxy_score_2": 0.0,
            "selected": False,
            "text": rewritten,
        }
    except Exception as e:
        print(f"  [WARN] {candidate_id} failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="LLM evasion candidate generator")
    parser.add_argument("--source-csv", required=True, help="Input machine texts CSV")
    parser.add_argument("--output-csv", required=True, help="Output candidates CSV")
    parser.add_argument("--candidates-per-text", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--base-url", default=None, help="LLM API base URL (or set LLM_BASE_URL)")
    parser.add_argument("--model", default=None, help="LLM model name (or set LLM_MODEL)")
    parser.add_argument("--api-key", default=None, help="LLM API key (or set LLM_API_KEY)")
    args = parser.parse_args()

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    model = args.model or os.getenv("TEXTGENADVTRACK_DEFAULT_MODEL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not all([base_url, model, api_key]):
        print("ERROR: Set OPENAI_BASE_URL, TEXTGENADVTRACK_DEFAULT_MODEL, OPENAI_API_KEY in .env or pass --base-url/--model/--api-key", file=sys.stderr)
        sys.exit(1)

    # Load source rows
    with open(args.source_csv, "r", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    print(f"[info] Loaded {len(source_rows)} source texts")

    # Check resume progress
    done_parents = load_progress(args.output_csv)
    if done_parents:
        print(f"[resume] Found {len(done_parents)} already-processed texts, skipping")

    # Filter remaining
    remaining = [r for r in source_rows if r["sample_id"] not in done_parents]
    print(f"[info] Processing {len(remaining)} texts × {args.candidates_per_text} candidates = {len(remaining) * args.candidates_per_text} API calls")
    print(f"[info] Concurrency: {args.concurrency}, Model: {model}")

    # Prepare output file
    write_header = not os.path.exists(args.output_csv) or os.path.getsize(args.output_csv) == 0

    # Process with thread pool
    total = len(remaining)
    completed = 0
    failed = 0
    start_time = time.time()

    print(f"[start] 开始调用 LLM API ...")
    sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for source_row in remaining:
            for ci in range(1, args.candidates_per_text + 1):
                future = executor.submit(
                    process_one_text, source_row, ci, base_url, model, api_key
                )
                futures[future] = source_row["sample_id"]

        batch = []
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result:
                batch.append(result)
            else:
                failed += 1

            # Write in batches of 50
            if len(batch) >= 50:
                write_rows(batch, args.output_csv, write_header)
                write_header = False
                batch = []

            # Progress
            if completed % 10 == 0 or completed == len(futures):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(futures) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(futures)}] ok={completed - failed} fail={failed} "
                      f"rate={rate:.1f}/s eta={eta / 60:.0f}min")

        # Write remaining
        if batch:
            write_rows(batch, args.output_csv, write_header)

    elapsed = time.time() - start_time
    print(f"\n[done] Generated {completed - failed} candidates in {elapsed:.0f}s")
    print(f"  Output: {args.output_csv}")


if __name__ == "__main__":
    main()
