"""
Back-translation + rule-based evasion candidate generator.
No LLM API needed - uses free translation APIs.
"""
import argparse
import csv
import os
import random
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ============================================================
# Back-translation via free APIs
# ============================================================

def translate_google(text: str, src: str, tgt: str, timeout: int = 30) -> str:
    """Translate using Google Translate free API."""
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": src,
        "tl": tgt,
        "dt": "t",
        "q": text[:5000],  # limit length
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return "".join(item[0] for item in data[0] if item[0])
    except Exception as e:
        print(f"  [WARN] translate failed: {e}", file=sys.stderr)
        return text


def back_translate(text: str, pivot_lang: str = "zh") -> str:
    """Translate to pivot language then back to original."""
    # Detect original language
    if re.search(r'[\u0400-\u04FF]', text):
        src_lang = "en"  # most "ru" texts are actually mixed
    elif re.search(r'[\u4e00-\u9fff]', text):
        src_lang = "zh"
        pivot_lang = "en"
    else:
        src_lang = "en"

    # Forward: src -> pivot
    translated = translate_google(text, src_lang, pivot_lang)
    if translated == text:
        return text

    time.sleep(0.3)  # rate limit

    # Backward: pivot -> src
    back = translate_google(translated, pivot_lang, src_lang)
    return back if back else text


# ============================================================
# Rule-based augmentation
# ============================================================

def split_and_rejoin(text: str) -> str:
    """Split sentences and rejoin with slight variations."""
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    if len(sentences) < 2:
        return text
    # Shuffle middle sentences slightly
    mid = sentences[1:-1]
    if mid and random.random() > 0.5:
        random.shuffle(mid)
        sentences[1:-1] = mid
    return " ".join(sentences)


def add_human_touch(text: str) -> str:
    """Add subtle human-like touches."""
    touches = [
        (r'\.$', '. Actually, that about sums it up.'),
        (r'!$', '! That is really something.'),
        (r'\.$', ' -- at least that is how I see it.'),
    ]
    if random.random() > 0.7:
        for pattern, replacement in touches:
            if re.search(pattern, text[-20:]):
                text = re.sub(pattern, replacement, text, count=1)
                break
    return text


def vary_punctuation(text: str) -> str:
    """Slightly vary punctuation."""
    replacements = [
        (' -- ', ' - '),
        (' - ', ' -- '),
        ('; ', '. '),
        (': ', ' - '),
        ('e.g.', 'for example'),
        ('i.e.', 'that is'),
    ]
    for old, new in replacements:
        if old in text and random.random() > 0.5:
            text = text.replace(old, new, 1)
    return text


def apply_rules(text: str) -> str:
    """Apply a combination of rule-based transformations."""
    text = split_and_rejoin(text)
    text = vary_punctuation(text)
    if random.random() > 0.8:
        text = add_human_touch(text)
    return text


# ============================================================
# Main logic
# ============================================================

def process_one(source_row: dict, cand_index: int) -> dict | None:
    """Generate one candidate using back-translation + rules."""
    sample_id = source_row["sample_id"]
    source_text = source_row["source_text"]
    candidate_id = f"{sample_id}-bt-{cand_index:02d}"

    try:
        # Method alternation
        if cand_index == 1:
            # Back-translate
            rewritten = back_translate(source_text)
        elif cand_index == 2:
            # Back-translate via different pivot
            rewritten = back_translate(source_text, pivot_lang="ja")
        else:
            # Back-translate + rules
            rewritten = back_translate(source_text)
            rewritten = apply_rules(rewritten)

        if not rewritten or len(rewritten) < 10:
            rewritten = apply_rules(source_text)

        return {
            "candidate_id": candidate_id,
            "parent_id": sample_id,
            "language": source_row.get("language", "unknown"),
            "domain": source_row.get("domain", "official_val"),
            "source_model": source_row.get("source_model", "unknown"),
            "prompt_type": source_row.get("prompt_type", "unknown"),
            "rewrite_model": "back_translate",
            "rewrite_type": "back_translate",
            "semantic_score": 0.0,
            "proxy_score_1": 0.0,
            "proxy_score_2": 0.0,
            "selected": False,
            "text": rewritten,
        }
    except Exception as e:
        print(f"  [WARN] {candidate_id} failed: {e}", file=sys.stderr)
        return None


def load_progress(output_csv: str) -> set:
    done = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["parent_id"])
    return done


def write_rows(rows: list[dict], output_csv: str, write_header: bool):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--candidates-per-text", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    with open(args.source_csv, "r", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    print(f"[info] Loaded {len(source_rows)} source texts")

    done_parents = load_progress(args.output_csv)
    if done_parents:
        print(f"[resume] Found {len(done_parents)} already-processed texts, skipping")

    remaining = [r for r in source_rows if r["sample_id"] not in done_parents]
    print(f"[info] Processing {len(remaining)} texts x {args.candidates_per_text} candidates")

    write_header = not os.path.exists(args.output_csv) or os.path.getsize(args.output_csv) == 0

    completed = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for source_row in remaining:
            for ci in range(1, args.candidates_per_text + 1):
                future = executor.submit(process_one, source_row, ci)
                futures[future] = source_row["sample_id"]

        batch = []
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result:
                batch.append(result)
            else:
                failed += 1

            if len(batch) >= 50:
                write_rows(batch, args.output_csv, write_header)
                write_header = False
                batch = []

            if completed % 10 == 0 or completed == len(futures):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(futures) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(futures)}] ok={completed - failed} fail={failed} "
                      f"rate={rate:.1f}/s eta={eta / 60:.0f}min")
                sys.stdout.flush()

        if batch:
            write_rows(batch, args.output_csv, write_header)

    elapsed = time.time() - start_time
    print(f"\n[done] Generated {completed - failed} candidates in {elapsed:.0f}s")
    print(f"  Output: {args.output_csv}")


if __name__ == "__main__":
    main()
