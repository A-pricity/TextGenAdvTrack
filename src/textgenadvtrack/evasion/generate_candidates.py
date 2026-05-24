from pathlib import Path

from textgenadvtrack.io import load_evasion_source_rows, save_csv


def planned_rewrite_mix() -> list[tuple[str, int]]:
    return [
        ("paraphrase", 2),
        ("reorder", 1),
        ("compress", 1),
        ("expand", 1),
        ("synonym", 1),
    ]


def _paraphrase(text: str, variant: int) -> str:
    prefix = "Please consider this restated version: " if variant == 0 else "Restated: "
    return f"{prefix}{text}"


def _reorder(text: str) -> str:
    parts = text.split()
    if len(parts) < 4:
        return _paraphrase(text, 1)
    midpoint = len(parts) // 2
    rewritten = " ".join(parts[midpoint:] + parts[:midpoint])
    return rewritten if rewritten != text else _paraphrase(text, 1)


def _compress(text: str) -> str:
    parts = text.split()
    if len(parts) <= 8:
        return f"{text} In short, that is the main point."
    rewritten = " ".join(parts[: len(parts) // 2])
    return rewritten if rewritten != text else f"{text} In short, that is the main point."


def _expand(text: str) -> str:
    return f"{text} In other words, the same point can be described more naturally and explicitly."


def _synonymize(text: str) -> str:
    replacements = {
        "important": "notable",
        "good": "strong",
        "bad": "weak",
        "because": "since",
        "但是": "不过",
        "因此": "所以",
    }
    result = text
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result if result != text else f"{text} The wording can be made a little more direct."


def _rewrite_text(text: str, rewrite_type: str, variant_index: int = 0) -> str:
    if rewrite_type == "paraphrase":
        return _paraphrase(text, variant_index)
    if rewrite_type == "reorder":
        return _reorder(text)
    if rewrite_type == "compress":
        return _compress(text)
    if rewrite_type == "expand":
        return _expand(text)
    if rewrite_type == "synonym":
        return _synonymize(text)
    return text


def generate_candidates(source_csv: Path, output_csv: Path, rewrite_model: str = "rule_rewriter") -> list[dict]:
    sources = load_evasion_source_rows(source_csv)
    rows: list[dict] = []
    for source in sources:
        per_source_index = 0
        for rewrite_type, count in planned_rewrite_mix():
            for variant_index in range(count):
                per_source_index += 1
                rows.append(
                    {
                        "candidate_id": f"{source.sample_id}-cand-{per_source_index}",
                        "parent_id": source.sample_id,
                        "language": source.language,
                        "domain": source.domain,
                        "source_model": source.source_model,
                        "prompt_type": source.prompt_type,
                        "rewrite_model": rewrite_model,
                        "rewrite_type": rewrite_type,
                        "semantic_score": 0.0,
                        "proxy_score_1": 0.0,
                        "proxy_score_2": 0.0,
                        "selected": False,
                        "text": _rewrite_text(source.source_text, rewrite_type, variant_index),
                    }
                )
    save_csv(rows, output_csv)
    return rows
