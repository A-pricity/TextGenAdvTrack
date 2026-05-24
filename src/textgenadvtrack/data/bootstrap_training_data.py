from __future__ import annotations

from pathlib import Path

import pandas as pd


TEMPLATE_FILES = {
    "human": "human_template.csv",
    "ai_original": "ai_original_template.csv",
    "ai_rewritten": "ai_rewritten_template.csv",
}


def bootstrap_training_data(template_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for key, filename in TEMPLATE_FILES.items():
        source = template_dir / filename
        target = output_dir / filename.replace("_template", "")
        if not source.exists():
            raise FileNotFoundError(f"Missing template file: {source}")
        if not target.exists():
            pd.read_csv(source).to_csv(target, index=False)
            created.append(str(target))
    return {"output_dir": str(output_dir), "created_files": created}
