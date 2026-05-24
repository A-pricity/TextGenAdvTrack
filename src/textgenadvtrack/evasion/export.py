from pathlib import Path

from textgenadvtrack.io import load_evasion_official_rows, save_csv


def build_evasion_submission(
    official_input_csv: Path,
    selected_csv: Path,
    output_csv: Path,
) -> dict:
    import pandas as pd

    official_rows = load_evasion_official_rows(official_input_csv)
    selected_frame = pd.read_csv(selected_csv).fillna("")
    selected_lookup = {
        row["parent_id"]: row["final_text"]
        for row in selected_frame.to_dict(orient="records")
    }

    output_rows = []
    machine_index = 0
    for row in official_rows:
        if row.label == 0:
            machine_index += 1
            source_id = f"eva-{machine_index:04d}"
            text = selected_lookup.get(source_id, row.text)
        else:
            text = row.text
        if row.label == 1:
            text = row.text
        output_rows.append({"prompt": row.prompt, "text": text})

    save_csv(output_rows, output_csv)
    return {"rows": len(output_rows), "output_csv": str(output_csv)}
