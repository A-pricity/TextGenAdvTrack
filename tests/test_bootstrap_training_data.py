from pathlib import Path

from textgenadvtrack.data.bootstrap_training_data import bootstrap_training_data


def test_bootstrap_training_data_copies_templates(tmp_path):
    result = bootstrap_training_data(Path("data/templates"), tmp_path)
    assert len(result["created_files"]) == 3
    assert (tmp_path / "human.csv").exists()
    assert (tmp_path / "ai_original.csv").exists()
    assert (tmp_path / "ai_rewritten.csv").exists()
