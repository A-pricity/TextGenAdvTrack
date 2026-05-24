from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)


def default_provider() -> str:
    return os.getenv("TEXTGENADVTRACK_DEFAULT_PROVIDER", "mock")


def default_model() -> str:
    return os.getenv("TEXTGENADVTRACK_DEFAULT_MODEL", "mock-model")
