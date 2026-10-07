"""Load config.yaml. All pipeline parameters come from here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    env_agent = os.environ.get("SEC_USER_AGENT")
    if env_agent:
        cfg["edgar"]["user_agent"] = env_agent
    return cfg
