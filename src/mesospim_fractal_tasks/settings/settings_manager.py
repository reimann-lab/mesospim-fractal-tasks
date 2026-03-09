from __future__ import annotations

import json
import logging
import shutil
from importlib import resources
from pathlib import Path

from platformdirs import user_data_path

logger = logging.getLogger(__name__)


DEFAULTS_PKG = "mesospim_fractal_tasks"
DEFAULTS_SUBDIR = "settings"


def get_writable_channel_settings_dir() -> Path:
    """
    Return the user-writable directory where channel preset JSON files live.
    """
    data_dir = user_data_path(DEFAULTS_PKG, appauthor=DEFAULTS_PKG)
    data_dir.mkdir(parents=True, exist_ok=True)

    settings_dir = data_dir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir


def ensure_default_channel_presets_copied() -> Path:
    """
    Copy packaged default preset JSON files into the writable settings directory
    if they are not already present. Existing user files are never overwritten.
    """
    dst_dir = get_writable_channel_settings_dir()
    packaged_dir = resources.files(DEFAULTS_PKG) / DEFAULTS_SUBDIR

    for res in packaged_dir.iterdir():
        if not res.is_file():
            continue

        name = res.name
        if not (name.startswith("channel_color_") and name.endswith(".json")):
            continue

        dst = dst_dir / name
        if dst.exists():
            continue

        with resources.as_file(res) as src_path:
            shutil.copyfile(src_path, dst)

    return dst_dir


def get_channel_settings_dir() -> Path:
    """
    Public API for runtime use.
    Ensures defaults exist, then returns the writable settings directory.
    """
    return ensure_default_channel_presets_copied()


def list_channel_settings_files() -> list[Path]:
    """
    Return all available channel preset JSON files from the writable settings dir.
    """
    settings_dir = get_channel_settings_dir()
    return sorted(
        p for p in settings_dir.glob("channel_color_*.json")
        if p.is_file()
    )


def find_channel_settings_file(user_channels_path: str | Path = "default") -> Path:
    """
    Resolve a channel settings input to a real JSON file in the writable settings dir.

    Accepted inputs:
    - existing file path
    - keyword such as 'default', 'lectin', etc., matched against filenames
    """
    # Explicit file path wins
    candidate = Path(user_channels_path)
    if candidate.exists():
        logger.info("Loading channel-specific information from %s", candidate)
        return candidate

    keyword = str(user_channels_path)
    settings_dir = get_channel_settings_dir()
    json_files = [
        p for p in settings_dir.glob("channel_color_*.json")
        if p.is_file() and keyword in p.name
    ]

    if len(json_files) != 1:
        msg = (
            f"Expected exactly one JSON file for keyword '{keyword}' in "
            f"{settings_dir}, found {len(json_files)}."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Loading channel-specific information from %s", json_files[0])
    return json_files[0]


def load_channel_colors(user_channels_path: str | Path = "default") -> dict:
    """
    Load channel color settings from a real path or from a matching preset name
    in the writable settings directory.
    """
    resolved_path = find_channel_settings_file(user_channels_path)
    with resolved_path.open("r", encoding="utf-8") as f:
        return json.load(f)