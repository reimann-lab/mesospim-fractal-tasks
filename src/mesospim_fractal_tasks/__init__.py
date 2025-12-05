"""
Collection of Fractal tasks to convert and process mesoSPIM lightsheet data.
"""
from importlib.metadata import PackageNotFoundError, version
import subprocess

try:
    __version__ = version("mesospim-fractal-tasks")
except PackageNotFoundError:
    __version__ = "uninstalled"

try:
    __commit__ = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    __commit__ = "unknown"
