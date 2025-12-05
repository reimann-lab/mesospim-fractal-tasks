import json
from pathlib import Path

import mesospim_fractal_tasks

PACKAGE = "mesospim_fractal_tasks"
PACKAGE_DIR = Path(mesospim_fractal_tasks.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
    TASK_LIST = MANIFEST["task_list"]
