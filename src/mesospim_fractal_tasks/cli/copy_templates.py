import shutil
from pathlib import Path

def copy_templates():
    """
    Copy local run templates into the chosen directory (default: current dir).
    """

    src = Path(__file__).parent / "templates"
    dst = Path(".")

    if not src.exists():
        raise RuntimeError(f"Template directory not found: {src}")

    for file in src.glob("*.py"):
        target = dst / file.name
        if target.exists():
            print(f"⚠️ {file.name} already exists, overwriting...")
        else:
            print(f"Copied {file.name} → {target}")
        shutil.copy(file, target)

    print("Done! You can edit the copied templates safely.")
