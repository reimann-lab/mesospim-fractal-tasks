import shutil
from pathlib import Path
import json
import typer

app = typer.Typer()

def set_channel_setting(
    json_file: str,
    setting_name: str = "default"
):
    """
    Set a JSON channel settings as new setting to use during conversion.
    """

    src = Path(json_file)
    if not src.exists():
        raise RuntimeError(f"JSON file not found: {src}")

    with open(src, "r") as f:
        channel_settings = json.load(f)
    
    # Check json integrity
    for key, entry in channel_settings.items():
        required = ["label", "laser_wavelength", "color"]
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid channel setting: {key}. Must be a dictionary.")
        if not set(required).issubset(set(entry.keys())):
            raise ValueError(f"Invalid channel setting: {key}. Missing required entry : "
                             "among 'label', 'laser_wavelength', 'color'.")
        for k, v in entry.items():
            if not isinstance(v, str):
                raise ValueError(f"Invalid channel setting: {key}. "
                                 f"Entry '{k}' must be a string.")
    
    dst = Path(".", "src", "mesospim_fractal_tasks", "settings")

    target = dst / f"channel_color_{setting_name}.json"
    if target.exists():
        print(f"⚠️ {target.name} already exists, overwriting...")
    else:
        print(f"Copied {src.name} → {target}")
    shutil.copy(src, target)

    print("Done! The new setting can now be used during conversion.")

@app.command()
def copy_run_templates():
    """
    Copy local run templates into the chosen directory (default: current dir).
    """

    src = Path(__file__).parent / "templates"
    dst = Path(".")

    if not src.exists():
        raise RuntimeError(f"Template directory not found: {src}")

    for file in src.glob("run_*.py"):
        target = dst / file.name
        if target.exists():
            print(f"⚠️ {file.name} already exists, overwriting...")
        else:
            print(f"Copied {file.name} → {target}")
        shutil.copy(file, target)

    print("Done! You can edit the run templates safely.")

@app.command()
def copy_channel_template():
    """
    Copy local run templates into the chosen directory (default: current dir).
    """

    src = Path(__file__).parent / "templates"
    dst = Path(".")

    if not src.exists():
        raise RuntimeError(f"Template directory not found: {src}")

    source = src / "channel_template.json"
    target = dst / "channel_template.json"
    if target.exists():
        print(f"⚠️ {target.name} already exists, overwriting...")
    else:
        print(f"Copied {source.name} → {target}")
    shutil.copy(source, target)

    print("Done! You can edit the channel template safely.")

@app.command()
def set_channel_setting_cli(
    json_file: str = typer.Argument(..., help="Path to JSON file."),
    setting_name: str = typer.Option("default", help="Name for this channel setting.")
):
    """
    Set a JSON channel settings file as a named template for conversion.
    """
    set_channel_setting(json_file, setting_name)

def main():
    app()
