import shutil
from pathlib import Path
import json
import typer

get_app = typer.Typer()
set_app = typer.Typer()

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

@set_app.command()
def set_channel_setting_parser(
    json_file: str = typer.Argument(..., help="Path to JSON file."),
    setting_name: str = typer.Option("default", help="Name for this channel setting.")
):
    """
    Set a JSON channel settings file as a named template for conversion.
    """
    set_channel_setting(json_file, setting_name)

def set_channel_setting_cli():
    set_app()

def get_channel_keys(
) -> None:
    """
    Display the available channel settings keys.
    """
    src = Path(__file__).parents[1] / "settings"

    print("Available channel settings keys:")
    for i, file in enumerate(src.glob("*.json")):
        file_name_parts = file.name.split("_")
        print((f"Key {i}: {file_name_parts[2].replace('.json', '')}"))

def get_channel_key(
    key: str
) -> None:
    """
    Display nicely the channel settings for a given key.
    """
    src = Path(__file__).parents[1] / "settings" / f"channel_color_{key}.json"
    print(key)

    if not src.exists():
        raise RuntimeError(f"Channel settings not found: {src}")

    with open(src, "r") as f:
        channel_settings = json.load(f)

    print(f"Channel settings for key {key}:")
    for key, item in channel_settings.items():
        print(f"{    key}:")
        for k, v in item.items():
            print(f"    {k}: {v}")

@get_app.command()
def get_channel_key_parser(
    key: str = typer.Argument(..., help="Key for the channel settings.")
):
    """
    Set a JSON channel settings file as a named template for conversion.
    """
    get_channel_key(key)

def get_channel_key_cli(
):
    """
    Set a JSON channel settings file as a named template for conversion.
    """
    get_app()