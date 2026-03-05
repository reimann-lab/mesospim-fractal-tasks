"""
Initializes the parallelisation list to perform global and z illumination correction.
"""
import logging
from typing import Any
from fractal_tasks_core.channels import get_omero_channel_list
from pydantic import validate_call
import anndata as ad
import zarr
from pathlib import Path

from mesospim_fractal_tasks.utils.zarr_utils import (
    _get_pyramid_structure, create_zarr_pyramid)

logger = logging.getLogger(__name__)


def group_by_channel(
    zarr_path: Path
) -> dict[str, list[str]]:
    """
    Create channel dictionaries for the zarr_urls.

    Keys are channel ids, values are a list of zarr_urls that belong to that channel; 
    the number of FOVs per channel and the index of the channel.
    
    Parameters:
        zarr_path: Path to the OME-Zarr image to be processed.

    Returns:
        channel_dict (dict[str, list[str]]): Dictionary containing the channel 
            information.
    """
    channel_dict = {}
    channels = get_omero_channel_list(image_zarr_path=zarr_path)
    if len(channels) == 0:
        raise ValueError("No channels found in the OME Zarr metadata.")
    FOV_ROI_table = ad.read_zarr(f"{zarr_path}/tables/FOV_ROI_table")

    n_FOVs = len(FOV_ROI_table)

    for c, channel in enumerate(channels):
        if channel.index is None:
            channel.index = c
        channel_dict[channel.label] = {"zarr_url": str(zarr_path), 
                                       "n_FOVs": n_FOVs, 
                                       "index": channel.index}
    return channel_dict

@validate_call
def init_correct_illumination(
    *,
    zarr_urls: list[str],
    zarr_dir: str
) -> dict[str, list[dict[str, Any]]]:
    """
    Initializes illumination correction task.

    Parameters:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    if len(zarr_urls) != 1:
        raise ValueError(
            "Error! Expected only one zarr_url for this task."
        )
    zarr_path = Path(zarr_urls[0])
    new_zarr_path = Path(zarr_path.parent, zarr_path.name + "_illum_corr")
    logger.info(f"Start task: `Illumination Correction (Initialisation)` "
                f"for {zarr_path.parent.name}/{zarr_path.name}")

    logger.info(
        f"Calculating illumination profiles (flatfield/darkfield) based on "
        f"randomly sampled z planes across all ROIs for each channel.")
    channels_dict = group_by_channel(zarr_path)

    is_proxy = False
    fractal_tasks = zarr.open_group(zarr_path, mode="r").attrs.get("fractal_tasks", {})
    if "prepare_mesospim_omezarr" in fractal_tasks and zarr_path.name == "fake_raw_image":
        is_proxy = True

    # Create new zarr pyramid
    pyramid_dict = _get_pyramid_structure(zarr_path)   
    create_zarr_pyramid(zarr_path, new_zarr_name=new_zarr_path.name, pyramid_dict=pyramid_dict)

    parallelization_list = []

    for channel, channel_dict in channels_dict.items():
        parallelization_list.append(
            dict(
                zarr_url=channel_dict["zarr_url"],
                init_args=dict(
                    channel_name=channel,
                    channel_index=channel_dict["index"],
                    n_FOVs=channel_dict["n_FOVs"],
                    is_proxy=is_proxy,
                ),
            )
        )
    return dict(parallelization_list=parallelization_list)
    

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_correct_illumination,
        logger_name=logger.name,
    )