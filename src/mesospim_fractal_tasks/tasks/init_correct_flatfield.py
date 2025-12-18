# This file is partly inspired from the file "init_calculate_basicpy_illumination_models.py" 
# of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
# Adrian Tschan <atschan@apricotx.com>

"""
Initializes the parallelisation list to perform flatfield correction using BaSiCPy.
"""
import logging
from typing import Any, Optional
from fractal_tasks_core.channels import get_omero_channel_list
from pydantic import validate_call
import anndata as ad
import zarr
from pathlib import Path

from fractal_tasks_core.ngff import load_NgffImageMeta

logger = logging.getLogger(__name__)


def group_by_channel(
    zarr_path: Path
) -> dict[str, list[str]]:
    """
    Create channel dictionaries for the zarr_urls.

    Keys are channel ids, values are a list of zarr_urls that belong to that channel
    and the index of the channel.
    
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
    for c, channel in enumerate(channels):
        if channel.index is None:
            channel.index = c
        channel_dict[channel.label] = {"zarr_url": str(zarr_path),  
                                       "index": channel.index}
    return channel_dict

@validate_call
def init_correct_flatfield(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    FOV_list: Optional[list[int]] = None,
    z_levels: Optional[list[int]] = None,
    save_models: bool = False
) -> dict[str, list[dict[str, Any]]]:
    """
    Initialized BaSiCPy illumination correction task
    
    This task prepares a parallelization list of all zarr_urls that need to be
    used to perform illumination correction with BaSiCPy.

    Parameters:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Present for compatibility, not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        FOV_list: List of FOVs to process. If provided, illumination profiles will be 
            computed from this list of FOVs without BaSiCPy. Default: None.
        z_levels: Two integers indicating the maximum number of z planes to process 
            at the top and bottom of the 3D tile stack. If provided, illumination 
            profiles will be computed using z planes up to the first number of z_levels 
            at the bottom and down to the 2nd number of z_levels at the top of the FOVs 
            (expecting empty FOVs) without BaSiCPy.
            If FOV_list is not empty the subvolumes will be extracted from the FOVs 
            in FOV_list, otherwise from the four corner FOVs. Default: None.
        save_models: If `True`, illumination profiles will be saved in the parent folder
            of the currently processed OME-Zarr. Default: False.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    if len(zarr_urls) != 1:
        raise ValueError(
            "Error! Expected only one zarr_url for this task."
        )
    zarr_path = Path(zarr_urls[0])
    logger.info(f"Start task: `Flatfield Correction (Initialisation)` "
                f"for {zarr_path.parent.name}/{zarr_path.name}")

    logger.info(
        f"Calculating illumination profiles (flatfield/darkfield) based on "
        f"randomly sampled z planes across all ROIs for each channel.")

    channels_dict = group_by_channel(zarr_path)

    image_meta = load_NgffImageMeta(zarr_path)
    num_level = image_meta.num_levels
    coarsening_xy = image_meta.coarsening_xy
    if coarsening_xy is None:
        coarsening_xy = 2
    raw_array = zarr.open(zarr_path)["0"]
    for level in range(num_level):
        shape = (raw_array.shape[0], raw_array.shape[1],
                 raw_array.shape[2] // 2**level, 
                 raw_array.shape[3] // 2**level)
        _ = zarr.create(
            shape=shape,
            chunks=raw_array.chunks,
            store=zarr.storage.FSStore(Path(zarr_path.parent, 
                                            zarr_path.name + "_flatfield_corr", 
                                            str(level))),
            dtype=raw_array.dtype,
            overwrite=True,
            dimension_separator="/",
        )

    if FOV_list is None and z_levels is not None:
        FOV_list = []
        ROI_table = ad.read_zarr(zarr_path / "tables" / "FOV_ROI_table").to_df()
        max_y_FOV = ROI_table["y_micrometer"].max()
        max_x_FOV = ROI_table["x_micrometer"].max()
        for i in range(len(ROI_table)):
            if (ROI_table.iloc[i]["y_micrometer"] == 0 and 
                ROI_table.iloc[i]["x_micrometer"] == 0):
                FOV_list.append(i)
            if (ROI_table.iloc[i]["y_micrometer"] == 0 and 
                ROI_table.iloc[i]["x_micrometer"] == max_x_FOV):
                FOV_list.append(i)
            if (ROI_table.iloc[i]["y_micrometer"] == max_y_FOV and 
                ROI_table.iloc[i]["x_micrometer"] == 0):
                FOV_list.append(i)
            if (ROI_table.iloc[i]["y_micrometer"] == max_y_FOV and 
                ROI_table.iloc[i]["x_micrometer"] == max_x_FOV):
                FOV_list.append(i)

    if save_models:
        if FOV_list is not None or z_levels is not None:
            folder_path = Path(zarr_path.parent, "IllumModels")
        else:
            folder_path = Path(zarr_path.parent, "BaSiCPyModels")
        for channel in channels_dict.keys():
            channel_model_folder = Path(folder_path, channel)
            if not channel_model_folder.exists():
                channel_model_folder.mkdir(parents=True, exist_ok=True)
    else:
        folder_path = None
    
    parallelization_list = []
    for channel, channel_dict in channels_dict.items():
        parallelization_list.append(
            dict(
                zarr_url=channel_dict["zarr_url"],
                init_args=dict(
                    channel_name=channel,
                    channel_index=channel_dict["index"],
                    saving_path=str(channel_model_folder),
                    FOV_list=FOV_list,
                    z_levels=z_levels
                ),
            )
        )
    return dict(parallelization_list=parallelization_list)

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_correct_flatfield,
        logger_name=logger.name,
    )