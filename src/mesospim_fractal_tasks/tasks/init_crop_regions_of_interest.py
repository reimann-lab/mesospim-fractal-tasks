"""
Initialises the parallelisation list for `Crop Regions of Interest Task`.
"""
from pathlib import Path
import logging
from typing import Any, Optional
from pydantic import validate_call
import pandas as pd

logger = logging.getLogger(__name__)

@validate_call
def init_crop_regions_of_interest(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    roi_table_name: Optional[str] = None,
    num_levels: Optional[int] = None,
    coarsening_xy: int = 2,

) -> dict[str, list[dict[str, Any]]]:
    """
    Initialise the parallelisation list for `Crop Regions of Interest Task`.
    
    This task prepares a parallelization list of all rois that are to be
    cropped from a given Zarr image.

    Parameters:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server). 
            Expected to be a list of length 1 for this task.
        zarr_dir: path to the directory where the OME-Zarr image is located and
            where the ROI coordinates table can be found.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_table_name: Name/identifier of the ROI coordinates table to identify 
            it in the `zarr_dir`. If not provided, the default `roi_coords` is used.
        num_levels: Number of pyramid levels to generate for the ROI image (including 
            the full resolution image).
        coarsening_xy: Coarsening factor in XY for the ROI image. Optional, if different
            from the coarsening factor used for the full resolution image.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """

    if len(zarr_urls) != 1:
        raise ValueError(
            "Error! Expected only one zarr_url for this task."
        )
    zarr_url = zarr_urls[0]

    logger.info(
        f"Running {__name__} for {zarr_url}"
    )

    logger.info("Loading ROI coordinates table.")
    if roi_table_name is None:
        roi_table_name = "roi_coords"
    tables = []
    for path in Path(zarr_dir).rglob(f"*{roi_table_name}*.csv"):
        tables.append(path)
    if len(tables) != 1:
        logger.error(f"Unique ROI coordinates table not found in {zarr_dir}.")
        raise FileNotFoundError
    roi_table = pd.read_csv(tables[0], index_col=0)

    nb_rois = len(roi_table)
    logger.info(f"Preparing parallelisation list for {nb_rois} ROIs.")
    parallelisation_list = []
    for roi_id, roi_row in roi_table.iterrows():
        if nb_rois == 1:
            logger.info(f"Only one ROI to crop, removing numbering...")
            roi_id = roi_id[:-7]
        parallelisation_list.append(
            dict(
                zarr_url=zarr_url,
                init_args=dict(
                    roi_id=roi_id,
                    roi_coords=roi_row.to_dict(),
                    num_levels=num_levels,
                    coarsening_xy=coarsening_xy,
                ),
            )
        )

    return dict(parallelization_list=parallelisation_list)

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_crop_regions_of_interest,
        logger_name=logger.name,
    )
