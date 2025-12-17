"""
Initialises the parallelisation list for to crop regions of interest.
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
    crop_or_roi: str = "roi",
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
        crop_or_roi: Whether the coordinates are for a crop or a ROI. If `crop`, 
            the coordinates correspond to a crop of the full resolution image to remove
            empty space for example. A cropped_image will be output with adapted
            FOV_ROI_table. If `roi`, one or more small ROIs are to be extracted.
            Default: `roi`.
        roi_table_name: Name/identifier of the ROI coordinates table to identify 
            it in the `zarr_dir`. If not provided, the default `roi_coords` is used.
        num_levels: Number of pyramid levels to generate for the ROI image (including 
            the full resolution image). If not provided, the same multi-resolution
            pyramid size as the original image will be used. Default: None.
        coarsening_xy: Coarsening factor in XY for the ROI image. Optional, if different
            from the coarsening factor used for the full resolution image. Default: 2.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """

    if len(zarr_urls) != 1:
        raise ValueError(
            "Error! Expected only one zarr_url for this task."
        )
    zarr_path = Path(zarr_urls[0])
    logger.info(f"Start task: `Crop Region of Interest (Initialisation)"
                f" for {zarr_path.parent.name}/{zarr_path.name}")

    logger.info("Loading ROI coordinates table.")
    if roi_table_name is None:
        roi_table_name = "roi_coords"
    tables = []
    for path in Path(zarr_path.parent).glob(f"*{roi_table_name}*.csv"):
        tables.append(path)
    if len(tables) != 1:
        logger.error(f"Unique ROI coordinates table not found in {zarr_path.parent}.")
        raise FileNotFoundError
    roi_table = pd.read_csv(tables[0], index_col=0)

    nb_rois = len(roi_table)
    logger.info(f"Preparing parallelisation list for {nb_rois} ROIs.")
    parallelisation_list = []
    for roi_id, roi_row in roi_table.iterrows():
        if crop_or_roi == "crop":
            if len(roi_table) != 1:
                logger.error("Number of ROIs in table and crop_or_roi parameters are "
                             "inconsistent. Table cannot have more than one ROI if "
                             "crop_or_roi is set to 'crop'.")
                raise ValueError
            logger.info(f"Task set to produce a crop from original image to reduce size"
                        " for example.")
            roi_id = zarr_path.name + "_cropped"
        parallelisation_list.append(
            dict(
                zarr_url=str(zarr_path),
                init_args=dict(
                    roi_id=roi_id,
                    roi_coords=roi_row.to_dict(),
                    num_levels=num_levels,
                    crop_or_roi=crop_or_roi,
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
