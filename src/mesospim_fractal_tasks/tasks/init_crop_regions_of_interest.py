"""
Initialises the parallelisation list for to crop regions of interest.
"""
from pathlib import Path
import logging
from typing import Any, Optional
from pydantic import validate_call
import pandas as pd
import zarr

logger = logging.getLogger(__name__)

from mesospim_fractal_tasks.utils.models import DimTuple

@validate_call
def init_crop_regions_of_interest(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    crop_or_roi: str = "roi",
    roi_table_name: Optional[str] = None,
    num_levels: Optional[int] = None,
    chunksize: Optional[DimTuple] = None,
    overwrite: bool = False,
    erase_source_image: bool = False,

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
            empty space for example. A cropped image will be output with adapted
            FOV_ROI_table. If `roi`, one or more small ROIs are to be extracted.
            Default: `roi`.
        roi_table_name: Name/identifier of the ROI coordinates table to identify 
            it in the target image of the `zarr_dir` (e.g. if cropping zarr_dir/raw_image,
            the table must be in the raw_image folder). If not provided, 
            the default `roi_coords` is used.
        num_levels: Number of pyramid levels to generate for the ROI image (including 
            the full resolution image). If not provided, the same multi-resolution
            pyramid size as the original image will be used. Default: None.
        chunksize: Chunk size to use for the new ROI image(s). If None, the chunksize
            of the original image will be used. Default: None.
        overwrite: Whether to overwrite existing ROI images if they already exist 
            in the OME-Zarr folder. Default: False.
        erase_source_image: If `True`, the source image will be erased after cropping.
            It only works if `crop_or_roi` is set to `crop`. Default: False.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """

    if len(zarr_urls) != 1:
        raise ValueError(
            "Error! Expected only one zarr_url for this task."
        )
    zarr_path = Path(zarr_urls[0])
    logger.info(f"Starting task: `Crop Region of Interest (Initialisation)"
                f" for {zarr_path.parent.name}/{zarr_path.name}")
    
    is_proxy = False
    fractal_tasks = zarr.open_group(zarr_path, mode="r").attrs.get("fractal_tasks", {})
    if "prepare_mesospim_omezarr" in fractal_tasks and zarr_path.name == "fake_raw_image":
        is_proxy = True
    
    full_res_arr = zarr.open_array(zarr_path / "0", mode="r")
    chunk_size = list(full_res_arr.chunks)
    if chunksize is not None:
        for i, dim in enumerate(["z", "y", "x"]):
            if chunksize[dim] is not None:
                chunk_size[(i+1)] = int(chunksize[dim])
    chunk_size = tuple(chunk_size)
    logger.info(f"Chunksize set to: {chunk_size}")

    # Find coordinates table in OME-Zarr
    logger.info("Loading ROI coordinates table.")
    if roi_table_name is None:
        roi_table_path = Path("roi_coords.csv")
    else:
        roi_table_path = Path(roi_table_name).with_suffix(".csv")
    table_path = Path(zarr_path.parent, roi_table_path)
    if table_path.exists():
        roi_table = pd.read_csv(table_path, index_col=0)
    else:
        table_path = Path(zarr_path, roi_table_path)
        if table_path.exists():
            roi_table = pd.read_csv(table_path, index_col=0)
        else:
            possible_match = list(Path(zarr_path.parent).rglob(roi_table_path.name))
            if len(possible_match) == 1:
                roi_table = pd.read_csv(possible_match[0], index_col=0)
            else:
                logger.error(f"ROI coordinates table could not be found in {zarr_path.parent.name} folder "
                            f"with name {roi_table_path.name}.")
                raise FileNotFoundError

    nb_rois = len(roi_table)
    logger.info(f"Preparing parallelisation list for {nb_rois} ROIs.")
    parallelisation_list = []
    current_images = list(p.name for p in Path(zarr_path.parent).glob("*") if p.is_dir())
    for roi_id, roi_row in roi_table.iterrows():
        roi_id = str(roi_id).lower()
        if crop_or_roi == "crop":
            if len(roi_table) != 1:
                logger.error("Number of ROIs in table and crop_or_roi parameters are "
                             "inconsistent. Table cannot have more than one ROI if "
                             "crop_or_roi is set to 'crop'.")
                raise ValueError
            logger.info(f"Task set to produce a crop from original image (e.g. to reduce size).")
            
            if zarr_path.name == "fake_raw_image":
                roi_id = "raw_image_cropped"
            else:
                roi_id = zarr_path.name + "_cropped"
                
            if not overwrite and roi_id in current_images:
                logger.error(f"Crop {roi_id} already exists in {zarr_path.parent.name} and "
                             "overwrite set to `False`. Try setting overwrite to `True`.")
                raise FileExistsError
        else:
            while roi_id in current_images and not overwrite:
                logger.warning(f"ROI {roi_id} already exists in {zarr_path.parent.name} "
                    f"and overwrite set to `False`.") 
                prefix = roi_id.split("roi_")[0]
                suffix = int(roi_id.split("roi_")[-1]) + 1
                if suffix == 9:
                    roi_id = f"{prefix}roi_{suffix+1:03d}"
                else:    
                    roi_id = f"{prefix}roi_{suffix+1:02d}"
            current_images.append(roi_id)
        roi_path = Path(zarr_path.parent, str(roi_id))
        parallelisation_list.append(
            dict(
                zarr_url=str(zarr_path),
                init_args=dict(
                    roi_id=roi_id,
                    roi_coords=roi_row.to_dict(),
                    roi_path=roi_path,
                    num_levels=num_levels,
                    chunksize=chunk_size,
                    crop_or_roi=crop_or_roi,
                    erase_source_image=erase_source_image,
                    is_proxy=is_proxy,
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
