import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numcodecs
numcodecs.blosc.set_nthreads(1)

import logging
from pathlib import Path
import zarr
from dask.distributed import Client
import dask.array as da
import pandas as pd
import numpy as np
import anndata as ad
from typing import Dict, Any
from pydantic import validate_call

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import get_single_image_ROI, prepare_FOV_ROI_table
from fractal_tasks_core.tables import write_table

from mesospim_fractal_tasks.utils.zarr_utils import (_determine_optimal_contrast,
                                                     _update_omero_channels,
                                                     build_pyramid,
                                                     _store_label_to_zarr,
                                                     _write_label_metadata)
from mesospim_fractal_tasks.utils.parallelisation import _set_dask_cluster

logger = logging.getLogger(__name__)

def adapt_coordinates(
    start: float, 
    end: float, 
    dim: str, 
    scale: float, 
    table: pd.DataFrame
) -> None:
    """
    Adapt the entries of the FOV ROI table for the given coordinates per dimension.

    Parameters:
        start (float): New start pixel coordinate.
        end (float): New end pixel coordinate.
        dim (str): Dimension to adapt.
        scale (float): Pixel scale in um.
        table (pd.DataFrame): Original FOV ROI Table to be adapted.
    """
    new_start_um = start * scale
    new_end_um = end * scale
    table[f"pixel_size_{dim}"] = scale
    table[f"{dim}_micrometer"] = table[f"{dim}_micrometer"] - new_start_um
    table[f"{dim}_micrometer"] = table[f"{dim}_micrometer"].clip(
        lower=0, upper=(new_end_um-new_start_um))
    if dim != "z":
        table[f"{dim}_micrometer_original"] = (table[f"{dim}_micrometer_original"] - 
                                               new_start_um)
        table[f"{dim}_micrometer_original"] = table[f"{dim}_micrometer_original"].clip(
            lower=0, upper=(new_end_um-new_start_um))
    dim_max = table[f"{dim}_micrometer"].max()
    if dim_max == (new_end_um-new_start_um):
        table.drop(table[table[f"{dim}_micrometer"] == dim_max].index, inplace=True)
    
    dim_micrometers = sorted(table[f"{dim}_micrometer"].unique())
    dim_micrometers.append(new_end_um-new_start_um)
    dim_micrometers = np.array(dim_micrometers)
    for r, row in table.iterrows():
        i = np.argwhere(dim_micrometers == row[f"{dim}_micrometer"])[0][0]
        table.loc[r,f"len_{dim}_micrometer"] = (dim_micrometers[i+1] - 
                                                row[f"{dim}_micrometer"])
        table.loc[r,f"{dim}_pixel"] = int(round(table.loc[r, f"len_{dim}_micrometer"] 
                                                / scale))
    
def adapt_roi_table(
    zarr_path: Path, 
    roi_path: Path, 
    coords: dict[str, int], 
    scale: tuple[float, float, float]
) -> pd.DataFrame:
    """
    Adapt the FOV ROI table to the new crop coordinates.

    Parameters:
        zarr_path (Path): Path to the OME-Zarr image.
        roi_path (Path): Path to the ROI table.
        coords (dict): Coordinates to adapt.
        scale (tuple[float, float, float]): Pixel scale in um.

    Returns:
        pandas.DataFrame: Adaptated FOV ROI table.
    """
    logger.info(f"Adapting FOV ROI table to new crop coordinates for {roi_path}")

    # Load original table
    source_table = ad.read_zarr(zarr_path/ "tables" / "FOV_ROI_table").to_df()

    # Update z, y, x coordinates
    x_start, x_end = coords["x_start"], coords["x_end"]
    y_start, y_end = coords["y_start"], coords["y_end"]
    z_start, z_end = coords["z_start"], coords["z_end"]
    adapt_coordinates(x_start, x_end, "x", scale[2], source_table)
    adapt_coordinates(y_start, y_end, "y", scale[1], source_table)
    adapt_coordinates(z_start, z_end, "z", scale[0], source_table)

    # Update index
    source_table.index = [i for i in range(len(source_table))]
    return source_table

def check_binary_compatibility(
    slice_start: float, 
    slice_end: float,
    max_end: int,
    scale: float,
    power: int = 6
) -> tuple[int, int]:
    """
    Check if the crop slices can be divided by a power of 2.
    
    Parameters:
        slice_start (float): Beginning of slice in microns.
        slice_end (float): End of slice in microns.
        max_end (int): Maximum possible end of slice in px.
        scale (float): Scale of the image.
        power (int): Power of 2 to check (should match pyramid resolution).
    
    Returns:
        tuple[int, int]: New slice start and end in pixels.
    """
    modulo = abs(round(slice_end / scale) - round(slice_start / scale)) % 2**power
    if modulo > 0:
        add_start = (2**power - modulo) // 2
        temp_start = round(slice_start / scale)
        if add_start > temp_start:
            add_start = temp_start
        new_slice_start = temp_start - add_start
        new_slice_end = round(slice_end / scale) + (2**power - modulo - add_start)
    else:
        new_slice_start = round(slice_start / scale)
        new_slice_end = round(slice_end / scale)
    if new_slice_end > max_end:
        diff = new_slice_end - slice_end
        new_slice_end = max_end
        add_start = 2**power - (diff % 2**power)
        new_slice_start = new_slice_start - add_start
        if new_slice_start < 0:
            new_slice_start = 0
            logger.warning(f"Crop is outside of array boundary. Keeping original size.")
        
    return int(new_slice_start), int(new_slice_end)

@validate_call
def crop_regions_of_interest(
    *,
    zarr_url: str,
    init_args: Dict[str, Any],
) -> None:
    """
    Crop regions of interest from a multi-channel OME-Zarr image. It loads the full
    resolution image, crops the ROI, and saves it in the same well.
    
    Parameters:
        zarr_url: Path to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_crop_regions_of_interest`.
    """
    zarr_path = Path(zarr_url)
    logger.info(f"Starting task: `Crop Region of Interest` "
                f"for {zarr_path.parent.name}/{zarr_path.name}")
    
    cluster = _set_dask_cluster()

    # Load full resolution image and NGFF metadata
    logger.info(f"Loading full resolution image.")
    full_res_arr = da.from_zarr(zarr_path/"0")
    full_shape = full_res_arr.shape
    image_meta = load_NgffImageMeta(zarr_path)
    scale = image_meta.get_pixel_sizes_zyx(level=0)
    if init_args["num_levels"] is None:
        init_args["num_levels"] = image_meta.num_levels

    # Read ROI coordinates
    coords = init_args["roi_coords"]
    roi_id = init_args["roi_id"]
    z_start, z_end = check_binary_compatibility(max(coords['z_start'], 0),
                                                coords['z_end'] + scale[0],
                                                full_shape[1],
                                                scale[0], 
                                                power=0)
    y_start, y_end = check_binary_compatibility(max(coords['y_start'], 0),
                                                coords['y_end'] + scale[1], 
                                                full_shape[2],
                                                scale[1],
                                                power=init_args["num_levels"])
    x_start, x_end = check_binary_compatibility(max(coords['x_start'], 0),
                                                coords['x_end'] + scale[2],
                                                full_shape[3], 
                                                scale[2],
                                                power=init_args["num_levels"])

    # Crop region
    logger.info(f"Cropping ROI region from full resolution image at "
                f"{z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}.")
    crop = full_res_arr[:,
                        z_start:z_end,
                        y_start:y_end,
                        x_start:x_end]
        
    logger.info(f"Saving cropped region as {roi_id}.")
    root_path = zarr_path.parent
    roi_path = Path(root_path, roi_id)
    roi_arr = zarr.create(
            shape=crop.shape,
            chunks=full_res_arr.chunksize,
            dtype=full_res_arr.dtype,
            store=zarr.storage.FSStore(f"{roi_path}/0"),
            overwrite=True,
            dimension_separator="/"
    )
    with Client(cluster) as client:
        z_chunk = full_res_arr.chunksize[1]
        for z in range(0, z_end-z_start, z_chunk):
            region = (slice(None),
                    slice(z, z+z_chunk),
                    slice(None),
                    slice(None))
            crop[region].to_zarr(roi_arr, compute=True, region=region)
        logger.info(f"ROI {roi_id} saved!")

    # Copy NGFF metadata from the raw image to the roi image
    logger.info(f"Copying NGFF metadata from {zarr_path.name} to {roi_path.name}")
    source_group = zarr.open_group(zarr_url, mode="a")
    source_attrs = source_group.attrs.asdict()
    roi_group = zarr.open(roi_path, mode="a")
    roi_group.attrs.put(source_attrs)

    logger.info(f"Saving cropping metadata to {roi_path.name}")
    roi_group = zarr.open(roi_path, mode="a")
    roi_group.attrs["crop_info"] = {
        "roi_id": roi_id,
        "crop_coordinates": {
            "z_start_um": coords['z_start'],
            "z_end_um": coords['z_end'],
            "y_start_um": coords['y_start'],
            "y_end_um": coords['y_end'],
            "x_start_um": coords['x_start'],
            "x_end_um": coords['x_end']
        },
        "origin": f"{zarr_path.name}"

    }
    multiscales = roi_group.attrs["multiscales"]
    multiscales[0]["name"] = roi_id
    roi_datasets = [multiscales[0]["datasets"][i] \
                    for i in range(init_args["num_levels"])]
    multiscales[0]["datasets"] = roi_datasets
    roi_group.attrs["multiscales"] = multiscales

    # Add roi masks in the source image
    if init_args["crop_or_roi"] == "roi":
        lowest_level = image_meta.num_levels - 1
        lowest_res_arr = zarr.open_array(f"{zarr_path}/{lowest_level}", mode="r")
        low_scale = image_meta.get_pixel_sizes_zyx(level=lowest_level)
        low_shape = lowest_res_arr.shape
        roi_mask = np.zeros(low_shape[1:], dtype=np.uint8)
        z_start, z_end = check_binary_compatibility(max(coords['z_start'], 0),
                                                    coords['z_end'] + low_scale[0],
                                                    low_shape[1], # type: ignore
                                                    low_scale[0], 
                                                    power=0)
        y_start, y_end = check_binary_compatibility(max(coords['y_start'], 0),
                                                    coords['y_end'] + low_scale[1], 
                                                    low_shape[2], # type: ignore
                                                    low_scale[1],
                                                    power=0)
        x_start, x_end = check_binary_compatibility(max(coords['x_start'], 0),
                                                    coords['x_end'] + low_scale[2],
                                                    low_shape[3], # type: ignore
                                                    low_scale[2],
                                                    power=0)
        roi_mask[z_start:z_end, y_start:y_end, x_start:x_end] = 1

        mask_dict = dict(
            colors=[
                {
                    "label-value": 1,
                    "rgba": [255, 255, 255, 125],
                    "label": "roi_mask"
                }
            ]
        )
        _write_label_metadata(
            source_group,
            roi_id,
            mask_dict,
            num_levels=1,
            analysis_resolution_level=lowest_level,
            overwrite=True
        )
        _store_label_to_zarr(
            zarr_path / "labels" / roi_id,
            label_mask=roi_mask,
            chunksize=lowest_res_arr.chunks[1:],
            overwrite=True
        )

    # Update FOV ROI table in case of crop
    if init_args["crop_or_roi"] == "crop":
        coords = dict(x_start=x_start, x_end=x_end, 
                      y_start=y_start, y_end=y_end, 
                      z_start=z_start, z_end=z_end)
        fov_roi_table = adapt_roi_table(zarr_path, roi_path, coords, scale)
        
        # Write table
        logger.info(f"Writing FOV ROI table for {roi_path.name}")
        fov_roi_table = prepare_FOV_ROI_table(fov_roi_table)
        write_table(
            zarr.open(roi_path, mode="a"),
            "FOV_ROI_table",
            fov_roi_table,
            overwrite=True,
            table_attrs={"type": "roi_table"},
        )

    # Write well ROI table
    logger.info(f"Writing well ROI table for {roi_path.name}")
    well_table = get_single_image_ROI(roi_arr.shape[1:], 
                                      scale) 
    write_table(
        zarr.open(roi_path, mode="a"),
        "well_ROI_table",
        well_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )

    # Write pyramid of resolution
    with Client(cluster) as client:
        logger.info(f"Building pyramid of resolution for {roi_path.name}")
        build_pyramid(
            zarrurl=roi_path,
            overwrite=True,
            num_levels=init_args["num_levels"],
            coarsening_xy=init_args["coarsening_xy"],
            chunksize=roi_arr.chunks,
        )

    # Re-compute optimal contrast limits for ROI
    if init_args["crop_or_roi"] == "crop":
        contrast_limits = _determine_optimal_contrast(roi_path, init_args["num_levels"], segment_sample=True)
    else:
        contrast_limits = _determine_optimal_contrast(roi_path, init_args["num_levels"])
    
    _update_omero_channels(roi_path, {"window": contrast_limits})
    
    # Update image list
    if init_args["crop_or_roi"] == "crop":
        type = "is_crop"
    else:
        type = "is_roi"
    image_list_updates = dict(
        image_list_updates=[dict(origin=str(zarr_path),
                                 zarr_url=str(roi_path), 
                                 attributes=dict(image=roi_id),
                                 types={type: True})]
    )
    return image_list_updates

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=crop_regions_of_interest,
        logger_name=logger.name,
    )