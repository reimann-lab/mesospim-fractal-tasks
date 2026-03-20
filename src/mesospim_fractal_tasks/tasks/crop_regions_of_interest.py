import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

import numcodecs
numcodecs.blosc.set_nthreads(1)

import logging
import shutil
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
                                                     _write_label_metadata, 
                                                     _estimate_pyramid_depth)
from mesospim_fractal_tasks.tasks.crop_regions_of_interest_dask import (
    adapt_coordinates,
    adapt_roi_table,
    check_binary_compatibility,
    check_tile_size,
    save_roi_parallel,
)
from mesospim_fractal_tasks.utils.parallelisation import _set_dask_cluster
from mesospim_fractal_tasks.utils.models import ProxyArray

logger = logging.getLogger(__name__)


@validate_call
def crop_regions_of_interest(
    *,
    zarr_url: str,
    init_args: Dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Crop regions of interest from a multi-channel OME-Zarr image. It loads the full
    resolution image, crops the ROI, and saves it in the same well.
    
    Parameters:
        zarr_url: Path to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_crop_regions_of_interest`.

    Returns:
        dict: A dictionary containing the updated image list.
    """
    zarr_path = Path(zarr_url)
    logger.info(f"Starting task: `Crop Region of Interest` "
                f"for {zarr_path.parent.name}/{zarr_path.name}")

    # Load full resolution image and NGFF metadata
    logger.info(f"Loading full resolution image.")
    if init_args["is_proxy"]:
        full_res_arr = ProxyArray.open(zarr_path, requested_level=0)
    else:
        full_res_arr = da.from_zarr(zarr_path/"0")
    full_shape = full_res_arr.shape
    image_meta = load_NgffImageMeta(zarr_path)
    scale = image_meta.get_pixel_sizes_zyx(level=0)
    shape = (
        full_res_arr.shape[0], full_res_arr.shape[1], full_res_arr.shape[2], full_res_arr.shape[3])
    pyramid_dict = _estimate_pyramid_depth(
        shape, scale=tuple(scale), roi_coords=init_args["roi_coords"], num_levels=init_args["num_levels"])

    # Read ROI coordinates
    coords = init_args["roi_coords"]
    if init_args["crop_or_roi"] == "crop":
        coords = check_tile_size(zarr_path, coords)
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
                                                power=len(pyramid_dict))
    x_start, x_end = check_binary_compatibility(max(coords['x_start'], 0),
                                                coords['x_end'] + scale[2],
                                                full_shape[3], 
                                                scale[2],
                                                power=len(pyramid_dict))
    
    assert z_start < z_end
    assert y_start < y_end
    assert x_start < x_end

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
            chunks=init_args["chunksize"],
            dtype=full_res_arr.dtype,
            store=zarr.storage.FSStore(f"{roi_path}/0"),
            overwrite=True,
            dimension_separator="/",
            fill_value=0,
            write_empty_chunks=False,
    )

    # Set dask cluster
    client = None
    cluster = None
    try:
        cluster = _set_dask_cluster()
        client = Client(cluster)
        client.forward_logging(logger_name = "mesospim_fractal_tasks", level=logging.INFO)

        z_chunk = init_args["chunksize"][1]
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
        roi_datasets = []
        for key in pyramid_dict.keys():
            roi_datasets.append(
                {
                    "coordinateTransformations": [
                        {
                            "scale": [
                                1.0,
                                pyramid_dict[key]["scale"][0],
                                pyramid_dict[key]["scale"][1],
                                pyramid_dict[key]["scale"][2]
                            ],
                            "type": "scale"
                        }
                    ],
                    "path": key
                })
        multiscales[0]["datasets"] = roi_datasets
        roi_group.attrs["multiscales"] = multiscales

        # Add roi masks in the source image
        if init_args["crop_or_roi"] == "roi":
            lowest_level = image_meta.num_levels - 1
            if init_args["is_proxy"]:
                lowest_res_arr = ProxyArray.open(zarr_path, requested_level=lowest_level)
                low_scale = lowest_res_arr.pyramid_dict[str(lowest_level)]["scale"]
                chunksize = lowest_res_arr.chunksize
            else:
                lowest_res_arr = zarr.open_array(f"{zarr_path}/{lowest_level}", mode="r")
                low_scale = image_meta.get_pixel_sizes_zyx(level=lowest_level)
                chunksize = lowest_res_arr.chunks[1:]
            
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
                        "rgba": [255/255, 255/255, 255/255, 125/255],
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
                chunksize=chunksize,
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
                zarr.open_group(roi_path, mode="a"),
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
            zarr.open_group(roi_path, mode="a"),
            "well_ROI_table",
            well_table,
            overwrite=True,
            table_attrs={"type": "roi_table"},
        )

        # Write pyramid of resolution
        logger.info(f"Building pyramid of resolution for {roi_path.name}")
        build_pyramid(
            zarr_url=roi_path,
            pyramid_dict=pyramid_dict
        )

    finally:
        if client is not None:
            try:
                client.close(timeout="300s")  # give it more time
            except TimeoutError:
                pass

        if cluster is not None:
            try:
                cluster.close(timeout=300)
            except TimeoutError:
                pass

    # Re-compute optimal contrast limits for ROI
    if init_args["crop_or_roi"] == "crop":
        contrast_limits = _determine_optimal_contrast(roi_path, len(pyramid_dict), segment_sample=True)
        if init_args["erase_source_image"]:
            logger.info("Erasing source image...")
            shutil.rmtree(zarr_path)
    else:
        contrast_limits = _determine_optimal_contrast(roi_path, len(pyramid_dict))
    
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