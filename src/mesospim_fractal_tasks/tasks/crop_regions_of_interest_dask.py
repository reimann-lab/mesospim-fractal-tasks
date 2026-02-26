import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numcodecs
numcodecs.blosc.set_nthreads(1)

import shutil
import logging
from pathlib import Path
import zarr
from dask.distributed import Client
import dask.array as da
import pandas as pd
import numpy as np
import anndata as ad
from typing import Dict, Any, Optional
from pydantic import validate_call

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import get_single_image_ROI, prepare_FOV_ROI_table
from fractal_tasks_core.tables import write_table

from mesospim_fractal_tasks.utils.zarr_utils import (_determine_optimal_contrast,
                                                     _update_omero_channels,
                                                     build_pyramid,
                                                     _write_label_metadata,
                                                     _store_label_to_zarr,
                                                     _estimate_pyramid_depth)
from mesospim_fractal_tasks.utils.models import DimTuple
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

def save_roi_parallel(
    zarr_path: Path,
    roi_id: str,
    coords: dict[str, int],
    scale: tuple[float, float, float],
    pyramid_dict: dict[str, dict],
    chunksize: tuple[int, int, int, int],
    crop_or_roi: str,
) -> None:
    """
    Save a ROI from a multi-channel OME-Zarr image to a new OME-Zarr image.

    Parameters:
        zarr_path (Path): Path to the OME-Zarr image to be processed.
        roi_id (str): ID of the ROI to save.
        coords (dict[str, int]): Coordinates of the ROI to save.
        scale (tuple[float, float, float]): Pixel scale in um.
        pyramid_dict (dict[str, dict]): Dictionary containing the scale and coarsening factors for each level.
        chunksize (tuple[int, int, int]): Chunk size to use for the new ROI image(s).
        crop_or_roi (str): Whether the ROI is a crop or a ROI.
    """
    full_res_arr = da.from_zarr(zarr_path/"0")
    full_shape = full_res_arr.shape
    z_start, z_end = check_binary_compatibility(max(coords['z_start'], 0),
                                                    coords['z_end'] + scale[0],
                                                    full_shape[1], # type: ignore
                                                    scale[0], 
                                                    power=0)
    y_start, y_end = check_binary_compatibility(max(coords['y_start'], 0),
                                                    coords['y_end'] + scale[1], 
                                                    full_shape[2], # type: ignore
                                                    scale[1],
                                                    power=len(pyramid_dict))
    x_start, x_end = check_binary_compatibility(max(coords['x_start'], 0),
                                                    coords['x_end'] + scale[2],
                                                    full_shape[3], # type: ignore
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
            shape=crop.shape, # type: ignore
            chunks=chunksize,
            dtype=full_res_arr.dtype,
            store=zarr.storage.FSStore(f"{roi_path}/0"), # type: ignore
            overwrite=True,
            dimension_separator="/",
            fill_value=0,
            write_empty_chunks=False,
    )
    z_chunk = chunksize[1]
    for z in range(0, z_end-z_start, z_chunk):
        logger.info(f"Progress: {z / (z_end-z_start) * 100:.2f}%")
        region = (slice(None),
                slice(z, z+z_chunk),
                slice(None),
                slice(None))
        crop[region].to_zarr(roi_arr, compute=True, region=region) # type: ignore
    logger.info(f"ROI {roi_id} saved!")

    # Copy NGFF metadata from the raw image to the roi image
    logger.info(f"Copying NGFF metadata from {zarr_path.name} to {roi_path.name}")
    source_group = zarr.open_group(zarr_path, mode="r")
    source_attrs = source_group.attrs.asdict()
    logger.info(f"Saving cropping metadata to {roi_path.name}")
    roi_group = zarr.open_group(roi_path, mode="a")
    roi_group.attrs.put(source_attrs)
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

    # Update FOV ROI table in case of crop
    if crop_or_roi == "crop":
        coords = dict(x_start=x_start, x_end=x_end, 
                    y_start=y_start, y_end=y_end, 
                    z_start=z_start, z_end=z_end)
        fov_roi_table = adapt_roi_table(zarr_path, roi_path, coords, scale)
        
        # Write table
        logger.info(f"Writing FOV ROI table for {roi_path.name}")
        fov_roi_table = prepare_FOV_ROI_table(fov_roi_table)
        write_table(
            roi_group,
            "FOV_ROI_table",
            fov_roi_table,
            overwrite=True,
            table_attrs={"type": "roi_table"},
        )

    # Write well ROI table
    logger.info(f"Writing well ROI table for {roi_path.name}")
    well_table = get_single_image_ROI(roi_arr.shape[1:], list(scale)) # type: ignore
    write_table(
        roi_group,
        "well_ROI_table",
        well_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )


@validate_call
def crop_regions_of_interest(
    *,
    zarr_url: str,
    crop_or_roi: str = "roi",
    roi_table_name: str = "roi_coords",
    num_levels: Optional[int] = None,
    chunksize: Optional[DimTuple] = None,
    overwrite: bool = False,
    erase_source_image: bool = False,
) -> Dict[str, Any]:
    """
    Crop regions of interest from a multi-channel OME-Zarr image. It loads the full
    resolution image, crops the ROI, and saves it in the same well.
    
    Parameters:
        zarr_url: Path to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        crop_or_roi: Whether the coordinates are for a crop or a ROI. If `crop`, 
            the coordinates correspond to a crop of the full resolution image to remove
            empty space for example. A cropped_image will be output with adapted
            FOV_ROI_table. If `roi`, one or more small ROIs are to be extracted.
            Default: `roi`.
        roi_table_name: Name/identifier of the ROI coordinates table to identify 
            it in the OME-Zarr folder of the image to crop (e.g. if cropping zarr_dir/raw_image,
            the table must be in the `zarr_dir` folder). If not provided, 
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
     """
    zarr_path = Path(zarr_url)
    logger.info(f"Start task: `Crop Region of Interest` "
                f"for {zarr_path.parent.name}/{zarr_path.name}")

    # Load full resolution image and NGFF metadata
    logger.info(f"Loading full resolution image...")
    full_res_arr = zarr.open_array(zarr_path/"0", mode="r")
    image_meta = load_NgffImageMeta(str(zarr_path))
    scale = image_meta.get_pixel_sizes_zyx(level=0)
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

    # Prepare parallelisation list
    nb_rois = len(roi_table)
    logger.info(f"Preparing parallelisation list for {nb_rois} ROIs.")
    parallelisation_list = []
    image_list_updates = []
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
            roi_id = zarr_path.name + "_cropped"
            if not overwrite and roi_id in current_images:
                logger.error(f"Crop {roi_id} already exists in {zarr_path.parent.name} and "
                             "overwrite set to `False`. Try setting overwrite to `True`.")
                raise FileExistsError
            roi_type = "is_crop"
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
            roi_type = "is_roi"
        roi_path = Path(zarr_path.parent, str(roi_id))
        logger.info(f"New ROI will be saved with name {roi_id}.")
        roi_coords = roi_row.to_dict()
        shape = (
            full_res_arr.shape[0], full_res_arr.shape[1], full_res_arr.shape[2], full_res_arr.shape[3])
        pyramid_dict = _estimate_pyramid_depth(
            shape, scale=tuple(scale), roi_coords=roi_coords, num_levels=num_levels)
        parallelisation_list.append(
            dict(
                roi_id=roi_id,
                roi_coords=roi_coords,
                roi_path=roi_path,
                roi_pyramid=pyramid_dict
            )
        )
        image_list_updates.append(dict(zarr_url=str(roi_path), 
                                       origin=str(zarr_path), 
                                       attributes=dict(image=roi_id),
                                       types={roi_type: True}))
    if crop_or_roi == "crop":    
        with _set_dask_cluster(n_workers=4) as cluster:
            with Client(cluster) as client:
                client.forward_logging(logger_name = "mesospim_fractal_tasks", level=logging.INFO)
                crop_id = parallelisation_list[0]["roi_id"]
                crop_coords = parallelisation_list[0]["roi_coords"]
                crop_path = parallelisation_list[0]["roi_path"]
                pyramid_dict = parallelisation_list[0]["roi_pyramid"]
                save_roi_parallel(zarr_path,
                                roi_id=crop_id, 
                                coords=crop_coords, 
                                scale=tuple(scale),
                                pyramid_dict=pyramid_dict,
                                chunksize=chunk_size,
                                crop_or_roi=crop_or_roi)
                
                # Write pyramid of resolution
                logger.info(f"Building pyramid of resolution for {crop_path.name}")
                build_pyramid(
                    zarr_url=crop_path,
                    pyramid_dict=pyramid_dict
                )
                
                # Re-compute optimal contrast limits for ROI
                contrast_limits = _determine_optimal_contrast(crop_path, len(pyramid_dict), segment_sample=True)
                _update_omero_channels(crop_path, {"window": contrast_limits})

        if erase_source_image:
            logger.info("Erasing source image...")
            shutil.rmtree(zarr_path)

    else:
        with _set_dask_cluster(n_workers=len(parallelisation_list)) as cluster:
            with Client(cluster) as client:
                client.forward_logging(logger_name = "mesospim_fractal_tasks", level=logging.INFO)
                futures = []
                for _, roi_params in enumerate(parallelisation_list):
                    fut = client.submit(save_roi_parallel, 
                                zarr_path=zarr_path,
                                roi_id=roi_params["roi_id"],
                                coords=roi_params["roi_coords"],
                                scale=scale,
                                pyramid_dict=roi_params["roi_pyramid"],
                                chunksize=chunk_size,
                                crop_or_roi=crop_or_roi,
                                pure=False,
                                retries=1)
                    futures.append(fut)
                client.gather(futures)
                for _, roi_params in enumerate(parallelisation_list):
                    fut = client.submit(build_pyramid, 
                                zarr_url=roi_params["roi_path"],
                                pyramid_dict=roi_params["roi_pyramid"],
                                pure=False,
                                retries=1)
                    futures.append(fut)
                client.gather(futures)

            # Re-compute optimal contrast limits for ROI
            for _, roi_params in enumerate(parallelisation_list):
                contrast_limits = _determine_optimal_contrast(
                    roi_params["roi_path"], 
                    len(roi_params["roi_pyramid"]),
                    segment_sample=True)
                _update_omero_channels(roi_params["roi_path"], {"window": contrast_limits})
        
        # Add roi masks in the source image
        for _, roi_params in enumerate(parallelisation_list):
            lowest_level = image_meta.num_levels - 1
            lowest_res_arr = zarr.open_array(f"{zarr_path}/{lowest_level}", mode="r")
            low_scale = image_meta.get_pixel_sizes_zyx(level=lowest_level)
            low_shape = lowest_res_arr.shape
            roi_mask = np.zeros(low_shape[1:], dtype=np.uint8)
            coords = roi_params["roi_coords"]
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
                zarr.open_group(zarr_path, mode="a"),
                roi_params["roi_id"],
                mask_dict,
                num_levels=1,
                analysis_resolution_level=lowest_level,
                overwrite=True
            )
            _store_label_to_zarr(
                zarr_path / "labels" / roi_params["roi_id"],
                label_mask=roi_mask,
                chunksize=lowest_res_arr.chunks[1:],
                overwrite=True
            )

    image_list_updates = dict(
        image_list_updates=image_list_updates
    )
    return image_list_updates

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=crop_regions_of_interest,
        logger_name=logger.name,
    )