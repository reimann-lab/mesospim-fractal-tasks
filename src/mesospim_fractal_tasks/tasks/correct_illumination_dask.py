import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numcodecs
numcodecs.blosc.set_nthreads(1)

from typing import  Any
from pydantic import validate_call
import numpy as np
import dask.array as da
import dask
from dask.delayed import delayed
from dask.distributed import Client
import zarr
from scipy.optimize import least_squares
import anndata as ad
from pathlib import Path
import logging
from scipy.ndimage import gaussian_filter1d

from mesospim_fractal_tasks.utils.zarr_utils import (
    _determine_optimal_contrast,
    _update_omero_channels,
    create_zarr_pyramid)
from mesospim_fractal_tasks.utils.parallelisation import (
    _set_dask_cluster,
    build_pyramid_per_channel,
    correct_per_channel)
from mesospim_fractal_tasks import __version__, __commit__

from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.roi import convert_ROI_table_to_indices
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url

logger = logging.getLogger(__name__)

def print_dict(
    d: dict, 
    float_precision: int = 3
) -> str:
    lines = []
    for k in sorted(d, key=lambda x: int(x.split("_")[1])):
        v = float(d[k])
        lines.append(f"  {k:<7}: {v:.{float_precision}f}")
    return "\n".join(lines)

def compute_z_correction_profile(
    zarr_path: Path,
    channel_name: str,
    channel_index: int,
) -> np.ndarray:
    """
    Compute model of z-correction factor to correct uneven illumination in Z direction 
    (z-banding).
    
    Parameters:
        zarr_path (Path): Path to the OME-Zarr image to be processed.
        channel_name (str): Name of the channel to process.
        channel_index (int): Index of the channel to process.

    Returns:
        z_profile (np.ndarray): Array of shape (1, Z, 1, 1) with the per-z correction 
            factors.
    """
    logging.info(f"Computing z-correction profile for channel {channel_name} "
                 "for all FOVs.")
    ngff_image_meta = load_NgffImageMeta(str(zarr_path))
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    if coarsening_xy is None:
        coarsening_xy = 2
    channel_arr = da.from_zarr(Path(zarr_path, str(num_levels-1)))[channel_index]

    z_profile_percentile = []
    z_profile_percentile= np.median(np.median(
            channel_arr[:,:,:].compute(), axis=1), axis=1) + 1
    z_profile_percentile = np.concatenate([np.repeat(z_profile_percentile[0], 60),
                                    z_profile_percentile,
                                    np.repeat(z_profile_percentile[-1], 60)])
    z_percentile_smooth = gaussian_filter1d(z_profile_percentile, sigma=15)
    correction_factor = z_percentile_smooth[60:-60] / z_profile_percentile[60:-60]
    return correction_factor[None, :, None, None]

def gain_residuals(
    gains: np.ndarray,
    gain_graph: list[tuple[str, str, float]],
    ROIs_indices: dict[str, int]
) -> np.ndarray:
    """
    Compute the residuals of the gains for each pair of ROIs.
    
    Parameters:
        gains (np.ndarray): Array of gains for each ROI.
        gain_graph (list[tuple[str, str, float]]): List of tuples containing the 
            indices of the ROIs and their corresponding gains.
        ROIs_indices (dict[str, int]): Dictionary mapping ROI names to their indices.
    
    Returns:
        np.ndarray: Array of residuals.
    """

    residuals = []
    for ROI1, ROI2, gain in gain_graph:
        i = ROIs_indices[ROI1]
        j = ROIs_indices[ROI2]
        residuals.append(np.log(gains[i]) - np.log(gains[j]) - np.log(gain))
    return np.array(residuals)

@delayed
def _gain_from_stats(
    mean1: float, 
    mean2: float, 
    cnt1: int, 
    cnt2: int, 
    thresh: float
) -> float:
    """
    Decide gain from overlap statistics.
    """
    if (cnt1 < thresh) or (cnt2 < thresh):
        return 1.0
    
    # Protect against division by 0
    if mean1 <= 0:
        return 1.0
    return float(mean2 / mean1)

def compute_global_normalisation(
    zarr_path: Path,
    channel_name: str,
    channel_index: int,
    z_profile: da.Array
) -> dict[str, float]:
    """
    Compute the global normalisation factors for each ROI to correct for uneven 
    illumination across tiles.
    
    Parameters:
        zarr_path (Path): Path to the OME-Zarr image to be processed.
        channel_name (str): Name of the channel to process.
        channel_index (int): Index of the channel to process.
        z_profile (np.ndarray): Array of shape (1, Z, 1, 1) with the 
            per-z correction factors.
    
    Returns:
        dict[str, float]: Dictionary mapping ROI names to their corresponding gains.
    """

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(str(zarr_path))
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    if coarsening_xy is None:
        coarsening_xy = 2

    # Lazily load highest-res level from original zarr array
    logger.info(f"Loading lowest resolution image for channel {channel_name}.")
    image_arr = da.from_zarr(Path(zarr_path, str(num_levels-1)))

    # Get FOVs coordinates
    FOV_ROI_table = ad.read_zarr(Path(zarr_path, "tables", "FOV_ROI_table"))
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    original_indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=(num_levels-1),
        coarsening_xy=coarsening_xy,
        cols_xyz_pos= [
        "x_micrometer_original",
        "y_micrometer_original",
        "z_micrometer"],
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    zarr_indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=(num_levels-1),
        coarsening_xy=coarsening_xy,
        cols_xyz_pos= [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer"],
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )

    logger.info(f"Start computing difference of the mean between FOVs...")
    num_ROIs = len(original_indices)
    ROIs = [f"ROI_{i}" for i in range(num_ROIs)]
    i_c = channel_index
    gain_graph = []
    gain_tasks = []
    pair_meta = []

    for i_ROI1 in range(num_ROIs-1):
        for i_ROI2 in range(i_ROI1+1, num_ROIs):
            s_z1, e_z1, s_y1, e_y1, s_x1, e_x1 = original_indices[i_ROI1][:]
            s_z2, e_z2, s_y2, e_y2, s_x2, e_x2 = original_indices[i_ROI2][:]
            overlap = np.array([e_y1 - s_y2, e_x1 - s_x2])
            if ((overlap[0] > 0 and overlap[1] > 0) and 
                overlap[0] <= (e_y1-s_y1) and overlap[1] <= (e_x1-s_x1)):
                _, _, s_y1, e_y1, s_x1, e_x1 = zarr_indices[i_ROI1][:]
                _, _, s_y2, e_y2, s_x2, e_x2 = zarr_indices[i_ROI2][:]
                overlap_tile1 = image_arr[i_c, :, 
                                          e_y1 - overlap[0]:e_y1, 
                                          e_x1 - overlap[1]:e_x1] * z_profile
                overlap_tile2 = image_arr[i_c, :, 
                                          s_y2:s_y2 + overlap[0], 
                                          s_x2:s_x2 + overlap[1]] * z_profile
                mask1 = overlap_tile1 > 0
                mask2 = overlap_tile2 > 0
                sum1 = da.sum(da.where(mask1, overlap_tile1, 0))
                sum2 = da.sum(da.where(mask2, overlap_tile2, 0))
                cnt1 = da.sum(mask1)
                cnt2 = da.sum(mask2)

                mean1 = sum1 / da.maximum(cnt1, 1)
                mean2 = sum2 / da.maximum(cnt2, 1)
                overlap_thresh = 0.1 * overlap[0] * overlap[1] * image_arr.shape[1]
                
                g = _gain_from_stats(mean1, mean2, cnt1, cnt2, overlap_thresh)
                gain_tasks.append(g)
                pair_meta.append((i_ROI1, i_ROI2))

    if gain_tasks:
        gain_vals = dask.compute(*gain_tasks)
        gain_graph = [
            (f"ROI_{a}", f"ROI_{b}", float(g))
            for (a, b), g in zip(pair_meta, gain_vals)
        ]
    else:
        gain_graph = []

    logger.info(f"Solving gains for channel {channel_name} using least squares...")
    ROI_indices = {section: i for i, section in enumerate(ROIs)}
    if len(gain_graph) > 0:
        gains = least_squares(
            gain_residuals, 
            x0=np.ones(len(ROIs), dtype=np.float32), 
            args=(gain_graph, ROI_indices), 
            bounds=(np.ones(num_ROIs, dtype=np.float32), 
                    np.full(num_ROIs, np.inf, dtype=np.float32))).x
    else:
        gains = np.ones(len(ROIs), dtype=np.float32)
    max_idx = np.argmax(gains)
    gain_map = {tile: (gains[i] / gains[max_idx]) for i, tile in enumerate(ROIs)}
    logger.info(f"Gain map computed for {channel_name}:\n" + print_dict(gain_map))

    return gain_map

def correct_FOV(
    FOV_dask: da.Array,
    i_FOV: int,
    gain_factors: dict[str, float],
    z_profile: da.Array,
) -> da.Array:
    gain = gain_factors[f"ROI_{i_FOV}"]
    return da.clip(FOV_dask * gain * z_profile, 
                    0, 65535).astype(np.uint16)

@validate_call
def correct_illumination(
    *,
    zarr_url: str,
    z_correction: bool = False,
) -> dict[str, Any]:

    # Define new zarr path
    zarr_path = Path(zarr_url)
    new_zarr_path = Path(zarr_path.parent, zarr_path.name + "_illum_corr")
    logger.info(f"Start task: `Illumination Correction` "
                f"for {zarr_path.parent.name}/{zarr_path.name}")
    
    # Create zarr pyramid
    create_zarr_pyramid(zarr_path, new_zarr_name = new_zarr_path.name)
    
    # Map channel name to channel index
    channel_dict = {}
    channels = get_omero_channel_list(image_zarr_path=str(zarr_path))
    if len(channels) == 0:
        raise ValueError("No channels found in the OME Zarr metadata.")
    for c, channel in enumerate(channels):
        if channel.index is None:
            channel.index = c
        channel_dict[channel.label] = channel.index

    # Get relevant metadata
    image_array = da.from_zarr(zarr_path / "0")
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    coarsening_xy = ngff_image_meta.coarsening_xy
    if coarsening_xy is None:
        coarsening_xy = 2
    num_levels = ngff_image_meta.num_levels
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    z_chunk = image_array.chunks[1][0]

    gain_factors = {}
    z_profile = {}
    for channel_name, channel_index in channel_dict.items():
        
        # Compute correction factors
        if z_correction:
            z_profile[channel_name] = compute_z_correction_profile(zarr_path, 
                                                        channel_name=channel_name, 
                                                        channel_index=channel_index)
            
            # Make z_profile dask-aligned with input chunks to avoid bloated graphs
            z_profile[channel_name] = da.from_array(z_profile[channel_name], 
                                                    chunks=(1, z_chunk, 1, 1))
        else:
            z_profile[channel_name] = da.ones((1, image_array.shape[1], 1, 1), 
                                              chunks=(1, z_chunk, 1, 1))
        gain_factors[channel_name] = compute_global_normalisation(zarr_path, 
                                                    channel_name=channel_name, 
                                                    channel_index=channel_index,
                                                    z_profile=z_profile[channel_name])

    with _set_dask_cluster(n_workers=len(channel_dict.keys())) as cluster:
        with Client(cluster) as client:
            futures = []
            for channel_name, channel_idx in channel_dict.items():
                fut = client.submit(
                    correct_per_channel,
                    zarr_path=zarr_path,
                    new_zarr_path=new_zarr_path,
                    channel_name=channel_name,
                    channel_index=channel_idx,
                    full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
                    correct_func=correct_FOV,
                    correct_func_kwargs={"gain_factors": gain_factors[channel_name],
                                        "z_profile": z_profile[channel_name]},
                    pure=False,
                    retries=1
                )
                futures.append(fut)
            client.gather(futures)

            futures = []
            for channel_name, channel_idx in channel_dict.items():
                fut = client.submit(
                    build_pyramid_per_channel,
                    new_zarr_path=new_zarr_path,
                    channel_index=channel_idx,
                    num_levels=num_levels,
                    coarsening_xy=coarsening_xy,
                    chunksize=image_array.chunksize,
                    pure=False,
                    retries=1
                )
                futures.append(fut)
            client.gather(futures)

    # Copy ROI tables from the old zarr_url
    _copy_tables_from_zarr_url(str(zarr_path), str(new_zarr_path))

    # Copy NGFF metadata from the old zarr_url to the new zarr
    logger.info(f"Copying NGFF metadata from {zarr_path.name}"
                f" to {new_zarr_path.name}.")
    source_group = zarr.open_group(zarr_path, mode="r")
    source_attrs = source_group.attrs.asdict()
    image_name = source_attrs["multiscales"][0]["name"] + "_illum_corr"
    source_attrs["multiscales"][0]["name"] = image_name
    fractal_tasks = source_attrs.get("fractal_tasks", {})
    task_dict = dict(
        version=__version__.split("dev")[0][:-1],
        commit=__commit__,
        input_parameters=dict(
            z_correction=z_correction,
        )
    )
    fractal_tasks["correct_illumination"] = task_dict
    source_attrs["fractal_tasks"] = fractal_tasks # type: ignore
    new_group = zarr.open_group(str(new_zarr_path), mode="a")
    new_group.attrs.put(source_attrs)

    contrast_limits = _determine_optimal_contrast(new_zarr_path, 
                                                  num_levels, 
                                                  segment_sample=True)
    _update_omero_channels(new_zarr_path, {"window": contrast_limits})

    image_list_updates = dict(
        image_list_updates=[dict(zarr_url=str(new_zarr_path), 
                                 origin=str(zarr_path), 
                                 attributes=dict(image=new_zarr_path.name))]
        )
    return image_list_updates

if __name__ == "__main__":

    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=correct_illumination,
        logger_name=logger.name,
    )