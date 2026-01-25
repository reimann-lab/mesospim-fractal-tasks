from typing import Dict, Any
from pydantic import validate_call
import numpy as np
import dask
from dask.delayed import delayed
import dask.array as da
from dask.distributed import Client
import zarr
from scipy.optimize import least_squares
import anndata as ad
from pathlib import Path
import logging
from scipy.ndimage import gaussian_filter1d

from mesospim_fractal_tasks import __version__, __commit__
from mesospim_fractal_tasks.utils.parallelisation import _set_dask_cluster

from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
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
    ngff_image_meta = load_NgffImageMeta(zarr_path)
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

@validate_call
def correct_illumination(
    *,
    zarr_url: str,
    init_args: Dict[str, Any],
    z_correction: bool = False,
) -> dict[str, Any]:
    
    cluster = _set_dask_cluster()

    zarr_path = Path(zarr_url)
    logger.info(f"Start task: `Illumination Correction` "
                f"for {zarr_path.parent.name}/{zarr_path.name}")
    
    # Define new zarr path
    new_zarr_path = Path(zarr_path.parent, zarr_path.name + "_illum_corr")
    if not new_zarr_path.exists():
        logger.error(f"Error! {new_zarr_path.name} does not exist.")
        raise FileNotFoundError
    image_array = da.from_zarr(Path(zarr_url, "0"))
    
    # Get channel name from init task
    channel_name = init_args["channel_name"]
    channel_index = init_args["channel_index"]

    # Get relevant metadata
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    coarsening_xy = ngff_image_meta.coarsening_xy
    if coarsening_xy is None:
        coarsening_xy = 2
    num_levels = ngff_image_meta.num_levels
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    z_chunk = image_array.chunks[1][0]

    # Get FOVs coordinates
    FOV_ROI_table = ad.read_zarr(Path(zarr_url, "tables", "FOV_ROI_table"))
    indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=0,
        coarsening_xy=2,
        cols_xyz_pos= [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer"],
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )

    # Compute correction factors
    if z_correction:
        z_profile = compute_z_correction_profile(zarr_path, 
                                                 channel_name=channel_name, 
                                                 channel_index=channel_index)
        # Make z_profile dask-aligned with input chunks to avoid bloated graphs
        z_profile[channel_name] = da.from_array(z_profile[channel_name], 
                                                chunks=(1, z_chunk, 1, 1))
    else:
        z_profile = da.ones((1, image_array.shape[1], 1, 1),
                            chunks=(1, z_chunk, 1, 1))
    gain_factors = compute_global_normalisation(zarr_path, 
                                                channel_name=channel_name, 
                                                channel_index=channel_index,
                                                z_profile=z_profile)

    logger.info(f"Starting illumination correction...")
    with _set_dask_cluster() as cluster:
        with Client(cluster) as client:
            for i_ROI, idxs_ROI in enumerate(indices):
                s_z, e_z, s_y, e_y, s_x, e_x = idxs_ROI[:]
                region = (
                    slice(channel_index, channel_index + 1),
                    slice(s_z, e_z),
                    slice(s_y, e_y),
                    slice(s_x, e_x),
                )
                gain = gain_factors[f"ROI_{i_ROI}"]
                corrected_FOV = da.clip(image_array[region] * gain * z_profile, 
                                                0, 65535).astype(np.uint16)
                
                # Write to disk
                logger.info(f"Saving corrected FOV to {new_zarr_path.name}.")
                corrected_FOV.to_zarr(
                    url=zarr.open(str(new_zarr_path / "0")),
                    region=region,
                    compute=True,
                )

            logger.info(f"Building the pyramid of resolution levels for {new_zarr_path.name}.")
            for level in range(0, num_levels-1):
                up_channel_arr = da.from_zarr(new_zarr_path / str(level))[channel_index:channel_index+1]
                down_channel_arr = da.coarsen(
                    reduction=np.mean,
                    x=up_channel_arr,
                    axes={0:1, 1:1, 2: coarsening_xy, 3: coarsening_xy},
                    trim_excess=True)
                region = (slice(channel_index, channel_index+1),
                        slice(None),
                        slice(None),
                        slice(None))
                down_channel_arr = down_channel_arr.rechunk(image_array.chunksize)
                down_channel_arr.to_zarr(
                    url=zarr.open(str(new_zarr_path / str(level+1))), 
                    region=region, 
                    overwrite=True)
        
    # Copy NGFF metadata from the old zarr_url to the new zarr if needed
    if channel_index == 0:

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
        source_attrs["fractal_tasks"] = fractal_tasks
        new_group = zarr.open_group(str(new_zarr_path), mode="a")
        new_group.attrs.put(source_attrs)

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