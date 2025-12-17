from typing import Dict, Any
from pydantic import validate_call
import numpy as np
import dask.array as da
import zarr
from scipy.optimize import least_squares
import anndata as ad
from pathlib import Path
import logging
from filelock import FileLock
from scipy.ndimage import gaussian_filter1d

from mesospim_fractal_tasks.utils.zarr_utils import (_determine_optimal_contrast,
                                                     _update_omero_channels)
from mesospim_fractal_tasks import __version__, __commit__

from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url

logger = logging.getLogger(__name__)

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

def compute_global_normalisation(
    zarr_path: Path,
    channel_name: str,
    channel_index: int,
    z_profile: np.ndarray
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
                mean1 = da.mean(overlap_tile1[mask1])
                mean2 = da.mean(overlap_tile2[mask2])
                overlap_thresh = 0.1 * overlap[0] * overlap[1] * image_arr.shape[1]
                if da.sum(mask1) < overlap_thresh or da.sum(mask2) < overlap_thresh:
                    logger.warning(f"Manually setting gain to 1 for ROI pair {i_ROI1} "
                                   f" and {i_ROI2} because "
                                   "of meaningless overlap (almost no sample present).")
                    gain = 1
                else:
                    gain = (mean2 / mean1).compute()
                logger.info(f"ROI_{i_ROI1}, ROI_{i_ROI2}: {gain}")
                gain_graph.append((f"ROI_{i_ROI1}", f"ROI_{i_ROI2}", gain))

    logger.info(f"Start computing gain map fo channel {channel_name}...")
    ROI_indices = {section: i for i, section in enumerate(ROIs)}
    if len(gain_graph) > 0:
        gains = least_squares(gain_residuals, x0=np.ones(len(ROIs)), 
                            args=(gain_graph, ROI_indices), 
                            bounds=(np.ones(num_ROIs), np.full(num_ROIs, np.inf))).x
    else:
        gains = np.ones(len(ROIs))
    max_idx = np.argmax(gains)
    gain_map = {tile: (gains[i] / gains[max_idx]) for i, tile in enumerate(ROIs)}
    logger.info(f"Gain map computed: {gain_map}")

    return gain_map

@validate_call
def correct_illumination(
    *,
    zarr_url: str,
    init_args: Dict[str, Any],
    z_correction: bool = False,
) -> dict[str, Any]:
    
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
    else:
        z_profile = np.ones((1, image_array.shape[1], 1, 1))
    gain_factors = compute_global_normalisation(zarr_path, 
                                                channel_name=channel_name, 
                                                channel_index=channel_index,
                                                z_profile=z_profile)

    logger.info(f"Starting illumination correction...")
    for i_ROI, indices in enumerate(indices):
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
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
            url=zarr.open(Path(new_zarr_path, "0")),
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
            url=zarr.open(new_zarr_path / str(level+1)), 
            region=region, 
            overwrite=True)

    sync_path = new_zarr_path / ".zarr_process.lock"
    synchronizer = zarr.sync.ProcessSynchronizer(str(sync_path))
    store = zarr.storage.DirectoryStore(str(new_zarr_path))
        
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
        new_group = zarr.open_group(store=store, synchronizer=synchronizer, mode="a")
        new_group.attrs.put(source_attrs)

    # Determine optimal contrast limits
    contrast_limits = _determine_optimal_contrast(
        new_zarr_path, 
        num_levels, 
        channel_index=channel_index, 
        segment_sample=True, 
        synchronizer=synchronizer
    )
    _update_omero_channels(
        new_zarr_path, 
        {"window": contrast_limits}, 
        synchronizer=synchronizer
    )

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