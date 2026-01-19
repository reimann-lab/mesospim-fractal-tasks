import os
import anndata as ad
import dask.array as da
import numpy as np
from dask.distributed import LocalCluster
import logging
from pathlib import Path
import zarr
from typing import Any, Callable

from fractal_tasks_core.roi import convert_ROI_table_to_indices

logger = logging.getLogger(__name__)

def _set_dask_cluster(
) -> LocalCluster:
    """
    Set up a dask cluster for distributed computing.
    
    Returns:
        Dask cluster.
    """

    workers = os.environ.get("SLURM_CPUS_PER_TASK", None)
    if workers is None:
        workers = os.cpu_count()
        if workers is None:
            workers = 1
    workers = int(workers)

    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
        silence_logs=logging.ERROR,
    )
    return cluster

def correct_per_channel(
    *,
    zarr_path: Path,
    new_zarr_path: Path,
    channel_name: str,
    channel_index: int,
    full_res_pxl_sizes_zyx: tuple[float, float, float, float],
    correct_func: Callable[..., da.Array],
    correct_func_kwargs: dict[str, Any],
) -> None:

    image_array = da.from_zarr(zarr_path / "0")
    new_image_array = zarr.open_array(new_zarr_path / "0")

    # Get FOVs coordinates
    FOV_ROI_table = ad.read_zarr(Path(zarr_path, "tables", "FOV_ROI_table"))
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

    logger.info(f"Starting illumination correction for {channel_name}...")
    for i_ROI, idxs_ROI in enumerate(indices):
        s_z, e_z, s_y, e_y, s_x, e_x = idxs_ROI[:]
        region = (
            slice(channel_index, channel_index + 1),
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )

        corrected_FOV = correct_func(image_array[region], i_ROI,
                                     **correct_func_kwargs)
        
        # Write to disk
        logger.info(f"{i_ROI+1}/{len(indices)} corrected and saved to {new_zarr_path.name}.")
        corrected_FOV.to_zarr(
            url=new_image_array,
            region=region,
            compute=True,
        )

def build_pyramid_per_channel(
    new_zarr_path: Path,
    channel_index: int,
    num_levels: int,
    coarsening_xy: int,
    chunksize: tuple[int, int, int, int],
) -> None:
    
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
        down_channel_arr = down_channel_arr.rechunk(chunksize)
        down_channel_arr.to_zarr(
            url=zarr.open(str(new_zarr_path / str(level+1))), 
            region=region, 
            overwrite=True)
