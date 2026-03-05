import os
import anndata as ad
import dask.array as da
import numpy as np
from dask.distributed import LocalCluster
import logging
from pathlib import Path
import zarr
from typing import Any, Callable, Optional

from mesospim_fractal_tasks.utils.zarr_utils import convert_ROI_table_to_indices

from mesospim_fractal_tasks.utils.models import ProxyArray

logger = logging.getLogger(__name__)

def _set_dask_cluster(
    n_workers: Optional[int] = None,
) -> LocalCluster:
    """
    Set up a dask cluster for distributed computing.
    
    Returns:
        Dask cluster.
    """

    
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", None)
    if cpus is None:
        cpus = os.cpu_count()
        if cpus is None:
            raise ValueError("Number of CPUs not found.")

    if n_workers is None:
        n_workers = int(cpus)
    else:
        min(n_workers, int(cpus))

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=max(int(int(cpus) / n_workers), 1),
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
    is_proxy: bool,
    correct_func: Callable[..., da.Array],
    correct_func_kwargs: dict[str, Any],
) -> None:

    if is_proxy:
        image_array = ProxyArray.open(zarr_path, requested_level=0)
    else:
        image_array = da.from_zarr(zarr_path / "0")
    new_image_array = zarr.open_array(new_zarr_path / "0")

    # Get FOVs coordinates
    FOV_ROI_table = ad.read_zarr(Path(zarr_path, "tables", "FOV_ROI_table"))
    indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        scale_zyx=full_res_pxl_sizes_zyx,
        cols_xyz_pos= [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer"]
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
    logger.info(f"Illumination correction for {channel_name} completed.")
