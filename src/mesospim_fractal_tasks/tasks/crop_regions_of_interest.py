import logging
from pathlib import Path
import zarr
import dask.array as da
from typing import Dict, Any
from pydantic import validate_call

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import get_single_image_ROI
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.pyramids import build_pyramid

from skinnervation3d_fractal_tasks.utils.zarr_utils import (_determine_optimal_contrast,
                                                            _update_omero_channels)

logger = logging.getLogger(__name__)

def check_binary_compatibility(
    slice_start: float, 
    slice_end: float,
    scale: float,
    power: int = 4
) -> tuple[int, int]:
    """
    Check if the crop slices can be divided by a power of 2.
    
    Parameters:
        slice_start (float): Beginning of slice in microns.
        slice_end (float): End of slice in microns.
        scale (float): Scale of the image.
        power (int): Power of 2 to check.
    
    Returns:
        tuple[int, int]: New slice start and end in pixels.
    """

    modulo = abs(round(slice_end / scale) - round(slice_start / scale)) % 2**power
    if modulo > 0:
        add_start = (2**power - modulo) // 2
        new_slice_start = round(slice_start / scale) - add_start
        new_slice_end = round(slice_end / scale) + (2**power - modulo - add_start)
    else:
        new_slice_start = round(slice_start / scale)
        new_slice_end = round(slice_end / scale)
    
    return new_slice_start, new_slice_end

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
    
    logger.info(f"Start task: {__name__} for {zarr_url}")

    # Load full resolution image and NGFF metadata
    logger.info(f"Loading full resolution image.")
    full_res_arr = da.from_zarr(f"{zarr_url}/0")
    image_meta = load_NgffImageMeta(zarr_url)
    scale = image_meta.get_pixel_sizes_zyx(level=0)
    if init_args["num_levels"] is None:
        init_args["num_levels"] = image_meta.num_levels

    # Read ROI coordinates
    coords = init_args["roi_coords"]
    roi_id = init_args["roi_id"]
    z_start, z_end = check_binary_compatibility(coords['z_start_um'], 
                                                coords['z_end_um'], 
                                                scale[0], 
                                                power = 0)
    y_start, y_end = check_binary_compatibility(coords['y_start_um'], 
                                                coords['y_end_um'], 
                                                scale[1],
                                                power = init_args["num_levels"])
    x_start, x_end = check_binary_compatibility(coords['x_start_um'], 
                                                coords['x_end_um'], 
                                                scale[2],
                                                power = init_args["num_levels"])

    # Crop region
    logger.info(f"Cropping ROI region from full resolution image at "
                f"{z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}.")
    crop = full_res_arr[:,
                        z_start:z_end,
                        y_start:y_end,
                        x_start:x_end]
    
    
    logger.info(f"Saving cropped region as {roi_id}.")
    root_path = Path(zarr_url).parent
    roi_path = Path(root_path, roi_id)
    roi_arr = zarr.create(
            shape=crop.shape,
            chunks=full_res_arr.chunksize,
            dtype=full_res_arr.dtype,
            store=zarr.storage.FSStore(f"{roi_path}/0"),
            overwrite=False,
            dimension_separator="/"
    )
    z_chunk = full_res_arr.chunksize[1]
    for z in range(0, z_end-z_start, z_chunk):
        region = (slice(None),
                  slice(z, z+z_chunk),
                  slice(None),
                  slice(None))
        crop[region].to_zarr(roi_arr, compute=True, region=region)
    logger.info(f"ROI {roi_id} saved!")

    # Copy NGFF metadata from the raw image to the roi image
    logger.info(f"Copying NGFF metadata from {zarr_url} to {roi_path}")
    source_group = zarr.open_group(zarr_url, mode="r")
    source_attrs = source_group.attrs.asdict()
    roi_group = zarr.open(roi_path, mode="a")
    roi_group.attrs.put(source_attrs)

    logger.info(f"Saving cropping metadata to {roi_path}")
    roi_group = zarr.open(roi_path, mode="a")
    roi_group.attrs["crop_info"] = {
        "roi_id": roi_id,
        "crop_coordinates": {
            "z_start_um": coords['z_start_um'],
            "z_end_um": coords['z_end_um'],
            "y_start_um": coords['y_start_um'],
            "y_end_um": coords['y_end_um'],
            "x_start_um": coords['x_start_um'],
            "x_end_um": coords['x_end_um']
        },
        "origin": f"{Path(zarr_url).name}"

    }
    multiscales = roi_group.attrs["multiscales"]
    multiscales[0]["name"] = roi_id
    roi_datasets = [multiscales[0]["datasets"][i] \
                    for i in range(init_args["num_levels"])]
    multiscales[0]["datasets"] = roi_datasets
    roi_group.attrs["multiscales"] = multiscales

    # Write pyramid of resolution
    logger.info(f"Building pyramid of resolution for {roi_path}")
    build_pyramid(
        zarrurl=roi_path,
        overwrite=True,
        num_levels=init_args["num_levels"],
        coarsening_xy=init_args["coarsening_xy"],
        chunksize=roi_arr.chunks,
    )

    # Re-compute optimal contrast limits for ROI
    contrast_limits = _determine_optimal_contrast(roi_path, init_args["num_levels"], 
                                                  percentile=None)
    
    _update_omero_channels(roi_path, {"window": contrast_limits})

    # Write well ROI table
    logger.info(f"Writing well ROI table for {roi_path}")
    well_table = get_single_image_ROI(roi_arr.shape[1:], 
                                      scale) 
    write_table(
        zarr.open(roi_path, mode="a"),
        "well_ROI_table",
        well_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )

    image_list_updates = dict(
        image_list_updates=[dict(zarr_url=roi_path, 
                                 attributes=dict(image=roi_id))]
    )
    return image_list_updates

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=crop_regions_of_interest,
        logger_name=logger.name,
    )