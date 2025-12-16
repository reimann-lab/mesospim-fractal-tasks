"""This is the Python module for sitching FOVs from an OME-Zarr image."""

import logging
import shutil
from pathlib import Path
from typing import Optional
import os

import anndata as ad
import numpy as np
import zarr
import dask.array as da
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import get_single_image_ROI
from fractal_tasks_core.tables import write_table
from multiview_stitcher import fusion, msi_utils, param_utils, registration
from multiview_stitcher import spatial_image_utils as si_utils
from pydantic import validate_call

from mesospim_fractal_tasks.utils.stitching import (
    StitchingChannelInputModel,
    get_sim_from_multiscales,
    get_tiles_from_sim,
    patched_get_sim_from_array,
)
from mesospim_fractal_tasks.utils.models import DimTuple
si_utils.get_sim_from_array = patched_get_sim_from_array
from mesospim_fractal_tasks import __version__, __commit__

logger = logging.getLogger(__name__)

@validate_call
def stitch_with_multiview_stitcher(
    *,
    zarr_url: str,
    channel: StitchingChannelInputModel,
    registration_resolution_level: int = 0,
    registration_on_z_proj: bool = True,
    registration_function: str = "phase_correlation",
    overlap_tolerance: DimTuple = DimTuple(z=0, y=0, x=0),
    transform_type: str = "translation",
    pre_registration_pruning_method: str = "keep_axis_aligned",
    n_batches: int = 1,
    fusion_chunksize: Optional[DimTuple] = None,
) -> None:
    """Stitches FOVs from an OME-Zarr image.

    Performs registration and fusion of FOVs indicated
    in the FOV_ROI_table of the OME-Zarr image. Writes the
    fused image back to a "fused" group in the same Zarr array.

   Parameters:
        zarr_url: Absolute path to the OME-Zarr image.
        channel: Channel for registration; requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`), but not
            both.
        registration_resolution_level: Resolution level to use for registration.
        registration_on_z_proj: Whether to perform registration on a maximum
            projection along z in case of 3D data. Recommended for large z step or when
            there are empty tiles.
        registration_function: Type of transformation to use for registration.
            Available functions:
            - 'phase_correlation': (default).
            - 'antspy': see ANTsPy documentation for more information.
        overlap_tolerance: float or dict, optional
            Extend overlap regions considered for pairwise registration.
            - if 0, the overlap region is the intersection of the tiles.
            - if > 0, the overlap region is the intersection of the tiles
                extended by this value in the given spatial dimensions.
            Default: 0 for all dimensions.
        transform_type: Type of transformation to use for registration. 
            Available types:
            - 'translation': translation (default)
            - 'rigid': rigid body transformation
            - 'similarity': similarity transformation
            - 'affine': affine transformation
        pre_registration_pruning_method: Method to use for selecting a subset
            of all overlapping tiles for pairwise registration. By default,
            only lower, upper, right and left neighbors are considered. Set
            this parameter to no_pruning if pairs of tiles which deviate
            from this pattern need to be registered. Available methods:
            - None: No pruning, useful when no regular arrangement is present.
            - 'alternating_pattern': Prune to edges between squares of differering
                colors in checkerboard pattern. Useful for regular 2D tile arrangements 
                (of both 2D or 3D data).
            - 'shortest_paths_overlap_weighted': Prune to shortest paths in overlap 
                graph (weighted by overlap). Useful to minimize the number of pairwise 
                registrations.
            - 'otsu_threshold_on_overlap': Prune to edges with overlap above Otsu 
                threshold. This is useful for regular 2D or 3D grid arrangements, 
                as diagonal edges will be pruned.
            - 'keep_axis_aligned': Keep only edges that align with tile axes. This is 
                useful for regular grid arrangements and to explicitely prune diagonals, 
                e.g. when other methods fail.
        fusion_chunksize: Chunksize for the dimension (Z, Y, X) to use when performing 
            the fusion. It impacts the memory usage and the time to fuse the tiles. 
            If None, the chunksize of the raw image is used. In case of 
            registration_on_z_proj=True, chunksize for the Z dimension is set to the 
            number of z planes.
    """

    zarr_path = Path(zarr_url)
    logger.info(f"Start task: `Stitching with Multiview Stitcher` "
                f"for {zarr_path.parent}/{zarr_path.name}")

    # Parse and log several NGFF-image metadata attributes
    ngff_image_meta = load_NgffImageMeta(zarr_path)
    fov_roi_table = ad.read_zarr(Path(zarr_path, "tables/FOV_ROI_table")).to_df()
    input_transform_key = "fractal_input"

    # Load FOVs for registration as spatial image
    xim_well_reg = get_sim_from_multiscales(
        Path(zarr_url), resolution=registration_resolution_level
    )
    original_chunksize = xim_well_reg.data.chunksize

    # Determine whether to perform registration on maximum projection in Z
    z_dim = xim_well_reg.shape[1]
    if registration_on_z_proj:
        xim_well_reg = xim_well_reg.max("z")
        overlap_tolerance = DimTuple(y=0, x=0)

    # Define the registration grid
    msims_reg = get_tiles_from_sim(
        xim_well_reg, fov_roi_table, transform_key=input_transform_key
    )
    reg_spatial_dims = si_utils.get_spatial_dims_from_sim(
        xim_well_reg.squeeze(drop=True)
    )

    logger.info("Started registration...")
    logger.info(f"Registration resolution level: {registration_resolution_level}")
    logger.info(f"Registration spatial dimsensions: {reg_spatial_dims}")

    # Find channel index
    omero_channel = channel.get_omero_channel(zarr_path)
    if omero_channel:
        reg_channel_index = omero_channel.index
    else:
        logger.error(
            f"Error. {channel} is not available in that OME-Zarr image."
        )
        raise ValueError
    
    if transform_type not in ["translation", "rigid", "similarity", "affine"]:
        raise ValueError(f"Error. Unknown transformation type: {transform_type}."
                    " Available types are 'translation', 'rigid', 'similarity', "
                    "'affine'.")

    if registration_function not in ["phase_correlation", "antspy"]:
        raise ValueError(f"Error. Unknown registration function: {registration_function}."
                    " Available functions are 'phase_correlation', 'antspy'.")
    elif registration_function == "antspy":
        registration_function = registration.registration_ANTsPy
    else:
        registration_function = registration.phase_correlation_registration

    fusion_transform_key = "translation_registered"
    overlap_tolerance = overlap_tolerance.get_dict()
    n_cpus = os.cpu_count()
    if n_cpus == None:
        n_cpus = 1
    registration.register(
        msims_reg,
        transform_key=input_transform_key,
        new_transform_key=fusion_transform_key,
        reg_channel_index=reg_channel_index,
        pairwise_reg_func=registration_function,
        overlap_tolerance=overlap_tolerance,
        registration_binning={dim: 1 for dim in reg_spatial_dims},
        groupwise_resolution_kwargs={"transform": transform_type},
        n_parallel_pairwise_regs=n_cpus,
        pre_registration_pruning_method=pre_registration_pruning_method
    )

    logger.info("Finished registration.")

    # Preparing for fusion
    output_zarr_path = Path(zarr_path.parent, zarr_path.name + "_fused")
    logger.info(f"Saving fused image to {output_zarr_path.name}") 
    output_zarr_temp = Path(output_zarr_path, "temp")
    output_zarr_final = Path(output_zarr_path, "0")

    if fusion_chunksize is None:
        fusion_chunksize = original_chunksize[-3:]
    else:
        fusion_chunksize = (fusion_chunksize.z, fusion_chunksize.y, fusion_chunksize.x)
    logger.info(f"Fusion Chunk size set to: {fusion_chunksize}.")
    
    if registration_resolution_level == 0 and not registration_on_z_proj:
        xim_well = xim_well_reg
        msims_fusion = msims_reg
    else:
        if registration_on_z_proj:
            fusion_chunksize = (z_dim, fusion_chunksize[1], 
                                fusion_chunksize[2])
        
        # Load the full-resolution image for fusion
        xim_well = get_sim_from_multiscales(zarr_path, 
                                            resolution=0, 
                                            chunks=(1,) + fusion_chunksize)
        msims_fusion = get_tiles_from_sim(
            xim_well, fov_roi_table, transform_key=input_transform_key
        )

    # assign the registration parameters to the tiles to be fused
    for itile in range(len(msims_fusion)):
        affine = msi_utils.get_transform_from_msim(
            msims_reg[itile], fusion_transform_key
        )

        # if the registration was performed on a maximum projection in Z, we need to
        # broadcast the obtained affine parameters to 3D
        if registration_on_z_proj:
            affine_3d = param_utils.identity_transform(
                ndim=3, t_coords=affine.coords["t"] if "t" in affine.dims else None
            )
            affine_3d.loc[{pdim: affine.coords[pdim] for pdim in affine.dims}] = affine
            affine = affine_3d

        msi_utils.set_affine_transform(
            msims_fusion[itile], affine, fusion_transform_key
        )

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims_fusion]
    sdims = si_utils.get_spatial_dims_from_sim(xim_well)
    ndim = len(sdims)

    if fusion_chunksize is None:
        fusion_chunksize_dict = {
            dim: xim_well.data.chunksize[(-ndim + idim)] \
            for idim, dim in enumerate(sdims)
        }
    else:
        fusion_chunksize_dict = {
            dim: cs for dim, cs in zip(sdims, fusion_chunksize)
        }
    
    logger.info(f"Starting fusing tiles with chunksize {fusion_chunksize_dict}...")
    fused = fusion.fuse(
        sims,
        transform_key=fusion_transform_key,
        output_chunksize=fusion_chunksize_dict,
        output_spacing=si_utils.get_spacing_from_sim(sims[0]),
        output_zarr_url=output_zarr_temp,
        batch_options={"zarr_array_creation_kwargs": {"dimension_separator": "/", 
                                                      "compressor": None}, 
                       "n_batch": n_batches},
    )

    logger.info("Resizing (dropping t dimension) and rechunking to"
                f" {original_chunksize}...")
    
    # Open the zarr group (read/write)
    temp_array = da.from_zarr(output_zarr_temp)[0]
    new_shape = temp_array.shape
    
    # Refactor the tmep output zarr array (drop t and rechunk)
    final_fused_arr = zarr.create(
        shape=new_shape,
        chunks=original_chunksize,
        dtype=np.uint16,
        store=zarr.storage.FSStore(output_zarr_final),
        overwrite=True,
        dimension_separator="/",
    )
    for i in range(0, new_shape[-2], original_chunksize[-2] * n_batches):
        for j in range(0, new_shape[-1], original_chunksize[-1] * n_batches):
            region = (slice(None), 
                      slice(None), 
                      slice(i, i+original_chunksize[-2] * n_batches), 
                      slice(j, j+original_chunksize[-1] * n_batches))
            rechunked_array = temp_array[region].rechunk(original_chunksize)
            da.to_zarr(rechunked_array, final_fused_arr, region=region)
    logger.info("Finished fusing tiles.")

    logger.info("Started building resolution pyramid")
    build_pyramid(
        zarrurl=output_zarr_path,
        overwrite=True,
        num_levels=ngff_image_meta.num_levels,
        chunksize=original_chunksize,
        coarsening_xy=ngff_image_meta.coarsening_xy,
        open_array_kwargs={"write_empty_chunks": False, "fill_value": 0},
    )
    logger.info("Finished building resolution pyramid")

    # Copy NGFF metadata from the old zarr_url to the new zarr
    logger.info(f"Copying NGFF metadata from {zarr_path.name}"
                f" to {output_zarr_path.name}.")
    source_group = zarr.open_group(zarr_path, mode="r")
    source_attrs = source_group.attrs.asdict()
    source_attrs["multiscales"][0]["name"] = (source_attrs["multiscales"][0]["name"] + 
                                              "_fused")
    fractal_tasks = source_attrs.get("fractal_tasks", {})
    task_dict = dict(
        version=__version__.split("dev")[0][:-1],
        commit=__commit__,
        input_parameters=dict()
    )
    fractal_tasks["stitching_with_multiview_stitcher"] = task_dict
    source_attrs["fractal_tasks"] = fractal_tasks
    new_group = zarr.open(output_zarr_path, mode="a")
    new_group.attrs.put(source_attrs)
    logger.info("Finished copying NGFF metadata.")

    # Add ROI table to the image
    ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pixels_ZYX = (
        ngff_image_meta.multiscales[0]
        .datasets[0]
        .coordinateTransformations[0]
        .scale[-3:]
    )
    image_ROI_table = get_single_image_ROI(new_shape, pixels_ZYX=pixels_ZYX)
    write_table(
        new_group,
        "well_ROI_table",
        image_ROI_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )

    # Clean up temporary Zarr file
    shutil.rmtree(output_zarr_temp)

    # Prepare the image list update
    image_list_updates = dict(
        image_list_updates=[dict(zarr_url=str(output_zarr_path), 
                                 origin=zarr_url,
                                 attributes=dict(image=output_zarr_path.name))
        ]
    )

    return image_list_updates

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stitch_with_multiview_stitcher)