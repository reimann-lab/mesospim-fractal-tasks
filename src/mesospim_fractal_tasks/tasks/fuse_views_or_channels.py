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
from pathlib import Path
from typing import Optional
import os
import numpy as np
import shutil
from dask.distributed import Client
import dask.array as da

import anndata as ad
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
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
    parallel_block_processing,
    fuse,
    phase_correlation_registration
)

si_utils.get_sim_from_array = patched_get_sim_from_array
fusion.fuse = fuse
registration.phase_correlation_registration = phase_correlation_registration

from mesospim_fractal_tasks.utils.models import DimTuple
from mesospim_fractal_tasks.utils.zarr_utils import (
    build_pyramid, _get_pyramid_structure, convert_ROI_table_to_indices)
from mesospim_fractal_tasks.utils.parallelisation import _set_dask_cluster
from mesospim_fractal_tasks import __version__, __commit__

logger = logging.getLogger(__name__)



import copy
import logging
import shutil
from pathlib import Path
from typing import Optional

import dask.array as da
import numcodecs
import zarr
from dask.distributed import Client
from pydantic import validate_call

from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.ngff import load_NgffImageMeta
from multiview_stitcher import fusion, msi_utils, registration
from multiview_stitcher import spatial_image_utils as si_utils

from mesospim_fractal_tasks import __commit__, __version__
from mesospim_fractal_tasks.utils.models import DimTuple
from mesospim_fractal_tasks.utils.parallelisation import _set_dask_cluster
from mesospim_fractal_tasks.utils.stitching import (
    StitchingChannelInputModel,
    fuse,
    get_sim_from_multiscales,
    parallel_block_processing,
    patched_get_sim_from_array,
    phase_correlation_registration,
)
from mesospim_fractal_tasks.utils.zarr_utils import (
    _get_pyramid_structure,
    build_pyramid,
)

numcodecs.blosc.set_nthreads(1)

si_utils.get_sim_from_array = patched_get_sim_from_array
fusion.fuse = fuse
registration.phase_correlation_registration = phase_correlation_registration

logger = logging.getLogger(__name__)

DEFAULT_INPUT_TRANSFORM_KEY = "fractal_input"
DEFAULT_REGISTERED_TRANSFORM_KEY = "translation_registered"


def _is_proxy_image(zarr_path: Path) -> bool:
    fractal_tasks = zarr.open_group(zarr_path, mode="r").attrs.get("fractal_tasks", {})
    return (
        "prepare_mesospim_omezarr" in fractal_tasks
        and zarr_path.name == "fake_raw_image"
    )


def _get_channel_index(
    zarr_path: Path,
    channel: StitchingChannelInputModel,
) -> int:
    channel.verify_label(zarr_path)
    omero_channel = channel.get_omero_channel(zarr_path)
    if omero_channel is None or omero_channel.index is None:
        raise ValueError(f"Channel {channel} not found in {zarr_path}.")
    return int(omero_channel.index)


def _load_xim(
    zarr_path: Path,
    *,
    resolution: int = 0,
    chunks: Optional[tuple[int, int, int, int]] = None,
):
    return get_sim_from_multiscales(
        zarr_path,
        resolution=resolution,
        chunks=chunks,
        is_proxy=_is_proxy_image(zarr_path),
    )


def _make_msim_from_single_sim(
    xim, 
    transform_key: str = DEFAULT_INPUT_TRANSFORM_KEY):

    sim = si_utils.get_sim_from_array(
            xim.data,
            dims=xim.dims,
            c_coords=xim.coords["c"].data,
            scale=si_utils.get_spacing_from_sim(xim),
            translation=None,
            transform_key=transform_key,
        )

    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    return msim


    

def prepare_and_fuse(
    *,
    zarr_image_paths: list,
    msims_reg_list: list,
    registration_on_z_proj: bool,
    fusion_chunksize_dict: dict,
    fusion_chunks: tuple,
    output_zarr_path: Path,
    batch_options: dict,
    fov_roi_tables: list,
    mode: str
) -> list[str]:

    # Load full-res sims and attach registered transforms
    all_sims = []
    channel_labels = []
    ref_spacing = {}
    for zarr_path, msim_reg_tiles in zip(zarr_image_paths, msims_reg_list):
        xim_ch = _load_xim(zarr_path, resolution=0, chunks=(1,) + fusion_chunks)
        ref_spacing = si_utils.get_spacing_from_sim(xim_ch)
        channel_labels.append(str(xim_ch.coords["c"].values[0]))

        # If multitile, load the tiles and attach registered transforms
        if len(fov_roi_tables) != 0:
            msims_tiles, _ = get_tiles_from_sim(
                xim_ch, fov_roi_tables[0], transform_key=DEFAULT_INPUT_TRANSFORM_KEY
            )
            for itile in range(len(msims_tiles)):
                affine = msi_utils.get_transform_from_msim(
                    msim_reg_tiles[itile], 
                    DEFAULT_REGISTERED_TRANSFORM_KEY)
            
                # if the registration was performed on a maximum projection in Z, we need to
                # broadcast the obtained affine parameters to 3D
                if registration_on_z_proj:
                    affine_3d = param_utils.identity_transform(
                        ndim=3, t_coords=affine.coords["t"] if "t" in affine.dims else None
                    )
                    affine_3d.loc[{pdim: affine.coords[pdim] for pdim in affine.dims}] = affine
                    affine = affine_3d

                msi_utils.set_affine_transform(
                    msims_tiles[itile], affine, DEFAULT_REGISTERED_TRANSFORM_KEY
                )
            sims_ch = [msi_utils.get_sim_from_msim(msim) for msim in msims_tiles]
            all_sims.append(sims_ch)
        else:
            msim_ch = _make_msim_from_single_sim(xim_ch, DEFAULT_INPUT_TRANSFORM_KEY)

            affine = msi_utils.get_transform_from_msim(msim_reg_tiles, DEFAULT_REGISTERED_TRANSFORM_KEY)
            if registration_on_z_proj:
                affine_3d = param_utils.identity_transform(
                    ndim=3,
                    t_coords=affine.coords["t"] if "t" in affine.dims else None,
                )
                affine_3d.loc[{d: affine.coords[d] for d in affine.dims}] = affine
                affine = affine_3d
            msi_utils.set_affine_transform(msim_ch, affine, DEFAULT_REGISTERED_TRANSFORM_KEY)
            all_sims.append(msi_utils.get_sim_from_msim(msim_ch))

    # Compute the shared output space once across all channel sims
    common_osp = fusion.process_output_stack_properties(
        sims=[sim for sims in all_sims for sim in sims],
        output_spacing=ref_spacing,
        output_origin=None,
        output_shape=None,
        output_stack_properties=None,
        output_stack_mode="union",
        transform_key=DEFAULT_REGISTERED_TRANSFORM_KEY,
    )
    spatial_shape = tuple(int(common_osp["shape"][d]) for d in ["z", "y", "x"])
    logger.info(f"Common output spatial shape: {spatial_shape}")

    # Pre-create the final output zarr with full (n_ch, z, y, x) shape
    n_ch = len(channel_labels)
    full_shape = (n_ch,) + spatial_shape
    out_chunks = (1,) + fusion_chunks
    output_zarr_path.mkdir(parents=True, exist_ok=True)
    zarr.open_array(
        str(output_zarr_path / "0"),
        mode="w",
        shape=full_shape,
        chunks=out_chunks,
        dtype=np.uint16,
        dimension_separator="/",
    )

    # Don't overwrite zarr array in fuse function
    zarr_options = {"overwrite": False}

    # Each channel fuses directly into slice [ic:ic+1, ...] of the final zarr if mode=channels
    for ic, sim_ch in enumerate(all_sims):
        logger.info(f"Fusing {mode[:-1]} {ic + 1}/{n_ch} ({channel_labels[ic]})...")

        if mode == "channels":

            # Pass channel coordinates through batch_options -> meta
            logger.info(f"Fusing channel {ic + 1}/{n_ch} ({channel_labels[ic]})...")
            ch_batch_options = copy.deepcopy(batch_options)
            ch_batch_options["channel_index"] = ic 
        else:
            logger.info(f"Fusing view {ic + 1}/{n_ch}...")
            ch_batch_options = batch_options

        if type(sim_ch) != list:
            sim_ch = [sim_ch]
        fusion.fuse(
            sim_ch,
            transform_key=DEFAULT_REGISTERED_TRANSFORM_KEY,
            drop_t=True,
            output_chunksize=fusion_chunksize_dict,
            output_stack_properties=common_osp,
            output_zarr_url=str(output_zarr_path / "0"),  # same store, every time
            batch_options=ch_batch_options,
            zarr_options=zarr_options,
        )
    return channel_labels

def _copy_registered_transform(
    msim_source,
    msim_target,
    registration_on_z_proj: bool,
    transform_key: str = DEFAULT_REGISTERED_TRANSFORM_KEY,
):
    affine = msi_utils.get_transform_from_msim(msim_source, transform_key)
    
    # if the registration was performed on a maximum projection in Z, we need to
    # broadcast the obtained affine parameters to 3D
    if registration_on_z_proj:
        affine_3d = param_utils.identity_transform(
            ndim=3, t_coords=affine.coords["t"] if "t" in affine.dims else None
        )
        affine_3d.loc[{pdim: affine.coords[pdim] for pdim in affine.dims}] = affine
        affine = affine_3d
    msi_utils.set_affine_transform(msim_target, affine, transform_key=transform_key)
    return msim_target


def _get_spacing_dict(sim) -> dict[str, float]:
    spacing = si_utils.get_spacing_from_sim(sim)
    return {k: float(v) for k, v in spacing.items()}


def _get_spatial_shape_dict(sim) -> dict[str, int]:
    sdims = si_utils.get_spatial_dims_from_sim(sim)
    return {dim: int(sim.sizes[dim]) for dim in sdims}


def _get_spatial_origin_dict(sim) -> dict[str, float]:
    sdims = si_utils.get_spatial_dims_from_sim(sim)
    origin = {}
    for dim in sdims:
        coords = sim.coords[dim].values
        origin[dim] = float(coords[0])
    return origin


def _write_level0_array(output_zarr_path: Path, data: da.Array):
    da.to_zarr(data, str(output_zarr_path / "0"), overwrite=True)


def _update_omero_channels(attrs: dict, channel_labels: list[str]) -> dict:
    attrs = copy.deepcopy(attrs)
    omero = attrs.get("omero", {})
    old_channels = omero.get("channels", [])

    new_channels = []
    for i, label in enumerate(channel_labels):
        if i < len(old_channels):
            ch = copy.deepcopy(old_channels[i])
        else:
            ch = {
                "active": True,
                "coefficient": 1,
                "color": "FFFFFF",
                "family": "linear",
                "inverted": False,
                "label": label,
                "window": {"start": 0, "end": 65535, "min": 0, "max": 65535},
            }
        ch["label"] = label
        new_channels.append(ch)

    omero["channels"] = new_channels
    attrs["omero"] = omero
    return attrs


def _copy_ngff_metadata(
    *,
    source_zarr_path: Path,
    output_zarr_path: Path,
    task_name: str,
    task_params: dict,
    channel_labels: list[str],
):
    source_group = zarr.open_group(source_zarr_path, mode="r")
    output_group = zarr.open_group(output_zarr_path, mode="a")

    attrs = source_group.attrs.asdict()
    attrs = _update_omero_channels(attrs, channel_labels)

    if "multiscales" in attrs and len(attrs["multiscales"]) > 0:
        attrs["multiscales"][0]["name"] = f"{source_zarr_path.name}_registered"

    fractal_tasks = attrs.get("fractal_tasks", {})
    fractal_tasks[task_name] = dict(
        version=__version__.split("dev")[0][:-1],
        commit=__commit__,
        input_parameters=task_params,
    )
    attrs["fractal_tasks"] = fractal_tasks

    output_group.attrs.put(attrs)


def _build_pyramid_like_source(source_zarr_path: Path, output_zarr_path: Path):
    logger.info("Start building multi-resolution pyramid.")
    with _set_dask_cluster(n_workers=4) as cluster:
        with Client(cluster):
            pyramid_dict = _get_pyramid_structure(source_zarr_path)
            build_pyramid(
                zarr_url=output_zarr_path,
                pyramid_dict=pyramid_dict,
            )
    logger.info("Finished building multi-resolution pyramid.")


def _register(
    *,
    xim_reg_list,
    reg_channel_index: int,
    registration_function: str,
    transform_type: str,
    registration_on_z_proj: bool,
    overlap_tolerance: Optional[DimTuple],
    fov_roi_tables: list,
):
    if transform_type not in ["translation", "rigid", "similarity", "affine"]:
        raise ValueError(f"Error. Unknown transformation type: {transform_type}."
                    " Available types are 'translation', 'rigid', 'similarity', "
                    "'affine'.")

    if registration_function not in ["phase_correlation", "antspy"]:
        raise ValueError(f"Error. Unknown registration function: {registration_function}."
                    " Available functions are 'phase_correlation', 'antspy'.")

    registration_callable = (
        registration.registration_ANTsPy
        if registration_function == "antspy"
        else registration.phase_correlation_registration
    )

    if overlap_tolerance is None:
        overlap_tolerance = DimTuple(z=0, y=0, x=0)
    else:
        for dim in ["z", "y", "x"]:
            if overlap_tolerance[dim] is None:
                overlap_tolerance[dim] = 0

    if registration_on_z_proj:
        xim_reg_list = [xim_reg.max("z") for xim_reg in xim_reg_list]
        overlap_tolerance["z"] = None

    msims_reg_list = []
    for i, xim_reg in enumerate(xim_reg_list):
        if len(fov_roi_tables) != 0:
            msims_tiles_reg, _ = get_tiles_from_sim(
                xim_reg, fov_roi_tables[i], transform_key=DEFAULT_INPUT_TRANSFORM_KEY
            )
            msims_reg_list.append(msims_tiles_reg)
        else:
            msims_reg_list.append(_make_msim_from_single_sim(xim_reg, DEFAULT_INPUT_TRANSFORM_KEY))

    overlap_tolerance_dict = overlap_tolerance.get_dict()

    n_cpus = os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())
    n_cpus = 1 if n_cpus is None else int(n_cpus)

    if len(fov_roi_tables) == 0:
        registration.register(
            msims_reg_list,
            transform_key=DEFAULT_INPUT_TRANSFORM_KEY,
            new_transform_key=DEFAULT_REGISTERED_TRANSFORM_KEY,
            reg_channel_index=reg_channel_index,
            pairwise_reg_func=registration_callable,
            overlap_tolerance=overlap_tolerance_dict,
            registration_binning={'z': 1, 'y': 1, 'x': 1},
            groupwise_resolution_kwargs={"transform": transform_type},
            n_parallel_pairwise_regs=4,
            pre_registration_pruning_method=None,
        )
    else:
        for i in range(len(msims_reg_list)):
            assert len(msims_reg_list[0]) == len(msims_reg_list[i]), \
                f"All channels must have the same number of tiles."
        for tile in range(len(msims_reg_list[0])):
            logger.info(f"Registering tile {tile+1}/{len(msims_reg_list[0])}...")
            registration.register(
                [msims_reg_list[i][tile] for i in range(len(msims_reg_list))],
                transform_key=DEFAULT_INPUT_TRANSFORM_KEY,
                new_transform_key=DEFAULT_REGISTERED_TRANSFORM_KEY,
                reg_channel_index=reg_channel_index,
                pairwise_reg_func=registration_callable,
                overlap_tolerance=overlap_tolerance_dict,
                registration_binning={'z': 1, 'y': 1, 'x': 1},
                groupwise_resolution_kwargs={"transform": transform_type},
                n_parallel_pairwise_regs=4,
                pre_registration_pruning_method=None,
            )
    logger.info("Finished registration.")

    return msims_reg_list

@validate_call
def fuse_views_or_channels(
    *,
    zarr_url: str,
    output_zarr_name: Optional[str | Path] = None,
    zarr_image_paths: Optional[list[Path | str]] = None,
    registration_channel: Optional[StitchingChannelInputModel] = None,
    registration_resolution_level: Optional[int] = None,
    registration_on_z_proj: bool = False,
    registration_function: str = "phase_correlation",
    transform_type: str = "translation",
    max_workers: int = 4,
    fusion_chunksize: Optional[DimTuple] = None,
    overwrite: bool = False,
) -> dict[str, list]:
    """
    Register two OME-Zarr views and fuse them into one OME-Zarr.

    Assumptions:
    - All views share the same name and are differentiated by a numbering suffix: e.g. 
        common_name_1, common_name_2, common_name_3, or common_name1, common_name2, etc.
    - Both input images contain the same channels in the same order.
    - Registration is estimated on `registration_channel`.
    - Fusion is then applied channel-by-channel using the same transform.
    """
    if registration_channel is None:
        mode = "channels"
    else:
        mode = "views"
    ref_zarr_path = Path(zarr_url)

    common_name = None
    image_name = ref_zarr_path.name
    
    if zarr_image_paths == None:
  
        # Search in parent directories for the views to register
        if ref_zarr_path.parent.stem[-1].isdigit():
            
            common_name = ref_zarr_path.parent.stem[:-1]
            while common_name[-1].isdigit():
                common_name = common_name[:-1]
            if common_name[-3:] == "_sh" or common_name[-3:] == "_ch":
                common_name = common_name[:-3]
            if common_name[-1] == "_":
                common_name = common_name[:-1]

        elif ref_zarr_path.parent.stem[-2:].lower() in ["_a", "_b", "_c", "_d"]:
            common_name = ref_zarr_path.parent.stem[:-2]

        else:
            raise ValueError(f"Could not determine common name of the {mode}. Expected to " \
            "end with a numbering or a letter, e.g. common_name_1, common_name_2," \
            " common_name_3, or common_name1, common_name2, etc.")
        zarr_paths = [path for path in ref_zarr_path.parents[1].glob(f"{common_name}*.zarr") if path.is_dir()]
        zarr_image_paths = [path / image_name for path in zarr_paths]
    else:
        zarr_image_paths = [Path(zarr_path) for zarr_path in zarr_image_paths]

    if output_zarr_name is None:
        if common_name is None:
            names = [zarr_path.name for zarr_path in zarr_image_paths]
            common_name = os.path.commonprefix(names)
            if common_name == "":
                raise ValueError("Could not determine a common prefix between the view filenames."
                                 "Either provide the output_zarr_name or make sure all views share a common prefix.")
            if common_name[-3:] == "_sh":
                common_name = common_name[:-3]
            if common_name[-1] == "_":
                common_name = common_name[:-1]
        output_zarr_name = common_name + f"_{mode}_fused"
    output_zarr_name = Path(output_zarr_name).with_suffix(".zarr")
    output_zarr_path = Path(ref_zarr_path.parents[1], output_zarr_name, image_name)
    if output_zarr_path.exists() and not overwrite:
        raise ValueError(f"Output zarr {output_zarr_name} already exists in "
                         f"{ref_zarr_path.parents[1].name}. "
                         "Hint: try setting overwrite=True if you want to overwrite it.")
    elif output_zarr_path.exists() and overwrite:
        shutil.rmtree(output_zarr_path.parent)
        if output_zarr_path in zarr_image_paths:
            zarr_image_paths.remove(output_zarr_path)
    
    for path in zarr_image_paths:
        if not path.exists():
            raise ValueError(f"Could not find {path.name} in {path.parent.name}.")

    logger.info(
        f"Starting task `Fuse Views or Channels` "
        f"with files of common name {common_name} in OME-Zarr directory {ref_zarr_path.parents[1].name}"
    )

    if registration_resolution_level is None:
        registration_resolution_levels = []
        for path in zarr_image_paths:
            path_meta = load_NgffImageMeta(str(path))
            registration_resolution_levels.append(path_meta.num_levels - 1)
        registration_resolution_level = min(registration_resolution_levels)
    else:
        for path in zarr_image_paths:
            path_meta = load_NgffImageMeta(str(path))
            if path_meta.num_levels - 1 < registration_resolution_level:
                raise ValueError(f"All {mode} must have the "
                                 f"provided resolution level: {registration_resolution_level}.")
    
    if registration_channel is not None:
        reg_channel_index = _get_channel_index(ref_zarr_path, registration_channel)
    else:
        reg_channel_index = 0

    logger.info(f"Loading the {mode} with resolution level {registration_resolution_level}.")
    xim_reg_list = [_load_xim(zarr_path, resolution=registration_resolution_level) for zarr_path in zarr_image_paths]
    if mode == "channels":
        fake_channel_name = xim_reg_list[0].coords["c"].data
        for i, xim_reg in enumerate(xim_reg_list):
            xim_reg_list[i] = xim_reg.assign_coords(c=fake_channel_name)
    
    fov_roi_tables = []
    if Path(ref_zarr_path, "tables/FOV_ROI_table").exists():
        logger.info("Found FOV_ROI_table in the reference zarr, "
                    "assuming a multi-FOV dataset and using it for registration.")
        
        for zarr_path in zarr_image_paths:
            fov_roi_df = ad.read_zarr(Path(zarr_path, "tables/FOV_ROI_table")).to_df()
            for dim in ["z", "y", "x"]:
                fov_roi_df[f"{dim}_micrometer_original"] = fov_roi_df[f"{dim}_micrometer"]
            fov_roi_tables.append(fov_roi_df)

    logger.info(f"Started registration of the {mode}.")
    overlap_tolerance = DimTuple(z=0, y=0, x=0)
    msims_reg_list = _register(
        xim_reg_list=xim_reg_list,
        reg_channel_index=reg_channel_index,
        registration_function=registration_function,
        transform_type=transform_type,
        registration_on_z_proj=registration_on_z_proj,
        overlap_tolerance=overlap_tolerance,
        fov_roi_tables=fov_roi_tables,
    )

    original_chunks = da.from_zarr(ref_zarr_path / "0").chunksize
    fusion_chunks = list(original_chunks[-3:])
    if fusion_chunksize is not None:
        for i, dim in zip([-3, -2, -1], ["z", "y", "x"]):
            if fusion_chunksize[dim] is not None:
                fusion_chunks[i] = fusion_chunksize[dim]
    fusion_chunks = tuple(int(v) for v in fusion_chunks)
    logger.info(f"Fusion chunk size set to: {fusion_chunks}.")

    sdims = si_utils.get_spatial_dims_from_sim(xim_reg_list[0])

    fusion_chunksize_dict = {
        dim: cs for dim, cs in zip(sdims, fusion_chunks)
    }

    n_cpus = os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())
    n_cpus = 1 if n_cpus is None else int(n_cpus)
    if max_workers > n_cpus:
        logger.warning("Number of processes is greater than available number of workers"
                       "... Setting to the number of available CPUs. ")
        max_workers = n_cpus

    batch_options = {
        "zarr_array_creation_kwargs": {"dimension_separator": "/"},
        "max_workers": max_workers,
        "batch_func": parallel_block_processing,
    }
    if max_workers == 1:
        batch_options = {
            "zarr_array_creation_kwargs": {"dimension_separator": "/"},
            "max_workers": 1,
        }

    logger.info("Starting fusing {mode}...")
    channel_labels = prepare_and_fuse(
        zarr_image_paths=zarr_image_paths,
        msims_reg_list=msims_reg_list,
        registration_on_z_proj=registration_on_z_proj,
        fusion_chunksize_dict=fusion_chunksize_dict,
        fusion_chunks=fusion_chunks,
        output_zarr_path=output_zarr_path,
        batch_options=batch_options,
        fov_roi_tables=fov_roi_tables,
        mode=mode
    )
    logger.info("Finished fusing {mode}.")

    
    # Copy NGFF metadata from the old zarr_url to the new zarr
    _copy_ngff_metadata(
        source_zarr_path=ref_zarr_path,
        output_zarr_path=output_zarr_path,
        task_name="fuse_views_or_channels",
        task_params=dict(
            ref_zarr_url=str(ref_zarr_path),
            registration_channel=registration_channel.label,
            registration_resolution_level=registration_resolution_level,
            registration_on_z_proj=registration_on_z_proj,
            registration_function=registration_function,
            overlap_tolerance=overlap_tolerance_dict,
            transform_type=transform_type,
            max_workers=max_workers,
        ),
        channel_labels=[str(c) for c in channel_labels],
    )

    _build_pyramid_like_source(ref_zarr_path, output_zarr_path)

    return {
        "image_list_updates": [
            dict(
                zarr_url=str(output_zarr_path),
                origin=str(ref_zarr_path),
                attributes=dict(image=output_zarr_path.name),
            )
        ]
    }


    









if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    # Change manually depending on which task you want to expose in the manifest
    #run_fractal_task(task_function=fuse_views_or_channels)

    fuse_views_or_channels(
        zarr_url="data/Multitile/IENFD25-5-67TH/IENFD25-5-67TH_downsampled_ch561.zarr/raw_image",
        #output_zarr_name="test",
        #zarr_image_paths=["data/Multitile/IENFD25-6-1/IENFD25-6-1_fiber_pgp.zarr/raw_image",
        #                   "data/Multitile/IENFD25-6-1/IENFD25-6-1_fiber_lectin.zarr/raw_image"],
        registration_channel=None,
        registration_resolution_level=None,
        registration_on_z_proj=False,
        registration_function="phase_correlation",
        transform_type="translation",
        max_workers=2,
        fusion_chunksize=None,
        overwrite=True
    )