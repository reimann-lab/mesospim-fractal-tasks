import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

import copy
import logging
import shutil
from pathlib import Path
from typing import Optional
from pydantic import validate_call

import anndata as ad
import numpy as np
import dask.array as da
import numcodecs
import zarr
from dask.distributed import Client

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import get_single_image_ROI
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url
from multiview_stitcher import fusion, msi_utils, registration, param_utils
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
    get_tiles_from_sim,
    _copy_registered_transform,
)
from mesospim_fractal_tasks.utils.zarr_utils import (
    _get_pyramid_structure,
    build_pyramid,
    _copy_ngff_metadata,
    _determine_optimal_contrast,
    _update_omero_channels,
)

numcodecs.blosc.set_nthreads(1)

si_utils.get_sim_from_array = patched_get_sim_from_array
fusion.fuse = fuse
registration.phase_correlation_registration = phase_correlation_registration

logger = logging.getLogger(__name__)

DEFAULT_INPUT_TRANSFORM_KEY = "fractal_input"
DEFAULT_REGISTERED_TRANSFORM_KEY = "translation_registered"


def _is_proxy_image(
    zarr_path: Path
) -> bool:
    fractal_tasks = zarr.open_group(zarr_path, mode="r").attrs.get("fractal_tasks", {})
    return (
        "prepare_mesospim_omezarr" in fractal_tasks
        and zarr_path.name == "fake_raw_image"
    )

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

    sim = patched_get_sim_from_array(
            xim.data,
            dims=xim.dims,
            c_coords=xim.coords["c"].data,
            scale=si_utils.get_spacing_from_sim(xim),
            translation=None,
            transform_key=transform_key,
        )

    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    return msim

def _update_channels_metadata(
    source_zarr_paths: list,
    output_zarr_path: Path,
) -> None:
    
    attrs_list = [zarr.open_group(source_zarr_path, mode="r").attrs.asdict() for source_zarr_path in source_zarr_paths]
    ref_attrs = attrs_list[0]
    ref_omero = ref_attrs.get("omero", {})

    assert len(ref_omero["channels"]) == 1, "Metadata has more than one channel."
    ref_channel = ref_omero["channels"][0]["label"]

    ref_acquisition = ref_attrs["acquisition_metadata"]
    assert len(ref_acquisition["channels"]) == 1, "Metadata has more than one channel."
    assert ref_acquisition["channels"][0]["label"] == ref_channel, "Channel metadata is not corrupted. " \
        "Label for omero channel and acquisition metadata are not congruent"
    
    for attrs in attrs_list:
        omero = attrs.get("omero", {})
        new_channel = omero.get("channels", [])
        new_channel_acquisiton = attrs["acquisition_metadata"].get("channels", [])
        assert len(new_channel) == 1, "Metadata has more than one channel."

        if new_channel[0]["label"] == ref_channel:
            continue

        ref_omero["channels"].append(new_channel[0])
        ref_acquisition["channels"].append(new_channel_acquisiton[0])

    ref_attrs["omero"] = ref_omero
    ref_attrs["acquisition_metadata"] = ref_acquisition

    out_attrs = zarr.open_group(output_zarr_path, mode="a").attrs.asdict()
    out_attrs["omero"] = ref_omero
    out_attrs["acquisition_metadata"] = ref_acquisition

    zarr.open_group(output_zarr_path, mode="a").attrs.put(out_attrs)

def find_zarr_images(
    ref_zarr_path: Path,
    mode: str,
) -> tuple[list[Path], str]:
    
    image_name = ref_zarr_path.name
    
    # Search in parent directories for the views to register
    if ref_zarr_path.parent.stem[-1].isdigit():
        
        common_name = ref_zarr_path.parent.stem[:-1].lower()
        while common_name[-1].isdigit():
            common_name = common_name[:-1]
        if common_name[-3:] == "_sh" or common_name[-3:] == "_ch":
            common_name = common_name[:-3]
        elif common_name[-4:] == "_sh_" or common_name[-4:] == "_ch_":
            common_name = common_name[:-4]
        elif common_name[-6:] == "_view_":
            common_name = common_name[:-6]
        elif common_name[-5:] == "_view":
            common_name = common_name[:-5]
        
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

    return zarr_image_paths, common_name

def find_common_name(
    common_name: str | None,
    zarr_image_paths: list[Path],
    mode: str,
) -> str:
    if common_name is None:
        names = [zarr_path.name for zarr_path in zarr_image_paths]
        common_name = os.path.commonprefix(names)
        if common_name == "":
            raise ValueError("Could not determine a common prefix between the view filenames."
                                "Either provide the output_zarr_name or make sure all views share a common prefix.")
        if common_name[-3:].lower() == "_sh" or common_name[-3:].lower() == "_ch":
            common_name = common_name[:-3]
        if common_name[-1] == "_":
            common_name = common_name[:-1]
    output_zarr_name = common_name + f"_{mode}_fused"
    
    return output_zarr_name

def prepare_and_register(
    *,
    xim_reg_list,
    reg_channel_index: int,
    registration_function: str,
    transform_type: str,
    registration_on_z_proj: bool,
    fov_roi_tables: list,
    max_workers: int,
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

    if registration_on_z_proj:
        xim_reg_list = [xim_reg.max("z") for xim_reg in xim_reg_list]

    msims_reg_list = []
    for i, xim_reg in enumerate(xim_reg_list):
        if len(fov_roi_tables) != 0:
            msims_tiles_reg, _ = get_tiles_from_sim(
                xim_reg, fov_roi_tables[i], transform_key=DEFAULT_INPUT_TRANSFORM_KEY
            )
            msims_reg_list.append(msims_tiles_reg)
        else:
            msims_reg_list.append(_make_msim_from_single_sim(xim_reg, DEFAULT_INPUT_TRANSFORM_KEY))

    if len(fov_roi_tables) == 0:
        registration.register(
            msims_reg_list,
            transform_key=DEFAULT_INPUT_TRANSFORM_KEY,
            new_transform_key=DEFAULT_REGISTERED_TRANSFORM_KEY,
            reg_channel_index=reg_channel_index,
            pairwise_reg_func=registration_callable,
            registration_binning={'z': 1, 'y': 1, 'x': 1},
            groupwise_resolution_kwargs={"transform": transform_type},
            n_parallel_pairwise_regs=max_workers,
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
                registration_binning={'z': 1, 'y': 1, 'x': 1},
                groupwise_resolution_kwargs={"transform": transform_type},
                n_parallel_pairwise_regs=max_workers,
                pre_registration_pruning_method=None,
            )
    logger.info("Finished registration.")

    return msims_reg_list

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
) -> None:

    # Load full-res sims and attach registered transforms
    all_sims = []
    channel_labels = []
    ref_spacing = {}
    original_shape = None
    for zarr_path, msim_reg in zip(zarr_image_paths, msims_reg_list):
        xim_ch = _load_xim(zarr_path, resolution=0, chunks=(1,) + fusion_chunks)
        ref_spacing = si_utils.get_spacing_from_sim(xim_ch)
        channel_labels.append(str(xim_ch.coords["c"].values[0]))

        # If multitile, load the tiles separately and attach registered transforms
        if len(fov_roi_tables) != 0:
            msims_tiles, _ = get_tiles_from_sim(
                xim_ch, fov_roi_tables[0], transform_key=DEFAULT_INPUT_TRANSFORM_KEY
            )
            original_shape = xim_ch.shape
            for itile in range(len(msims_tiles)):
                msims_tiles[itile] = _copy_registered_transform(
                    msim_reg[itile],
                    msims_tiles[itile],
                    registration_on_z_proj,
                    transform_key=DEFAULT_REGISTERED_TRANSFORM_KEY,
                )
            sims_ch = [msi_utils.get_sim_from_msim(msim) for msim in msims_tiles]
            dtype = sims_ch[0].dtype
            all_sims.append(sims_ch)
        else:
            msim_ch = _make_msim_from_single_sim(xim_ch, DEFAULT_INPUT_TRANSFORM_KEY)
            msim_ch = _copy_registered_transform(
                msim_reg,
                msim_ch,
                registration_on_z_proj,
                transform_key=DEFAULT_REGISTERED_TRANSFORM_KEY,
            )
            all_sims.append(msi_utils.get_sim_from_msim(msim_ch))
            dtype = all_sims[0].dtype

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

    # Make sure to ignore original transform of the tiles (not stitching mode)
    if len(fov_roi_tables) != 0 and original_shape is not None:
        common_osp["shape"] = {
            'z': original_shape[-3], 
            'y': original_shape[-2], 
            'x': original_shape[-1]
        }
    spatial_shape = tuple(int(common_osp["shape"][d]) for d in ["z", "y", "x"])
    logger.info(f"Common output spatial shape: {spatial_shape}")

    # Each channel fuses directly into slice [ic:ic+1, ...] of the final zarr if mode=channels
    if mode == "channels":
        
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
            dtype=dtype,
            dimension_separator="/",
        )

        # Don't overwrite zarr array in fuse function
        zarr_options = {"overwrite": False}
        for ic, sim_ch in enumerate(all_sims):

            # Pass channel coordinates through batch_options -> meta
            logger.info(f"Fusing channel {ic + 1}/{n_ch} ({channel_labels[ic]})...")
            ch_batch_options = copy.deepcopy(batch_options)
            ch_batch_options["channel_index"] = ic 

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
    else:
        logger.info(f"Fusing views...")
        ch_batch_options = batch_options
        if type(all_sims[0]) is list:
            all_sims = [sim for sims in all_sims for sim in sims]
        fusion.fuse(
            all_sims,
            transform_key=DEFAULT_REGISTERED_TRANSFORM_KEY,
            drop_t=True,
            output_chunksize=fusion_chunksize_dict,
            output_stack_properties=common_osp,
            output_zarr_url=str(output_zarr_path / "0"),  # same store, every time
            batch_options=ch_batch_options
        )

def copy_roi_tables(
    *,
    fov_roi_tables: list,
    output_zarr_path: Path,
    ref_zarr_path: Path,
):
    if len(fov_roi_tables) != 0:
        _copy_tables_from_zarr_url(str(ref_zarr_path), str(output_zarr_path))
    else:
    
        # Open the zarr group (read/write)
        new_shape = zarr.open_array(output_zarr_path / "0").shape
        ngff_image_meta = load_NgffImageMeta(str(output_zarr_path))
        pixels_ZYX = ngff_image_meta.get_pixel_sizes_zyx(level=0)

        # Add ROI table to the image
        new_group = zarr.open_group(output_zarr_path, mode="a")
        image_ROI_table = get_single_image_ROI(new_shape, pixels_ZYX=pixels_ZYX)
        write_table(
            new_group,
            "well_ROI_table",
            image_ROI_table,
            overwrite=True,
            table_attrs={"type": "roi_table"},
        )

@validate_call
def fuse_views_or_channels(
    *,
    zarr_url: str,
    output_zarr_name: Optional[str] = None,
    zarr_image_paths: Optional[list[str]] = None,
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
    Register two OME-Zarr views or channels and fuse them into one OME-Zarr.

    Parameters:
        zarr_url: Absolute path to the OME-Zarr image.
        output_zarr_name: Name for the output OME-Zarr.
        zarr_image_paths: List of absolute paths to the OME-Zarr images to fuse.
        registration_channel: You should only provide this parameter if fusing views! Otherwise, 
            leave it blank. When provided it tells which channel is to be used to 
            compute the correct positions of the views. It requires either 
            `wavelength_id` (e.g. `488`) or `label` (e.g. `PGP9.5`), but not both.
        registration_resolution_level: Resolution level to use for registration.
            Recommended to set the lowest level possible, e.g. 5 (highest is 0).
            If None, the lowest resolution level available will be used for registration.
            Default: None.
        registration_on_z_proj: Whether to perform registration on a maximum
            projection along z in case of 3D data. Recommended when memory is
            limited but results are generally less good. Default: False.
        registration_function: Type of transformation to use for registration.
            Available functions:
            - 'phase_correlation'
            - 'antspy': see ANTsPy documentation for more information.
            Default: 'phase_correlation'.
        transform_type: Type of transformation to use for registration.
            Available types:
            - 'translation': translation
            - 'rigid': rigid body transformation
            - 'similarity': similarity transformation
            - 'affine': affine transformation
            Default: 'translation'.
        max_workers: Maximum number of workers to process blocks in parallel. Should not
            be more than number of available workers. If set to one, it falls back to
            sequential processing. Default: 4.
        fusion_chunksize: Chunksize for the dimension (Z, Y, X) to use when performing
            the fusion. It impacts the memory usage and the time to fuse the tiles. It
            also corresponds to the chunksize of the output zarr. Setting smaller chunks
            can reduce memory usage but increase the time to fuse the tiles.
            If None, the chunksize of the raw image is used. Default: None.
        overwrite: If `True`, existing output zarr will be overwritten. Default: False.
    """

    if registration_channel is None:
        mode = "channels"
    else:
        mode = "views"
    ref_zarr_path = Path(zarr_url)

    common_name = None
    if zarr_image_paths == None:
        zarr_image_path_list, common_name = find_zarr_images(ref_zarr_path, mode)
    else:
        zarr_image_path_list = [Path(zarr_path) for zarr_path in zarr_image_paths]
    
    if output_zarr_name is None:
        output_zarr_name = find_common_name(common_name, zarr_image_path_list, mode)
    output_zarr_name = str(Path(output_zarr_name).with_suffix(".zarr"))
    output_zarr_path = Path(ref_zarr_path.parents[1], output_zarr_name, ref_zarr_path.name)
    if output_zarr_path.exists() and not overwrite:
        raise ValueError(f"Output zarr {output_zarr_name} already exists in "
                         f"{ref_zarr_path.parents[1].name}. "
                         "Hint: try setting overwrite=True if you want to overwrite it.")
    elif (output_zarr_path.exists() or output_zarr_path.parent.exists()) and overwrite:
        shutil.rmtree(output_zarr_path.parent)
        if output_zarr_path in zarr_image_path_list:
            zarr_image_path_list.remove(output_zarr_path)
    
    for path in zarr_image_path_list:
        if not path.exists():
            raise ValueError(f"Could not find {path.name} in {path.parent.name}.")

    # Open zarr group at root of output zarr
    zarr.open_group(output_zarr_path.parent)

    logger.info(
        f"Starting task `Fuse Views or Channels` "
        f"with files of common name {common_name} in OME-Zarr directory {ref_zarr_path.parents[1].name}"
    )

    if registration_resolution_level is None:
        registration_resolution_levels = []
        for path in zarr_image_path_list:
            path_meta = load_NgffImageMeta(str(path))
            registration_resolution_levels.append(path_meta.num_levels - 1)
        registration_resolution_level = min(registration_resolution_levels)
    else:
        for path in zarr_image_path_list:
            path_meta = load_NgffImageMeta(str(path))
            if path_meta.num_levels - 1 < registration_resolution_level:
                raise ValueError(f"All {mode} must have the "
                                 f"provided resolution level: {registration_resolution_level}.")
    
    if registration_channel is not None:
        reg_channel_index = registration_channel.get_omero_channel_index(ref_zarr_path)
    else:
        reg_channel_index = 0

    logger.info(f"Loading the {mode} with resolution level {registration_resolution_level}.")
    xim_reg_list = [_load_xim(zarr_path, resolution=registration_resolution_level) for zarr_path in zarr_image_path_list]
    if mode == "channels":
        fake_channel_name = xim_reg_list[0].coords["c"].data
        for i, xim_reg in enumerate(xim_reg_list):
            xim_reg_list[i] = xim_reg.assign_coords(c=fake_channel_name)
    else:
        channel_labels = set(xim_reg_list[0].coords["c"].data)
        for i, xim_reg in enumerate(xim_reg_list):
            if i == 0:
                continue
            if set(xim_reg.coords["c"].data) != channel_labels:
                raise ValueError("All views must have the same channels."
                                 f"Found channels {xim_reg.coords['c'].data} in view {i} "
                                 f"and {channel_labels} in view 0.")
    
    fov_roi_tables = []
    if Path(ref_zarr_path, "tables/FOV_ROI_table").exists():
        logger.info("Found FOV_ROI_table in the reference zarr, "
                    "assuming a multi-FOV dataset and using it for registration.")
        
        for zarr_path in zarr_image_path_list:
            fov_roi_df = ad.read_zarr(Path(zarr_path, "tables/FOV_ROI_table")).to_df()
            for dim in ["z", "y", "x"]:
                fov_roi_df[f"{dim}_micrometer_original"] = fov_roi_df[f"{dim}_micrometer"]
            fov_roi_tables.append(fov_roi_df)
    
    n_cpus = os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())
    n_cpus = 1 if n_cpus is None else int(n_cpus)
    if max_workers > n_cpus:
        logger.warning("Number of processes is greater than available number of workers"
                       "... Setting to the number of available CPUs. ")
        max_workers = n_cpus

    logger.info(f"Started registration of the {mode}.")
    msims_reg_list = prepare_and_register(
        xim_reg_list=xim_reg_list,
        reg_channel_index=reg_channel_index,
        registration_function=registration_function,
        transform_type=transform_type,
        registration_on_z_proj=registration_on_z_proj,
        fov_roi_tables=fov_roi_tables,
        max_workers=max_workers,
    )

    original_chunks = da.from_zarr(ref_zarr_path / "0").chunksize
    fusion_chunks = list(original_chunks[-3:])
    if fusion_chunksize is not None:
        for i, dim in zip([-3, -2, -1], ["z", "y", "x"]):
            if fusion_chunksize[dim] is not None:
                fusion_chunks[i] = fusion_chunksize[dim]
    fusion_chunks = tuple(int(v) for v in fusion_chunks)
    fusion_chunksize_dict = {
        dim: cs for dim, cs in zip(["z", "y", "x"], fusion_chunks)
    }
    logger.info(f"Fusion chunk size set to: {fusion_chunks}.")

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
    prepare_and_fuse(
        zarr_image_paths=zarr_image_path_list,
        msims_reg_list=msims_reg_list,
        registration_on_z_proj=registration_on_z_proj,
        fusion_chunksize_dict=fusion_chunksize_dict,
        fusion_chunks=fusion_chunks,
        output_zarr_path=output_zarr_path,
        batch_options=batch_options,
        fov_roi_tables=fov_roi_tables,
        mode=mode
    )
    logger.info(f"Finished fusing {mode}.")

    
    # Copy NGFF metadata from the old zarr_url to the new zarr
    _copy_ngff_metadata(
        source_zarr_path=zarr_image_path_list[0],
        output_zarr_path=output_zarr_path,
        fractal_task_name="fuse_views_or_channels",
        task_params=dict(
            zarr_image_paths=[zarr_image_path.parent.name + "/" + zarr_image_path.name for zarr_image_path in zarr_image_path_list],
            registration_channel=registration_channel.label if registration_channel else None,
            registration_resolution_level=registration_resolution_level,
            registration_on_z_proj=registration_on_z_proj,
            registration_function=registration_function,
            transform_type=transform_type,
            max_workers=max_workers,
            mode=mode,
        ),
        commit=__commit__,
        version=__version__,
    )

    # Copy all channels metadata in channels mode
    if mode == "channels":
        _update_channels_metadata(zarr_image_path_list, output_zarr_path)
    
    # Copy original metadata
    for i, source_zarr_path in enumerate(zarr_image_path_list):
        shutil.copyfile(source_zarr_path /".zattrs", output_zarr_path /f".{mode[:-1]}_{i}_zattrs")

    copy_roi_tables(
        fov_roi_tables=fov_roi_tables,
        output_zarr_path=output_zarr_path,
        ref_zarr_path=ref_zarr_path,
    )

    logger.info("Start building multi-resolution pyramid.")
    with _set_dask_cluster(n_workers=max_workers) as cluster:
        with Client(cluster) as client:
            pyramid_dict = _get_pyramid_structure(output_zarr_path)
            build_pyramid(
                zarr_url=output_zarr_path,
                pyramid_dict=pyramid_dict
            )
            logger.info("Finished building resolution pyramid")

    contrast_limits = _determine_optimal_contrast(output_zarr_path, len(pyramid_dict), segment_sample=True)
    
    _update_omero_channels(output_zarr_path, {"window": contrast_limits})

    # Prepare the image list update
    image_list_updates = dict(
        image_list_updates=[
            dict(
                zarr_url=str(output_zarr_path),
                origin=str(ref_zarr_path),
                attributes=dict(image=output_zarr_path.name),
            )
        ]
    )
    return image_list_updates


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=fuse_views_or_channels)