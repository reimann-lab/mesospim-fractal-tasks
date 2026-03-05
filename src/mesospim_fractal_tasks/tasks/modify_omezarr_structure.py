"""
Task: Modify OME-Zarr Structure

Allows users to modify the structure of an existing OME-Zarr image:
- Rename the image in the metadata
- Rechunk all pyramid levels
- Add / remove / consolidate pyramid levels
- Update channel colors and contrast limits
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import dask.array as da
import numpy as np
import zarr
from pydantic import validate_call
from dask.distributed import Client

from fractal_tasks_core.ngff import load_NgffImageMeta

from mesospim_fractal_tasks.utils.models import DimTuple, Channel
from mesospim_fractal_tasks.utils.zarr_utils import (
    _estimate_pyramid_depth, _build_single_level, _update_omero_channels)
from mesospim_fractal_tasks.utils.parallelisation import _set_dask_cluster

logger = logging.getLogger(__name__)



def _update_omero_channels(zarr_path: Path, update_dict: dict[str, Any]) -> None:
    """
    Update OMERO channel metadata in-place.

    ``update_dict`` keys map to channel attribute names.  The special key
    ``"window"`` is expected to be a dict keyed by *string* channel index with
    ``{"start": ..., "end": ...}`` sub-dicts.  All other keys are expected to
    be lists indexed by channel position.

    Parameters:
        zarr_path: Path to the OME-Zarr group.
        update_dict: Mapping of attribute name → new values.
    """
    zarr_group = zarr.open_group(str(zarr_path), mode="r+")
    channels_attrs = zarr_group.attrs["omero"]["channels"]

    for c in range(len(channels_attrs)):
        for key, value in update_dict.items():
            if key == "window":
                if str(c) in value:
                    logger.info(f"Updating window contrast limits for channel {c}: "
                                f"{channels_attrs[c][key]} → {value[str(c)]}")
                    if "start" in value[str(c)]:
                        channels_attrs[c][key]["start"] = value[str(c)]["start"]
                    if "end" in value[str(c)]:
                        channels_attrs[c][key]["max"] = min(value[str(c)]["end"] * 10, 2**16-1)
                        channels_attrs[c][key]["end"] = value[str(c)]["end"]
            else:
                if str(c) in value:
                    logger.info(f"Updating channel {c} '{key}': "
                                f"{channels_attrs[c].get(key)} → {value[str(c)]}")
                    channels_attrs[c][key] = value[str(c)]

    zarr_group.attrs["omero"] = {"channels": channels_attrs}

def _update_multiscales_name(zarr_path: Path, new_name: str) -> None:
    """
    Update the ``name`` field inside the ``multiscales`` metadata.

    Parameters:
        zarr_path: Path to the OME-Zarr group.
        new_name: New name string to write.
    """
    zarr_group = zarr.open_group(str(zarr_path), mode="r+")
    attrs = zarr_group.attrs.asdict()
    attrs["multiscales"][0]["name"] = new_name
    zarr_group.attrs.update(attrs)
    logger.info(f"Updated multiscales name to '{new_name}'.")

def _update_multiscales_datasets(zarr_path: Path, pyramid_dict: dict[str, dict]) -> None:
    """
    Rewrite the ``datasets`` list in the ``multiscales`` metadata to match
    ``pyramid_dict`` (levels present on disk).

    Parameters:
        zarr_path: Path to the OME-Zarr group.
        pyramid_dict: Pyramid dictionary as returned by ``_estimate_pyramid_depth``
            or ``_get_pyramid_structure``.
    """
    zarr_group = zarr.open_group(str(zarr_path), mode="r+")
    attrs = zarr_group.attrs.asdict()
    axes = attrs["multiscales"][0]["axes"]

    new_datasets = []
    for level_str, info in pyramid_dict.items():
        scale = info["scale"]
        
        # OME-NGFF expects one scale entry per axis; prepend t-scale=1 if present
        n_axes = len(axes)
        if n_axes == 4:
            full_scale = [1.0] + list(scale)
        else:
            full_scale = list(scale)
        new_datasets.append({
            "path": level_str,
            "coordinateTransformations": [{"type": "scale", "scale": full_scale}],
        })

    attrs["multiscales"][0]["datasets"] = new_datasets
    zarr_group.attrs.update(attrs)
    logger.info(f"Updated multiscales datasets metadata ({len(new_datasets)} levels).")

def _check_level_complete(
    zarr_path: Path, 
    level: int
) -> bool:
    """
    Check whether all chunks of a given pyramid level are present on disk.

    Parameters:
        zarr_path: Path to the OME-Zarr group (e.g. ``plate.zarr/B/03/0``).
        level: Pyramid level index (0 = full resolution).

    Returns:
        ``True`` if every expected chunk file exists, ``False`` otherwise.
    """
    root = zarr.open(str(zarr_path), mode="r")
    zarr_array = root[str(level)]

    shape = zarr_array.shape
    chunks = zarr_array.chunks
    expected_chunks_per_axis = tuple(int(np.ceil(s / c)) for s, c in zip(shape, chunks))
    total_expected = int(np.prod(expected_chunks_per_axis))

    level_path = zarr_path / str(level)
    actual_chunks = [
        p for p in level_path.rglob("*")
        if p.is_file() and p.name not in (".zarray", ".zattrs", ".zgroup")
    ]
    return len(actual_chunks) == total_expected

def _rechunk_level(
    zarr_path: Path,
    level: int,
    new_chunksize: tuple[int, ...],
) -> None:
    """
    Rechunk a single pyramid level in-place by writing to a temporary array
    and atomically replacing the original.

    Parameters:
        zarr_path: Path to the OME-Zarr group.
        level: Pyramid level index.
        new_chunksize: Target chunk shape (c, z, y, x).
    """
    level_path = zarr_path / str(level)
    tmp_path = zarr_path / f"_tmp_rechunk_{level}"

    arr = da.from_zarr(str(level_path))
    arr_rechunked = arr.rechunk(new_chunksize)

    logger.info(f"Rechunking level {level}: {arr.chunksize} → {new_chunksize}")

    # Write to temp store
    tmp = zarr.open(
        str(tmp_path),
        shape=arr.shape,
        chunks=new_chunksize,
        dtype=arr.dtype,
        mode="w",
        dimension_separator="/",
        write_empty_chunks=False,
        fill_value=0,
    )
    z_chunk = new_chunksize[1]
    z_end = arr.shape[1]
    for z in range(0, z_end, z_chunk):
        logger.info(f"Progress: {z/z_end*100:.2f}%")
        region = (slice(None), slice(z, z + z_chunk), slice(None), slice(None))
        arr_rechunked[region].to_zarr(tmp, region=region, compute=True)

    # Atomic swap
    shutil.rmtree(str(level_path))
    shutil.move(str(tmp_path), str(level_path))
    logger.info(f"Rechunking level {level} complete.")

def rechunk_omezarr(
    zarr_path: Path, 
    new_chunksize: tuple[int, ...],
    num_levels: Optional[int] = None,
) -> None:
    """
    Rechunk every pyramid level of an OME-Zarr image.

    Parameters:
        zarr_path: Path to the OME-Zarr group.
        new_chunksize: Target chunk shape (c, z, y, x).
        num_levels: Number of pyramid levels to build.  If ``None`` (default)
    """
    image_meta = load_NgffImageMeta(str(zarr_path))
    scale = tuple(image_meta.get_pixel_sizes_zyx(level=0))
    old_num_levels = image_meta.num_levels
    if num_levels is None:
        num_levels = old_num_levels
    else:
        if num_levels < old_num_levels:
            for level in range(num_levels, old_num_levels):
                shutil.rmtree(str(zarr_path / str(level)))

    for level in range(min(num_levels, old_num_levels)):
        if not _check_level_complete(zarr_path, level) and level == 0: 
            raise ValueError(
                f"Level 0 of {zarr_path} is incomplete. "
                "Cannot rechunk a corrupted image, aborting task..."
            )
        elif not _check_level_complete(zarr_path, level) and level > 0:
            logger.info(f"Level {level} of {zarr_path} is incomplete, "
                        f"rebuilding level using level {level - 1} instead.")
            old_num_levels = level
            break
        _rechunk_level(zarr_path, level, new_chunksize)
    
    if num_levels > old_num_levels:
        src_array = zarr.open_array(str(zarr_path / "0"), mode="r")
        new_pyramid_dict = _estimate_pyramid_depth(
            shape=src_array.shape,
            scale=scale,
            num_levels=num_levels,
        )
        for level in range(old_num_levels, num_levels):
            _build_single_level(
                zarr_path, level, 
                channel_index=None, 
                pyramid_dict=new_pyramid_dict, 
                chunksize=new_chunksize)
    
    logger.info("Rechunking complete for all levels.")

def modify_omezarr_pyramid(
    zarr_path: Path,
    num_levels: int,
    chunksize: tuple[int, ...]
) -> None:
    """
    Add, remove, or consolidate pyramid levels to reach ``num_levels``.

    Rules
    -----
    - Level 0 must always be complete; raises ``ValueError`` otherwise.
    - Existing levels that are already complete are reused — they are not recomputed.
    - Incomplete levels are removed and rebuilt from the nearest complete
      lower level.
    - If ``num_levels`` is smaller than what currently exists, the extra
      levels are deleted.

    Parameters:
        zarr_path: Path to the OME-Zarr group.
        num_levels: Desired total number of levels (including level 0).
        chunksize: Chunk shape
    """
    image_meta = load_NgffImageMeta(str(zarr_path))
    old_num_levels = image_meta.num_levels
    scale = tuple(image_meta.get_pixel_sizes_zyx(level=0))
    full_res_shape = zarr.open_array(str(zarr_path / "0"), mode="r").shape 

    # Validate level 0 is complete
    if not _check_level_complete(zarr_path, 0):
        raise ValueError(
            f"Level 0 of {zarr_path} is incomplete. "
            "The full resolution OME-Zarr image is corrupted, aborting task..."
        )

    # Compute the desired pyramid structure
    new_pyramid_dict = _estimate_pyramid_depth(
        shape=full_res_shape,
        scale=scale,
        num_levels=num_levels,
    )

    # Remove levels beyond num_levels that currently exist
    if num_levels < old_num_levels:
        for level in range(num_levels, old_num_levels):
            level_path = zarr_path / str(level)
            if level_path.exists():
                logger.info(f"Removing surplus pyramid level {level}.")
                shutil.rmtree(str(level_path))

    # Determine which levels need to be (re)built
    for level in range(1, num_levels):
        level_path = zarr_path / str(level)
        if level_path.exists():
            
            # Check completeness
            is_complete = _check_level_complete(zarr_path, level)
            if not is_complete:
                logger.info(f"Level {level} is incomplete, "
                        f"rebuilding using level {level-1}...")
                shutil.rmtree(str(level_path))
                _build_single_level(
                    zarr_path, level, 
                    channel_index=None,
                    pyramid_dict=new_pyramid_dict, 
                    chunksize=chunksize
                )
        else:
            logger.info(f"Building pyramid level {level}...")
            _build_single_level(
                zarr_path, level, 
                    channel_index=None,
                    pyramid_dict=new_pyramid_dict, 
                    chunksize=chunksize
                )

    # Update metadata
    _update_multiscales_datasets(zarr_path, new_pyramid_dict)
    logger.info(f"Pyramid modification complete: {num_levels} levels.")


@validate_call
def modify_omezarr_structure(
    *,
    zarr_url: str,
    new_image_name: Optional[str] = None,
    chunksize: Optional[DimTuple] = None,
    num_levels: Optional[int] = None,
    channels_list: Optional[list[Channel]] = None,
) -> None:
    """
    Modify the structure of an existing OME-Zarr image.

    Parameters:
        zarr_url: Path of the image zarr group (e.g. ``some/path/plate.zarr/B/03/0``).
            new_image_name: If provided, changes the name of the OME-Zarr image. 
            Default: None.
        new_image_name: If provided, changes the name of the OME-Zarr image.
        chunksize: New chunk shape as ``(c, z, y, x)``. Default: None.
        num_levels: Desired total number of pyramid levels (including full resolution).
            Triggers pyramid modification (adding, removing, or consolidating
            levels).  Incomplete levels are detected automatically and rebuilt.
        channels_list: Modify the channels information such as channel name (e.g. `PGP9.5`, 
            `Lectin`,...), channel color (stored as hex color code: e.g. `FF0000`. 
            You can find hex color code on the website:
            https://www.color-hex.com/color-wheel/ ), or contrast limits. Default: None.

    Returns
        None
    """
    zarr_path = Path(zarr_url)
    logger.info(
        f"Start task `Modify OME-Zarr structure` for "
        f"{zarr_path.parent.name}/{zarr_path.name}"
    )

    # Set new chunk size if necessary
    new_chunksize = list(zarr.open_array(str(zarr_path / "0"), mode="r").chunks)
    if chunksize is not None:
        for d, dim in enumerate(["z", "y", "x"]):
            if chunksize[dim] is not None:
                new_chunksize[d+1] = chunksize[dim]

    cluster = None
    client = None
    try:
        cluster = _set_dask_cluster(n_workers = 1)
        client = Client(cluster)
        client.forward_logging(logger_name = "mesospim_fractal_tasks", level=logging.INFO)

        # Rechunk and modify pyramid if necessary
        if chunksize is not None:
            logger.info(f"Rechunking all pyramid levels to {new_chunksize}." +
                        (f"Pyramid levels set to {num_levels}." if num_levels is not None else ""))
            rechunk_omezarr(zarr_path, tuple(new_chunksize), num_levels=num_levels)

        # Pyramid modification only (no rechunking)
        if num_levels is not None and chunksize is None:
            logger.info(
                f"Modifying pyramid: target num_levels={num_levels}"
            )
            modify_omezarr_pyramid(zarr_path, num_levels, tuple(new_chunksize))
    
    finally:
        if client is not None:
            try:
                client.close(timeout="300s")  # give it more time
            except TimeoutError:
                pass

        if cluster is not None:
            try:
                cluster.close(timeout=300)
            except TimeoutError:
                pass

    # Channel updates
    zarr_group = zarr.open_group(str(zarr_path), mode="r+")
    image_attrs = zarr_group.attrs.asdict()
    if channels_list is not None:
        omero_update: dict[str, Any] = {}
        channel_labels = {channel.laser_wavelength: channel.label for channel in channels_list}
        print(channel_labels)
        for channel in image_attrs["acquisition_metadata"]["channels"]:
            print(channel)
            current_wavelength = channel["excitation_wavelength"]
            if current_wavelength in channel_labels.keys():
                channel["label"] = channel_labels[current_wavelength]
        
        for channel in image_attrs["omero"]["channels"]:
            current_wavelength = int(channel["wavelength_id"])
            if current_wavelength in channel_labels.keys():
                channel["label"] = channel_labels[current_wavelength]
        zarr_group.attrs.update(image_attrs)
    
        # _update_omero_channels expects a list indexed by channel position
        channel_order = {channel["label"]: str(i) for i, channel in enumerate(image_attrs["omero"]["channels"])}
        for channel in channels_list:
            omero_update["color"] = {}
            omero_update["window"] = {}
            if channel.color is not None:
                omero_update["color"][channel_order[channel.label]] = channel.color

            omero_update["window"][channel_order[channel.label]] = {}
            if channel.start_contrast is not None:
                omero_update["window"][channel_order[channel.label]]["start"] = channel.start_contrast
            if channel.end_contrast is not None:
                omero_update["window"][channel_order[channel.label]]["end"] = channel.end_contrast


        if omero_update:
            logger.info("Updating OMERO channel metadata.")
            _update_omero_channels(zarr_path, omero_update)

        # Rename image and update metadata
    if new_image_name is not None:
        _update_multiscales_name(zarr_path, new_image_name)
        shutil.move(str(zarr_path), str(zarr_path.parent / new_image_name))

    logger.info("Task `Modify OME-Zarr structure` complete.")


if __name__ == "__main__":

    from fractal_task_tools.task_wrapper import run_fractal_task
  
    run_fractal_task(
        task_function=modify_omezarr_structure,
        logger_name=logger.name,
    )
