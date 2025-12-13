"""
Convert mesoSPIM data to OME-NGFF zarr array.
"""
from pathlib import Path
from typing import Optional, Callable, Any
import json
from importlib import resources
import fractal_tasks_core
from fractal_tasks_core.tasks.io_models import ChunkSizes
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.ngff.specs import NgffImageMeta
from fractal_tasks_core.roi import prepare_FOV_ROI_table, prepare_well_ROI_table
from fractal_tasks_core.tables import write_table
import numpy as np
import pandas as pd
from pydantic import validate_call
import zarr
import tifffile as tiff
import h5py
import psutil
import os

from mesospim_fractal_tasks.utils.zarr_utils import _determine_optimal_contrast
from mesospim_fractal_tasks import __version__, __commit__

__OME_NGFF_VERSION__ =  fractal_tasks_core.__OME_NGFF_VERSION__ 

import logging
logger = logging.getLogger(__name__)

def estimate_available_memory(mem_fraction: float = 0.3
) -> int:
    """
    Estimate the available memory on the system, whether it is SLURM or not.

    Returns:
        int: Estimated available memory in bytes.
    """
    slurm_mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    logger.info(f"SLURM_MEM_PER_NODE: {slurm_mem_per_node}")
    if slurm_mem_per_node is not None:
        slurm_mem_per_node = int(slurm_mem_per_node) * 1024**2
        nb_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
        if nb_cpus is None:
            nb_cpus = 1
            logger.warning(f"SLURM_CPUS_ON_NODE is not set. Assuming 1 CPU.")
        else:
            nb_cpus = int(nb_cpus)
        available_mem = slurm_mem_per_node * nb_cpus
    else:
        available_mem = psutil.virtual_memory().available
    available_fraction = available_mem * mem_fraction
    if available_fraction < 1e9:
        logger.warning(f"Available memory is less than 1GB. Consider increasing "
                       f"the available memory for the task.")
    return available_fraction

def load_channel_colors(
    user_channels_path: str = "default"
) -> dict:
    """
    Load the channel colors from a JSON file.

    Parameters:
        user_channels_path (str): Path to the JSON file or keyword identifying the JSON 
            file containing the channel colors information. 

    Returns:
        dict: Dictionary containing the channel colors.
    """
    if Path(user_channels_path).exists():
        user_channels_path = Path(user_channels_path)
        logger.info(f"Loading channel-specific information from {user_channels_path}.")
    else:
        keyword = user_channels_path
        settings_dir = resources.files("mesospim_fractal_tasks.settings")
        json_files = [file for file in settings_dir.iterdir() if ((file.is_file()) and 
                                                                  (keyword in file.name))]
        if len(json_files) != 1:
            logger.error(f"No JSON file found for the given parameter {user_channels_path}.")
            raise FileNotFoundError
        else:
            user_channels_path = settings_dir / json_files[0]

    with open(user_channels_path, "r") as f:
        return json.load(f)

def write_ome_zarr_metadata(
    zarr_group: zarr.Group,
    meta_df: pd.DataFrame,
    num_levels: int,
    coarsening_xy: int,
    contrast_limits: Optional[dict[str, dict[str, int]]] = None,
    input_param: dict[str, Any] = {},
    user_channels_path: str = "default",
) -> None:
    """
    Parameters: 
        zarr_group (zarr.Group): Zarr group to write metadata into.
        meta_df (pd.DataFrame): DataFrame containing metadata information.
        contrast_limits (tuple): Contrast window for all channels.
        num_levels (int): Number of pyramid levels (no pyramid if num_levels=1).
        coarsening_factor (int): Coarsening factor for the pyramid.
        input_param (dict[str, Any]): Input parameters for the task.
        user_channels_path (str): Path to the JSON file or keyword identifying the JSON 
            file containing the channel colors information.
    Returns:
        None
    """

    logger.info(f"Writing NGFF compliant metadata for {zarr_group.name }")

    channel_mapping = load_channel_colors(user_channels_path)
    
    try:
        
        # Prepare the dataset info with scale transformations
        logger.info("Preparing multiscales metadata...")
        dataset = [
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                    "type": "scale",
                    "scale": [
                        1.0,
                        meta_df.loc[0, "z_scale"],   
                        meta_df.loc[0, "y_scale"] * coarsening_xy**(level),
                        meta_df.loc[0, "x_scale"] * coarsening_xy**(level)
                        ]
                    }
                ]
            }
            for level in range(num_levels)
        ]

        # Write multiscales metadata
        zarr_group.attrs["multiscales"] = [{
            "version": __OME_NGFF_VERSION__,
            "name": zarr_group.name.lstrip("/"),
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": dataset,
        }]
        logger.info("Multiscales metadata written.")

        # Prepare the OMERO channel info
        logger.info("Preparing OMERO display metadata...")
        omero_channels = []
        for c, label in enumerate(meta_df["channel"].unique()):
            channel = channel_mapping[str(label)]
            if contrast_limits is None:
                contrast_end = (2**16 - 1)
                contrast_start = 0
            else:
                contrast_end = contrast_limits[str(c)]["end"]
                contrast_start = contrast_limits[str(c)]["start"]
            omero_channels.append({
                "active": True,
                "label": channel["label"],
                "wavelength_id": channel["laser_wavelength"],
                "color": channel["color"],
                "index": c,
                "window": {
                    "max": (2**16 - 1),
                    "end": contrast_end,
                    "start": contrast_start,
                    "min": 0
                },
            })

        # Write OMERO metadata
        zarr_group.attrs["omero"] = {
            "id": 1,
            "version": __OME_NGFF_VERSION__,
            "name": zarr_group.name.lstrip("/"),
            "channels": omero_channels,
            "rdefs": {
                "model": "color"
            }
        }
        logger.info("OMERO metadata written.")

        # Prepare the acquisition metadata
        logger.info("Writing acquisition metadata...")
        acquisition_info = []
        for label in meta_df["channel"].unique():
            channel_df = meta_df[meta_df["channel"] == label]
            channel_df.index = range(len(channel_df))
            acquisition_info.append({
                "label": channel_mapping[str(label)]["label"],
                "excitation_wavelength": channel_mapping[str(label)]["laser_wavelength"],
                "emission_wavelength": str(channel_df.loc[0, "filter"]),
                "shutter": str(channel_df.loc[0, "shutter"]),
                "zoom": str(channel_df.loc[0, "zoom"]),
                "laser_intensity": str(channel_df.loc[0, "intensity"])
            })
        
        # Write acquisition metadata
        zarr_group.attrs["acquisition_metadata"] = {
            "channels": acquisition_info
        }
        zarr_group.attrs["fractal_tasks"] = {
            "mesospim_to_omezarr": {
                "version": __version__.split("dev")[0][:-1],
                "commit": __commit__,
                "input_parameters": input_param,
            }
        }

        logger.info("Metadata writing complete! Zarr file is now NGFF compliant.")

        NgffImageMeta(**zarr_group.attrs)

    except Exception as e:
        logger.error(f"Failed to write metadata for group '{zarr_group.name}'")
        raise e

def check_n_pixels(
    meta_df: pd.DataFrame
) -> None:
    """
    Check if there is a positive overlap between tiles given the number of pixels 
    in the x and y directions referenced in the metadata.
    If swapping the x and y pixel sizes leads to positive overlap, it is corrected.

    Parameters:
        meta_df (pd.DataFrame): DataFrame containing metadata information.

    Returns:
        None
    """

    x_pixels = meta_df.loc[0, "x_n_pixels"]
    y_pixels = meta_df.loc[0, "y_n_pixels"]

    x_scale = meta_df.loc[0, "x_scale"]
    y_scale = meta_df.loc[0, "y_scale"]

    x_pos = meta_df["x_pos"].sort_values().unique()
    y_pos = meta_df["y_pos"].sort_values().unique()

    if len(x_pos) == 1 or len(y_pos) == 1:
        return

    x_overlap = x_pixels - (round(abs(x_pos[1] - x_pos[0]) / x_scale))
    y_overlap = y_pixels - (round(abs(y_pos[1] - y_pos[0]) / y_scale))


    if x_overlap < 0 or y_overlap < 0:
        logger.warning(f"Tiles have negative overlap!")
        logger.warning(f"Overlap in x direction: {x_overlap} pixels.")
        logger.warning(f"Overlap in y direction: {y_overlap} pixels.")
        swapped_x_overlap = y_pixels - (round(abs(x_pos[0] - x_pos[1]) / x_scale))
        swapped_y_overlap = x_pixels - (round(abs(y_pos[0] - y_pos[1]) / y_scale))
        if swapped_x_overlap >= 0 and swapped_y_overlap >= 0:
            logger.info(f"Overlap is non-negative if the x and y pixel sizes are "
                        f"swapped. It is likely it was wrongly referenced in the "
                        "metadata. Correcting...")
            meta_df["y_n_pixels"] = x_pixels
            meta_df["x_n_pixels"] = y_pixels

def read_metadata(
    metadata_path: Path,
    exclusion_list: list[int]
) -> pd.DataFrame:
    """
    Read structured metadata from a microscope metadata text file.

    The metadata text file contains acquisition parameters for each channel,
    including laser settings, pixel sizes, zoom, and z stack dimensions.

    Args:
        metadata_path (Path): Path to the metadata file.
        exclusion_list (list[int]): List of rows to exclude from the metadata.

    Returns:
        pd.DataFrame: DataFrame with parsed and typed metadata. Columns include:
            - channel (alias laser wavelength)
            - intensity
            - zoom, filter, shutter
            - x/y/z_scale: micrometer spacing
            - x/y/z_n_pixels: pixel dimensions
            - x/y_pos: position of the down-right corner of the image in micrometers
    """

    logger.info(f"Reading metadata from {metadata_path.name}")

    columns = ["laser", "intensity", "zoom", "filter", "shutter", "x_scale", "y_scale", 
                "z_scale", "x_pos", "y_pos",
                "x_n_pixels", "y_n_pixels", "z_n_pixels", "ignore"]
    meta_df = pd.DataFrame(columns=columns)

    # Define important keys present in the metadata
    keys = ["laser", "intensity", "zoom", "filter", "shutter", "pixelsize", "z_stepsize", 
            "z_planes", "x_pixels", "y_pixels", "x_pos", "y_pos"]
    
    # Initialize a row counter (= number of files referenced in the metadata
    nb_rows = -1
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.lower()
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace(" (%)", "")
            line = line.replace(" in um", "")
            line = line.replace("\n", "")
            contents = line.split(" ")
            key = contents[0].replace(":", "")
            if key == "metadata":
                nb_rows += 1
            if key in keys:
                if key == "x_pixels":
                    meta_df.loc[nb_rows, "x_n_pixels"] = contents[1]
                elif key == "y_pixels":
                    meta_df.loc[nb_rows, "y_n_pixels"] = contents[1]
                elif key == "pixelsize":
                    meta_df.loc[nb_rows, "x_scale"] = contents[1]
                    meta_df.loc[nb_rows, "y_scale"] = contents[1]
                elif key == "z_stepsize":
                    meta_df.loc[nb_rows, "z_scale"] = contents[1]
                elif key == "z_planes":
                    meta_df.loc[nb_rows, "z_n_pixels"] = contents[1]
                elif key == "zoom":
                    meta_df.loc[nb_rows, key] = contents[1].split("_")[0]
                elif key == "filter":
                    meta_df.loc[nb_rows, key] = " ".join(contents[1:])
                else:
                    meta_df.loc[nb_rows, key] = contents[1]
    if len(meta_df) == 0:
        raise ValueError("No metadata found in the provided file.")
    
    meta_df.rename(columns={"laser": "channel"}, inplace=True)

    # Correct dtypes
    meta_df[["intensity", 
            "x_pos", 
            "y_pos",
            "x_scale", 
            "y_scale", 
            "z_scale"]] = meta_df[["intensity", 
                                    "x_pos", 
                                    "y_pos",
                                    "x_scale", 
                                    "y_scale", 
                                    "z_scale"]].astype("float")
    meta_df[["x_n_pixels", 
            "y_n_pixels", 
            "z_n_pixels"]] = meta_df[["x_n_pixels", 
                                    "y_n_pixels", 
                                    "z_n_pixels"]].astype("int")
    if meta_df.loc[0, "intensity"] > 1:
        meta_df["intensity"] = meta_df["intensity"] / 100

    logger.info(f"Parsed metadata for {len(meta_df['channel'].unique())} channels.")

    # Exclude rows according to list from the metadata
    meta_df["ignore"] = False
    indices_to_exclude = []
    exclusion_set = set(exclusion_list)
    assert exclusion_set.issubset(meta_df.index), (
            "Exclusion list contains invalid indices: "
            f"{exclusion_set.difference(meta_df.index)}")
    for row in exclusion_set:
        indices_to_exclude += list(meta_df.groupby("channel").nth(row).index)
    for index in indices_to_exclude:
        meta_df.at[index, "ignore"] = True

    # Check for missing values in required columns
    required_columns = ["channel", "x_scale", "y_scale", "z_scale", "x_n_pixels",
                        "y_n_pixels", "z_n_pixels", "x_pos", "y_pos", "filter"]
    nan_cols = [c for c in required_columns if meta_df[c].isna().any()]
    if nan_cols:
        rows = meta_df[meta_df[nan_cols].isna().any(axis=1)]
        raise ValueError(
            f"Metadata contains missing values in required fields: {nan_cols}\n"
            f"Problematic rows:\n{rows}"
        )
    
    check_n_pixels(meta_df)

    return meta_df

def convert_raw(
    file_dir: str,
    basename: str,
    image_group: zarr.Group,
    image_path: str,
    meta_df: pd.DataFrame,
    chunk_sizes: ChunkSizes,
    mem_fraction: float = 0.3
) -> None:
    """
    Convert raw files that matches the basename in provided directory to zarr.

    Parameters:
        file_dir (str): Path to the directory containing the tiff files.
        basename (str): Common basename of the tiff files.
        image_group (zarr.Group): Image group to store the image from the tiff files.
        zarr_image_path (str): Filepath to the image group to store the tiff files.
        meta_df (pd.DataFrame): DataFrame containing metadata information.
        chunk_sizes (ChunkSizes): Chunk sizes for the zarr dataset.

    Returns:
        None
    """

    path = Path(file_dir)
    files = [file for file in path.rglob("*")
             if file.is_file() and basename in str(file)
             and (file.suffix == (".raw"))
             ]
    
    if len(files) == 0:
        logger.error(f"No raw files found for {basename} in {file_dir}")
        raise FileNotFoundError
    else:
        logger.info(f"Found {len(files)} raw files for {basename} in {file_dir}")
        if len(files) != len(meta_df):
            raise ValueError("Number of raw files does not match expected number of "
                             "files/channels extracted from metadata file "
                             f"({len(meta_df)} rows).")
        else:
            concat_filenames = ""
            for file in files:
                concat_filenames += file.stem
            channels = [meta_df.loc[i, "channel"] for i in range(len(meta_df))]
            for channel in channels:
                if channel not in concat_filenames:
                    raise ValueError("No raw file found that correspond to expected "
                                     f"channel {channel}.")
    
    zarr_shape = [len(files), meta_df.loc[0,"z_n_pixels"], meta_df.loc[0, "y_n_pixels"], 
                  meta_df.loc[0, "x_n_pixels"]]
    logger.info(f"Creating zarr dataset of size {zarr_shape[0]} x "
                f"{zarr_shape[1]} x {zarr_shape[2]} x "
                f"{zarr_shape[3]} to store raw files")
    logger.info(f"Chunk size set to: {chunk_sizes.get_chunksize()}")
    
    image_arr = zarr.create(
        shape=zarr_shape,
        chunks=chunk_sizes.get_chunksize(),
        dtype=np.uint16,
        store=zarr.storage.FSStore(Path(image_path, "0")),
        overwrite=True,
        dimension_separator="/",
    )
    
    for i, channel in enumerate(meta_df["channel"]):
        for file in files:
            if channel in file.stem:
                logger.info(f"Converting {file.name} to zarr")
                mmap_file = np.memmap(file, dtype=np.uint16, mode="r", shape=tuple(zarr_shape[1:]))
                for z in range(meta_df.loc[0, "z_n_pixels"]):
                    plane = mmap_file[z]
                    plane = plane[None, None, :, :]
                    region = (slice(i, i+1), slice(z, z+1), slice(None), slice(None))
                    image_arr[region] = plane
                logger.info(f"Converted {file} to zarr")
    logger.info(f"Converted {len(files)} raw files to zarr")

    logger.info(f"Writing FOV and well ROI table to {image_path}")
    roi_df = pd.DataFrame()
    roi_df.loc[0, "x_micrometer"] = 0
    roi_df.loc[0, "y_micrometer"] = 0
    roi_df.loc[0, "z_micrometer"] = 0
    roi_df.loc[0, "x_micrometer_original"] = 0
    roi_df.loc[0, "y_micrometer_original"] = 0
    roi_df.loc[0, "z_micrometer_original"] = 0
    roi_df.loc[0, "len_x_micrometer"] = meta_df.loc[0,"x_n_pixels"] * meta_df.loc[0,"x_scale"]
    roi_df.loc[0, "len_y_micrometer"] = meta_df.loc[0,"y_n_pixels"] * meta_df.loc[0,"y_scale"]
    roi_df.loc[0, "len_z_micrometer"] = meta_df.loc[0,"z_n_pixels"] * meta_df.loc[0,"z_scale"]
    roi_df.loc[0, "pixel_size_x"] = meta_df.loc[0,"x_scale"]
    roi_df.loc[0, "pixel_size_y"] = meta_df.loc[0,"y_scale"]
    roi_df.loc[0, "pixel_size_z"] = meta_df.loc[0,"z_scale"]
    roi_df.loc[0, "x_pixel"] = meta_df.loc[0,"x_n_pixels"]
    roi_df.loc[0, "y_pixel"] = meta_df.loc[0,"y_n_pixels"]
    roi_df.loc[0, "z_pixel"] = meta_df.loc[0,"z_n_pixels"]
    well_roi_table = prepare_well_ROI_table(roi_df)

    # Write AnnData tables into the `tables` zarr group
    write_table(
        image_group,
        "well_ROI_table",
        well_roi_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )

def convert_tiff(
    file_dir: str,
    basename: str,
    image_group: zarr.Group,
    image_path: str,
    meta_df: pd.DataFrame,
    chunk_sizes: ChunkSizes,
    mem_fraction: float = 0.3
) -> None:
    """
    Convert tiff files that matches the basename in provided directory to zarr.

    Parameters:
        file_dir (str): Path to the directory containing the tiff files.
        basename (str): Common basename of the tiff files.
        image_group (zarr.Group): Image group to store the image from the tiff files.
        image_path (str): Filepath to the image group to store the tiff files.
        meta_df (pd.DataFrame): DataFrame containing metadata information.
        chunk_sizes (ChunkSizes): Chunk sizes for the zarr dataset.

    Returns:
        None
    """

    path = Path(file_dir)
    files = [file for file in path.rglob("*")
             if file.is_file() and basename in str(file)
             and (file.suffix == (".tif") or file.suffix == (".tiff"))
             ]
    
    if len(files) == 0:
        logger.error(f"No tiff files found for {basename} in {file_dir}")
        raise FileNotFoundError
    else:
        logger.info(f"Found {len(files)} tiff files for {basename} in {file_dir}")
        if len(files) != len(meta_df):
            raise ValueError("Number of tiff files does not match expected number of "
                            "files/channels extracted from metadata file "
                            f"({len(meta_df)} rows).")
        else:
            concat_filenames = ""
            for file in files:
                concat_filenames += file.stem
            channels = [meta_df.loc[i, "channel"] for i in range(len(meta_df))]
            for channel in channels:
                if channel not in concat_filenames:
                    raise ValueError("No tiff file found that correspond to expected "
                                    f"channel {channel}.")

    logger.info(f"Creating zarr dataset of size {len(files)} x \
                {meta_df.loc[0,'z_n_pixels']} x {meta_df.loc[0, 'y_n_pixels']} x \
                {meta_df.loc[0, 'x_n_pixels']} to store tiff files")
    logger.info(f"Chunk size set to: {chunk_sizes.get_chunksize()}")
    
    image_arr = zarr.create(
        shape=(len(files), 
               meta_df.loc[0,"z_n_pixels"], 
               meta_df.loc[0, "y_n_pixels"], 
               meta_df.loc[0, "x_n_pixels"]),
        chunks=chunk_sizes.get_chunksize(),
        dtype=np.uint16,
        store=zarr.storage.FSStore(Path(image_path, "0")),
        overwrite=True,
        dimension_separator="/",
    )
    
    for i, channel in enumerate(meta_df["channel"]):
        for file in files:
            if channel in file.stem:
                logger.info(f"Converting {file.name} to zarr")
                for z in range(meta_df.loc[0, "z_n_pixels"]):
                    plane = tiff.imread(file, key=z)
                    plane = plane[None, None, :, :]
                    region = (slice(i, i+1), slice(z, z+1), slice(None), slice(None))
                    image_arr[region] = plane
                logger.info(f"Converted {file} to zarr")
    logger.info(f"Converted {len(files)} tiff files to zarr")

    logger.info(f"Writing FOV and well ROI table to {image_path}")
    roi_df = pd.DataFrame()
    roi_df.loc[0, "x_micrometer"] = 0
    roi_df.loc[0, "y_micrometer"] = 0
    roi_df.loc[0, "z_micrometer"] = 0
    roi_df.loc[0, "x_micrometer_original"] = 0
    roi_df.loc[0, "y_micrometer_original"] = 0
    roi_df.loc[0, "z_micrometer_original"] = 0
    roi_df.loc[0, "len_x_micrometer"] = meta_df.loc[0,"x_n_pixels"] * meta_df.loc[0,"x_scale"]
    roi_df.loc[0, "len_y_micrometer"] = meta_df.loc[0,"y_n_pixels"] * meta_df.loc[0,"y_scale"]
    roi_df.loc[0, "len_z_micrometer"] = meta_df.loc[0,"z_n_pixels"] * meta_df.loc[0,"z_scale"]
    roi_df.loc[0, "pixel_size_x"] = meta_df.loc[0,"x_scale"]
    roi_df.loc[0, "pixel_size_y"] = meta_df.loc[0,"y_scale"]
    roi_df.loc[0, "pixel_size_z"] = meta_df.loc[0,"z_scale"]
    roi_df.loc[0, "x_pixel"] = meta_df.loc[0,"x_n_pixels"]
    roi_df.loc[0, "y_pixel"] = meta_df.loc[0,"y_n_pixels"]
    roi_df.loc[0, "z_pixel"] = meta_df.loc[0,"z_n_pixels"]
    well_roi_table = prepare_well_ROI_table(roi_df)

    # Write AnnData tables into the `tables` zarr group
    write_table(
        image_group,
        "well_ROI_table",
        well_roi_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )

def convert_h5_multitile(
    file_dir: str,
    pattern: str,
    image_group: zarr.Group,
    image_path: str,
    meta_df: pd.DataFrame,
    chunk_sizes: ChunkSizes, 
    mem_fraction: float = 0.3
) -> None:
    """
    Convert tiles stored in an h5 file that matches the pattern in provided directory 
    to zarr.

    Parameters:
        file_dir (str): Path to the directory containing the h5 files.
        pattern (str): Common pattern of the h5 files.
        image_group (zarr.Group): Image group to store the image from the h5 file.
        image_path (str): Filepath to the image group to store the h5 file.
        meta_df (pd.DataFrame): DataFrame containing metadata information.
        chunk_sizes (ChunkSizes): Chunk sizes for the zarr dataset.

    Returns:
        None.
    """
    
    path = Path(file_dir)
    files = [file for file in path.rglob((f"*{pattern}*.h5").replace("**", "*")) \
             if file.is_file()]
    if len(files) == 0:
        logger.error(f"No h5 file matches pattern:\"{pattern}\" in {file_dir}.")
        raise FileNotFoundError
    elif len(files) > 1:
        logger.error(f"Found more than one h5 file matches for pattern:\"{pattern}\" "
                    f"in {file_dir}.")
        raise FileNotFoundError
    else:
        filename = files[0]
        logger.info(f"Found {filename.name} in {file_dir}.")

    nb_channels = len(meta_df["channel"].unique())
    logger.info(f"Found {nb_channels} channels in {filename.name}")

    tile_names = [f"t00000/s{i:02}/0/cells" for i in range(len(meta_df))]
    if len(tile_names) % nb_channels != 0:
        logger.error(f"The number of tiles and the number of channels don't match.")
        raise ValueError
    else:
        nb_tiles = len(tile_names) // nb_channels
        logger.info(f"Found {nb_tiles} tiles per channel in {filename.name}")

    z_scale = meta_df.loc[0, "z_scale"]
    y_scale = meta_df.loc[0, "y_scale"]
    x_scale = meta_df.loc[0, "x_scale"]

    z_pixels = meta_df.loc[0, "z_n_pixels"]
    y_pixels = meta_df.loc[0, "y_n_pixels"]
    x_pixels = meta_df.loc[0, "x_n_pixels"]
    final_y_pixels = meta_df[~meta_df["ignore"]].groupby("y_pos")["y_n_pixels"].unique().sum()[0] 
    final_x_pixels = meta_df[~meta_df["ignore"]].groupby("x_pos")["x_n_pixels"].unique().sum()[0] 

    available_mem = estimate_available_memory(mem_fraction=mem_fraction)
    necessary_mem = (x_pixels * y_pixels * z_pixels * 2)
    max_z_planes = min(z_pixels, int(z_pixels * available_mem / necessary_mem))
    logger.info(f"Based on available memory ({(available_mem/1e9):.2f}), the maximum "
                f"number of z planes loaded at once is set to {max_z_planes}.")

    logger.info(f"Chunk size set to: {chunk_sizes.get_chunksize()}")
    image_arr = zarr.create(
        shape=(nb_channels, z_pixels, final_y_pixels, final_x_pixels),
        chunks=chunk_sizes.get_chunksize(),
        dtype=np.uint16,
        store=zarr.storage.FSStore(Path(image_path, "0")),
        overwrite=True,
        dimension_separator="/",
    )
    logger.info(f"Creating zarr dataset of size {nb_channels} x" \
            f" {z_pixels} x {final_y_pixels} x {final_x_pixels} to store h5 tiles.")

    roi_df = pd.DataFrame()
    for c, channel in enumerate(meta_df["channel"].unique()):
        logger.info(f"Converting channel {channel}")
        channel_df = meta_df[meta_df["channel"] == channel]
        channel_df.index = tile_names[nb_tiles*c:nb_tiles*(c+1)]
        channel_df = channel_df[~channel_df["ignore"]]
        channel_df = channel_df.sort_values(by=["x_pos", "y_pos"], 
                                            ascending=[False, False])

        y_counter = 0
        x_counter = 0
        for t, tile_name in enumerate(channel_df.index):
            if not channel_df.iloc[t]["ignore"]:
                logger.info(f"Converting {tile_name} to zarr")
                y_pixels = channel_df.iloc[t]["y_n_pixels"]
                x_pixels = channel_df.iloc[t]["x_n_pixels"]

                n_zslices = 0
                with h5py.File(filename, "r") as f:
                    while n_zslices < z_pixels:
                        z_start = n_zslices
                        z_end = min(z_start + max_z_planes, z_pixels)
                        z_plane = f[tile_name][z_start:z_end,
                                               :y_pixels,
                                               :x_pixels]
                        z_plane = z_plane[None, :, :, :]
                        region = (slice(c, c+1), 
                            slice(z_start, z_end), 
                            slice(y_counter, y_counter + y_pixels), 
                            slice(x_counter, x_counter + x_pixels))
                        image_arr[region] = z_plane
                        n_zslices += max_z_planes
                logger.info(f"Converted {tile_name} to zarr") 
                
                if c == 0:
                    x_pos = abs(channel_df.iloc[t]["x_pos"] - channel_df["x_pos"].max())
                    y_pos = abs(channel_df.iloc[t]["y_pos"] - channel_df["y_pos"].max())
                    roi_df.loc[t, "z_micrometer"] = 0
                    roi_df.loc[t, "y_micrometer"] = y_counter * y_scale 
                    roi_df.loc[t, "x_micrometer"] = x_counter * x_scale 
                    roi_df.loc[t, "y_micrometer_original"] = y_pos 
                    roi_df.loc[t, "x_micrometer_original"] = x_pos 
                    roi_df.loc[t, "len_z_micrometer"] = z_pixels * z_scale
                    roi_df.loc[t, "len_y_micrometer"] = y_pixels * y_scale
                    roi_df.loc[t, "len_x_micrometer"] = x_pixels * x_scale
                    roi_df.loc[t, "x_pixel"] = x_pixels
                    roi_df.loc[t, "y_pixel"] = y_pixels
                    roi_df.loc[t, "z_pixel"] = z_pixels
                    roi_df.loc[t, "pixel_size_x"] = x_scale
                    roi_df.loc[t, "pixel_size_y"] = y_scale
                    roi_df.loc[t, "pixel_size_z"] = z_scale
                if t != (len(channel_df)-1):
                    if channel_df.iloc[t]["y_pos"] != channel_df.iloc[t+1]["y_pos"]:
                        y_counter += y_pixels
                        if y_counter == (final_y_pixels):
                            y_counter = 0
                    if channel_df.iloc[t]["x_pos"] != channel_df.iloc[t+1]["x_pos"]:
                        x_counter += x_pixels
                        if x_counter == (final_x_pixels):
                            x_counter = 0

    logger.info(f"Conversion completed.")
    logger.info(f"Writing FOV and well ROI table to {image_path}")
    fov_roi_table = prepare_FOV_ROI_table(roi_df)
    well_roi_table = prepare_well_ROI_table(roi_df)

    # Write AnnData tables into the `tables` zarr group
    write_table(
        image_group,
        "FOV_ROI_table",
        fov_roi_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )
    write_table(
        image_group,
        "well_ROI_table",
        well_roi_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )

def dispatcher(
    extension: str
) -> Callable[[str], None]:
    """
    Dispatch files to convert to the appropriate loader based on its extension.

    Parameters:
        extension (str): File extension of the files to convert.

    Returns:
        Callable[[str], None]: Function to convert the files.
    """

    if extension == "h5":
        logger.info(f"Dispatching files to h5 converter")
        return convert_h5_multitile
    elif extension == "tiff" or extension == "tif":
        logger.info(f"Dispatching files to tiff converter")
        return convert_tiff
    elif extension == "raw":
        logger.info(f"Dispatching files to raw converter")
        return convert_raw
    else:
        logger.error(f"Unsupported extension: {extension}")
        raise ValueError

@validate_call
def mesospim_to_omezarr(
    *,
    zarr_dir: str,
    pattern: str = "",
    extension: str = "h5",
    zarr_name: Optional[str] = None,
    image_name: Optional[str] = None,
    metadata_file: Optional[str] = None,
    channel_color_file: str = "default",
    exclusion_list: list[int] = [],
    num_levels: int = 6,
    coarsening_factor: int = 2,
    chunksize: tuple[int, int, int] = (32, 1024, 1024),
    overwrite: bool = True,
    mem_fraction: float = 0.3
) -> dict[str, Any]:
    """
    Convert mesoSPIM data (TIFFs or H5) to OME-NGFF zarr array.

    Parameters:
        zarr_dir (str): Path of the directory where the new OME-ZARR will be created. 
            (standard argument for Fractal tasks, managed by Fractal server).
        pattern (str): Common pattern to identify which files in the dataset directory 
            are to be converted (for example: if the files are image_name1.tiff, 
            image_name2.tiff, ... then pattern = image_name). Default: "".
        extension (str): File extension of the files to convert (currently support TIFF, 
            raw and H5 format). Default: "h5".
        zarr_name (Optional[str]): Name of the OME-Zarr to create/open. If not provided,
            the name of the dataset directory will be used. If the OME-Zarr already
            exists, the new image will be appended. The `overwrite` argument handles the
            overwriting or not of the image if it exists.
        image_name (Optional[str]): Name of the new image to be created. 
            Default: 'raw_image'.
        metadata_file (Optional[str]): Name of the metadata file. It is expected to be
            in the same folder as the acquisition files. Note: if not provided,
            a _meta.txt will be searched using the provided pattern.
        channel_color_file (str): Path to a JSON file or keyword identifying the JSON 
            file among provided defaults containing the channel colors information. 
            Default: "default".
        exclusion_list (list[int]): List of tiles to exclude from being converted, e.g.
            empty signal tiles.
        num_levels (int): Number of pyramid levels (including the full resolution level, 
            so if no pyramid, then num_levels=1).
        chunksize (tuple[int, int, int]): Chunk size to use for the OME-Zarr image.
            Default: (32, 1024, 1024).
        coarsening_factor (int): Coarsening factor to apply to the pyramid. Default: 2.
        overwrite (bool): Whether to overwrite OME-Zarr image if it already exists. 
            Default: True.
        mem_fraction (float): Fraction of available memory to use for conversion. 
            Default: 0.3

    Returns:
        None
    """
    logger.info(f"Starting task {__name__}. Zarr Directory: {zarr_dir}")

    # Check if the zarr directory exists
    if not Path(zarr_dir).exists():
        raise FileNotFoundError(f"Zarr directory {zarr_dir} does not exist.")

    # Determine the OME-Zarr path
    if zarr_name is not None:
        if zarr_name == "":
            raise ValueError("zarr_name cannot be empty.")
        zarr_path = Path(zarr_dir, zarr_name).with_suffix(".zarr")
    else:
        zarr_path = Path(zarr_dir, Path(zarr_dir).name).with_suffix(".zarr")

    # Create the OME-ZARR
    logger.info(f"Opening OME-ZARR: {zarr_path.name}")
    zarr_group = zarr.open_group(zarr_path, mode="a")
    
    # Open image group
    if image_name is None:
        image_name = "raw_image"
    image_group = zarr_group.create_group(image_name, overwrite=overwrite)
    image_path = Path(zarr_path, image_name)

    # Read the metadata 
    extension = extension.lower().replace(".", "")
    if metadata_file is None:
        metadata_path = []
        if extension == "tiff" or extension == "tif":
            extensions = ("tif", "tiff")
        else:
            extensions = (extension,)
        for ext in extensions:
            meta_pattern = f"*{pattern}*.{ext}_meta.txt".replace("**", "*")
            for path in Path(zarr_dir).rglob(meta_pattern):
                metadata_path.append(path)
        if len(metadata_path) != 1:
            logger.error(f"Unique metadata file for pattern="
                            f"\"{pattern}\" and"
                            f" {extension} not found in {zarr_dir}.")
            raise FileNotFoundError
        else:
            metadata_path = metadata_path[0]
            logger.info(f"Using {metadata_path.name} as metadata file.")
    else: 
        metadata_path = Path(zarr_dir, metadata_file)
        if not metadata_path.exists():
            logger.error(f"Metadata file {metadata_file} not found in {zarr_dir}.")
            raise FileNotFoundError
    
    # Convert files based on file extension
    convert_fn = dispatcher(extension)
    meta_df = read_metadata(metadata_path, exclusion_list)
    chunk_sizes = ChunkSizes()
    chunk_sizes.c = 1
    chunk_sizes.z = chunksize[0]
    chunk_sizes.y = chunksize[1]
    chunk_sizes.x = chunksize[2]
    convert_fn(zarr_dir, pattern, image_group, image_path, meta_df, 
               chunk_sizes, mem_fraction=mem_fraction)

    # Build the pyramid
    build_pyramid(
        zarrurl=str(image_path),
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_factor,
        chunksize=chunk_sizes.get_chunksize()
    )

    # Determine optimal contrast limits
    contrast_limits = _determine_optimal_contrast(image_path, 
                                                  num_levels, 
                                                  segment_sample=True)

    # Write OME-ZARR metadata
    write_ome_zarr_metadata(
        zarr_group=image_group,
        meta_df=meta_df,
        num_levels=num_levels,
        coarsening_xy=coarsening_factor,
        contrast_limits=contrast_limits,
        input_param=dict(pattern=pattern, 
                         extension=extension, 
                         metadata_file=metadata_file, 
                         exclusion_list=exclusion_list),
        user_channels_path=channel_color_file
    )

    # Update Fractal attributes metadata
    attributes = dict(image=image_name)

    # Update Fractal image list
    image_list_updates = [
        dict(
            zarr_url=str(image_path),
            attributes=attributes,
            types=dict(is_3D=True)
        )
    ]

    return {"image_list_updates": image_list_updates}

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=mesospim_to_omezarr, 
        logger_name=logger.name,
        )