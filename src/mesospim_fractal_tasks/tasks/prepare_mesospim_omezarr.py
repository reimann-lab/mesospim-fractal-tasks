import json
from pathlib import Path
from typing import Any, Optional
import re
import pandas as pd
import zarr
from pydantic import validate_call
import logging

from fractal_tasks_core.roi import (
    prepare_FOV_ROI_table, prepare_well_ROI_table)
from fractal_tasks_core.tables import write_table

from mesospim_fractal_tasks.tasks.mesospim_to_omezarr import check_n_pixels, write_ome_zarr_metadata
from mesospim_fractal_tasks.utils.zarr_utils import _get_pyramid_structure, _estimate_pyramid_depth
from mesospim_fractal_tasks.utils.models import DimTuple

logger = logging.getLogger(__name__)


def update_translation_field(
    zarr_path: Path,
    y_translation: float,
    x_translation: float,
) -> None:
    """
    Remove the translation field from the metadata.

    Parameters:
        zarr_path (Path): Path to the OME-Zarr image.
    """
    zarr_group = zarr.open_group(zarr_path, mode="r+")
    attrs = zarr_group.attrs.asdict()
    datasets = attrs["multiscales"][0]["datasets"]
    for dataset in datasets:
        for t in dataset["coordinateTransformations"]:
            if t["type"] == "translation":
                t["translation"] = [0, y_translation, x_translation]
    attrs["multiscales"][0]["datasets"] = datasets
    zarr_group.attrs.put(attrs)
    logger.info("Removed translation field from metadata.")

def read_metadata_file(
    metadata_path: Path
) -> dict[str, Any]:
    """
    Read a metadata file and return a dictionary.
    """

    # Define important keys present in the metadata
    keys = ["laser", "intensity", "zoom", "filter", "shutter", "pixelsize", "z_stepsize",
            "z_planes", "x_pixels", "y_pixels", "x_pos", "y_pos"]

    metadata = {}

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
            if key in keys:
                if key == "x_pixels":
                    metadata["x_n_pixels"] = contents[1]
                elif key == "y_pixels":
                    metadata["y_n_pixels"] = contents[1]
                elif key == "pixelsize":
                    metadata["x_scale"] = contents[1]
                    metadata["y_scale"] = contents[1]
                elif key == "z_stepsize":
                    metadata["z_scale"] = contents[1]
                elif key == "z_planes":
                    metadata["z_n_pixels"] = contents[1]
                elif key == "zoom":
                    metadata[key] = contents[1].split("_")[0]
                elif key == "filter":
                    metadata[key] = " ".join(contents[1:])
                else:
                    metadata[key] = contents[1]
    if len(metadata.keys()) == 0:
        raise ValueError("No metadata found in the provided file.")

    return metadata

def find_per_tile_omezarr(
    zarr_dir: Path,
    root_zarr: Path
) -> pd.DataFrame:
    """
    Find tile stores. Returns a dataframe with at least:
      tile_id, channel, store_relpath

    Assumes naming: <root_zarr_name>*_Tile{t}_Ch{c}_*.ome.zarr_meta.txt.
    """

    # Quick check if number of metadata files matches number of tiles
    nb_tiles = len(list(root_zarr.glob("*.ome.zarr")))
    if nb_tiles == 0:
        raise ValueError(f"No per-tile *.ome.zarr stores found in {root_zarr}")

    # Search for per tile OME-Zarr in root OME_Zarr
    rows = []
    for p in sorted(zarr_dir.glob("*.ome.zarr_meta.txt")):
        m = re.search(r"Tile(\d+)_Ch(\d+)_.*", p.name)
        if not m:
            continue
        tile_omezarr_name = (p.name.split(".ome.zarr_")[1] + ".ome.zarr").replace(" ", "_")
        metadata_dict = read_metadata_file(p)
        metadata_dict["tile_id"] = int(m.group(1))
        channel = int(m.group(2))
        if channel != int(metadata_dict["laser"]):
            raise ValueError("Channel in filename and laser wavelength referenced in metadata do not match.")
        metadata_dict["tile_omezarr"] = tile_omezarr_name
        rows.append(metadata_dict)

    if len(rows) == 0:
        raise ValueError(f"No metadata txt tile found in the Zarr directory {zarr_dir}.")
    if len(rows) != nb_tiles:
        raise ValueError(f"Number of metadata txt tile found in the Zarr directory {len(rows)} "
                         f"does not match the number of tiles in the source image {nb_tiles}.")

    meta_df = pd.DataFrame(rows)
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
            "z_n_pixels",
            "channel"]] = meta_df[["x_n_pixels",
                                    "y_n_pixels",
                                    "z_n_pixels",
                                    "channel"]].astype("int")
    if meta_df.loc[0, "intensity"] > 1: # type: ignore
        meta_df["intensity"] = meta_df["intensity"] / 100

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

    return meta_df

def build_fov_roi_table(
    meta_df: pd.DataFrame,
    root_omezarr: Path
) -> pd.DataFrame:
    """
    Build the FOV ROI table to ensure compatibility with downstream tasks.
    """
    # Open one representative tile (channel 0) to read shape + pixel sizes (if available).
    channels = meta_df["channel"].unique()
    ch0_df = meta_df[meta_df["channel"] == meta_df["channel"].min()].copy()
    ch0_df = ch0_df.sort_values("tile_id")
    ch0_df = ch0_df.reset_index(drop=True)

    z_pixels, y_pixels, x_pixels = ch0_df.loc[0, "z_n_pixels"], ch0_df.loc[0, "y_n_pixels"], ch0_df.loc[0, "x_n_pixels"]
    final_y_pixels = ch0_df.groupby("y_pos")["y_n_pixels"].unique().sum()[0]
    final_x_pixels = ch0_df.groupby("x_pos")["x_n_pixels"].unique().sum()[0]

    z_scale = ch0_df.loc[0, "z_scale"]
    y_scale = ch0_df.loc[0, "y_scale"]
    x_scale = ch0_df.loc[0, "x_scale"]

    # Simple single-row layout; replace with stage-position based row/col if available
    x_counter = 0
    y_counter = 0

    roi_df = pd.DataFrame()
    for i, row in ch0_df.iterrows():
        x_pos = abs(row["x_pos"] - ch0_df.loc[0, "x_pos"])
        y_pos = abs(row["y_pos"] - ch0_df.loc[0, "y_pos"])
        roi_df.loc[i, "z_micrometer"] = 0.0
        roi_df.loc[i, "y_micrometer"] = y_counter * y_scale
        roi_df.loc[i, "x_micrometer"] = x_counter * x_scale
        roi_df.loc[i, "y_micrometer_original"] = y_pos
        roi_df.loc[i, "x_micrometer_original"] = x_pos
        roi_df.loc[i, "len_z_micrometer"] = z_pixels * z_scale
        roi_df.loc[i, "len_y_micrometer"] = y_pixels * y_scale
        roi_df.loc[i, "len_x_micrometer"] = x_pixels * x_scale
        roi_df.loc[i, "x_pixel"] = x_pixels
        roi_df.loc[i, "y_pixel"] = y_pixels
        roi_df.loc[i, "z_pixel"] = z_pixels
        roi_df.loc[i, "pixel_size_x"] = x_scale
        roi_df.loc[i, "pixel_size_y"] = y_scale
        roi_df.loc[i, "pixel_size_z"] = z_scale

        # Update translation field for all channels
        for channel in channels:
            ch_df = meta_df[meta_df["channel"] == channel]
            ch_df = ch_df.sort_values("tile_id")
            ch_df = ch_df.reset_index(drop=True)
            tile_path = root_omezarr / ch_df.iloc[i]["tile_omezarr"]
            update_translation_field(
                tile_path,
                y_translation = roi_df.loc[i, "y_micrometer"],
                x_translation = roi_df.loc[i, "x_micrometer"])

        if i != (len(ch0_df)-1):
            if ch0_df.iloc[i]["y_pos"] != ch0_df.iloc[i+1]["y_pos"]:
                y_counter += y_pixels
                if y_counter == (final_y_pixels):
                    y_counter = 0
            if ch0_df.iloc[i]["x_pos"] != ch0_df.iloc[i+1]["x_pos"]:
                x_counter += x_pixels
                if x_counter == (final_x_pixels):
                    x_counter = 0
    return roi_df

def build_proxy(
    proxy_image_path: Path,
    meta_df: pd.DataFrame,
    source_zarr_path: Path,
    chunksize: list[int]
) -> None:
    """
    Create proxy OME-Zarr folder with manifest in attrs and FOV_ROI_table as AnnData.
    """

    # Choose pixel size and z step from metadata (ref channel)
    ch0_df = meta_df[meta_df["channel"] == meta_df["channel"].min()].copy()
    z_planes = int(ch0_df["z_n_pixels"].max())
    y_pixels = int(ch0_df.loc[0, "y_n_pixels"].max())
    x_pixels = int(ch0_df.loc[0, "x_n_pixels"].max())
    n_rows = len(ch0_df["y_pos"].unique())
    n_cols = len(ch0_df["x_pos"].unique())

    # Build TileSpec list
    channels = [int(c) for c in meta_df["channel"].sort_values().unique()]
    tile_specs: dict[int, Any] = {}
    for c, channel in enumerate(channels):
        channel_spec: list[Any] = []
        ch_df = meta_df[meta_df["channel"] == channel].copy()
        ch_df = ch_df.sort_values("tile_id")
        ch_df = ch_df.set_index("tile_id")
        for row in range(n_rows):
            row_specs : list[Any] = []
            for col in range(n_cols):
                tile_id = int(n_rows*col + row)
                tile = ch_df.loc[tile_id]
                row_specs.append(
                    dict(
                        tile_id=tile_id,
                        channel_index=int(c),
                        channel_label=str(channel),
                        store_relpath=str(tile["tile_omezarr"]),
                        row=int(row),
                        col=int(col),
                    )
                )
            channel_spec.append(row_specs)
        tile_specs[channel] = channel_spec

    pyramid_dict = _get_pyramid_structure(source_zarr_path / ch0_df.loc[0, "tile_omezarr"])
    for level, level_dict in pyramid_dict.items():
        new_level_dict = dict(level_dict)
        new_level_dict["scale"] = [float(s) for s in level_dict["scale"]]
        pyramid_dict[level] = new_level_dict

    tile_array = zarr.open_array(source_zarr_path / ch0_df.loc[0, "tile_omezarr"] / "0", mode="r")
    dtype = str(tile_array.dtype)
    final_y_pixels = int(ch0_df.groupby("y_pos")["y_n_pixels"].unique().sum()[0])
    final_x_pixels = int(ch0_df.groupby("x_pos")["x_n_pixels"].unique().sum()[0])
    shape = (len(channels), int(z_planes), final_y_pixels, final_x_pixels)

    _ = zarr.open(
        str(proxy_image_path / "0"),
        shape=shape,
        dtype=dtype,
        write_empty_chunks=False,
        chunks=(1,) + tuple(chunksize),
        dimension_separator="/",
        fill_value=0,
        mode="w",
    )

    manifest = {
        "manifest":
        {
            "type": "mesospim_proxy_v1",
            "version": "1.0",
            "source_omezarr": str(source_zarr_path),
            "axes": ["c","z","y","x"],
            "tile_shape_zyx": [z_planes, y_pixels, x_pixels],
            "shape": list(shape),
            "dtype": dtype,
            "channels": list(channels),
            "pyramid": pyramid_dict,
            "chunksize_zyx": chunksize,
            "tiles": tile_specs,
        }
    }

    manifest_path = Path(proxy_image_path, "proxy_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return


@validate_call
def prepare_mesospim_omezarr(
    *,
    zarr_dir: str,
    pattern: str = "",
    zarr_name: Optional[str] = None,
    channel_color_settings: str = "default",
    num_levels: Optional[int] = None,
    chunksize: DimTuple = DimTuple(z=64, y=1024, x=1024),
    overwrite: bool = False
) -> None:
    """
    Create a tiny proxy OME-Zarr-like directory on disk to insure compatibility between analysis pipeline and
    the OME-Zarr structure output by mesoSPIM.

    Parameters:
        zarr_dir: Path to the OME-Zarr directory.
        pattern: Common pattern to identify which OME-Zarr in the dataset directory
            are to be prepared in case of several OME-Zarr present. Default: "".
        zarr_name: Name of the OME-Zarr to create/open. If not provided,
            the name of the dataset directory will be used. If the OME-Zarr already
            exists, the new image will be appended. The `overwrite` argument handles the
            overwriting or not of the image if it exists. Default: None.
        channel_color_settings: Keyword identifying the channel color settings
            among all saved settings. Default: "default".
        num_levels: Number of pyramid levels (including the full resolution level,
            so with no extra pyramid, the number of levels is 1). For a 1Tb dataset, it is
            recommended to have at least 6 levels. If not provided, the code will use
            the existing pyramid depth of the OME-Zarr if it is present, otherwise
            it will estimate the optimal depth based on the size of the image. Default: None.
            the optimal pyramid depth based on the size of the image. Default: None.
        chunksize: Chunk size to use for the OME-Zarr image. Smaller chunksizes improve
            visualisation smoothness but impairs processing efficiency.
            Default: (64, 1024, 1024).
        overwrite: Whether to overwrite OME-Zarr image if it already exists. It will
            not overwrite the OME-Zarr folder if it already exists. Default: False.
    """
    zarr_dir = Path(zarr_dir)
    logger.info(f"Starting task: `Prepare mesoSPIM OME-Zarr for analysis.`")

    # Find mesoSPIM OME-Zarr in zarr dir
    root_omezarr = list(Path(zarr_dir).glob(f"*{pattern}*.ome.zarr".replace("**", "*")))
    if len(root_omezarr) != 1:
        raise ValueError(f"Found {len(root_omezarr)} OME-Zarr in {zarr_dir} for given "
                         f"pattern {pattern}. Expected only one.")
    root_omezarr = root_omezarr[0]
    logger.info(f"Found mesoSPIM OME-Zarr {root_omezarr.name} in {zarr_dir}.")

    # Create fake ome zarr image for compatibility
    if zarr_name is None:
        zarr_name = root_omezarr.name.split("_Mag")[0]
    fake_zarr_path = Path(zarr_dir, zarr_name + ".zarr")
    fake_root = zarr.open_group(fake_zarr_path, mode="a")
    image_name = "fake_raw_image"
    fake_image_group = fake_root.create_group(image_name, overwrite=overwrite)

    # Discover tile stores and build metadata df
    meta_df = find_per_tile_omezarr(zarr_dir, root_omezarr)

    # Check the congruency of x and y number of pixels
    check_n_pixels(meta_df)

    # Rebuild ROI df and AnnData tables
    roi_df = build_fov_roi_table(meta_df, root_omezarr)
    fov_roi_table = prepare_FOV_ROI_table(roi_df)
    well_roi_table = prepare_well_ROI_table(roi_df)

    # Write AnnData tables into the `tables` zarr group
    write_table(
        fake_image_group,
        "FOV_ROI_table",
        fov_roi_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )
    write_table(
        fake_image_group,
        "well_ROI_table",
        well_roi_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )

    # Write multiscales metadata
    source_file_name = root_omezarr.name
    scale = (float(meta_df.loc[0, "z_scale"]), float(meta_df.loc[0, "y_scale"]), float(meta_df.loc[0, "x_scale"]))
    nb_channels = len(meta_df["channel"].unique())
    final_y_pixels = int(meta_df.groupby("y_pos")["y_n_pixels"].unique().sum()[0])
    final_x_pixels = int(meta_df.groupby("x_pos")["x_n_pixels"].unique().sum()[0])
    shape = (nb_channels, int(meta_df.loc[0, "z_n_pixels"]), final_y_pixels, final_x_pixels)

    if num_levels is None:
        pyramid_dict = _get_pyramid_structure(
            zarr_path=Path(root_omezarr, meta_df.loc[0, "tile_omezarr"]), # type: ignore
        )
        if len(pyramid_dict) == 1:
            pyramid_dict = _estimate_pyramid_depth(
                shape=shape,
                scale=scale
            )
    else:
        pyramid_dict = _estimate_pyramid_depth(
            shape=shape,
            scale=scale,
            num_levels=num_levels,
        )

    default_chunksize = [64, 1024, 1024]
    for d, dim in enumerate(["z", "y", "x"]):
        if chunksize[dim] is not None:
            default_chunksize[d] = chunksize[dim]

    write_ome_zarr_metadata(
        zarr_group=fake_image_group,
        meta_df=meta_df,
        pyramid_dict=pyramid_dict,
        contrast_limits=None,
        input_param=dict(pattern=pattern,
                         source_file=source_file_name,
                         channel_color_settings=channel_color_settings,
                         num_levels=num_levels,
                         chunksize=default_chunksize),
        user_channels_path=channel_color_settings,
        is_proxy=True
    )

    # Build manifest
    build_proxy(
        proxy_image_path=fake_zarr_path / image_name,
        meta_df=meta_df,
        source_zarr_path=root_omezarr,
        chunksize=default_chunksize,
    )


if __name__ == "__main__":

    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=prepare_mesospim_omezarr,
        logger_name=logger.name,
    )
