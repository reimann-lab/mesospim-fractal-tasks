"""Fractal multiview stitcher utils."""
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import dask.array as da
import zarr
import pandas as pd
import xarray as xr
import numpy as np
from numpy._typing import ArrayLike
from fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    OmeroChannel,
    get_channel_from_image_zarr,
    get_omero_channel_list,
)
from fractal_tasks_core.ngff import load_NgffImageMeta
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils
from spatial_image import to_spatial_image

DEFAULT_TRANSFORM_KEY = "affine_metadata"

logger = logging.getLogger(__name__)


# --- Monkey-patch get_sim_from_array ---
def patched_get_sim_from_array(
    array: ArrayLike, 
    dims: Optional[Union[list, tuple]] = None, 
    scale: Optional[dict] = None, 
    translation: Optional[dict] = None,
    affine: Optional[xr.DataArray] = None, 
    transform_key: str = DEFAULT_TRANSFORM_KEY,
    c_coords: Optional[Union[list, tuple, ArrayLike]] = None,
    t_coords: Optional[Union[list, tuple, ArrayLike]] = None
):
    """
    Get a spatial-image (multiview-stitcher flavor)
    from an array-like object.

    Parameters
    ----------
    array : ArrayLike
        Image data
    dims : Optional[Union[list, tuple]], optional
        Image dimension. Subset of ('t', 'c', 'z', 'y', 'x')
    scale : Optional[dict], optional
        Pixel spacing, e.g. {'z': 1.0, 'y': 0.3, 'x': 0.3}
    translation : Optional[dict], optional
        Image offset {'z': 50.0, 'y': 50. 'x': 50.}
    affine : Optional[xr.DataArray], optional
        Affine transform, e.g. xr.DataArray(np.eye(4), dims=["x_in", "x_out"])
    transform_key : str, optional
        By default DEFAULT_TRANSFORM_KEY
    c_coords : Optional[Union[list, tuple, ArrayLike]], optional
        Channel coordinates, e.g. ['DAPI', 'GFP', 'RFP']
    t_coords : Optional[Union[list, tuple, ArrayLike]], optional
        Time coordinates, e.g. [0.0, 0.2, 0.4, 0.6, 0.8]

    Returns
    -------
    spatial_image.SpatialImage
        spatial-image with multiview-stitcher flavor
        (SpatialImage + affine transform attributes)
    """


    # Ensure a Dask array is handed to xarray (lazy)
    if not isinstance(array, da.Array):
        if isinstance(array, zarr.Array):
            array = da.from_zarr(array)
        else:
            array = da.from_array(array, chunks="auto", asarray=False, lock=False)

    if dims is None:
        dims = ["t", "c", "z", "y", "x"][-array.ndim:]
    else:
        assert len(dims) == array.ndim

    xim = xr.DataArray(array, dims=dims)

    nsdims = ["c", "t"]
    for nsdim in nsdims:
        if nsdim not in xim.dims:
            xim = xim.expand_dims([nsdim])

    new_dims = [dim for dim in si_utils.SPATIAL_IMAGE_DIMS if dim in xim.dims]
    if new_dims != xim.dims:
        xim = xim.transpose(*new_dims, transpose_coords=False)

    spatial_dims = [dim for dim in xim.dims if dim in si_utils.SPATIAL_DIMS]
    ndim = len(spatial_dims)

    if scale is None:
        scale = {dim: 1 for dim in spatial_dims}
    if translation is None:
        translation = {dim: 0 for dim in spatial_dims}

    sim = si_utils.si.to_spatial_image(
        xim.data,
        dims=xim.dims,
        scale=scale,
        translation=translation,
        c_coords=c_coords,
        t_coords=t_coords,
    )

    if affine is None:
        affine_xr = si_utils.param_utils.identity_transform(ndim, t_coords=None)
    else:
        affine_xr = si_utils.param_utils.affine_to_xaffine(affine)

    si_utils.set_sim_affine(sim, affine_xr, transform_key=transform_key)
    return sim


def get_sim_from_multiscales(
    multiscales_path: Path,
    resolution: int = 0,
    chunks: Optional[tuple[int,int, int, int]] = None,
):
    """Get a spatial image from a multiscales ngff zarr file
    representing a given resolution level.

    Parameters
    ----------
    multiscales_path : Path
        Path to the multiscales group in the Zarr file.
    resolution : int, optional
        Resolution level index, by default 0

    Returns:
    -------
    spatial_image.SpatialImage
    """
    ngff_image_meta = load_NgffImageMeta(multiscales_path)
    axes = ngff_image_meta.axes_names
    spatial_dims = [dim for dim in axes if dim in ["z", "y", "x"]]
    scales = ngff_image_meta.pixel_sizes_zyx

    channel_names = [
        oc.label for oc in get_omero_channel_list(image_zarr_path=multiscales_path)
    ]

    data = da.from_zarr(f"{multiscales_path / Path(str(resolution))}")

    if chunks is not None:
        data = data.rechunk(chunks)

    sim = to_spatial_image(
        data,
        dims=axes,
        c_coords=channel_names,
        scale={dim: scales[resolution][idim] for idim, dim in enumerate(spatial_dims)},
        translation={dim: 0 for dim in spatial_dims},
    )

    return sim

def get_tiles_from_sim(
    xim_well,
    fov_roi_table: pd.DataFrame,
    transform_key: str = "fractal_input",
):
    """_summary_

    Parameters
    ----------
    xim_well : spatial_image.SpatialImage
        Array representing the well.
    fov_roi_table : pd.DataFrame
        Table with the FOV ROIs.

    Returns:
    -------
    list of multiscale_spatial_image (multiview-stitcher flavor)
    """
    input_spatial_dims = [dim for dim in xim_well.dims if dim in ["z", "y", "x"]]
    msims = []
    for _, row in fov_roi_table.iterrows():
        origin = {dim: row[f"{dim}_micrometer"] for dim in input_spatial_dims}
        extent = {dim: row[f"len_{dim}_micrometer"] for dim in input_spatial_dims}

        origin_original = {
            dim: row[f"{dim}_micrometer_original"] if dim != "z" else 0
            for dim in input_spatial_dims
        }

        tile = xim_well.sel(
            {
                dim: slice(origin[dim], origin[dim] + extent[dim] - 1e-6)
                for dim in input_spatial_dims
            }
        )

        tile = tile.squeeze(drop=True)

        sim = si_utils.get_sim_from_array(
            tile.data,
            dims=tile.dims,
            c_coords=xim_well.coords["c"].data,
            scale=si_utils.get_spacing_from_sim(tile),
            translation=origin_original,
            transform_key=transform_key,
        )

        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])

        msims.append(msim)

    return msims


class StitchingChannelInputModel(ChannelInputModel):
    """Channel input for stitching.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
    """

    def get_omero_channel(self, zarr_url) -> OmeroChannel:
        """Get the omero channel from the zarr url"""
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {e!s}"
            )
            return None


class PreRegistrationPruningMethod(Enum):
    """PreRegistrationPruningMethod Enum class

    Attributes:
        NOPRUNING: All overlapping tiles are used for registration.
        KEEPAXISALIGNED: Use only orthogonal tile pairs for registration.
            This excludes diagonal tile pairs and can lead to more robust
            registration results when tiles are not positioned irregularily.
        SHORTESTPATHSOVERLAPWEIGHTED: Only the tile pairs required to connect
            all tiles to an automatically determined reference tile are used
            for registration. Tile pairs with high overlaps are preferred.
    """

    NOPRUNING = "no_pruning"
    KEEPAXISALIGNED = "keep_axis_aligned"
    SHORTESTPATHSOVERLAPWEIGHTED = "shortest_paths_overlap_weighted"

    def get_pruning_method(self):
        """Get the pruning method to use

        Complex mapping for no pruning, see
        https://github.com/fractal-analytics-platform/fractal-web/issues/714
        for context. NOPRUNING should return a None, not the string.
        """
        return None if self == PreRegistrationPruningMethod.NOPRUNING else self.value