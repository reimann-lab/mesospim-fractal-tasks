from typing import Optional, Any
from pathlib import Path
from pydantic import BaseModel, ConfigDict
import numpy as np
import logging
import dask.array as da
import json
from mesospim_fractal_tasks.utils.zarr_utils import _estimate_pyramid_depth

logger = logging.getLogger(__name__)

class Channel(BaseModel):
    """
    Channel information.

    Attributes:
        label: Channel label.
        laser_wavelength: Laser wavelength in nm.
        color: Channel color in hex format.
        start_contrast: Start contrast of the channel.
        end_contrast: End contrast of the channel.
    """
    label: str = "channel_name"
    laser_wavelength: int = 488
    color: Optional[str] = None
    start_contrast: Optional[float] = None
    end_contrast: Optional[float] = None

class DimTuple(BaseModel):
    """
    Tuple for the dimensions (Z, Y, X) used to retrieve numerical information per dimension.

    Attributes:
        z: Z dimension size.
        y: Y dimension size.
        x: X dimension size.
    """
    z: Optional[int] = None
    y: Optional[int] = None
    x: Optional[int] = None

    def get_dict(self):
        d = dict()
        for key, value in self.model_dump().items():
            if value is not None:
                d[key] = value
        return d

    def __getitem__(self, key: str) -> Optional[int]:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Optional[int]) -> None:
        setattr(self, key, value)

class IlluminationModel(BaseModel):
    """
    Illumination correction profiles.

    Attributes:
        flatfield: Flatfield image profile.
        darkfield: Darkfield image profile.
        baseline: Baseline value of the image.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    flatfield: Optional[np.ndarray | da.Array] = None
    darkfield: Optional[np.ndarray | da.Array] = None
    baseline: Optional[float] = 0

    def save_models(
        self,
        folder: str
    ) -> None:
        """
        Save illumination correction profiles to a folder as npz files.

        Args:
            folder (str): Folder name where the illumination
                correction profiles will be saved.
        """
        if self.flatfield is None:
            raise ValueError("Flatfield correction profile not found.")
        if type(self.flatfield) == da.Array:
            self.flatfield = self.flatfield.compute()
        if self.darkfield is None:
            logger.warning("Darkfield correction profile not found. "
                           "Saving only flatfield profile and baseline.")
            np.savez(
                Path(Path(folder), "profiles.npz"),
                flatfield=np.array(self.flatfield),
                baseline=np.array(self.baseline),
            )
        else:
            if type(self.darkfield) == da.Array:
                self.darkfield = self.darkfield.compute()
            np.savez(
                Path(Path(folder), "profiles.npz"),
                flatfield=np.array(self.flatfield),
                darkfield=np.array(self.darkfield),
                baseline=np.array(self.baseline),
            )

class BaSiCPyModelParams(BaseModel):
    """
    Advanced parameters for BaSiCPy illumination correction.

    Attributes:
        autosegment: When set to `True`, automatically segment the image before fitting.
            A threshold method is used and only the brighter pixels of the image
            are taken.
        autosegment_margin: Only meaningful when autosegment=True. It sets the margin of
            the segmentation mask to the thresholded region.
        smoothness_flatfield: Increase to obtain a smoother flatfield image.
        smoothness_darkfield: Increase to obtain a smoother darkfield image.
        get_darkfield: When True, will estimate the darkfield shading component.
        epsilon: Weight regularization term.
        max_workers: Maximum number of threads used for processing. Increase for
            faster processing.
        working_size: Maximal size in pixels of the XY plane analysed by BaSiCPy. If
            not set, there is no resizing of the image performed by BaSiCPy.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    autosegment: bool = False
    autosegment_margin: int = 10
    epsilon: float = 0.1
    get_darkfield: bool = True
    max_workers: int = 2
    smoothness_darkfield: float = 1.0
    smoothness_flatfield: float = 1.0
    working_size: Optional[list[int]] = None


class ProxyArray:
    """
    Array-like object that mimics (C,Z,Y,X) slicing but reads per-tile stores.
    """

    def __init__(
        self,
        *,
        source_path: Path,
        proxy_dask: da.Array,
        shape: tuple[int,int,int,int],
        dtype: np.dtype,
        pyramid_dict: dict[str, Any],
        requested_level: int | str = 0,
        requested_chunksize: tuple[int,int,int],
    ):
        self.source_path = Path(source_path)
        self._dask = proxy_dask
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.pyramid_dict = pyramid_dict
        self.requested_level = requested_level
        self.chunksize = requested_chunksize
        self.ndim = len(self.shape)

    @classmethod
    def open(
        cls,
        proxy_zarr_path: Path,
        requested_level: int | str = 0,
    ) -> "ProxyArray":

        proxy_manifest_path = Path(proxy_zarr_path / "proxy_manifest.json")
        manifest = json.load(open(proxy_manifest_path, "r"))
        if manifest.get("manifest", {}).get("type") != "mesospim_proxy_v1":
            raise ValueError("Not a mesospim proxy OME-Zarr")
        manifest = manifest["manifest"]

        source_zarr_path = Path(manifest["source_omezarr"])
        channels = manifest["channels"]
        nb_channels = len(channels)

        # Global fake mosaic extents derived from ROI table
        final_z_pixels = int(manifest["shape"][1])
        final_y_pixels = int(manifest["shape"][2])
        final_x_pixels = int(manifest["shape"][3])
        dtype = np.dtype(manifest["dtype"])
        pyramid_dict = manifest["pyramid"]
        chunksize = manifest["chunksize_zyx"]
        shape = (nb_channels, final_z_pixels, final_y_pixels, final_x_pixels)

        # Verify that requested level is available
        if int(requested_level) > (len(pyramid_dict)-1):
            level_to_build = requested_level
            requested_level = len(pyramid_dict)-1
            logger.info(f"Requested pyramid level {level_to_build} not available. "
                        f"Building pyramid level {level_to_build} using smallest available level"
                        f" {len(pyramid_dict)-1}.")
        else:
            level_to_build = requested_level

        # Create proxy dask array
        tiles = manifest["tiles"]
        dasks_per_channel = []
        for channel_tiles in tiles.values():
            col_grid = []
            for col in channel_tiles:
                row_grid = []
                for row in col:
                    row_grid.append(da.from_zarr(source_zarr_path / row["store_relpath"] / str(requested_level))[None,:,:,:])
                row_grid = da.concatenate(row_grid, axis=-1)
                col_grid.append(row_grid)
            col_grid = da.concatenate(col_grid, axis=-2)
            dasks_per_channel.append(col_grid)
        proxy_dask = da.concatenate(dasks_per_channel, axis=0)

        if level_to_build != requested_level:
            scale = pyramid_dict[str(0)]["scale"]
            desired_pyramid = _estimate_pyramid_depth(
                shape=shape,
                scale=scale,
                num_levels=int(level_to_build)+1 
            )
            coarsening_z = 0
            coarsening_xy = 0
            for level in range(int(requested_level), int(level_to_build)):
                coarsening_z += desired_pyramid[str(level)]["coarsening_z"]
                coarsening_xy += desired_pyramid[str(level)]["coarsening_xy"]
            proxy_dask = da.coarsen(
                np.mean,
                proxy_dask,
                {1: int(coarsening_z), 2: int(coarsening_xy), 3: int(coarsening_xy)},
                trim_excess=True,
            ).astype(proxy_dask.dtype)
            proxy_dask = proxy_dask.rechunk([1,] + chunksize)

        return cls(
            source_path=source_zarr_path,
            proxy_dask=proxy_dask,
            shape=shape,
            dtype=dtype,
            pyramid_dict=pyramid_dict,
            requested_level=requested_level,
            requested_chunksize=chunksize,
        )

    def get_dask(self) -> da.Array:
        return self._dask

    def __getitem__(self, key) -> da.Array:
        return self._dask[key]
