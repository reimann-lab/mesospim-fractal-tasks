from typing import Optional, Any
from pathlib import Path
from pydantic import BaseModel, ConfigDict
import numpy as np
import logging
import dask.array as da
import anndata as ad
import json
from dataclasses import dataclass

from fractal_tasks_core.roi import convert_ROI_table_to_indices

logger = logging.getLogger(__name__)

class Channel(BaseModel):
    """
    Channel information.

    Attributes:
        label: Channel label.
        laser_wavelength: Laser wavelength in nm.
        color: Channel color in hex format.
    """
    label: str
    laser_wavelength: int
    color: str
    max_intensity: Optional[float] = None
    start_contrast: Optional[float] = None
    end_contrast: Optional[float] = None
    min_intensity: Optional[float] = None

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


@dataclass(frozen=True)
class TileEntry:
    tile_id: int
    channel_index: int
    store_relpath: str


class ProxyArray:
    """
    Array-like object that mimics (C,Z,Y,X) slicing but reads per-tile stores.
    Designed to be compatible with code that does image_array[region] where
    region matches one tile bbox.
    """

    def __init__(
        self,
        *,
        source_path: Path,
        proxy_dask: da.Array,
        #tiles: dict[tuple[int,int], TileEntry],  # (tile_id, channel_index)->entry
        #bboxes: dict[tuple[int,int,int,int], int],  # (y start,y end,x start,x end)->tile_id
        shape: tuple[int,int,int,int],
        dtype: np.dtype,
        pyramid_dict: dict[str, Any],
        requested_level: int | str = 0,
        requested_chunksize: tuple[int,int,int],
    ):
        self.source_path = Path(source_path)
        self._dask = proxy_dask
        #self.tiles = tiles
        #self.bboxes = bboxes
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.pyramid_dict = pyramid_dict
        self.requested_level = requested_level
        self.chunksize = requested_chunksize

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

        # Load ROI table
        #FOV_ROI_table = ad.read_zarr(proxy_zarr_path / "tables" / "FOV_ROI_table")
        #indices = convert_ROI_table_to_indices(
        #    FOV_ROI_table,
        #    level=0,
        #    cols_xyz_pos= [
        #    "x_micrometer",
        #    "y_micrometer",
        #    "z_micrometer"],
        #    full_res_pxl_sizes_zyx=(5, 1, 1),
        #)

        # Build `zarr indices -> tile` map
        #bbox_map = {}
        #for i in range(len(indices)):
        #    _, _, y_s, y_e, x_s, x_e = indices[i][:]
        #    bbox_map[(int(y_s), int(y_e), int(x_s), int(x_e))] = int(i)

        # Global fake mosaic extents derived from ROI table
        final_z_pixels = manifest["shape"][1]
        final_y_pixels = manifest["shape"][2]
        final_x_pixels = manifest["shape"][3]
        dtype = np.dtype(manifest["dtype"])
        pyramid_dict = manifest["pyramid"]
        chunksize = manifest["chunksize_zyx"]
        shape = (nb_channels, final_z_pixels, final_y_pixels, final_x_pixels)

        # Verify that requested level is available
        if int(requested_level) > (len(pyramid_dict)-1):
            raise ValueError(f"Requested pyramid level {requested_level} not available."
                             f"Maximum available level is {len(pyramid_dict)-1}.")
        
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

        # Tiles mapping
        #tiles: dict[tuple[int,int], TileEntry] = {}
        #for tile in manifest["tiles"]:
        #    new_tile = TileEntry(
        #        tile_id=int(tile["tile_id"]),
        #        channel_index=int(tile["channel_index"]),
        #        store_relpath=str(tile["store_relpath"]),
        #    )
        #    tiles[(new_tile.tile_id, new_tile.channel_index)] = new_tile

        return cls(
            source_path=source_zarr_path,
            #tiles=tiles,
            proxy_dask=proxy_dask,
            #bboxes=bbox_map,
            shape=shape,
            dtype=dtype,
            pyramid_dict=pyramid_dict,
            requested_level=requested_level,
            requested_chunksize=chunksize,
        )

    def _open_tile_level(
        self, 
        tile_id: int, 
        ch_id: int,
    ) -> da.Array:
        pass
        #tile = self.tiles[(tile_id, ch_id)]
        #level = str(self.requested_level)
        
        #return da.from_zarr(self.source_path / tile.store_relpath / level)

    def __getitem__(self, key) -> da.Array:
        return self._dask[key]
    
    def __getitem__2(self, key):
        """
        Supports key = (slice(c,c+1), slice(z0,z1), slice(y0,y1), slice(x0,x1)).
        Returns a dask array with shape (1,Z,Y_tile,X_tile) (channel dim retained).
        """
        if not isinstance(key, tuple) or len(key) != 4:
            raise TypeError("Expected 4D slicing key (c,z,y,x)")

        c_sl, z_sl, y_sl, x_sl = key
        if not all(isinstance(s, slice) for s in key):
            raise TypeError("Only slice indexing is supported")

        c0 = 0 if c_sl.start is None else int(c_sl.start)
        c1 = self.shape[0] if c_sl.stop is None else int(c_sl.stop)
        if (c1 - c0) != 1:
            raise ValueError("ProxyArray expects a single-channel slice at a time")

        y0 = 0 if y_sl.start is None else int(y_sl.start)
        y1 = int(y_sl.stop)
        x0 = 0 if x_sl.start is None else int(x_sl.start)
        x1 = int(x_sl.stop)

        tile_id = self.bboxes.get((y0, y1, x0, x1))
        if tile_id is None:
            raise KeyError(
                "Requested (y,x) slice does not match a single tile bbox. "
                "This ProxyArray currently supports per-tile ROIs only."
            )

        tile = self._open_tile_level(tile_id, c0)  # (Z,Y,X)
        tile = tile[z_sl]  # apply Z slice
        tile = tile.rechunk(chunks=self.chunksize) # type: ignore

        return tile[None, ...]  # (1,Z,Y,X)