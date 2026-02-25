from typing import Optional
from pathlib import Path
from pydantic import BaseModel, ConfigDict
import numpy as np
import logging
import dask.array as da

logger = logging.getLogger(__name__)


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

