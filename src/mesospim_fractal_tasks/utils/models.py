from typing import Optional
from pathlib import Path
from pydantic import BaseModel, ConfigDict
import numpy as np
from typing_extensions import Literal
from mesospim_fractal_tasks.utils.basicpy_nojax import FittingMode, ResizeMode
import logging

logger = logging.getLogger(__name__)


class DimTuple(BaseModel):
    """
    Tuple for the dimensions (Z, Y, X) used to retrieve numerical information per dimension.
    """
    z: float | None = None
    y: float | None = None
    x: float | None = None

    def get_dict(self):
        d = dict()
        for key, value in self.model_dump().items():
            if value is not None:
                d[key] = value
        return d

class IlluminationModel(BaseModel):
    """
    Illumination correction profiles.

    Attributes:
        flatfield: Flatfield correction profile.
        darkfield: Darkfield correction profile.
        baseline: Baseline correction profile.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    flatfield: Optional[np.ndarray] = None
    darkfield: Optional[np.ndarray] = None
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
        if self.darkfield is None:
            logger.warning("Darkfield correction profile not found. "
                           "Saving only flatfield profile and baseline.")
            np.savez(
                Path(Path(folder), "profiles.npz"),
                flatfield=np.array(self.flatfield),
                baseline=np.array(self.baseline),
            )
        else:
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
        autosegment: When not False, automatically segment the image before fitting.
            When True, threshold_otsu from scikit-image is used and the brighter pixels 
            are taken. When a callable is given, it is used as the segmentation 
            function.
        autosegment_margin: Margin of the segmentation mask to the thresholded region.
        epsilon: Weight regularization term.
        fitting_mode: Must be one of [‘ladmap’, ‘approximate’]
        get_darkfield: When True, will estimate the darkfield shading component.
        max_iterations: Maximum number of iterations for single optimization.
        max_mu_coef: Maximum allowed value of mu, divided by the initial value.
        max_reweight_iterations: Maximum number of reweighting iterations.
        max_reweight_iterations_baseline: Maximum number of reweighting 
            iterations for baseline.
        max_workers: Maximum number of threads used for processing.
        mu_coef: Coefficient for initial mu value.
        optimization_tol: Optimization tolerance.
        optimization_tol_diff: Optimization tolerance for update diff.
        resize_mode: Resize mode to use when downsampling images. 
            Must be one of ‘skimage’ or ‘skimage_dask’
        reweighting_tol: Reweighting tolerance in mean absolute difference of images.
        rho: Parameter rho for mu update.
        smoothness_darkfield: Weight of the darkfield term in the Lagrangian.
        smoothness_flatfield: Weight of the flatfield term in the Lagrangian.
        sort_intensity: Whether or not to sort the intensities of the image.
        sparse_cost_darkfield: Size for running computations. None means no rescaling.
        working_size: Maximal size in pixels of the XY plane analysed by BaSiCPy.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    autosegment: bool = False
    autosegment_margin: int = 10
    epsilon: float = 0.1
    fitting_mode: Literal[FittingMode.ladmap, 
                          FittingMode.approximate] = FittingMode.ladmap
    get_darkfield: bool = True
    max_iterations: int = 500
    max_mu_coef: float = 10000000.0
    max_reweight_iterations: int = 10
    max_reweight_iterations_baseline: int = 5
    max_workers: int = 2
    mu_coef: float = 12.5
    optimization_tol: float = 0.001
    optimization_tol_diff: float = 0.01
    resize_mode: Literal[ResizeMode.skimage, 
                         ResizeMode.skimage_dask] = ResizeMode.skimage_dask
    reweighting_tol: float = 0.01
    rho: float = 1.5
    smoothness_darkfield: float = 1.0
    smoothness_flatfield: float = 1.0
    sort_intensity: bool = False
    sparse_cost_darkfield: float = 0.01
    working_size: Optional[int] = None

