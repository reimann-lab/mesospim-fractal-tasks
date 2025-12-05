"""
BaSiC class (NumPy-only version)
--------------------------------

This module is a JAX-free adaptation of the BaSiC illumination correction model
originally provided by the `basicpy` package (PengLab).

It reproduces the core functionality of BaSiC — flatfield, darkfield, and baseline
correction — using only NumPy and SciPy operations for improved portability and
CPU-only environments.

Source: https://github.com/peng-lab/basicpy
Version: 1.2.0
Modifications: all JAX-specific components (jax.numpy, jax.lax, jit decorators,
device_put, and JaxDCT) have been replaced by equivalent NumPy/SciPy routines.
This allows BaSiC.fit() and related methods to run identically without having JAX 
as dependency.
"""

import logging
import os
import time
from enum import Enum
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial

# 3rd party modules
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ConfigDict
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_erosion
from skimage.transform import resize as skimage_resize
from scipy.fftpack import dct, idct

# Substitutejax term to numpy to avoid jax dependency
newax = np.newaxis
jnp = np

# Get number of available threads to limit CPU thrashing
# From preadator: https://pypi.org/project/preadator/
if hasattr(os, "sched_getaffinity"):
    # On Linux, we can detect how many cores are assigned to this process.
    # This is especially useful when running in a Docker container, when the
    # number of cores is intentionally limited.
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    # Default back to multiprocessing cpu_count, which is always going to count
    # the total number of cpus
    NUM_THREADS = cpu_count()

# initialize logger with the package name
logger = logging.getLogger(__name__)

class FittingMode(str, Enum):
    """Fit method enum."""

    ladmap: str = "ladmap"
    approximate: str = "approximate"

class ResizeMode(str, Enum):
    """Resize method enum."""

    #jax: str = "jax"
    skimage: str = "skimage"
    skimage_dask: str = "skimage_dask"

# multiple channels should be handled by creating a `basic` object for each channel
class BaSiC(BaseModel):
    """A class for fitting and applying BaSiC illumination correction profiles."""

    baseline: Optional[np.ndarray] = Field(
        None,
        description="Holds the baseline for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    darkfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the darkfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    fitting_mode: FittingMode = Field(
        FittingMode.ladmap, description="Must be one of ['ladmap', 'approximate']"
    )
    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    flatfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the flatfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    smoothness_flatfield: float = Field(
        1.0, description="Weight of the flatfield term in the Lagrangian."
    )
    smoothness_darkfield: float = Field(
        1.0, description="Weight of the darkfield term in the Lagrangian."
    )
    sparse_cost_darkfield: float = Field(
        0.01, description="Weight of the darkfield sparse term in the Lagrangian."
    )
    autosegment: bool = Field(
        False,
        description="When not False, automatically segment the image before fitting."
        "When True, `threshold_otsu` from `scikit-image` is used "
        "and the brighter pixels are taken."
        "When a callable is given, it is used as the segmentation function.",
    )
    autosegment_margin: int = Field(
        10,
        description="Margin of the segmentation mask to the thresholded region.",
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )
    max_reweight_iterations: int = Field(
        10,
        description="Maximum number of reweighting iterations.",
    )
    max_reweight_iterations_baseline: int = Field(
        5,
        description="Maximum number of reweighting iterations for baseline.",
    )
    max_workers: int = Field(
        NUM_THREADS,
        description="Maximum number of threads used for processing.",
        exclude=True,  # Don't dump to output json/yaml
    )
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    mu_coef: float = Field(12.5, description="Coefficient for initial mu value.")
    max_mu_coef: float = Field(
        1e7, description="Maximum allowed value of mu, divided by the initial value."
    )
    optimization_tol: float = Field(
        1e-3,
        description="Optimization tolerance.",
    )
    optimization_tol_diff: float = Field(
        1e-2,
        description="Optimization tolerance for update diff.",
    )
    resize_mode: ResizeMode = Field(
        ResizeMode.skimage,
        description="Resize mode to use when downsampling images. "
        + "Must be one of 'skimage', and 'skimage_dask'",
    )
    resize_params: Dict = Field(
        {},
        description="Parameters for the resize function when downsampling images.",
    )
    reweighting_tol: float = Field(
        1e-2,
        description="Reweighting tolerance in mean absolute difference of images.",
    )
    sort_intensity: bool = Field(
        False,
        description="Whether or not to sort the intensities of the image.",
    )
    working_size: Optional[Union[int, List[int]]] = Field(
        128,
        description="Size for running computations. None means no rescaling.",
    )

    # Private attributes for internal processing
    _score: float = PrivateAttr(None)
    _reweight_score: float = PrivateAttr(None)
    _weight: float = PrivateAttr(None)
    _weight_dark: float = PrivateAttr(None)
    _residual: float = PrivateAttr(None)
    _S: float = PrivateAttr(None)
    _B: float = PrivateAttr(None)
    _D_R: float = PrivateAttr(None)
    _D_Z: float = PrivateAttr(None)
    _smoothness_flatfield: float = PrivateAttr(None)
    _smoothness_darkfield: float = PrivateAttr(None)
    _sparse_cost_darkfield: float = PrivateAttr(None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    def debug_log_values(cls, values: Dict[str, Any]):
        """Use a validator to echo input values."""
        logger.debug("Initializing BaSiC with parameters:")
        for k, v in values.items():
            logger.debug(f"{k}: {v}")
        return values

    def __call__(
        self, images: np.ndarray, timelapse: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Shortcut for `BaSiC.transform`."""
        return self.transform(images, timelapse)

    def _resize(self, Im, target_shape):
        #if self.resize_mode == ResizeMode.jax:
        #    resize_params = dict(method=ResizeMethod.LINEAR)
        #    resize_params.update(self.resize_params)
        #    Im = device_put(Im).astype(jnp.float32)
         #   return jax_resize(Im, target_shape, **resize_params)

        if self.resize_mode == ResizeMode.skimage:
            Im = skimage_resize(
                np.array(Im), target_shape, preserve_range=True, **self.resize_params
            )
            return Im.astype(np.float32)

        elif self.resize_mode == ResizeMode.skimage_dask:
            assert np.array_equal(target_shape[:-2], Im.shape[:-2])
            import dask.array as da

            Im = (
                da.from_array(
                    [
                        skimage_resize(
                            np.array(Im[tuple(inds)]),
                            target_shape[-2:],
                            preserve_range=True,
                            **self.resize_params,
                        )
                        for inds in np.ndindex(Im.shape[:-2])
                    ]
                )
                .reshape((*Im.shape[:-2], *target_shape[-2:]))
                .compute()
            )
            return Im.astype(np.float32)

    def _resize_to_working_size(self, Im):
        """Resize the images to the working size."""
        if self.working_size is not None:
            if np.isscalar(self.working_size):
                working_shape = [self.working_size] * (Im.ndim - 2)
            else:
                if not Im.ndim - 2 == len(self.working_size):
                    raise ValueError(
                        "working_size must be a scalar or match the image dimensions"
                    )
                else:
                    working_shape = self.working_size
            target_shape = [*Im.shape[:2], *working_shape]
            Im = self._resize(Im, target_shape)

        return Im

    def _perform_segmentation(self, Im):
        """Perform segmentation on the images."""
        if not self.autosegment:
            return np.ones_like(Im, dtype=bool)
        elif self.autosegment is True:
            th = threshold_otsu(Im)
            mask = Im < th
            return np.array(
                [binary_erosion(m, ball(self.autosegment_margin)) for m in mask]
            )
        else:
            return self.autosegment(Im)

    def fit(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning=False,
    ) -> None:
        """Generate illumination correction profiles from images.

        Args:
            images: Input images to fit shading model.
                    Must be 3-dimensional or 4-dimensional array
                    with dimension of (T,Y,X) or (T,Z,Y,X).
                    T can be either of time or mosaic position.
                    Multichannel images should be
                    independently corrected for each channel.
            fitting_weight: Relative fitting weight for each pixel.
                    Higher value means more contribution to fitting.
                    Must has the same shape as images.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.

        Example:
            >>> from basicpy import BaSiC
            >>> from basicpy import datasets as bdata
            >>> images = bdata.wsi_brain()
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        """
        ndim = images.ndim
        if images.ndim == 3:
            images = images[:, np.newaxis, ...]
            if fitting_weight is not None:
                fitting_weight = fitting_weight[:, np.newaxis, ...]
        elif images.ndim == 4:
            if self.fitting_mode == FittingMode.approximate:
                raise ValueError(
                    "Only 2-dimensional images are accepted for the approximate mode."
                )
        else:
            raise ValueError(
                "Images must be 3 or 4-dimensional array, "
                + "with dimension of (T,Y,X) or (T,Z,Y,X)."
            )

        if images.shape[-1] < 10 and not skip_shape_warning:
            logger.warning(
                "Image last dimension is less than 10. "
                + "Are you supplying images with the channel dimension?"
                + "Multichannel images should be "
                + "independently corrected for each channel."
            )

        if fitting_weight is not None and fitting_weight.shape != images.shape:
            raise ValueError("fitting_weight must have the same shape as images.")

        logger.info("=== BaSiC fit started ===")
        start_time = time.monotonic()

        Im = self._resize_to_working_size(images)

        if fitting_weight is not None:
            Ws = fitting_weight.astype(np.float32)
            Ws = self._resize_to_working_size(Ws)
            # normalize relative weight to 0 to 1
            Ws_min = np.min(Ws)
            Ws_max = np.max(Ws)
            Ws = (Ws - Ws_min) / (Ws_max - Ws_min)
        else:
            Ws = np.ones_like(Im)

        Ws = Ws * self._perform_segmentation(Im)

        # Im2 and Ws2 will possibly be sorted
        if self.sort_intensity:
            inds = np.argsort(Im, axis=0)
            Im2 = np.take_along_axis(Im, inds, axis=0)
            Ws2 = np.take_along_axis(Ws, inds, axis=0)
        else:
            Im2 = Im
            Ws2 = Ws

        if self.fitting_mode == FittingMode.approximate:
            mean_image = np.mean(Im2, axis=0)
            mean_image = mean_image / np.mean(Im2)
            mean_image_dct = JaxDCT.dct3d(mean_image.T)
            self._smoothness_flatfield = (
                np.sum(np.abs(mean_image_dct)) / 800 * self.smoothness_flatfield
            )
            self._smoothness_darkfield = (
                self._smoothness_flatfield * self.smoothness_darkfield / 2.5
            )
            self._sparse_cost_darkfield = (
                self._smoothness_darkfield * self.sparse_cost_darkfield * 100
            )
        else:
            self._smoothness_flatfield = self.smoothness_flatfield
            self._smoothness_darkfield = self.smoothness_darkfield
            self._sparse_cost_darkfield = self.sparse_cost_darkfield

        logger.debug(f"_smoothness_flatfield set to {self._smoothness_flatfield}")
        logger.debug(f"_smoothness_darkfield set to {self._smoothness_darkfield}")
        logger.debug(f"_sparse_cost_darkfield set to {self._sparse_cost_darkfield}")

        # spectral_norm = jnp.linalg.norm(Im.reshape((Im.shape[0], -1)), ord=2)
        _temp = np.linalg.svd(Im2.reshape((Im2.shape[0], -1)), full_matrices=False)
        spectral_norm = _temp[1][0]

        if self.fitting_mode == FittingMode.approximate:
            init_mu = self.mu_coef / spectral_norm
        else:
            init_mu = self.mu_coef / spectral_norm / np.prod(Im2.shape)
        fit_params = self.dict()
        fit_params.update(
            dict(
                smoothness_flatfield=self._smoothness_flatfield,
                smoothness_darkfield=self._smoothness_darkfield,
                sparse_cost_darkfield=self._sparse_cost_darkfield,
                # matrix 2-norm (largest sing. value)
                init_mu=init_mu,
                max_mu=init_mu * self.max_mu_coef,
                D_Z_max=np.min(Im2),
                image_norm=np.linalg.norm(Im2.flatten(), ord=2),
            )
        )

        # Initialize variables
        W = np.ones_like(Im2, dtype=np.float32) * Ws2
        W_D = np.ones(Im2.shape[1:], dtype=np.float32)
        last_S = None
        last_D = None
        S = None
        D = None
        B = None

        if self.fitting_mode == FittingMode.ladmap:
            fitting_step = LadmapFit(**fit_params)
        else:
            fitting_step = ApproximateFit(**fit_params)

        for i in range(self.max_reweight_iterations):
            logger.debug(f"reweighting iteration {i}")
            if self.fitting_mode == FittingMode.approximate:
                S = jnp.zeros(Im2.shape[1:], dtype=jnp.float32)
            else:
                S = jnp.median(Im2, axis=0)
            D_R = jnp.zeros(Im2.shape[1:], dtype=jnp.float32)
            D_Z = 0.0
            if self.fitting_mode == FittingMode.approximate:
                B = jnp.ones(Im2.shape[0], dtype=jnp.float32)
            else:
                B = jnp.ones(Im2.shape[0], dtype=jnp.float32)
            I_R = jnp.zeros(Im2.shape, dtype=jnp.float32)
            S, D_R, D_Z, I_R, B, norm_ratio, converged = fitting_step.fit(
                Im2,
                W,
                W_D,
                S,
                D_R,
                D_Z,
                B,
                I_R,
            )
            logger.debug(f"single-step optimization score: {norm_ratio}.")
            logger.debug(f"mean of S: {float(jnp.mean(S))}.")
            self._score = norm_ratio
            if not converged:
                logger.debug("single-step optimization did not converge.")
            if S.max() == 0:
                logger.error(
                    "Estimated flatfield is zero. "
                    + "Please try to decrease smoothness_darkfield."
                )
                raise RuntimeError(
                    "Estimated flatfield is zero. "
                    + "Please try to decrease smoothness_darkfield."
                )
            self._S = S
            self._D_R = D_R
            self._B = B
            self._D_Z = D_Z
            D = fitting_step.calc_darkfield(S, D_R, D_Z)  # darkfield
            mean_S = jnp.mean(S)
            S = S / mean_S  # flatfields
            B = B * mean_S  # baseline
            I_B = B[:, newax, newax, newax] * S[newax, ...] + D[newax, ...]
            W = fitting_step.calc_weights(I_B, I_R) * Ws2
            W_D = fitting_step.calc_dark_weights(D_R)

            self._weight = W
            self._weight_dark = W_D
            self._residual = I_R

            logger.debug(f"Iteration {i} finished.")
            if last_S is not None:
                mad_flatfield = jnp.sum(jnp.abs(S - last_S)) / jnp.sum(np.abs(last_S))
                if self.get_darkfield:
                    mad_darkfield = jnp.sum(jnp.abs(D - last_D)) / max(
                        jnp.sum(jnp.abs(last_D)), 1
                    )  # assumes the amplitude of darkfield is more than 1
                    self._reweight_score = max(mad_flatfield, mad_darkfield)
                else:
                    self._reweight_score = mad_flatfield
                logger.debug(f"reweighting score: {self._reweight_score}")
                logger.info(
                    f"Iteration {i} elapsed time: "
                    + f"{time.monotonic() - start_time} seconds"
                )

                if self._reweight_score <= self.reweighting_tol:
                    logger.info("Reweighting converged.")
                    break
            if i == self.max_reweight_iterations - 1:
                logger.warning("Reweighting did not converge.")
            last_S = S
            last_D = D

        if not converged:
            logger.warning(
                "Single-step optimization did not converge "
                + "at the last reweighting step."
            )

        assert S is not None
        assert D is not None
        assert B is not None

        if self.sort_intensity:
            for i in range(self.max_reweight_iterations_baseline):
                B = jnp.ones(Im.shape[0], dtype=jnp.float32)
                if self.fitting_mode == FittingMode.approximate:
                    B = jnp.mean(Im, axis=(1, 2, 3))
                I_R = jnp.zeros(Im.shape, dtype=jnp.float32)
                logger.debug(f"reweighting iteration for baseline {i}")
                I_R, B, norm_ratio, converged = fitting_step.fit_baseline(
                    Im,
                    W,
                    S,
                    D,
                    B,
                    I_R,
                )

                I_B = B[:, newax, newax, newax] * S[newax, ...] + D[newax, ...]
                W = fitting_step.calc_weights_baseline(I_B, I_R) * Ws
                self._weight = W
                self._residual = I_R
                logger.debug(f"Iteration {i} finished.")

        self.flatfield = skimage_resize(S, images.shape[1:])
        self.darkfield = skimage_resize(D, images.shape[1:])
        if ndim == 3:
            self.flatfield = self.flatfield[0]
            self.darkfield = self.darkfield[0]
        self.baseline = B
        logger.info(
            f"=== BaSiC fit finished in {time.monotonic()-start_time} seconds ==="
        )






def dct2d(x):
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2d(x):
    return idct(idct(x, axis=0, norm='ortho'), axis=1, norm='ortho')

def dct3d(x):
    return dct(dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho'),
               axis=2, norm='ortho')

def idct3d(x):
    return idct(idct(idct(x, axis=0, norm='ortho'), axis=1, norm='ortho'),
               axis=2, norm='ortho')

def _jshrinkage(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)


class BaseFit(BaseModel):
    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    max_mu: float = Field(0, description="The maximum value of mu.")
    init_mu: float = Field(0, description="Initial value for mu.")
    D_Z_max: float = Field(0, description="Maximum value for D_Z.")
    image_norm: float = Field(0, description="The 2nd order norm for the images.")
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    optimization_tol: float = Field(
        1e-6,
        description="Optimization tolerance.",
    )
    optimization_tol_diff: float = Field(
        1e-6,
        description="Optimization tolerance for update diff.",
    )
    smoothness_darkfield: float = Field(
        0.0,
        description="Darkfield smoothness weight for sparse reguralization.",
    )
    sparse_cost_darkfield: float = Field(
        0.0,
        description="Darkfield sparseness weight for sparse reguralization.",
    )
    smoothness_flatfield: float = Field(
        0.0,
        description="Flatfield smoothness weight for sparse reguralization.",
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )
    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
    )

    def _cond(self, vals):
        k = vals[0]
        fit_residual = vals[-2]
        value_diff = vals[-1]
        norm_ratio = jnp.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm
        conv = jnp.any(
            jnp.array(
                [
                    norm_ratio > self.optimization_tol,
                    value_diff > self.optimization_tol_diff,
                ]
            )
        )
        return jnp.all(
            jnp.array(
                [
                    conv,
                    k < self.max_iterations,
                ]
            )
        )

    def _fit_jit(
        self,
        Im,
        W,
        W_D,
        S,
        D_R,
        D_Z,
        B,
        I_R,
    ):
        # initialize values
        Y = jnp.zeros_like(Im, dtype=jnp.float32)
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf
        value_diff = jnp.inf

        vals = (0, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff)
        step = partial(
            self._step,
            Im,
            W,
            W_D,
        )
        while self._cond(vals):
            vals = step(vals)
        #vals = lax.while_loop(self._cond, step, vals)
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff = vals
        norm_ratio = jnp.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm
        return S, D_R, D_Z, I_R, B, norm_ratio, k < self.max_iterations

    def _fit_baseline_jit(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
    ):
        # initialize values
        Y = jnp.zeros_like(Im, dtype=jnp.float32)
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf
        value_diff = jnp.inf

        vals = (0, I_R, B, Y, mu, fit_residual, value_diff)
        step = partial(
            self._step_only_baseline,
            Im,
            W,
            S,
            D,
        )

        while self._cond(vals):
            vals = step(vals)
        #vals = lax.while_loop(self._cond, step, vals)
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        norm_ratio = jnp.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm
        return I_R, B, norm_ratio, k < self.max_iterations

    def fit(
        self,
        Im,
        W,
        W_D,
        S,
        D_R,
        D_Z,
        B,
        I_R,
    ):
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D_R.shape != Im.shape[1:]:
            raise ValueError("D_R must have the same shape as images.shape[1:]")
        if not jnp.isscalar(D_Z):
            raise ValueError("D_Z must be a scalar.")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")
        if W_D.shape != Im.shape[1:]:
            raise ValueError(
                "darkfield weight must have the same shape as images.shape[1:]"
            )
        return self._fit_jit(Im, W, W_D, S, D_R, D_Z, B, I_R)

    def fit_baseline(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool]:
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D.shape != Im.shape[1:]:
            raise ValueError("D must have the same shape as images.shape[1:]")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")
        return self._fit_baseline_jit(Im, W, S, D, B, I_R)

    def tree_flatten(self):
        # all of the fields are treated as "static" values for JAX
        children = []
        aux_data = self.dict()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, _children):
        return cls(**aux_data)

class LadmapFit(BaseFit):
    def _step(
        self,
        Im,
        weight,
        dark_weight,
        vals,
    ):
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff = vals
        T_max = Im.shape[0]

        I_B = S[newax, ...] * B[:, newax, newax, newax] + D_R[newax, ...] + D_Z
        eta_S = jnp.sum(B**2) * 1.02 + 0.01
        S_new = (
            S
            + jnp.sum(B[:, newax, newax, newax] * (Im - I_B - I_R + Y / mu), axis=0)
            / eta_S
        )
        S_new = idct3d(
            _jshrinkage(dct3d(S_new), self.smoothness_flatfield / (eta_S * mu))
        )
        S_new = jnp.where(S_new.min() < 0, S_new - S_new.min(), S_new)
        dS = S_new - S
        S = S_new

        I_B = S[newax, ...] * B[:, newax, newax, newax] + D_R[newax, ...] + D_Z
        I_R_new = _jshrinkage(Im - I_B + Y / mu, weight / mu / T_max)
        dI_R = I_R_new - I_R
        I_R = I_R_new

        R = Im - I_R
        S_sq = jnp.sum(S**2)
        B_new = jnp.sum(S[newax, ...] * (R + Y / mu), axis=(1, 2, 3)) / S_sq
        B_new = jnp.where(S_sq > 0, B_new, B)
        B_new = jnp.maximum(B_new, 0)

        mean_B = jnp.mean(B_new)
        B_new = jnp.where(mean_B > 0, B_new / mean_B, B_new)
        S = jnp.where(mean_B > 0, S * mean_B, S)

        dB = B_new - B
        B = B_new

        BS = S[newax, ...] * B[:, newax, newax, newax]
        if self.get_darkfield:
            D_Z_new = jnp.mean(Im - BS - D_R[newax, ...] - I_R + Y / 2.0 / mu)
            D_Z_new = jnp.clip(D_Z_new, 0, self.D_Z_max)
            dD_Z = D_Z_new - D_Z
            D_Z = D_Z_new

            eta_D = Im.shape[0] * 1.02
            D_R_new = D_R + 1.0 / eta_D * jnp.sum(
                Im - BS - D_R[newax, ...] - D_Z - I_R + Y / mu, axis=0
            )
            D_R_new = idct3d(
                _jshrinkage(dct3d(D_R_new), self.smoothness_darkfield / eta_D / mu)
            )
            D_R_new = _jshrinkage(
                D_R_new, self.sparse_cost_darkfield * dark_weight / eta_D / mu
            )
            dD_R = D_R_new - D_R
            D_R = D_R_new

        I_B = BS + D_R[newax, ...] + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual

        value_diff = jnp.max(
            jnp.array(
                [
                    jnp.linalg.norm(dS.ravel(), ord=2) * jnp.sqrt(eta_S),
                    jnp.linalg.norm(dI_R.ravel(), ord=2) * jnp.sqrt(1.0),
                    # TODO find better form with theoretical evidence
                    jnp.linalg.norm(dB.ravel(), ord=2),
                ]
            )
        )

        if self.get_darkfield:
            value_diff = jnp.max(
                jnp.array(
                    [
                        value_diff,
                        jnp.linalg.norm(dD_R.ravel(), ord=2) * jnp.sqrt(eta_D),
                        # TODO find better form with theoretical evidence
                        dD_Z**2,
                    ]
                )
            )
        value_diff = value_diff / self.image_norm
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff)

    def _step_only_baseline(self, Im, weight, S, D, vals):
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        T_max = Im.shape[0]

        I_B = S[newax, ...] * B[:, newax, newax, newax] + D[newax, ...]
        I_R_new = _jshrinkage(Im - I_B + Y / mu, weight / mu / T_max)
        dI_R = I_R_new - I_R
        I_R = I_R_new

        R = Im - I_R
        B_new = jnp.sum(S[newax, ...] * (R + Y / mu), axis=(1, 2, 3)) / jnp.sum(S**2)
        B_new = jnp.maximum(B_new, 0)
        dB = B_new - B
        B = B_new

        I_B = S[newax, ...] * B[:, newax, newax, newax] + D[newax, ...]
        fit_residual = R - I_B
        Y = Y + mu * fit_residual

        value_diff = jnp.max(
            jnp.array(
                [
                    jnp.linalg.norm(dI_R.ravel(), ord=2) * jnp.sqrt(1.0),
                    # TODO find better form with theoretical evidence
                    jnp.linalg.norm(dB.ravel(), ord=2),
                ]
            )
        )
        value_diff = value_diff / self.image_norm

        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, I_R, B, Y, mu, fit_residual, value_diff)

    def calc_weights(self, I_B, I_R):
        Ws = jnp.ones_like(I_R, dtype=jnp.float32) / (
            jnp.abs(I_R / (I_B + self.epsilon)) + self.epsilon
        )
        return Ws / jnp.mean(Ws)

    def calc_dark_weights(self, D_R):
        Ws = np.ones_like(D_R, dtype=jnp.float32) / (jnp.abs(D_R) + self.epsilon)
        return Ws / jnp.mean(Ws)

    def calc_weights_baseline(self, I_B, I_R):
        return self.calc_weights(I_B, I_R)

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R + D_Z


class ApproximateFit(BaseFit):
    _ent1: float = PrivateAttr(1.0)
    _ent2: float = PrivateAttr(10.0)

    def _step(
        self,
        Im,
        weight,
        dark_weight,
        vals,
    ):
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, value_diff = vals

        # approximate fitting only accepts two-dimensional images.
        # XXX better coding?
        Im = Im[:, 0, ...]
        weight = weight[:, 0, ...]
        S = S[0]
        D_R = D_R[0]
        I_R = I_R[:, 0, ...]
        Y = Y[:, 0, ...]

        S_hat = dct2d(S)
        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...] + D_Z
        temp_W = (Im - I_B - I_R + Y / mu) / self._ent1
        #    plt.imshow(temp_W[0]);plt.show()
        #    print(type(temp_W))
        temp_W = jnp.mean(temp_W, axis=0)
        S_hat = S_hat + dct2d(temp_W)
        S_hat = _jshrinkage(S_hat, self.smoothness_flatfield / (self._ent1 * mu))
        S = idct2d(S_hat)
        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...] + D_Z
        I_R = (Im - I_B + Y / mu) / self._ent1
        I_R = _jshrinkage(I_R, weight / (self._ent1 * mu))
        R = Im - I_R
        B = jnp.mean(R, axis=(1, 2)) / jnp.mean(R)
        B = jnp.maximum(B, 0)

        if self.get_darkfield:
            B_valid = B < 1

            S_inmask = S > jnp.mean(S) * (1 - 1e-6)
            S_outmask = S < jnp.mean(S) * (1 + 1e-6)
            A = (
                jnp.sum(R * S_inmask[newax, ...], axis=(1, 2))
                / jnp.sum(S_inmask * R.shape[0])
                - jnp.sum(R * S_outmask[newax, ...], axis=(1, 2))
                / jnp.sum(S_outmask * R.shape[0])
            ) / jnp.mean(R)
            A = jnp.where(jnp.isnan(A), 0, A)

            # temp1 = jnp.sum(p['A1_coeff'][validA1coeff_idx]**2)
            B_sq_sum = jnp.sum(B**2 * B_valid)
            B_sum = jnp.sum(B * B_valid)
            A_sum = jnp.sum(A * B_valid)
            BA_sum = jnp.sum(B * A * B_valid)
            denominator = B_sum * A_sum - BA_sum * jnp.sum(B_valid)
            # limit B1_offset: 0<B1_offset<B1_uplimit

            D_Z = jnp.clip(
                (B_sq_sum * A_sum - B_sum * BA_sum) / (denominator + 1e-6),
                0,
                self.D_Z_max / jnp.mean(S),
            )

            Z = D_Z * (np.mean(S) - S)

            D_R = (R * B_valid[:, newax, newax]).sum(axis=0) / B_valid.sum() - (
                B * B_valid
            ).sum() / B_valid.sum() * S
            D_R = D_R - jnp.mean(D_R) - Z

            # smooth A_offset
            D_R = dct2d(D_R)
            D_R = _jshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))
            D_R = idct2d(D_R)
            D_R = _jshrinkage(D_R, self.sparse_cost_darkfield / (self._ent2 * mu))
            D_R = D_R + Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        # put the variables back to 4-dim input array
        S = S[newax, ...]
        D_R = D_R[newax, ...]
        I_R = I_R[:, newax, ...]
        Y = Y[:, newax, ...]
        fit_residual = fit_residual[:, newax, ...]
        return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual, 0.0)

    def _step_only_baseline(self, Im, weight, S, D, vals):
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        Im = Im[:, 0, ...]
        weight = weight[:, 0, ...]
        S = S[0]
        D = D[0]
        I_R = I_R[:, 0, ...]
        Y = Y[:, 0, ...]

        I_B = S[newax, ...] * B[:, newax, newax] + D[newax, ...]

        # update I_R using approximated l0 norm
        I_R = I_R + (Im - I_B - I_R + (1 / mu) * Y) / self._ent1
        I_R = _jshrinkage(I_R, weight / (self._ent1 * mu))

        R1 = Im - I_R
        # A1_coeff = mean(R1)-mean(A_offset);
        B = jnp.mean(R1, axis=(1, 2)) - jnp.mean(D)
        # A1_coeff(A1_coeff<0) = 0;
        B = jnp.maximum(B, 0)
        # Z1 = D - A1_hat - E1_hat;
        fit_residual = Im - I_B - I_R
        # Y1 = Y1 + mu*Z1;
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        I_R = I_R[:, newax, ...]
        Y = Y[:, newax, ...]
        fit_residual = fit_residual[:, newax, ...]
        return (k + 1, I_R, B, Y, mu, fit_residual, 0.0)

    def calc_weights(self, I_B, I_R):
        I_B = I_B[:, 0, ...]
        I_R = I_R[:, 0, ...]
        XE_norm = I_R / (jnp.mean(I_B, axis=(1, 2))[:, newax, newax] + 1e-6)
        weight = jnp.ones_like(I_R) / (jnp.abs(XE_norm) + self.epsilon)
        weight = weight / jnp.mean(weight)
        return weight[:, newax, ...]

    def calc_dark_weights(self, D_R):
        return jnp.ones_like(D_R)

    def calc_weights_baseline(self, I_B, I_R):
        I_B = I_B[:, 0, ...]
        I_R = I_R[:, 0, ...]
        mean_vec = jnp.mean(I_B, axis=(1, 2))
        XE_norm = mean_vec[:, newax, newax] / (I_R + 1e-6)
        weight = 1.0 / (jnp.abs(XE_norm) + self.epsilon)
        weight = weight / jnp.mean(weight)
        return weight[:, newax, ...]

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R + D_Z * (1 + S)