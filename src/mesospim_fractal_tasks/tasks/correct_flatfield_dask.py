# This file is partly inspired from the file "calculate_basicpy_illumination_models.py" 
# of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
# Adrian Tschan <atschan@apricotx.com>

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numcodecs
numcodecs.blosc.set_nthreads(1)

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field, validate_call
import anndata as ad
import dask.array as da
from dask.distributed import Client
import numpy as np
import zarr
from scipy.ndimage import zoom
from dask_image import ndfilters

from mesospim_fractal_tasks.utils.models import (
    BaSiCPyModelParams, IlluminationModel)
from mesospim_fractal_tasks.utils.basicpy_nojax import BaSiC
from mesospim_fractal_tasks.utils.zarr_utils import (
    _determine_optimal_contrast, _update_omero_channels, 
    create_zarr_pyramid, _get_pyramid_structure, build_pyramid)
from mesospim_fractal_tasks.utils.parallelisation import (
    _set_dask_cluster, correct_per_channel)
from mesospim_fractal_tasks import __version__, __commit__

from fractal_tasks_core.roi import convert_ROI_table_to_indices
from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url

logger = logging.getLogger(__name__)

def compute_baseline(
    empty_tiles: da.Array,
    flatfield: np.ndarray,
    percentile: float = 1
) -> float:
    """
    Estimate a global baseline offset using 90% intensity percentile (empty tiles) 

    Args:
        empty_tiles (np.ndarray): Array of empty FOVs.
        percentile (float): Percentile to use for estimating the baseline.

    Returns:
        float: Estimated baseline offset.
    """
    dtype_max = np.iinfo(empty_tiles.dtype).max
    flatfield = np.clip(flatfield, 0, dtype_max)
    corrected_tiles = empty_tiles / (flatfield + 1e-6)
    return float(np.round(da.percentile(corrected_tiles[:,::4,::4].flatten(), percentile).compute()))

def compute_flatfield(
    empty_tiles: da.Array,
    smooth_sigma: float = 1
) -> np.ndarray:
    """
    Estimate multiplicative flatfield by averaging empty tiles.

    Args:
        empty_tiles (np.ndarray): Array of empty FOVs.
        smooth_sigma (float): Sigma for Gaussian smoothing.

    Returns:
        np.ndarray: Flatfield profile.
    """
    flat_raw = (da.median(empty_tiles, axis=0)).astype(np.float32)

    # Get low-pass illumination field
    flat_smooth = ndfilters.gaussian_filter(flat_raw, sigma=smooth_sigma).astype(np.float32)

    # Normalize to mean 1
    flat_norm = flat_smooth / da.mean(flat_smooth)
    return flat_norm.compute()

def compute_empty_fov_models(
    FOV_data: da.Array,
    channel_label: str,
) -> IlluminationModel:
    """
    Calculates illumination correction profiles based on a random sample
    of FOVs for a given channel_label or wavelength.

    Parameters:
        FOV_data: Array of shape (n_FOVs, y_size, x_size) with the FOVs.
    
    Returns:
        illumination_profiles: IlluminationModel instance with the illumination 
            correction profiles.
    """
    logger.info(f"Start fitting illumination profile model using empty FOVs "
                f"for channel {channel_label}...")
    
    illumination_profiles = IlluminationModel()
    illumination_profiles.flatfield = compute_flatfield(FOV_data)
    illumination_profiles.baseline = compute_baseline(
        FOV_data, flatfield=illumination_profiles.flatfield)
    
    logger.info(f"Illumination profile model fitted "
                f"for channel {channel_label}!")
    return illumination_profiles

def compute_basicpy_models(
    FOV_data: da.Array,
    basicpy_model_params: BaSiCPyModelParams,
    channel_label: str,
) -> IlluminationModel:
    """
    Calculates illumination correction profiles based on a provided sample of FOVs.

    Parameters:
        FOV_data: Array of shape (n_ROIs, y_size, x_size) with the FOVs.
        basicpy_model_params: Parameters for the BaSiCPy model.
    
    Returns:
        illum_profiles: IlluminationModel instance with the illumination 
            correction profiles.
    """

    # calculate illumination correction profile
    logger.info(f"Start fitting BaSiCPy illumination model for channel {channel_label}...")
    basic = BaSiC(
        autosegment=basicpy_model_params.autosegment,
        autosegment_margin=basicpy_model_params.autosegment_margin,
        epsilon=basicpy_model_params.epsilon,
        get_darkfield=basicpy_model_params.get_darkfield,
        max_workers=basicpy_model_params.max_workers,
        smoothness_darkfield=basicpy_model_params.smoothness_darkfield,
        smoothness_flatfield=basicpy_model_params.smoothness_flatfield,
        working_size=basicpy_model_params.working_size
    ) # type: ignore

    if FOV_data.shape[0] == 1:
        logger.info(f"Stack of ROIs shape is {FOV_data[0, :, :, :].shape}.")
        basic.fit(FOV_data[0, :, :, :].compute())
    else:
        logger.info(f"Stack of ROIs shape is {da.squeeze(FOV_data).shape}.")
        basic.fit(np.squeeze(FOV_data.compute()))

    logger.info(
        f"BaSiCPy model fitted for channel {channel_label}!")
    
    illum_profiles = IlluminationModel()
    illum_profiles.flatfield = basic.flatfield
    illum_profiles.darkfield = basic.darkfield
    illum_profiles.baseline = np.median(basic.baseline) 
    
    return illum_profiles

def collect_fovs(
    zarr_path: Path,
    channel_index: int,
    FOV_list: list[int] | None,
    resolution_level: int,
    pixel_sizes_yx: list[float],
    n_zplanes: int,
    z_levels: Optional[list[int]],
) -> da.Array:
    """
    Collect FOVs.

    Depending on the input parameters, the function will either collect FOVs from a 
    list of indices (empty tiles) or from a random sample of FOVs. 

    Args:
        zarr_path (Path): Path to the OME-Zarr image to be processed.
        channel_index (int): Index of the channel to process.
        FOV_list (Optional[list[int]]): List of indices of the FOVs to collect. 
        resolution_level (int): Resolution level to collect FOVs from.
        pixel_sizes_yx (tuple[float, float]): Tuple of pixel sizes in y and x 
            directions.
        n_zplanes (int): Maximum number of z-planes to collect.
        z_levels (Optional[tuple[int, int]): Max z level of z-planes to collect 
            (top and bottom).
    
    Returns:
        ROI_data (np.ndarray): Array of FOVs.
    """

    # Load corresponding resolution image
    image_arr = da.from_zarr(zarr_path / str(resolution_level))[channel_index]
    FOV_ROI_df = ad.read_zarr(zarr_path / "tables" / "FOV_ROI_table").to_df()
    z_size = image_arr.shape[0]
    
    if FOV_list is not None:
        assert set(FOV_list).issubset(range(len(FOV_ROI_df.index))), ("FOV list contains FOVs "
            "that are not present in the FOV_ROI_table.")
        logger.info(f"Collecting {n_zplanes} empty FOVs from {FOV_list}...")
        n_FOVs = len(FOV_list)
        if z_levels is not None:
            if z_levels[0] > z_levels[1]:
                z_levels = [z_levels[1], z_levels[0]]
            min_z_subvolume_size = min(z_levels[0], (z_size - z_levels[1]))
            if n_FOVs * min_z_subvolume_size < (n_zplanes // 2):
                n_zplanes = (n_FOVs * min_z_subvolume_size)
                logger.warning("Number of z planes provided and max_z are not congruent"
                               "; there isn't enough FOVs to collect all z planes."
                               f" Only {n_zplanes} z planes in total will be collected.")
    else:
        logger.info(f"Collecting {n_zplanes} random FOVs from full stack of FOVs...")
        n_FOVs = len(FOV_ROI_df.index)
        FOV_list = list(range(n_FOVs))
    n_zplanes_per_FOV = max(-(-n_zplanes // n_FOVs), 1)
    if n_zplanes_per_FOV > z_size:
        n_zplanes_per_FOV = z_size
        logger.warning("Number of z planes provided is greater than the number of "
                       "available z planes in the image. Correcting to "
                       f"{n_zplanes_per_FOV}.")

    # Sample z planes in random FOVs per image
    ROI_data = []
    n_ROIs = 0

    # Read FOV ROIs
    for i_FOV, _ in enumerate(FOV_ROI_df.index):
        if i_FOV in FOV_list: # type: ignore
            logger.info(f"Collecting {n_zplanes_per_FOV} z planes from FOV stack "
                        f"{i_FOV}.")
            if z_levels is not None:
                z_idxs = np.random.randint(0, z_levels[0], max(n_zplanes_per_FOV, 1))
                z_idxs = np.concatenate([z_idxs, 
                                         np.random.randint(z_size-z_levels[1], 
                                                           z_size,
                                                           max(n_zplanes_per_FOV, 1))])
            else:
                z_idxs = np.random.randint(0, z_size, n_zplanes_per_FOV)
            x_start, x_end = (round(FOV_ROI_df.loc[f"FOV_{i_FOV}", "x_micrometer"] / 
                                    pixel_sizes_yx[1]),  # type: ignore
                            round(FOV_ROI_df.loc[f"FOV_{i_FOV}", "len_x_micrometer"] /
                                    pixel_sizes_yx[1])) # type: ignore
            x_end = x_end + x_start
            y_start, y_end = (round(FOV_ROI_df.loc[f"FOV_{i_FOV}", "y_micrometer"] /
                                    pixel_sizes_yx[0]),  # type: ignore
                            round(FOV_ROI_df.loc[f"FOV_{i_FOV}", "len_y_micrometer"] /
                                    pixel_sizes_yx[0])) # type: ignore
            y_end = y_end + y_start
            for idx in z_idxs:
                if n_ROIs < n_zplanes:
                    n_ROIs += 1
                    region = (slice(idx, idx+1), 
                              slice(y_start, y_end), 
                              slice(x_start, x_end))
                    ROI_data.append(image_arr[region])
                else: 
                    break
            logger.info(f"Total number of z planes collected: {n_ROIs}/{n_zplanes}")
            if n_ROIs >= n_zplanes:      
                logger.info("Maximum number of z planes reached."
                            " Stop collecting z planes from FOV stacks.")
                break
                
    ROI_data = da.concatenate(ROI_data, axis=0)
    return ROI_data

def correct_FOV(
    FOV_dask: da.Array,
    i_FOV: int,
    illum_profiles: IlluminationModel,
) -> da.Array:
    """
    Apply flatfield/darkfield correction to all fields of view.

    Corrects a stack of images, using a given illumination profile (e.g. bright
    in the center of the image, dim outside).

    Parameters:
        img_stack: 4D numpy array (czyx), with dummy size along c.
        i_FOV: compatibility with parallelisation (unused)
        illum_profiles: IlluminationModel instance with the illumination 
            correction profiles.

    Returns:
        Corrected image stack.
    """

    # Check shapes
    if FOV_dask.shape[0] != 1:
        raise ValueError(
            "Error! Unexpected shape (should be (1, z, y, x)): "
            f"{FOV_dask.shape}\n"
        )
    
    # Store info about dtype
    dtype = FOV_dask.dtype
    dtype_max = np.iinfo(dtype).max
    img_stack = FOV_dask.astype(np.float32)

    if illum_profiles.flatfield is None:
        logger.error("Flatfield correction matrix not found.")
        raise ValueError
    
    # Apply the correction matrices
    if illum_profiles.darkfield is not None:
        img_stack = ((img_stack - illum_profiles.darkfield[None, None,:,:]) / 
                     (illum_profiles.flatfield[None, None,:,:] + 1e-6))
    else:
        img_stack = img_stack / (illum_profiles.flatfield[None, None,:,:] + 1e-6)

    # Background subtraction
    if illum_profiles.baseline is not None:
        img_stack = da.where(img_stack > illum_profiles.baseline,
                             img_stack - illum_profiles.baseline, 0)
        
    # Clip lazily
    new_img_stack = da.clip(img_stack, 0, dtype_max)   

    # Cast back to original dtype and return
    return new_img_stack.astype(dtype)

def resample_to_shape(
    img: np.ndarray,
    output_shape: tuple[int, int],
    order: int = 3, 
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True
) -> np.ndarray:
    """
    Function resamples image to the desired shape.

    Typically used to up or downscale a pyramid image by
    a potency of 2 (e.g. 0.5, 1, 2 etc.)
    """
    dtype = img.dtype
    dtype_max = np.iinfo(dtype).max
    zoom_values = [o / i for i, o in zip(img.shape, output_shape)]
    resampled = zoom(img, zoom_values, order=order, mode=mode, cval=cval,
                prefilter=prefilter)
    return np.clip(resampled, 0, dtype_max)

def get_non_default_params(
    model_instance: BaSiCPyModelParams
) -> Dict[str, Any]:
    """
    Return a dictionary of parameters that differ from the defaults
    for a given BaSiCPyModelParams instance.

    Args:
        model_instance (BaSiCPyModelParams): Model instance to check.

    Returns:
        changed_params (Dict[str, Any]): Dictionary of parameters that differ
            from the defaults.
    """
    changed_params = {}
    for field_name, model_field in model_instance.model_fields.items():
        default_value = model_field.default
        current_value = getattr(model_instance, field_name)
        if current_value != default_value:
            changed_params[field_name] = current_value
    return changed_params

def define_FOV_list(
    zarr_path: Path
) -> list[int]:
    
    FOV_list = []
    ROI_table = ad.read_zarr(zarr_path / "tables" / "FOV_ROI_table").to_df()
    max_y_FOV = ROI_table["y_micrometer"].max()
    max_x_FOV = ROI_table["x_micrometer"].max()
    for i in range(len(ROI_table)):
        if (ROI_table.iloc[i]["y_micrometer"] == 0 and 
            ROI_table.iloc[i]["x_micrometer"] == 0):
            FOV_list.append(i)
        if (ROI_table.iloc[i]["y_micrometer"] == 0 and 
            ROI_table.iloc[i]["x_micrometer"] == max_x_FOV):
            FOV_list.append(i)
        if (ROI_table.iloc[i]["y_micrometer"] == max_y_FOV and 
            ROI_table.iloc[i]["x_micrometer"] == 0):
            FOV_list.append(i)
        if (ROI_table.iloc[i]["y_micrometer"] == max_y_FOV and 
            ROI_table.iloc[i]["x_micrometer"] == max_x_FOV):
            FOV_list.append(i)   
    return FOV_list

@validate_call
def correct_flatfield(
    *,
    zarr_url: str,
    models_folder: Optional[str] = None,
    FOV_list: Optional[list[int]] = None,
    z_levels: Optional[list[int]] = None,
    save_models: bool = False,
    resolution_level: Optional[int] = None,
    n_zplanes: int = 200,
    basicpy_model_params: Optional[BaSiCPyModelParams] = Field(
        default_factory=BaSiCPyModelParams),
    erase_source_image: bool = False
) -> dict[str, list]:

    """
    Perform flatfield (and darkfield) correction using either BaSiCPy or empty FOVs 
    for each channel label or wavelength.

    Parameters:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        models_folder: Folder name where illumination
            profiles are stored and can be used to perform flatfield correction. 
            If provided, fitting models is skipped and only the correction step is
            performed. Default: None.
        FOV_list: List of tiles to process. If provided, illumination profiles will be 
            computed from this list of tiles without BaSiCPy. They are expected to 
            contain only empty space. Default: None.
        z_levels: Two integers indicating the maximum number of z planes to process 
            at the top and bottom of the 3D tile stack. If provided, illumination 
            profiles will be computed using z planes up to the first number of z_levels 
            at the bottom and down to the 2nd number of z_levels at the top of the tiles 
            (expecting empty FOVs) without BaSiCPy.
            If FOV_list is not empty the subvolumes will be extracted from the tiles 
            in FOV_list, otherwise from the four tiles at the corners. Default: None.
        save_models: If `True`, illumination profiles will be saved in the parent folder
            of the currently processed OME-Zarr. Default: False.
        resolution_level: Resolution level at which to calculate the illumination
            correction profiles. If None, the lowest resolution level will be used for BaSiCPy
            and highest resolution level for empty FOVs. Default: None.
        n_zplanes: Number of z planes to use to calculate the illumination profile model.
            Greater number requires more memory. If using BaSiCPy, at least 150 is recommended
            for a good fit. If using empty FOVs, at least 50 is recommended. Default: 200.
        basicpy_model_params: Parameters for the BaSiC model. See documentation
            for more information. Default: None.
        erase_source_image: If `True`, the source image will be erased after the flatfield
            correction. Default: False.
    """

    zarr_path = Path(zarr_url)
    new_zarr_path = Path(zarr_path.parent, zarr_path.name + "_flatfield_corr")
    logger.info(f"Start task: `Flatfield Correction` "
                f"for {zarr_path.parent.name}/{zarr_path.name}")
    
    # Map channel name to channel index
    channel_dict = {}
    channels = get_omero_channel_list(image_zarr_path=str(zarr_path))
    if len(channels) == 0:
        raise ValueError("No channels found in the OME Zarr metadata.")
    for c, channel in enumerate(channels):
        if channel.index is None:
            channel.index = c
        channel_dict[channel.label] = channel.index

    # Read attributes from NGFF metadata (Note: all FOVs expected to have same
    # shape)
    ngff_image_meta = load_NgffImageMeta(str(zarr_path))
    num_levels = ngff_image_meta.num_levels
    if resolution_level is None:
        if FOV_list is None and z_levels is None:
            resolution_level = num_levels-1    # basicpy run
        else:
            resolution_level = 0               # empty tiles run
    if resolution_level not in range(num_levels):
        raise ValueError(f"Resolution level {resolution_level} not found in "
                         f"multiscale pyramid. Available levels go from: 0 to {num_levels-1}")
    pxl_sizes_yx = ngff_image_meta.get_pixel_sizes_zyx(
        level=resolution_level)[1:]
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pyramid_dict = _get_pyramid_structure(zarr_path)

    # Create new zarr pyramid
    create_zarr_pyramid(zarr_path, new_zarr_name=new_zarr_path.name, pyramid_dict=pyramid_dict)

    # Lazily load highest-res level from original zarr array
    image_arr = da.from_zarr(Path(zarr_path, "0"))
    z_size = image_arr.shape[1]

    # Define FOV list
    if FOV_list is None and z_levels is not None:
        FOV_list = define_FOV_list(zarr_path)
    
    if FOV_list is not None:
        assert min(FOV_list) > 0, ("FOV list must start at 1.")
        FOV_list = [int(i-1) for i in FOV_list]

    if z_levels is not None:
        assert len(z_levels) == 2, "z_levels must be a list of two numbers."
        assert (0 < z_levels[0]) and (0 < z_levels[1]), "z_levels must be non-negative."
        assert z_levels[0] < z_size and z_levels[1] < z_size, "z_levels must be smaller than the number of z planes."

    # Get FOV shape
    FOV_ROI_table = ad.read_zarr(Path(zarr_path, "tables", "FOV_ROI_table"))
    indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=0,
        coarsening_xy=2,
        cols_xyz_pos= [
        "x_micrometer",
        "y_micrometer",
        "z_micrometer"],
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    FOV_shape = (indices[0][-3], indices[0][-1])

    # Set dask cluster
    client = None
    cluster = None
    try:
        cluster = _set_dask_cluster(n_workers=len(channel_dict.keys()))
        client = Client(cluster)
        client.forward_logging(logger_name = "mesospim_fractal_tasks", level=logging.INFO)
            
        # Iterate over channels
        illum_profiles = {}
        for channel in channel_dict.keys():
            channel_index = channel_dict[channel]
            FOV_data = collect_fovs(
                zarr_path=zarr_path,
                channel_index=channel_index,
                FOV_list=FOV_list,
                resolution_level=resolution_level,
                pixel_sizes_yx=pxl_sizes_yx,
                n_zplanes=n_zplanes,
                z_levels=z_levels,
            )
            if models_folder is None:
                if FOV_list is not None:
                        assert len(FOV_list) != 0, "FOV list is empty!"
                        client.forward_logging(logger_name = "mesospim_fractal_tasks", level=logging.INFO)
                        illum_profiles[channel] = compute_empty_fov_models(
                            FOV_data=FOV_data,
                            channel_label=channel,
                        )
                else:
                    if basicpy_model_params is None:
                        basicpy_model_params = BaSiCPyModelParams()
                    illum_profiles[channel] = compute_basicpy_models(
                        FOV_data=FOV_data,
                        basicpy_model_params=basicpy_model_params,
                        channel_label=channel,
                    )

                if save_models:
                    if FOV_list is not None:
                        folder_path = Path(zarr_path.parent, "IllumModels")
                    else:
                        folder_path = Path(zarr_path.parent, "BaSiCPyModels")
                    channel_model_folder = Path(folder_path, channel)
                    if not channel_model_folder.exists():
                        channel_model_folder.mkdir(parents=True, exist_ok=True)
                    illum_profiles[channel].save_models(folder=str(channel_model_folder))
            else:
                # Load illumination model
                logger.info(
                    f"Loading illumination profiles for channel {channel} from "
                    f"{models_folder}."
                )
                illum_profiles[channel] = IlluminationModel()
                profiles = np.load(Path(models_folder, channel, "profiles.npz"))
                if "darkfield" not in profiles.keys():
                    logger.warning("Darkfield profile not found in "
                                    f"{models_folder}. Skipping darkfield correction.")
                else:
                    illum_profiles[channel].darkfield = profiles["darkfield"]
                if "baseline" not in profiles.keys():
                    logger.warning("Baseline profile not found in "
                                    f"{models_folder}. Skipping baseline correction.")
                else:
                    illum_profiles[channel].baseline = profiles["baseline"]
                if "flatfield" not in profiles.keys():
                    raise ValueError("Error! Illumination profiles not found in "
                                    f"{models_folder}.")
                illum_profiles[channel].flatfield = profiles["flatfield"]
        
            # Resampling flatfield and darkfield if necessary
            if illum_profiles[channel].flatfield.shape[-2:] != FOV_shape:
                logger.warning(
                    f"Flatfield correction matrix shape does not match FOV shape in"
                    f" x and y. FOV YX shape: {FOV_shape}\n"
                    f"Flatfield shape: {illum_profiles[channel].flatfield.shape}. Resampling ...")
                illum_profiles[channel].flatfield = resample_to_shape(
                    illum_profiles[channel].flatfield, 
                    FOV_shape)
            illum_profiles[channel].flatfield = da.from_array(illum_profiles[channel].flatfield, 
                                                            chunks=image_arr.chunksize[-2:])

            if illum_profiles[channel].darkfield is not None:
                if illum_profiles[channel].darkfield.shape[-2:] != FOV_shape:
                    logger.warning(
                        "Darkfield correction matrix shape does not match FOV shape"
                        f" in x and y. FOV YX shape: {FOV_shape}\n"
                        f"Darkfield shape: {illum_profiles[channel].darkfield.shape}. Resampling ...")
                    illum_profiles[channel].darkfield = resample_to_shape(
                        illum_profiles[channel].darkfield,
                        FOV_shape)
                illum_profiles[channel].darkfield = da.from_array(illum_profiles[channel].darkfield, 
                                                                chunks=image_arr.chunksize[-2:])
                
        futures = []
        for channel_name, channel_idx in channel_dict.items():
            illum_prof_f = client.scatter(illum_profiles[channel_name], broadcast=False)
            fut = client.submit(
                correct_per_channel,
                zarr_path=zarr_path,
                new_zarr_path=new_zarr_path,
                channel_name=channel_name,
                channel_index=channel_idx,
                full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
                correct_func=correct_FOV,
                correct_func_kwargs={
                    "illum_profiles": illum_prof_f,
                },
                pure=False,
                retries=1
            )
            futures.append(fut)
        client.gather(futures)

        futures = []
        for channel_name, channel_idx in channel_dict.items():
            fut = client.submit(
                build_pyramid,
                zarr_url=new_zarr_path,
                pyramid_dict=pyramid_dict,
                channel_index=channel_idx,
                channel_name=channel_name,
                pure=False,
                retries=1
            )
            futures.append(fut)
        client.gather(futures)
    
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

    # Copy ROI tables from the old zarr_url
    _copy_tables_from_zarr_url(str(zarr_path), str(new_zarr_path))

    # Copy NGFF metadata from the old zarr_url to the new zarr
    logger.info(f"Copying NGFF metadata from {zarr_path.name}"
                f" to {new_zarr_path.name}.")
    source_group = zarr.open_group(zarr_path, mode="r")
    source_attrs = source_group.attrs.asdict()
    image_name = source_attrs["multiscales"][0]["name"] + "_flatfield_corr"
    source_attrs["multiscales"][0]["name"] = image_name
    fractal_tasks = source_attrs.get("fractal_tasks", {})
    task_dict = dict(
        version=__version__.split("dev")[0][:-1],
        commit=__commit__,
        input_parameters=dict(
            models_folder=models_folder,
            resolution_level=resolution_level,
            n_zplanes=n_zplanes,
            basicpy_model_params=get_non_default_params(
                basicpy_model_params) if basicpy_model_params is not None else None,
            FOV_list=FOV_list,
            z_levels=z_levels,
            save_models=save_models
        )
    )
    fractal_tasks["correct_flatfield"] = task_dict
    source_attrs["fractal_tasks"] = fractal_tasks # type: ignore
    new_group = zarr.open_group(str(new_zarr_path), mode="a")
    new_group.attrs.put(source_attrs)

    contrast_limits = _determine_optimal_contrast(
        new_zarr_path, 
        num_levels, 
        segment_sample=True
    )
    _update_omero_channels(new_zarr_path, {"window": contrast_limits})

    image_list_updates = dict(
        image_list_updates=[dict(zarr_url=str(new_zarr_path), 
                                 origin=str(zarr_path),
                                 attributes=dict(image=new_zarr_path.name))]
        )
    
    if erase_source_image:
        logger.info("Erasing source image...")
        shutil.rmtree(zarr_path)

    return image_list_updates

if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=correct_flatfield,
        logger_name=logger.name,
    )