import zarr
import dask.array as da
import logging
from typing import Any, Union, Optional
from skimage.measure import block_reduce
import numpy as np
from pathlib import Path
#from filelock import FileLock
from zarr.storage import DirectoryStore
from zarr.sync import ProcessSynchronizer


from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.labels import prepare_label_group

logger = logging.getLogger(__name__)


def _determine_optimal_contrast(
    image_path: Path,
    num_levels: int,
    channel_index: Optional[int] = None,
    segment_sample: bool = False, 
    synchronizer: zarr.sync.ProcessSynchronizer = None
) -> dict[str, dict[str, int]]:
    """
    Determine the optimal contrast limits for the image.

    Parameters:
        image_path (Path): Path to the image to determine the contrast limits for.
        num_levels (int): Number of pyramid levels.
        channel_index (int): Channel index for which to determine the contrast limits.
        segment_sample (bool): If should segment sample to compute the intensity 
            percentiles for the contrast limits.

    Returns:
        contrast_limits (list[dict[str, int]]): Contrast limits for the image.
    """

    logger.info(f"Determining optimal contrast limits for {image_path.name}")

    # Load the lowest resolution image
    store = DirectoryStore(str(image_path))
    if synchronizer is not None:
        image_group = zarr.open_group(store=store, mode="r", synchronizer=synchronizer)
    else:
        image_group = zarr.open_group(store=store, mode="r")
    low_res_arr = image_group[str(num_levels-1)]

    # Determine the percentile for the contrast limits
    contrast_limits = {}
    if channel_index is not None:
        channel_list = [channel_index]
    else:
        channel_list = range(low_res_arr.shape[0])
    for c in channel_list:
        if segment_sample:
            sample_mask = low_res_arr[c] > np.percentile(low_res_arr[c], 50)
            contrast_down = int(np.percentile(low_res_arr[c][sample_mask], 0.1))
            contrast_up = int(np.percentile(low_res_arr[c][sample_mask], 99.9))
        else:
            contrast_down = int(np.percentile(low_res_arr[c], 0.1))
            contrast_up = int(np.percentile(low_res_arr[c], 99.9))
        contrast_limits[str(c)] = {"start": contrast_down, "end": contrast_up}
    return contrast_limits

def _update_omero_channels(
        zarr_path: Path,
        update_dict: dict[str, Any],
        synchronizer: zarr.sync.ProcessSynchronizer = None,
    ) -> None:
    """
    Update the OMERO channels in the OME-ZARR metadata.

    Args:
        zarr_path (Path): Path to the OME-ZARR store.
        update_dict (dict[str, Any]): Dictionary containing the updates to be applied to the OMERO channels.

    Returns:
        None
    """

    # Read the OME-ZARR metadata

    # Open store + group with synchronizer
    store = DirectoryStore(str(zarr_path))
    if synchronizer is not None:
        zarr_group = zarr.open_group(store=store, mode="r+", synchronizer=synchronizer)
    else:
        zarr_group = zarr.open_group(store=store, mode="r+")
    channels_attrs = zarr_group.attrs["omero"]["channels"]

    for c in range(len(channels_attrs)):
        for key, value in update_dict.items():
            if key == "window":
                if str(c) in value.keys():
                    logger.info("Updating window contrast limits for channel " + str(c))
                    logger.info(f"Old window: {channels_attrs[c][key]}")
                    logger.info(f"New window: {value[str(c)]}")
                    channels_attrs[c][key]["start"] = value[str(c)]["start"]
                    channels_attrs[c][key]["end"] = value[str(c)]["end"]
            else:
                logger.info("Updating channel " + str(c) + " attribute " + key)
                logger.info(f"Old value: {channels_attrs[c][key]}")
                logger.info(f"New value: {value[c]}")
                channels_attrs[c][key] = value[c]
    
    # Write the updated OME-ZARR metadata
    zarr_group.attrs["omero"]= {"channels": channels_attrs}

def _write_label_metadata(
    image_group: zarr.Group,
    label_name: str,
    label_info: dict[str, Any],
    num_levels: Optional[int] = None,
    analysis_resolution_level: Optional[int] = None,
    overwrite: bool = False
) -> None:
    """
    Write OME-NGFF label metadata for a given label.
    
    Parameters:
        image_group: OME-Zarr group of the image to which the label applies.
        label_name: Name of the label to write metadata for.
        label_info: Dictionary containing label-specific information.
        num_levels: Number of pyramid levels to write metadata for.
        analysis_resolution_level: Resolution level to write metadata for.
        overwrite: Whether to overwrite existing metadata.
    
    Returns:
        None
    """

    # Copy from the source image metadata
    image_attrs = image_group.attrs.asdict()

    # Delete channel axis
    image_attrs["multiscales"][0]["axes"] = image_attrs["multiscales"][0]["axes"][1:]
    
    # Delete channel scale and keep relevant levels
    datasets = image_attrs["multiscales"][0]["datasets"]
    if num_levels is not None:
        nb_datasets = num_levels
    else:
        nb_datasets = len(datasets)
    scale = image_attrs["multiscales"][0]["datasets"][0] \
            ["coordinateTransformations"][0]["scale"][1:]
    if analysis_resolution_level is not None:
        scale = [scale[0], scale[1] * 2**analysis_resolution_level, 
                 scale[2] * 2**analysis_resolution_level]
    image_attrs["multiscales"][0]["datasets"] = []
    for d in range(nb_datasets):
        image_attrs["multiscales"][0]["datasets"].append(
            {
                "coordinateTransformations" : [ 
                    {
                        "scale": [scale[0], scale[1] * 2**d, scale[2] * 2**d],
                        "type": "scale"
                    }
                ],
                "path": str(d)
            }
        )
    
    # Delete irrelevant info
    if "omero" in image_attrs.keys():
        del image_attrs["omero"]
    if "crop_info" in image_attrs.keys():
        del image_attrs["crop_info"]
    if "fractal_tasks" in image_attrs.keys():
        del image_attrs["fractal_tasks"]
    
    # Adapt to the label metadata and write it to each label group
    logger.info(f"Writing OME-NGFF label metadata for {label_name} label.")
        
    # Prepare the OME-Zarr label group (write multiscale + labels metadata)
    label_group = prepare_label_group(
        image_group,
        label_name,
        image_attrs,
        overwrite=overwrite)
    
    # Add label-specific metadata
    label_metadata = {
        "version": "0.4",
        "source": {
            "image": "../../"
        }
    }
    for key in label_info.keys():
        label_metadata[key] = label_info[key]
        
    label_group.attrs["image-label"] = label_metadata
    logger.info(f"OME-NGFF label metadata written for {label_name} label.")

def _store_label_to_zarr(
    label_path: Path,
    label_mask: Union[np.ndarray, np.memmap],
    chunksize: tuple[int, int, int] = [16, 256, 256],
    overwrite: bool = False,
) -> None:
    """
    Store a label array to a zarr array.
    
    Parameters:
        image_path: Path to the image group the label is associated with.
        label_path: Path to the label group to build the pyramid for.
        label_mask: 3D integer array of shape (X,Y,Z) with label information.
        analysis_resolution_level: Resolution level used for the data analysis.
        chunksize: Chunk size to use for the zarr array.
        overwrite: Whether to overwrite existing pyramid.
    """

    # Load relevant metadata from source image
    logger.info(f"Saving label mask at to {label_path} with chunk size {chunksize}")
    shape = label_mask.shape
    
    logger.info(f"Opening OME-Zarr array of shape {shape}.")
    mask_zarr = zarr.create(
            shape=shape,
            chunks=chunksize,
            dtype=np.uint16,
            store=zarr.storage.FSStore(f"{label_path}/0"),
            overwrite=overwrite,
            dimension_separator="/",
        )
    
    mask_zarr[:] = label_mask
    logger.info(f"saving to Zarr done!")

def _build_label_pyramid(
    image_path: Path,
    label_path: Path,
    analysis_resolution_level: int,
    label_mask: Union[np.ndarray, np.memmap],
    overwrite: bool = False,
) -> None:
    """
    Build a pyramid of multiscale labels.
    
    Parameters:
        image_path: Path to the image group the label is associated with.
        label_path: Path to the label group to build the pyramid for.
        analysis_resolution_level: Resolution level used for the data analysis.
        label_mask: 3D integer array of shape (X,Y,Z) with label information.
        overwrite: Whether to overwrite existing pyramid.
    """

    # Load relevant metadata from source image
    image_meta = load_NgffImageMeta(image_path)
    num_levels = image_meta.num_levels
    if image_meta.coarsening_xy is None:
        coarsening_xy = 2
    else:
        coarsening_xy = image_meta.coarsening_xy
    logger.info(f"Building multi-resolution pyramid of {num_levels} levels for label at" 
                f" {label_path}")
    
    # Create the zarr array for the label mask at each resolution level
    for level in range(num_levels):
        
        logger.info(f"Building level {level+1}/{num_levels}")
        image_arr = da.from_zarr(f"{image_path}/{level}")
        shape = image_arr.shape[1:]
        chunks = image_arr.chunksize[1:]
        
        logger.info(f"Opening OME-Zarr array of shape {shape}.")
        mask_zarr = zarr.create(
                shape=shape,
                chunks=chunks,
                dtype=np.uint16,
                store=zarr.storage.FSStore(f"{label_path}/{level}"),
                overwrite=overwrite,
                dimension_separator="/",
            )
        
        # Correct the size of the mask array if necessary and save to zarr
        correction_factor = analysis_resolution_level-level
        block_size=(1, coarsening_xy**abs(correction_factor), 
                    coarsening_xy**abs(correction_factor))
        if correction_factor > 0:
            #resized_mask = _check_array_size(label_mask, shape, 
            #                                 block_size[1])
            label_dask = da.from_array(label_mask, chunks=chunks)
            resized_dask = label_dask.map_blocks(
                _upscale_chunk,
                block_size=block_size,
                dtype=mask_zarr.dtype,
                chunks=(chunks[0], chunks[1] * block_size[1], chunks[2] * block_size[2])
            )
            resized_mask = _check_array_size(resized_dask, shape)
            da.to_zarr(resized_dask[:shape[0], :shape[1], :shape[2]], mask_zarr)
        elif correction_factor < 0:
            resized_mask = block_reduce(
                label_mask, 
                block_size=block_size, 
                func=np.max
            )
            resized_mask = _check_array_size(resized_mask, shape)#, 1)
            mask_zarr[:] = resized_mask[:shape[0], :shape[1], :shape[2]]
        else:
            mask_zarr[:] = label_mask

        logger.info(f"Level {level+1}/{num_levels} done!")

def _downscale_chunk(
        chunk: np.ndarray,
        block_size: tuple[int, int, int]
) -> np.ndarray:
    """
    Downscale a chunk of a label array (categorical data).
    
    Parameters:
        chunk: Chunk of the label array to be downscaled.
        block_size: Block size of the label array.
    
    Returns:
        Downscaled chunk.
    """
    return block_reduce(chunk, block_size=block_size, func=np.max)

def _upscale_chunk(
        chunk: np.ndarray,
        block_size: tuple[int, int, int]
) -> np.ndarray:
    """
    Upscale a chunk of a label array (categorical data).
    
    Parameters:
        chunk: Chunk of the label array to be upscaled.
        block_size: Block size of the label array.
    
    Returns:
        Upscaled chunk.
    """
    return np.kron(chunk, np.ones(block_size))

def _check_array_size(
        array: da.Array,
        shape: tuple[int, int, int],
        #upscale_factor: int,
) -> da.Array:
    """
    Check if the array has the correct size after upscaling. Note: it is assumed
    that the array is at least 3D and only the last two dimensions are checked.
    
    Parameters:
        array: Numpy array to be checked (3D).
        shape: Desired shape of the dask array (3D).
        upscale_factor: Size of the blocks used to upscale the array.
    
    Returns:
        Numpy array with the correct size.
    """

    if len(array.shape) < 3:
        raise ValueError("Array is not 3D.")
    if len(shape) > 3:
        shape = shape[1:]

    array_shape = (array[:,0,0].compute().size, 
                        array[0,:,0].compute().size,
                        array[0,0,:].compute().size)
    if array_shape[0] != shape[0]:
        raise ValueError("Array has inconsistent number of Z frames.")
    


    array_shape = np.array(array_shape)
    shape = np.array(shape)
    #out_shape = array_shape * np.array([1, upscale_factor, upscale_factor])
    diff = shape - array_shape #out_shape
    if np.any(diff > 0):
        pad_width = ((0,0),   # z axis, no padding
                     (0, int(max(np.ceil(diff[1]), 0))),
                     (0, int(max(np.ceil(diff[2]), 0)))) 
                     #(0, int(max(np.ceil(diff[1] / 2), 0))),   # y axis padding
                     #(0, int(max(np.ceil(diff[2] / 2), 0))))  # x axis padding
        new_array = np.pad(array, pad_width=pad_width, mode="edge")
    elif np.any(diff < 0):
        new_array = array[:shape[0], :shape[1], :shape[2]]
    else:
        new_array = array
    return new_array
