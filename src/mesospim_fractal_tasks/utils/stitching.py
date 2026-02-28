"""Fractal multiview stitcher utils."""
import logging
import warnings
import shutil
import itertools
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import os
import dask.array as da
import spatial_image as si
from dask.delayed import delayed
from dask import config as dask_config
import zarr
import pandas as pd
from scipy import ndimage
from skimage.exposure import rescale_intensity
from skimage.metrics import structural_similarity
import numpy as np
import xarray as xr
from numpy._typing import ArrayLike
from fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    OmeroChannel,
    get_channel_from_image_zarr,
    get_omero_channel_list,
)
from collections.abc import Callable
from fractal_tasks_core.ngff import load_NgffImageMeta
from multiview_stitcher import msi_utils, ngff_utils, mv_graph, misc_utils, weights, param_utils
from multiview_stitcher.fusion import (process_output_stack_properties,
                                       process_output_chunksize,
                                       normalize_chunks,
                                       weighted_average_fusion,
                                       fuse_np)
from multiview_stitcher.registration import (link_quality_metric_func)
from multiview_stitcher import spatial_image_utils as si_utils
from spatial_image import to_spatial_image

DEFAULT_TRANSFORM_KEY = "affine_metadata"

logger = logging.getLogger(__name__)

BoundingBox = dict[str, dict[str, Union[float, int]]]


# --- worker globals (one copy per process) ---
_G = {}

def _init_worker(meta: dict, fuse_kwargs: dict):
    """
    Runs once per worker process.
    Stores heavy, read-only objects in process-global memory
    so they are not pickled/sent for every block.
    """
    _G["meta"] = meta
    _G["fuse_kwargs"] = fuse_kwargs

def _worker(block_id):
    """Thin wrapper: only block_id is sent per task."""
    return fuse_one_block_worker(block_id, meta=_G["meta"], fuse_kwargs=_G["fuse_kwargs"])

def parallel_block_processing(
    *,
    max_workers: int,
    nblocks,
    meta: dict,
    fuse_kwargs: dict,
    max_in_flight: int | None = None,
):
    """
    Parallel chunk processing with backpressure.

    - max_workers controls compute/write concurrency
    - max_in_flight controls how many tasks are submitted/queued at once
      (prevents OOM from submitting all tasks up-front)
    """
    all_block_ids = list(np.ndindex(*nblocks))
    total = len(all_block_ids)

    if max_in_flight is None:
        max_in_flight = 2 * max_workers  # good default backpressure

    it = iter(all_block_ids)
    in_flight = set()

    milestones = np.linspace(0, total, num=20, dtype=int)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(meta, fuse_kwargs),
    ) as ex:
        # prime the pipeline
        for _ in range(min(max_in_flight, total)):
            bid = next(it, None)
            if bid is None:
                break
            in_flight.add(ex.submit(_worker, bid))

        n_completed = 0
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

            # collect results + surface exceptions
            for fut in done:
                _ = fut.result()
                n_completed += 1
            
            if n_completed in milestones:
                logger.info(f"{((n_completed/total)*100):.0f}% completed!") 

            # submit new tasks to keep the pipeline full (bounded)
            for _ in range(len(done)):
                bid = next(it, None)
                if bid is None:
                    break
                in_flight.add(ex.submit(_worker, bid))

def fuse_one_block_worker(block_id, *, meta: dict, fuse_kwargs: dict):
    """
    Fuse a single block and write it to the existing Zarr array.

    Safe for multiprocessing (spawn/fork) and SLURM.
    """
    osp = meta["output_stack_properties"]
    nsdims = meta["nsdims"]
    sdims = meta["sdims"]
    drop_t = meta["drop_t"]
    normalized_chunks = meta["normalized_chunks"]
    output_zarr_url = meta["output_zarr_url"]

    # Open existing array (do NOT create here)
    output_zarr_array = zarr.open(output_zarr_url, mode="r+")

    # Non-spatial coord for this block (e.g., T/C)
    ns_coord = {dim: block_id[idim] for idim, dim in enumerate(nsdims)}

    spatial_chunk_ind = block_id[len(nsdims):]

    # Chunk offset (voxel)
    chunk_offset = {
        sdims[idim]: int(np.sum(normalized_chunks[len(nsdims) + idim][:b])) if b > 0 else 0
        for idim, b in enumerate(spatial_chunk_ind)
    }

    # Chunk offset (physical)
    chunk_offset_phys = {
        dim: chunk_offset[dim] * float(osp["spacing"][dim]) + float(osp["origin"][dim])
        for dim in sdims
    }

    # Chunk shape
    chunk_shape = {
        sdims[idim]: int(normalized_chunks[len(nsdims) + idim][b])
        for idim, b in enumerate(spatial_chunk_ind)
    }

    # Compute fused chunk by calling fuse() WITHOUT output_zarr_url (avoid recursion)
    fused = fuse(
        sims=fuse_kwargs["sims"],
        **{k: v for k, v in fuse_kwargs.items() if k != "sims"},
        output_origin={dim: chunk_offset_phys[dim] for dim in sdims},
        output_shape={dim: chunk_shape[dim] for dim in sdims},
        output_spacing={dim: float(osp["spacing"][dim]) for dim in sdims},
        output_zarr_url=None,
    ).data
    if drop_t:
        fused = fused[0]

    # Slice out the non-spatial single index (keeps your behavior)
    fused = fused[tuple(slice(ns_coord[dim], ns_coord[dim] + 1) for dim in nsdims)]

    region = tuple(
        [slice(ns_coord[dim], ns_coord[dim] + 1) for dim in nsdims]
        + [slice(chunk_offset[dim], chunk_offset[dim] + chunk_shape[dim]) for dim in sdims]
    )

    # Ensure a local scheduler inside the worker
    with dask_config.set(scheduler="single-threaded"):
        #da.to_zarr(fused, output_zarr_array, region=region)
        fused_block = fused.compute()
    output_zarr_array[region] = fused_block

    return block_id

def prepare_block_fusion(
    output_zarr_url: str,
    fuse_kwargs: dict,
    zarr_array_creation_kwargs: dict | None = None,
):
    """
    Prepare picklable metadata for parallel block fusion.

    - Creates the Zarr array ONCE in the main process.
    - Returns only picklable objects (dicts/lists/ints).
    """

    # IMPORTANT: do not mutate caller's dict (your original pops mutate!)
    fuse_kwargs = dict(fuse_kwargs)

    sims = fuse_kwargs["sims"]

    # Pop only from our local copy
    output_stack_properties = process_output_stack_properties(
        sims=sims,
        output_stack_properties=fuse_kwargs.pop("output_stack_properties", None),
        output_spacing=fuse_kwargs.pop("output_spacing", None),
        output_origin=fuse_kwargs.pop("output_origin", None),
        output_shape=fuse_kwargs.pop("output_shape", None),
        output_stack_mode=fuse_kwargs.pop("output_stack_mode", "union"),
        transform_key=fuse_kwargs.get("transform_key", None),
    )

    output_chunksize = process_output_chunksize(
        sims, fuse_kwargs.get("output_chunksize", None)
    )

    dims = sims[0].dims
    nsdims = si_utils.get_nonspatial_dims_from_sim(sims[0])
    drop_t = fuse_kwargs.get("drop_t", False)
    if drop_t:
        nsdims.remove("t")
        dims = (i for i in dims if i != 't')
    sdims = si_utils.get_spatial_dims_from_sim(sims[0])
    ns_shape = {dim: len(sims[0].coords[dim]) for dim in nsdims}

    full_output_shape = [ns_shape[dim] for dim in nsdims]\
         + [output_stack_properties['shape'][dim] for dim in sdims]

    full_output_chunksize = [1,] * len(nsdims)\
         + [int(output_chunksize[dim]) for dim in sdims]
    
    normalized_chunks = normalize_chunks(
        shape=full_output_shape,
        chunks=full_output_chunksize)
    
    logger.info(f"Fusing into a an output stack:")
    logger.info(f"- shape: ", {dim: int(output_stack_properties['shape'][dim])
        if dim in sdims else ns_shape[dim] for dim in dims})
    logger.info(f"- spacing: ", {k: float(v)
        for k, v in output_stack_properties['spacing'].items()})
    logger.info(f"- origin: ", {k: float(v)
        for k, v in output_stack_properties['origin'].items()})

    # Create Zarr array ONCE
    zarr.create(
        shape=full_output_shape,
        chunks=full_output_chunksize,
        dtype=sims[0].data.dtype,
        store=output_zarr_url,
        overwrite=True,
        **(zarr_array_creation_kwargs or {}),
    )

    nblocks = [len(nc) for nc in normalized_chunks]

    # Everything below is picklable
    meta = {
        "output_zarr_url": output_zarr_url,
        "output_stack_properties": output_stack_properties,
        "output_chunksize": {k: int(v) for k, v in output_chunksize.items()},
        "nsdims": list(nsdims),
        "sdims": list(sdims),
        "drop_t": drop_t,
        "ns_shape": {k: int(v) for k, v in ns_shape.items()},
        "normalized_chunks": normalized_chunks,  # list-of-lists of ints
        "nblocks": nblocks,
    }

    # Return fuse_kwargs with the popped items removed (as you intended)
    return {
        "meta": meta,
        "fuse_kwargs": fuse_kwargs,  # still contains sims + other fuse settings
        "nblocks": nblocks,
        "output_stack_properties": output_stack_properties,
    }

def fuse(
    sims: list,
    transform_key: Optional[str] = None,
    fusion_func: Callable = weighted_average_fusion,
    fusion_method_kwargs: Optional[dict] = None,
    weights_func: Optional[Callable] = None,
    weights_func_kwargs: Optional[dict] = None,
    drop_t: bool = False,
    output_spacing: Optional[dict[str, float]] = None,
    output_stack_mode: str = "union",
    output_origin: Optional[dict[str, float]] = None,
    output_shape: Optional[dict[str, int]] = None,
    output_stack_properties: Optional[BoundingBox] = None,
    output_chunksize: Optional[Union[int, dict[str, int]]] = None,
    overlap_in_pixels: Optional[int] = None,
    interpolation_order: int = 1,
    blending_widths: Optional[dict[str, float]] = None,
    output_zarr_url: str | None = None,
    zarr_options: dict | None = None,
    batch_options: dict | None = None,
):
    """

    Fuse input views.

    This function fuses all (Z)YX views ("fields") contained in the
    input list of images, which can additionally contain C and T dimensions.

    Parameters
    ----------
    sims : list of SpatialImage
        Input views.
    transform_key : str, optional
        Which (extrinsic coordinate system) to use as transformation parameters.
        By default None (intrinsic coordinate system).
    fusion_func : Callable, optional
        Fusion function to be applied. This function receives the following
        inputs (as arrays if applicable): transformed_views, blending_weights, fusion_weights, params.
        By default weighted_average_fusion
    fusion_method_kwargs : dict, optional
    weights_func : Callable, optional
        Function to calculate fusion weights. This function receives the
        following inputs: transformed_views (as spatial images), params.
        It returns (non-normalized) fusion weights for each view.
        By default None.
    weights_func_kwargs : dict, optional,
    drop_t: bool, optional
        Whether to drop the time dimension from the input views. By default False.
    output_spacing : dict, optional
        Spacing of the fused image for each spatial dimension, by default None
    output_stack_mode : str, optional
        Mode to determine output stack properties. Can be one of
        "union", "intersection", "sample". By default "union"
    output_origin : dict, optional
        Origin of the fused image for each spatial dimension, by default None
    output_shape : dict, optional
        Shape of the fused image for each spatial dimension, by default None
    output_stack_properties : dict, optional
        Dictionary describing the output stack with keys
        'spacing', 'origin', 'shape'. Other output_* are ignored
        if this argument is present.
    output_chunksize : int or dict, optional
        Chunksize of the dask data array of the fused image. If the first tile is a chunked dask array,
        its chunksize is used as the default. If the first tile is not a chunked dask array,
        the default chunksize defined in spatial_image_utils.py is used.
    output_zarr_url : str or None, optional
        If not None, fuse directly into a Zarr store at this location and do so in batches of chunks,
        with each chunk being processed independently. This allows for efficient memory usage and
        works well for very large datasets (successfully tested ~0.5PB on a macbook).
        When provided, fuse() performs eager fusion and returns a SpatialImage backed by the written store.
    zarr_options: dict, optional
        Additional (dict of) options to pass when creating the Zarr store. Keys:
        - ome_zarr : bool, optional
            If True and output_zarr_url is provided, write a NGFF/OME-Zarr multiscale image under
            "<output_zarr_url>/". Otherwise, the fused array is written directly under output_zarr_url.
        - ngff_version : str, optional
            NGFF version used when ome_zarr=True. Default "0.4".
        - zarr_array_creation_kwargs: dict = None, optional
            Additional keyword arguments to pass when creating the Zarr array.
        - overwrite: bool, by default True
    batch_options : dict, optional
        Options for chunked fusion when output_zarr_url is provided. Keys:
        - batch_func: Callable, optional
            Function to process each batch of fused chunks. Inputs:
            1) block_id(s)
            2) function that performs fusion when passed a given block_id
            By default None, in which case each block is processed sequentially.
        - max_workers: int
            Number of workers to process the blocks in parallel
            (max_workers=1 means sequential processing). By default 1.
        - batch_func_kwargs: dict, optional
            Additional keyword arguments passed to batch_func.
    Returns
    -------
    SpatialImage
        Fused image.
    """
    # If writing directly to Zarr/OME-Zarr, run chunked fusion path and return eagerly.
    if output_zarr_url is not None:
        
        # Collect batch options with defaults
        batch_options = batch_options or {}
        batch_func = batch_options.get("batch_func", None)
        max_workers = batch_options.get("max_workers", 1)
        batch_func_kwargs = batch_options.get("batch_func_kwargs", None)
        zarr_array_creation_kwargs = batch_options.get("zarr_array_creation_kwargs", None)

        # Collect zarr options with defaults
        zarr_options = zarr_options or {}
        ome_zarr = zarr_options.get("ome_zarr", False)
        ngff_version = zarr_options.get("ngff_version", "0.4")
        overwrite = zarr_options.get("overwrite", True)

        # Resolve store path for data (OME-Zarr stores scale 0 under "<root>/0")
        store_url = os.path.join(output_zarr_url, "0") if ome_zarr else output_zarr_url

        if overwrite and os.path.exists(store_url):
            shutil.rmtree(store_url)
        if ome_zarr:
            # Ensure creation kwargs reflect NGFF version when writing OME-Zarr
            zarr_array_creation_kwargs = ngff_utils.update_zarr_array_creation_kwargs_for_ngff_version(
                ngff_version, zarr_array_creation_kwargs
            )

        # Build kwargs for per-chunk fuse() calls (exclude zarr-specific args to avoid recursion)
        per_chunk_fuse_kwargs = {
            "sims": sims,
            "transform_key": transform_key,
            "fusion_func": fusion_func,
            "fusion_method_kwargs": fusion_method_kwargs,
            "drop_t": drop_t,
            "weights_func": weights_func,
            "weights_func_kwargs": weights_func_kwargs,
            "output_spacing": output_spacing,
            "output_stack_mode": output_stack_mode,
            "output_origin": output_origin,
            "output_shape": output_shape,
            "output_stack_properties": output_stack_properties,
            "output_chunksize": output_chunksize,
            "overlap_in_pixels": overlap_in_pixels,
            "interpolation_order": interpolation_order,
            "blending_widths": blending_widths,
        }

        # Prepare block fusion and process in batches
        block_fusion_info = prepare_block_fusion(
            store_url,
            fuse_kwargs=per_chunk_fuse_kwargs,
            zarr_array_creation_kwargs=zarr_array_creation_kwargs,
        )

        nblocks = block_fusion_info["nblocks"]
        meta = block_fusion_info["meta"]
        worker_fuse_kwargs = block_fusion_info["fuse_kwargs"]

        batch_func_kwargs = (batch_func_kwargs or {})
        batch_func_kwargs.update({
            "meta": meta,
            "fuse_kwargs": worker_fuse_kwargs,
            "max_workers": max_workers,
        })

        if batch_func is None:
            print(f'Fusing {np.prod(nblocks)} blocks sequentially...')
            #for batch in tqdm(
            #    misc_utils.ndindex_batches(nblocks, n_batch),
            #    total=int(np.ceil(np.prod(nblocks) / n_batch)),
            #):
                    
            # Sequential fallback
            all_block_ids = list(np.ndindex(*nblocks))
            for block_id in all_block_ids:
                fuse_one_block_worker(block_id, meta=meta, 
                                      fuse_kwargs=worker_fuse_kwargs)

        else:
            batch_func(nblocks=nblocks, **batch_func_kwargs)

        osp = block_fusion_info["output_stack_properties"]
        osp["shape"] = {dim: int(v) for dim, v in osp["shape"].items()}

        #for batch in tqdm(
        #    misc_utils.ndindex_batches(nblocks, n_batch),
        #    total=int(np.ceil(np.prod(nblocks) / n_batch)),
        #):
        #    if batch_func is None:
        #        for block_id in batch:
        #            fuse_chunk(block_id)
        #    else:
        #        batch_func(fuse_chunk, batch, **(batch_func_kwargs or {}))

        # Build SpatialImage from zarr array
        fusion_transform_key = transform_key
        if drop_t:
            sim_dims = None
        else:
            sim_dims = list(sims[0].dims)
        fused = si_utils.get_sim_from_array(
            array=da.from_zarr(store_url),
            dims=sim_dims,
            transform_key=fusion_transform_key,
            scale=osp["spacing"],
            translation=osp["origin"],
            c_coords=sims[0].coords["c"].values,
            t_coords=sims[0].coords["t"].values,
        )

        # If requested, write OME-Zarr metadata
        # and multiscale pyramid
        if ome_zarr:
            ngff_utils.write_sim_to_ome_zarr(
                fused,
                output_zarr_url=output_zarr_url,
                overwrite=False,
                batch_options=batch_options,
            )

        return fused

    # Default in-memory fusion path (unchanged)
    output_chunksize = process_output_chunksize(sims, output_chunksize)

    output_stack_properties = process_output_stack_properties(
        sims=sims,
        output_spacing=output_spacing,
        output_origin=output_origin,
        output_shape=output_shape,
        output_stack_properties=output_stack_properties,
        output_stack_mode=output_stack_mode,
        transform_key=transform_key,
    )

    sdims = si_utils.get_spatial_dims_from_sim(sims[0])
    nsdims = si_utils.get_nonspatial_dims_from_sim(sims[0])

    params = [
        si_utils.get_affine_from_sim(sim, transform_key=transform_key)
        for sim in sims
    ]

    # determine overlap from weights method
    # (soon: fusion methods will also require overlap)
    overlap_in_pixels = 0
    if weights_func is not None:
        overlap_in_pixels = np.max(
            [
                overlap_in_pixels,
                weights.calculate_required_overlap(
                    weights_func, weights_func_kwargs
                ),
            ]
        )

    # calculate output chunk bounding boxes
    output_chunk_bbs, block_indices = mv_graph.get_chunk_bbs(
        output_stack_properties, output_chunksize # type: ignore
    )

    # add overlap to output chunk bounding boxes
    output_chunk_bbs_with_overlap = [
        output_chunk_bb
        | {
            "origin": {
                dim: output_chunk_bb["origin"][dim] # type: ignore
                - overlap_in_pixels * output_stack_properties["spacing"][dim]  # type: ignore
                for dim in sdims 
            }
        } # type: ignore
        | {
            "shape": {
                dim: output_chunk_bb["shape"][dim] + 2 * overlap_in_pixels # type: ignore
                for dim in sdims
            }
        }
        for output_chunk_bb in output_chunk_bbs
    ]

    views_bb = [si_utils.get_stack_properties_from_sim(sim) for sim in sims]

    merges = []
    for ns_coords in itertools.product(
        *tuple([sims[0].coords[nsdim] for nsdim in nsdims])
    ):
        sim_coord_dict = {
            ndsim: ns_coords[i] for i, ndsim in enumerate(nsdims)
        }
        params_coord_dict = {
            ndsim: ns_coords[i]
            for i, ndsim in enumerate(nsdims)
            if ndsim in params[0].dims
        }

        # ssims = [sim.sel(sim_coord_dict) for sim in sims]
        sparams = [param.sel(params_coord_dict) for param in params]

        # should this be done within the loop over output chunks?
        fix_dims = []
        for dim in sdims:
            other_dims = [odim for odim in sdims if odim != dim]
            if (
                any((param.sel(x_in=dim, x_out=dim) - 1) for param in sparams)
                or any(
                    any(param.sel(x_in=dim, x_out=other_dims))
                    for param in sparams
                )
                or any(
                    any(param.sel(x_in=other_dims, x_out=dim))
                    for param in sparams
                )
                or any(
                    output_stack_properties["spacing"][dim]
                    - views_bb[iview]["spacing"][dim]
                    for iview in range(len(sims))
                )
                or any(
                    float(
                        output_stack_properties["origin"][dim]
                        - param.sel(x_in=dim, x_out="1")
                    )
                    % output_stack_properties["spacing"][dim]
                    for param in sparams
                )
            ):
                continue
            fix_dims.append(dim)

        fused_output_chunks = np.empty(
            np.max(block_indices, 0) + 1, dtype=object
        )

        for output_chunk_bb, output_chunk_bb_with_overlap, block_index in zip(
            output_chunk_bbs, output_chunk_bbs_with_overlap, block_indices
        ):
            # calculate relevant slices for each output chunk
            # this is specific to each non spatial coordinate
            views_overlap_bb = [
                mv_graph.get_overlap_for_bbs(
                    target_bb=output_chunk_bb_with_overlap,
                    query_bbs=[view_bb],
                    param=sparams[iview],
                    additional_extent_in_pixels={
                        dim: 0 if dim in fix_dims else int(interpolation_order)
                        for dim in sdims
                    },
                )[0]
                for iview, view_bb in enumerate(views_bb)
            ]

            # append to output
            relevant_view_indices = np.where(
                [
                    view_overlap_bb is not None
                    for view_overlap_bb in views_overlap_bb
                ]
            )[0]

            if not len(relevant_view_indices):
                fused_output_chunks[tuple(block_index)] = da.zeros(
                    tuple([output_chunk_bb["shape"][dim] for dim in sdims]),
                    dtype=sims[0].dtype,
                )
                continue

            tol = 1e-6
            sims_slices = [
                sims[iview].sel(
                    sim_coord_dict
                    | {
                        dim: slice(
                            views_overlap_bb[iview]["origin"][dim] - tol,
                            views_overlap_bb[iview]["origin"][dim]
                            + (views_overlap_bb[iview]["shape"][dim] - 1)
                            * views_overlap_bb[iview]["spacing"][dim]
                            + tol,
                        )
                        for dim in sdims
                    },
                    drop=True,
                )
                for iview in relevant_view_indices
            ]

            # determine whether to fuse plany by plane
            #  to avoid weighting edge artifacts
            # fuse planewise if:
            # - z dimension is present
            # - params don't affect z dimension
            # - shape in z dimension is 1 (i.e. only one plane)
            # (the last criterium above could be dropped if we find a way
            # (to propagate metadata through xr.apply_ufunc)

            if (
                "z" in fix_dims
                and output_chunk_bb_with_overlap["shape"]["z"] == 1
            ):
                fuse_planewise = True

                sims_slices = [sim.isel(z=0) for sim in sims_slices]
                tmp_params = [
                    sparams[iview].sel(
                        x_in=["y", "x", "1"],
                        x_out=["y", "x", "1"],
                    )
                    for iview in relevant_view_indices
                ]

                output_chunk_bb_with_overlap = mv_graph.project_bb_along_dim(
                    output_chunk_bb_with_overlap, dim="z"
                )

                full_view_bbs = [
                    mv_graph.project_bb_along_dim(views_bb[iview], dim="z")
                    for iview in relevant_view_indices
                ]

            else:
                fuse_planewise = False
                tmp_params = [
                    sparams[iview] for iview in relevant_view_indices
                ]
                full_view_bbs = [
                    views_bb[iview] for iview in relevant_view_indices
                ]

            fused_output_chunk = delayed(
                lambda append_leading_axis, **kwargs: fuse_np(**kwargs)[
                    np.newaxis
                ]
                if append_leading_axis
                else fuse_np(**kwargs),
            )(
                append_leading_axis=fuse_planewise,
                sims=sims_slices,
                params=tmp_params,
                output_properties=output_chunk_bb_with_overlap,
                fusion_func=fusion_func,
                fusion_method_kwargs=fusion_method_kwargs,
                weights_func=weights_func,
                weights_func_kwargs=weights_func_kwargs,
                trim_overlap_in_pixels=overlap_in_pixels,
                interpolation_order=1,
                full_view_bbs=full_view_bbs,
                blending_widths=blending_widths,
            )

            fused_output_chunk = da.from_delayed(
                fused_output_chunk,
                shape=tuple([output_chunk_bb["shape"][dim] for dim in sdims]),
                dtype=sims[0].dtype,
            )

            fused_output_chunks[tuple(block_index)] = fused_output_chunk

        fused = da.block(fused_output_chunks.tolist())

        merge = si.to_spatial_image(
            fused,
            dims=sdims,
            scale=output_stack_properties["spacing"],
            translation=output_stack_properties["origin"],
        )

        merge = merge.expand_dims(nsdims)
        merge = merge.assign_coords(
            {ns_coord.name: [ns_coord.values] for ns_coord in ns_coords}
        )
        merges.append(merge)

    if len(merges) > 1:
        # suppress pandas future warning occuring within xarray.concat
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)

            # if sims are named, combine_by_coord returns a dataset
            res = xr.combine_by_coords([m.rename(None) for m in merges])
    else:
        res = merge

    res = si_utils.get_sim_from_xim(res)
    si_utils.set_sim_affine(
        res,
        param_utils.identity_transform(len(sdims)),
        transform_key,
    )

    # order channels in the same way as first input sim
    # (combine_by_coords may change coordinate order)
    if "c" in res.dims:
        res = res.sel({"c": sims[0].coords["c"].values})

    return res

def phase_correlation_registration(
    fixed_data,
    moving_data,
    disambiguate_region_mode=None,
    **skimage_phase_corr_kwargs,
):
    """
    Phase correlation registration using a modified version of skimage's
    phase_cross_correlation function.

    Parameters
    ----------
    fixed_data : array-like
    moving_data : array-like

    Returns
    -------
    dict
        'affine_matrix' : array-like
            Homogeneous transformation matrix.
        'quality' : float
            Quality metric.
    """

    im0 = fixed_data.data
    im1 = moving_data.data
    ndim = im0.ndim

    # normalize images
    im0, im1 = (
        rescale_intensity(
            im,
            in_range=(np.nanmin(im), np.nanmax(im)),
            out_range=(0, 1),
        )
        for im in [im0, im1]
    )

    im0nm = np.isnan(im0)
    im1nm = np.isnan(im1)

    # use intersection mode if there are nan pixels in either image
    if disambiguate_region_mode is None:
        if np.any([im0nm, im1nm]):
            disambiguate_region_mode = "intersection"
        else:
            disambiguate_region_mode = "union"

    valid_pixels1 = np.sum(~im1nm)

    if np.any([im0nm, im1nm]):
        im0nn = np.nan_to_num(im0)
        im1nn = np.nan_to_num(im1)
    else:
        im0nn = im0
        im1nn = im1

    if "upsample_factor" not in skimage_phase_corr_kwargs:
        skimage_phase_corr_kwargs["upsample_factor"] = 10 if ndim == 2 else 2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # strategy: compute phase correlation with and without
        # normalization and keep the one with the highest
        # structural similarity score during manual "disambiguation"
        # (which should be a metric orthogonal to the corr coef)

        shift_candidates = []
        for normalization in ["phase", None]:
            shift_candidates.append(
                phase_cross_correlation(
                    im0nn,
                    im1nn,
                    disambiguate=False,
                    normalization=normalization,
                    **skimage_phase_corr_kwargs,
                )[0]
            )

        if np.any([im0nm, im1nm]):
            shift_candidates.append(
                phase_cross_correlation(
                    im0,
                    im1,
                    reference_mask=~im0nm,
                    moving_mask=~im1nm,
                    disambiguate=False,
                    **skimage_phase_corr_kwargs,
                )[0]
            )

    # disambiguate shift manually
    # there seems to be a problem with the scikit-image implementation
    # of disambiguate_shift, but this needs to be checked

    # assume that the shift along any dimension isn't larger than the overlap
    # in the dimension with smallest overlap
    # e.g. if overlap is 50 pixels in x and 200 pixels in y, assume that
    # the shift along x and y is smaller than 50 pixels
    max_shift_per_dim = np.min([im.shape for im in [im0, im1]])

    data_range = np.nanmax([im0, im1]) - np.nanmin([im0, im1])
    im1_min = np.nanmin(im1)

    disambiguate_metric_vals = []
    quality_metric_vals = []

    t_candidates = []
    for shift_candidate in shift_candidates:
        for s in np.ndindex(
            tuple([1 if shift_candidate[d] == 0 else 4 for d in range(ndim)])
        ):
            t_candidate = []
            for d in range(ndim):
                if s[d] == 0:
                    t_candidate.append(shift_candidate[d])
                elif s[d] == 1:
                    t_candidate.append(-shift_candidate[d])
                elif s[d] == 2:
                    t_candidate.append(-(shift_candidate[d] - im1.shape[d]))
                elif s[d] == 3:
                    t_candidate.append(-shift_candidate[d] - im1.shape[d])
            if np.max(np.abs(t_candidate)) < max_shift_per_dim:
                t_candidates.append(t_candidate)

    if not len(t_candidates):
        t_candidates = [[np.float32(0) for i in range(ndim)]]

    def get_bb_from_nanmask(mask):
        bbs = []
        for idim in range(mask.ndim):
            axes = list(range(mask.ndim))
            axes.remove(idim)
            valids = np.where(np.max(mask, axis=tuple(axes)))
            bbs.append([np.min(valids), np.max(valids)])
        return bbs

    im0_bb = get_bb_from_nanmask(~im0nm)

    for t_ in t_candidates:
        im1t = ndimage.affine_transform(
            im1,
            param_utils.affine_from_translation(list(t_)),
            order=1,
            mode="constant",
            cval=np.nan,
        )
        mask = ~np.isnan(im1t) * ~im0nm

        if np.all(~mask) or float(np.sum(mask)) / valid_pixels1 < 0.1:
            disambiguate_metric_val = -1
            quality_metric_val = -1
        else:
            im1t_bb = get_bb_from_nanmask(~np.isnan(im1t))

            if disambiguate_region_mode == "union":
                mask_slices = tuple(
                    [
                        slice(
                            min(im0_bb[idim][0], im1t_bb[idim][0]),
                            max(im0_bb[idim][1], im1t_bb[idim][1]) + 1,
                        )
                        for idim in range(ndim)
                    ]
                )
            elif disambiguate_region_mode == "intersection":
                mask_slices = tuple(
                    [
                        slice(
                            max(im0_bb[idim][0], im1t_bb[idim][0]),
                            min(im0_bb[idim][1], im1t_bb[idim][1]) + 1,
                        )
                        for idim in range(ndim)
                    ]
                )

            if np.nanmax(im1t[mask_slices]) <= im1_min:
                disambiguate_metric_val = -1
                quality_metric_val = -1
                continue

            # structural similarity seems to be better than
            # correlation for disambiguation (need to solidify this)
            min_shape = np.min(im0[mask_slices].shape)
            ssim_win_size = np.min([7, min_shape - ((min_shape - 1) % 2)])
            if ssim_win_size < 3 or np.max(im1t[mask_slices]) <= im1_min:
                logger.debug("SSIM window size too small")
                disambiguate_metric_val = -1
            else:
                disambiguate_metric_val = structural_similarity(
                    np.nan_to_num(im0[mask_slices]),
                    np.nan_to_num(im1t[mask_slices]),
                    data_range=data_range,
                    win_size=ssim_win_size,
                )
            # spearman seems to be better than structural_similarity
            # for filtering out bad links between views
            quality_metric_val = link_quality_metric_func(
                im0[mask], im1t[mask] - 1
            )

        disambiguate_metric_vals.append(disambiguate_metric_val)
        quality_metric_vals.append(quality_metric_val)
    argmax_index = np.nanargmax(disambiguate_metric_vals)
    t = t_candidates[argmax_index]

    reg_result = {}
    reg_result["affine_matrix"] = param_utils.affine_from_translation(t)
    reg_result["quality"] = quality_metric_vals[argmax_index]

    return reg_result

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
    def verify_label(self, zarr_url) -> None:
        if self.label is not None:
            channels = get_omero_channel_list(image_zarr_path=zarr_url)
            concat_labels = " ".join([c.label for c in channels])
            if self.label.lower() not in concat_labels.lower():
                raise ValueError(
                    f"Label {self.label} not found in {Path(zarr_url).name}."
                )
            else:
                for c in channels:
                    if self.label.lower() in c.label.lower():
                        self.label = c.label
                        break

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