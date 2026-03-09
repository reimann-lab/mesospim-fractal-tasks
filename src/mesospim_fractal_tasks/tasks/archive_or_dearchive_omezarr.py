import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

import shutil
import tarfile
import copy
from typing import Iterable, Optional

import numcodecs
from numcodecs import Blosc
import logging
from pathlib import Path
import zarr
import dask.array as da
from pydantic import validate_call
from dask.distributed import Client
from mesospim_fractal_tasks.utils.parallelisation import _set_dask_cluster


numcodecs.blosc.set_nthreads(1)

logger = logging.getLogger(__name__)


def get_dimension_separator(arr: zarr.Array):
    # Works for common zarr v2 usage
    return getattr(arr, "_dimension_separator", None)


def rewrite_omezarr_images_only(
    src_path: str | Path,
    dst_path: str | Path,
    image_compressor,
    overwrite: bool = False,
) -> None:
    """
    Copy an OME-Zarr and rewrite only image arrays with a new compressor.
    labels/ and tables/ arrays are copied unchanged.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_path}")

    if dst_path.exists():
        if overwrite:
            shutil.rmtree(dst_path)
        else:
            raise FileExistsError(f"Destination already exists: {dst_path}")

    src_root = zarr.open_group(src_path, mode="r")
    dst_root = zarr.group(dst_path, overwrite=True)

    image_dataset_paths = collect_image_dataset_paths(src_root)

    recreate_hierarchy_and_copy(
        source_root=src_root,
        archive_root=dst_root,
        image_dataset_paths=image_dataset_paths,
        compressor=image_compressor,
    )

def rewrite_array_with_new_compressor(
    src_arr: zarr.Array,
    dst_parent: zarr.Group,
    name: str,
    compressor,
) -> zarr.Array:
    dst_arr = dst_parent.create_dataset(
        name=name,
        shape=src_arr.shape,
        chunks=src_arr.chunks,  # keep chunking unchanged
        dtype=src_arr.dtype,
        compressor=compressor,
        fill_value=src_arr.fill_value,
        order=src_arr.order,
        filters=src_arr.filters,
        dimension_separator=get_dimension_separator(src_arr),
        overwrite=True,
    )
    dst_arr.attrs.update(dict(src_arr.attrs))
    dst_arr[:] = src_arr[:]
    return dst_arr

def collect_multiscale_groups(root: zarr.Group) -> dict[str, list[str]]:
    """
    Return:
        image_group_path -> list of full dataset paths in pyramid order

    Skip anything under labels/ and tables/.
    """
    out: dict[str, list[str]] = {}

    def recurse(group: zarr.Group, group_path: str = "") -> None:
        if is_under_any(group_path, ("labels", "tables")):
            return

        attrs = dict(group.attrs)
        multiscales = attrs.get("multiscales")
        if multiscales:
            dataset_paths: list[str] = []
            for ms in multiscales:
                for ds in ms.get("datasets", []):
                    rel = ds.get("path")
                    if rel is None:
                        continue
                    full = normalize_zarr_path(
                        str(Path(group_path) / rel) if group_path else rel
                    )
                    if not is_under_any(full, ("labels", "tables")):
                        dataset_paths.append(full)

            if dataset_paths:
                out[normalize_zarr_path(group_path)] = dataset_paths

        for key in group.group_keys():
            sub = group[key]
            sub_path = normalize_zarr_path(str(Path(group_path) / key) if group_path else key)
            recurse(sub, sub_path)

    recurse(root)
    return out

def build_preview_omezarr(
    src_root: zarr.Group,
    preview_root: zarr.Group,
) -> None:
    """
    Create a tiny OME-Zarr containing only the last pyramid level
    for each multiscale image group.
    """

    root_attrs = dict(src_root.attrs)
    preview_root.attrs.update(root_attrs)

    multiscale_groups = collect_multiscale_groups(src_root)

    for image_group_path, dataset_paths in multiscale_groups.items():
        if not dataset_paths:
            continue

        last_dataset_path = dataset_paths[-1]

        if image_group_path:
            src_group = src_root[image_group_path]
            dst_group = preview_root.require_group(image_group_path)
        else:
            src_group = src_root
            dst_group = preview_root

        # Copy non-multiscales attrs, then rebuild multiscales to only point to "0"
        group_attrs = dict(src_group.attrs)
        multiscales = group_attrs.pop("multiscales", None)
        dst_group.attrs.update(group_attrs)

        src_arr = src_root[last_dataset_path]

        copy_array_verbatim(
            src_arr=src_arr,
            dst_parent=dst_group,
            name="0",
        )

        if multiscales:
            num_levels = len(multiscales[0]["datasets"])
            new_multiscales = copy.deepcopy(multiscales)
            for ms in new_multiscales:
                ms["datasets"] = [{"path": "0", "coordinateTransformations": ms["datasets"][num_levels-1]["coordinateTransformations"]}]
            dst_group.attrs["multiscales"] = new_multiscales

def create_preview_from_omezarr(
    src_path: str | Path,
    preview_path: str | Path,
    overwrite: bool = False,
) -> None:
    src_path = Path(src_path)
    preview_path = Path(preview_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_path}")

    if preview_path.exists():
        if overwrite:
            shutil.rmtree(preview_path)
        else:
            raise FileExistsError(f"Preview path already exists: {preview_path}")
        
    logger.info(f"Building downsampled OME-Zarr version {preview_path.name} for fast-viewing.")

    src_root = zarr.open_group(src_path, mode="r")
    preview_root = zarr.group(preview_path, overwrite=True)

    build_preview_omezarr(
        src_root=src_root,
        preview_root=preview_root,
    )

def normalize_zarr_path(path: str) -> str:
    """
    Normalize a Zarr internal path to a clean POSIX-like form.
    """
    return str(Path(path)).replace("\\", "/").strip("/")


def is_under_any(path: str, prefixes: Iterable[str]) -> bool:
    """
    Return True if `path` is equal to or inside any of the given prefixes.
    """
    p = normalize_zarr_path(path)
    for prefix in prefixes:
        prefix = normalize_zarr_path(prefix)
        if p == prefix or p.startswith(prefix + "/"):
            return True
    return False


def copy_group_attrs(src: zarr.Group, dst: zarr.Group) -> None:
    """
    Copy group attributes.
    """
    dst.attrs.update(dict(src.attrs))


def copy_array_verbatim(src_arr: zarr.Array, dst_parent: zarr.Group, name: str) -> zarr.Array:
    """
    Copy an array with the same metadata and chunk data.
    This preserves the original compressor.
    """
    dst_arr = dst_parent.create_dataset(
        name=name,
        shape=src_arr.shape,
        chunks=src_arr.chunks,
        dtype=src_arr.dtype,
        compressor=src_arr.compressor,
        fill_value=src_arr.fill_value,
        order=src_arr.order,
        filters=src_arr.filters,
        dimension_separator=getattr(src_arr, "_dimension_separator", None),
        overwrite=True,
    )
    dst_arr.attrs.update(dict(src_arr.attrs))
    
    # Copy without changing chunks
    if "tables" in str(dst_parent.name):
        dst_arr[:] = src_arr[:]
    else:
        src_arr = da.from_zarr(src_arr)
        if len(dst_arr.shape) == 4:
            z_idx = 1
        else:
            z_idx = 0
        z_chunk = dst_arr.chunks[z_idx]
        z_end = dst_arr.shape[z_idx]
        for z in range(0, z_end, z_chunk):
            region = (
                slice(None),
                slice(z, z+z_chunk),
                slice(None),
                slice(None))
            region = region[abs(z_idx-1):]
            src_arr[region].to_zarr(dst_arr, region=region, compute=True)
    return dst_arr


def recompress_array(
    src_arr: zarr.Array,
    dst_parent: zarr.Group,
    name: str,
    compressor,
) -> zarr.Array:
    """
    Recreate one Zarr v2 array with the same metadata except compressor,
    and copy the data over without rechunking.
    """
    dst_arr = dst_parent.create_dataset(
        name=name,
        shape=src_arr.shape,
        chunks=src_arr.chunks,  # keep original chunks: no rechunking
        dtype=src_arr.dtype,
        compressor=compressor,
        fill_value=src_arr.fill_value,
        order=src_arr.order,
        filters=src_arr.filters,
        dimension_separator=getattr(src_arr, "_dimension_separator", None),
        overwrite=True,
    )
    dst_arr.attrs.update(dict(src_arr.attrs))
    src_arr = da.from_zarr(src_arr)

    # Copy data chunk-by-chunk via regular array assignment.
    # This rewrites data with the new compressor but same chunk grid.
        # Copy without changing chunks
    if len(dst_arr.shape) == 3:
        z_idx = 0
    else:
        z_idx = 1
    z_chunk = dst_arr.chunks[z_idx]
    z_end = dst_arr.shape[z_idx]
    for z in range(0, z_end, z_chunk):
        region = (
            slice(None),
            slice(z, z+z_chunk),
            slice(None),
            slice(None))
        region = region[abs(z_idx-1):]
        src_arr[region].to_zarr(dst_arr, region=region, compute=True)
    return dst_arr


def collect_image_dataset_paths(root: zarr.Group) -> set[str]:
    """
    Find image dataset paths from OME-NGFF multiscales metadata, excluding
    anything inside labels/ or tables/.
    """
    image_dataset_paths: set[str] = set()

    for group_path, group in root.groups():
        gpath = normalize_zarr_path(group_path)

        if is_under_any(gpath, ("labels", "tables")):
            continue

        attrs = dict(group.attrs)
        multiscales = attrs.get("multiscales")
        if not multiscales:
            continue

        for ms in multiscales:
            for ds in ms.get("datasets", []):
                rel = ds.get("path")
                if rel is None:
                    continue
                full = normalize_zarr_path(str(Path(gpath) / rel) if gpath else rel)

                if is_under_any(full, ("labels", "tables")):
                    continue

                image_dataset_paths.add(full)

    return image_dataset_paths

def tar_directory(source_dir: Path, tar_path: Path) -> None:
    """
    Create an uncompressed .tar archive containing `source_dir`.
    """
    logger.info(f"Creating TAR archive file {tar_path.name}.")
    with tarfile.open(tar_path, mode="w") as tf:
        tf.add(source_dir, arcname=source_dir.name)

def untar_to_directory(tar_path: Path, extract_dir: Path) -> Path:
    """
    Extract tar into extract_dir and return the path to the extracted top-level folder.
    Assumes the tar contains one top-level ome-zarr directory.
    """
    if not tar_path.exists():
        raise FileNotFoundError(f"TAR file does not exist: {tar_path}")

    if extract_dir.exists():
        raise FileExistsError(f"Extraction directory already exists: {extract_dir}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, mode="r") as tf:
        members = tf.getmembers()
        if not members:
            raise ValueError(f"Empty TAR archive: {tar_path}")

        top_levels = {
            Path(m.name).parts[0]
            for m in members
            if m.name and not m.name.startswith("/") and len(Path(m.name).parts) > 0
        }
        if len(top_levels) != 1:
            raise ValueError(
                f"Expected exactly one top-level entry in tar, got {sorted(top_levels)}"
            )

        top_name = next(iter(top_levels))
        tf.extractall(path=extract_dir)

    extracted_root = extract_dir / top_name
    if not extracted_root.exists():
        raise RuntimeError(f"Expected extracted root not found: {extracted_root}")

    return extracted_root

def recreate_hierarchy_and_copy(
    source_root: zarr.Group,
    archive_root: zarr.Group,
    image_dataset_paths: set[str],
    compressor,
) -> None:
    """
    Recursively recreate the group hierarchy.
    - Image datasets listed in `image_dataset_paths` are recompressed.
    - Other arrays are copied as-is.
    - Group attrs are preserved.
    """
    # Copy attrs of root
    copy_group_attrs(source_root, archive_root)

    def recurse(src_group: zarr.Group, dst_group: zarr.Group, current_path: str = "") -> None:
        # First create/copy child groups

        for key in src_group.group_keys():
            sub_src = src_group[key]
            sub_dst = dst_group.require_group(key)
            copy_group_attrs(sub_src, sub_dst)

            sub_path = normalize_zarr_path(str(Path(current_path) / key) if current_path else key)
            recurse(sub_src, sub_dst, sub_path)

        # Then arrays in this group
        for key in src_group.array_keys():
            src_arr = src_group[key]
            arr_path = normalize_zarr_path(str(Path(current_path) / key) if current_path else key)

            if arr_path in image_dataset_paths:
                logger.info(f"Recompressing {arr_path}")
                recompress_array(src_arr, dst_group, key, compressor=compressor)
            else:
                logger.info(f"Copying as-is {arr_path}")
                copy_array_verbatim(src_arr, dst_group, key)

    recurse(source_root, archive_root)


@validate_call
def archive_or_dearchive_omezarr(
    *,
    zarr_url: str,
    compressor_name: Optional[str] = None,
    compression_level: Optional[int] = None,
    archive: bool = True,
    output_preview: bool = True,
    keep_tar_archive: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Archive an OME-Zarr folder by increasing the compression level and saving it to a tar archive, or
    otherwise, dearchive an OME-Zarr saved in a tar archive.

    Parameters:
        zarr_url: If archiving, it requires the path to any image of the OME-Zarr to archive. 
            The full OME-Zarr folder containing the image will be archived. If unarchiving, it requires
            the path to the TAR archive file containing the OME-Zarr folder you wish to unarchive.
        compressor_name: Compressor to use for the archived OME-Zarr. Default: "zstd".
        compression_level: Compression level to use for the archived OME-Zarr. Default: 9.
        archive: If True, archives the OME-Zarr. If False, dearchives the OME-Zarr. In dearchiving,
            the zarr_url parameter is expected to be a tar archive with an OME-Zarr folder in it.
            Default: True.
        output_preview: If True, creates a downsampled version of the OME-Zarr folder called 
            `preview`. Default: True.
        keep_tar_archive: In dearchinving mode, if True, keeps the tar archive 
            after dearchiving. Default: True.
        overwrite: If True, it will overwrite any existing files. In archiving mode, it will overwrite
            the compressed OME-Zarr and/or the TAR file if it already exists. In unarchiving mode, it will 
            overwrite the extracted uncompressed OME-Zarr if it already exists in the parent directory 
            of the TAR file. Default: False.

    Returns
        None
    """
    zarr_path = Path(zarr_url)
    logger.info(
        f"Start task `Archive or Dearchive OME-Zarr` for "
        f"{zarr_path.name}"
    )

    if archive:

        zarr_path = zarr_path.parent
        archive_path = Path(zarr_path.parent, zarr_path.stem + "_archive.zarr")

        if compressor_name is None:
            compressor_name = "zstd"
        if compression_level is None:
            compression_level = 9
        try:
            compressor = Blosc(
                cname=compressor_name,
                clevel=compression_level,
                shuffle=Blosc.BITSHUFFLE,
            )
        except Exception as e:
            logger.error(f"Compressor {compressor_name} not supported.")
            raise e
        
        logger.info(f"Recompressing source OME-Zarr "
                    f"{zarr_path.parent.name}/{zarr_path.name} to "
                    f"{archive_path.parent.name}/{archive_path.name}"
                    f" with compressor {compressor_name} at level {compression_level}.")
        
        cluster = None
        client = None
        try:
            cluster = _set_dask_cluster(n_workers = 1)
            client = Client(cluster)
            client.forward_logging(logger_name = "mesospim_fractal_tasks", level=logging.INFO)

            rewrite_omezarr_images_only(
                src_path=zarr_path,
                dst_path=archive_path,
                image_compressor=compressor,
                overwrite=overwrite,
            )

            if output_preview:
                preview_path = Path(zarr_path.parent / (f"{zarr_path.stem}"+'_preview.zarr'))
                create_preview_from_omezarr(
                    src_path=zarr_path,
                    preview_path=preview_path,
                    overwrite=overwrite,
                )

            tar_path = Path(zarr_path.parent / (f"{archive_path.name}"+'.tar'))
            if tar_path.exists() and overwrite:
                tar_path.unlink()
            
            tar_directory(archive_path, tar_path)
            logger.info(f"TAR archive file written to: {tar_path}")
            #shutil.rmtree(archive_path)

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

    else:
        logger.info("Dearchiving OME-Zarr...")
        extract_dir = Path(zarr_path.parent, "archive")

        if extract_dir.exists():
            logger.info(f"Extraction directory already exists: {extract_dir.name}")
            if overwrite:
                logger.info(f"Overwriting existing extraction directory.")
                shutil.rmtree(extract_dir)
            else:
                raise FileExistsError

        restored_zarr_path = Path(zarr_path.parent, zarr_path.stem.replace("_archive", ""))
        assert restored_zarr_path.suffix == ".zarr"
        if restored_zarr_path.exists():
            logger.info(f"OME-Zarr contained in the archive {zarr_path.name} already "
                        f"exists in directory: {restored_zarr_path.parent.name}/{restored_zarr_path.name}")
            if overwrite:
                logger.info(f"Overwriting existing OME-Zarr.")
                shutil.rmtree(restored_zarr_path)
            else:
                raise FileExistsError

        extracted_archived_zarr = untar_to_directory(
            tar_path=zarr_path,
            extract_dir=extract_dir
        )

        logger.info(f"Extracted archived zarr to: {extract_dir.name}/{extracted_archived_zarr.name}")

        if compressor_name is None:
            compressor_name = "lz4"
        if compression_level is None:
            compression_level = 5
        try:
            compressor = Blosc(
                cname=compressor_name,
                clevel=compression_level,
                shuffle=Blosc.BITSHUFFLE,
            )
        except Exception as e:
            logger.error(f"Compressor {compressor_name} not supported.")
            raise e

        rewrite_omezarr_images_only(
            src_path=extracted_archived_zarr,
            dst_path=restored_zarr_path,
            overwrite=overwrite,
            image_compressor=compressor,
        )
        shutil.rmtree(extract_dir)
        logger.info(f"Restored working zarr written to: {restored_zarr_path.parent.name}/{restored_zarr_path.name}")

        if not keep_tar_archive:
            zarr_path.unlink()
            logger.info(f"Removed TAR file {zarr_path.name} and archive directory.")


if __name__ == "__main__":

    from fractal_task_tools.task_wrapper import run_fractal_task
  
    run_fractal_task(
        task_function=archive_or_dearchive_omezarr,
        logger_name=logger.name,
    )