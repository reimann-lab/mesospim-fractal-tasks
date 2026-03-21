import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

from mesospim_fractal_tasks.tasks.fuse_views_or_channels import fuse_views_or_channels
from mesospim_fractal_tasks.utils.stitching import StitchingChannelInputModel
from mesospim_fractal_tasks.utils.models import DimTuple
from pathlib import Path





###############################################################################
# Set the parameters of the task function

# e.g. "/data/zarr/sampleA.zarr/raw_image"
zarr_url = "path/to/zarr/image"  

# e.g. sampleA_fused
output_zarr_name = "sampleA_fused"

# e.g. list of path to the different views or channels
zarr_image_paths: list[Path | str] = ["path/to/zarr_view1/raw_image", "path/to/zarr_view2/raw_image"]

# e.g. "Lectin" or "PGP9.5"
channel_label = "Lectin"

# e.g. Default. See documentation for other options.
registration_function = "phase_correlation"

# e.g. 5, recommended lowest resolution level
registration_resolution_level = None

# e.g. True, recommended as first step
registration_on_z_proj = False

# e.g. "translation", "rigid", "similarity", "affine"
transform_type = "translation" 

# e.g. "keep_axis_aligned", "alternating_pattern", "shortest_paths_overlap_weighted", "otsu_threshold_on_overlap"
pre_registration_pruning_method = "keep_axis_aligned"

# e.g. Set different chunks than original image. Warning: can impact memory.
# To provide a value use: DimTuple(z=0, y=0, x=0) and replace 0 with your value
fusion_chunksize = None

# e.g. 4, recommended if memory always
max_workers = 4 
                                             
###############################################################################



if __name__ == "__main__":
    
    registration_channel = StitchingChannelInputModel(label=channel_label)
    fuse_views_or_channels(
        zarr_url=zarr_url,
        output_zarr_name=output_zarr_name,
        zarr_image_paths=zarr_image_paths,
        registration_channel=registration_channel,
        registration_function=registration_function,
        registration_resolution_level = registration_resolution_level,
        transform_type=transform_type,
        fusion_chunksize=fusion_chunksize,
        registration_on_z_proj = registration_on_z_proj,
        max_workers=max_workers
    )