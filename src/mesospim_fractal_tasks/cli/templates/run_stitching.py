from mesospim_fractal_tasks.tasks.stitch_with_multiview_stitcher import stitch_with_multiview_stitcher
from mesospim_fractal_tasks.utils.stitching import StitchingChannelInputModel
from mesospim_fractal_tasks.utils.models import DimTuple


###############################################################################
# Set the parameters of the task function

# e.g. "/data/zarr/sampleA.zarr/raw_image"
zarr_url = "path/to/zarr/image"                             

# e.g. "Lectin" or "PGP9.5"
channel_label = "Lectin"

# e.g. Default. See documentation for other options.
registration_function = "phase_correlation"

# e.g. 5, recommended lowest resolution level
registration_resolution_level = 5

# e.g. True, recommended as first step
registration_on_z_proj = False

# e.g. "translation", "rigid", "similarity", "affine"
transform_type = "translation" 

# e.g. "keep_axis_aligned", "alternating_pattern", "shortest_paths_overlap_weighted", "otsu_threshold_on_overlap"
pre_registration_pruning_method = "keep_axis_aligned"

# e.g. Extend the overlap region considered for finding the optimal tile positions
# To provide a value use: DimTuple(z=0, y=0, x=0) and replace 0 with your value
overlap_tolerance = DimTuple(z=0, y=0, x=0)

# e.g. Set different chunks than original image. Warning: can impact memory.
# To provide a value use: DimTuple(z=0, y=0, x=0) and replace 0 with your value
fusion_chunksize = None

# e.g. 4, recommended if memory always
max_workers = 4 
                                             
###############################################################################




if __name__ == "__main__":
    
    channel = StitchingChannelInputModel(label=channel_label)
    stitch_with_multiview_stitcher(
        zarr_url=zarr_url,
        channel=channel,
        registration_function=registration_function,
        overlap_tolerance=overlap_tolerance,
        registration_resolution_level = registration_resolution_level,
        transform_type=transform_type,
        pre_registration_pruning_method=pre_registration_pruning_method,
        fusion_chunksize=fusion_chunksize,
        registration_on_z_proj = registration_on_z_proj,
        max_workers=max_workers
    )