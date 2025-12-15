from mesospim_fractal_tasks.tasks.stitch_with_multiview_stitcher import stitch_with_multiview_stitcher
from mesospim_fractal_tasks.utils.stitching import StitchingChannelInputModel


###############################################################################
# Set the parameters of the task function

zarr_url = "path/to/zarr/image"                             # e.g. "/data/zarr/sampleA.zarr/raw_image"     
channel_label = "DAPI"                                      # e.g. "DAPI" or "A01_C01"
registration_resolution_level = 1                           # e.g. 1, recommended lowest resolution level
registration_on_z_proj = True                               # e.g. True, recommended as first step
transform_type = "translation"                              # e.g. "translation", "rigid", "similarity", "affine"
pre_registration_pruning_method = "keep_axis_aligned"       # e.g. "keep_axis_aligned", "alternating_pattern", "shortest_paths_overlap_weighted", "otsu_threshold_on_overlap"
fusion_chunksize = None                                     # e.g. Set smaller chunks than original image if memory is limited
n_batches = 4                                               # e.g. 4, recommended if memory always

###############################################################################




if __name__ == "__main__":
    
    channel = StitchingChannelInputModel(label=channel_label)
    stitch_with_multiview_stitcher(
        zarr_url=zarr_url,
        channel=channel,
        registration_resolution_level = registration_resolution_level,
        transform_type=transform_type,
        pre_registration_pruning_method=pre_registration_pruning_method,
        fusion_chunksize=fusion_chunksize,
        registration_on_z_proj = registration_on_z_proj,
        n_batches=n_batches
    )