from mesospim_fractal_tasks.tasks.stitch_with_multiview_stitcher import stitch_with_multiview_stitcher
from mesospim_fractal_tasks.utils.stitching import StitchingChannelInputModel

zarr_url = "Data/Multitile/IENFD25-5-9TH/IENFD25-5-9TH_restained_downsampled.zarr/raw_image_illum_corr"
channel_label = "PGP9.5"
registration_resolution_level = 1
registration_on_z_proj = True
transform_type = "translation"
pre_registration_pruning_method = "keep_axis_aligned"
fusion_chunksize = [64, 256, 256]
n_batches = 4

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