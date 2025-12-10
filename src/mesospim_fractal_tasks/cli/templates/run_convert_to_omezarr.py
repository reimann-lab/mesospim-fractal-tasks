from mesospim_fractal_tasks.tasks.mesospim_to_omezarr import mesospim_to_omezarr

zarr_dir = "data/Multitile/IENFD25-6-3"
pattern = "3z"
extension = "h5"
metadata_file = ""
zarr_name = "IENFD25-6-3-3z_downsampled_test.zarr" #"IENFD25-4-1-Irchel_downsampled.zarr" #
exclusion_list = []
chunk_sizes = (16, 512, 512)
num_levels = 2
coarsening_factor = 2
overwrite = True

mesospim_to_omezarr(
    zarr_dir=zarr_dir,
    pattern=pattern,
    extension=extension,
    zarr_name=zarr_name,
    chunksize=chunk_sizes,
    #metadata_file=metadata_file,
    channel_color_file="v0",
    exclusion_list=exclusion_list,
    num_levels=num_levels,
    coarsening_factor=coarsening_factor,
    overwrite=overwrite
)