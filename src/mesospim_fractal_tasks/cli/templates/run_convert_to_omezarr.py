from mesospim_fractal_tasks.tasks.mesospim_to_omezarr import mesospim_to_omezarr

zarr_dir = "path/to/zarr/directory"
pattern = ""
extension = "h5"
metadata_file = None
zarr_name = "filename.zarr"
channel_color_file = "default"
exclusion_list = []
chunk_sizes = (16, 512, 512)
num_levels = 6
coarsening_factor = 2
overwrite = True

mesospim_to_omezarr(
    zarr_dir=zarr_dir,
    pattern=pattern,
    extension=extension,
    zarr_name=zarr_name,
    chunksize=chunk_sizes,
    metadata_file=metadata_file,
    channel_color_file=channel_color_file,
    exclusion_list=exclusion_list,
    num_levels=num_levels,
    coarsening_factor=coarsening_factor,
    overwrite=overwrite
)