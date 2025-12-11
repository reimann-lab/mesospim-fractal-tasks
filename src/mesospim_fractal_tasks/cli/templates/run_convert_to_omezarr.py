from mesospim_fractal_tasks.tasks.mesospim_to_omezarr import mesospim_to_omezarr


###############################################################################
# Set the parameters of the task function

zarr_dir = "path/to/zarr/directory"                    # e.g. "/data/zarr/sampleA"
pattern = ""                                           # e.g. "sampleA" 
extension = "h5"                                       # e.g. "h5", "tiff", "raw"
metadata_file = None                                   # e.g. "/data/sampleA/scan001_meta.txt" if metadata file is not named as pattern
zarr_name = "filename.zarr"                            # e.g. "sampleA.zarr"
channel_color_file = "default"                         # e.g. "default" or "path/to/channel_colors.json"
exclusion_list = []                                    # e.g. give tiles to exclude from conversion
chunk_sizes = (16, 512, 512)                           # e.g. (32, 1024, 1024)
num_levels = 6                                         # e.g. depends on data size
coarsening_factor = 2                                  # e.g. typical is 2
overwrite = True                                       # e.g. will overwrite existing zarr

###############################################################################




if __name__ == "__main__":

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