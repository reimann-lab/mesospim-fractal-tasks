from mesospim_fractal_tasks.tasks.crop_regions_of_interest_dask import crop_regions_of_interest


###############################################################################
# Set the parameters of the task function

# e.g. "data/zarr/sampleA/sampleA.zarr/raw_image"
zarr_url = "path/to/zarr/image" 

# e.g. name of the table holding the ROI crop coordinates                     
roi_table_name = "roi_coords"

# e.g. whether to crop main image or extract ROIs
crop_or_roi = "roi"

# e.g. number of pyramid levels to build for the new ROI.
num_levels = 4

# e.g. Downsampling factor for the XY plane. Typical is 2
coarsening_xy = 2

# e.g. Set different chunks than original image.
# To provide a value use: DimTuple(z=0, y=0, x=0) and replace 0 with your value
chunksize = None

# e.g. True or False, whether to overwrite existing ROIs
overwrite = False

###############################################################################





if __name__ == "__main__":

    crop_regions_of_interest(
        zarr_url=zarr_url,
        roi_table_name=roi_table_name,
        crop_or_roi=crop_or_roi,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=chunksize,
        overwrite=overwrite
    )