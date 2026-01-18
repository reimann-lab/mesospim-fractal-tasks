from mesospim_fractal_tasks.tasks.correct_illumination_dask import correct_illumination


###############################################################################
# Set the parameters of the task function

# e.g. "data/zarr/sampleA/sampleA.zarr/raw_image"
zarr_url = "path/to/zarr/image"

# e.g. True or False                  
z_correction = False                                  

###############################################################################


if __name__ == "__main__":

    correct_illumination(
        zarr_url=zarr_url,
        z_correction=z_correction
    )
