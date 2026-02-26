import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

from mesospim_fractal_tasks.tasks.correct_illumination_dask import correct_illumination



###############################################################################
# Set the parameters of the task function

# e.g. "data/zarr/sampleA/sampleA.zarr/raw_image"
zarr_url = "path/to/zarr/image"

# e.g. True or False                  
z_correction = False

# e.g. True or False, whether to erase the source image after correction
erase_source_image = False

###############################################################################


if __name__ == "__main__":

    correct_illumination(
        zarr_url=zarr_url,
        z_correction=z_correction,
        erase_source_image=erase_source_image
    )
