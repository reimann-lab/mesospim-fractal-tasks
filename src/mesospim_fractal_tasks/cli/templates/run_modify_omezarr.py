import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

from mesospim_fractal_tasks.tasks.modify_omezarr_structure import modify_omezarr_structure
from mesospim_fractal_tasks.utils.models import DimTuple, Channel




###############################################################################
# Set the parameters of the task function


# e.g. "/data/zarr/sampleA"
zarr_url = "/data/zarr/sampleA" 

# e.g. "sampleA" 
new_image_name = None     

# e.g. DimTuple(32, 1024, 1024) or None
chunksize = None

# e.g. 3 or 4, depends on data size, 
num_levels = None     

# e.g. [Channel(<channel1 info>), Channel(<channel2 info)...]
channels_list = [
    Channel(
        label="new_label",
        laser_wavelength = 488,
        start_contrast=0,
        end_contrast=1000
    )]

###############################################################################




if __name__ == "__main__":

    modify_omezarr_structure(
        zarr_url=zarr_url,
        new_image_name=new_image_name,
        chunksize=chunksize,
        num_levels=num_levels,
        channels_list=channels_list
    )