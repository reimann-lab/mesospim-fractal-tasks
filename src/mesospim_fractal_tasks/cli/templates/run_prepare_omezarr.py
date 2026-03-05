import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

from mesospim_fractal_tasks.tasks.prepare_mesospim_omezarr import prepare_mesospim_omezarr
from mesospim_fractal_tasks.utils.models import DimTuple


###############################################################################
# Set the parameters of the task function


# e.g. "/data/zarr/sampleA"
zarr_dir = "/data/zarr/sampleA" 

# e.g. "sampleA" 
pattern = ""               

# e.g. "sampleA.zarr"
zarr_name = ""   

# e.g. "raw_image"
image_name = None

# e.g. "default" or "path/to/channel_colors.json"
channel_color_settings = "default"            

# e.g. DimTuple(z=32, y=1024, x=1024)
chunk_sizes = DimTuple(z=64, y=1024, x=1024)

# e.g. depends on data size
num_levels = None     

# e.g. will overwrite existing image in OME-Zarr
overwrite = False                                       

###############################################################################




if __name__ == "__main__":

    prepare_mesospim_omezarr(
        zarr_dir=zarr_dir,
        pattern=pattern,
        zarr_name=zarr_name,
        chunksize=chunk_sizes,
        channel_color_settings=channel_color_settings,
        num_levels=num_levels,
        overwrite=overwrite
    )