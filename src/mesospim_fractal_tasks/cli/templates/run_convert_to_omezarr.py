import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

from mesospim_fractal_tasks.tasks.mesospim_to_omezarr import mesospim_to_omezarr
from mesospim_fractal_tasks.utils.models import DimTuple


###############################################################################
# Set the parameters of the task function


# e.g. "/data/zarr/sampleA"
zarr_dir = "path/to/zarr/directory" 

# e.g. "sampleA" 
pattern = ""     

# e.g. "h5", "tiff", "raw"
extension = "h5"  

# e.g. "/data/sampleA/scan001_meta.txt" if metadata file is not named as pattern
metadata_file = None                

# e.g. "sampleA.zarr"
zarr_name = "filename.zarr"   

# e.g. "default" or "path/to/channel_colors.json"
channel_color_settings = "default"            

# e.g. DimTuple(z=32, y=1024, x=1024)
chunk_sizes = DimTuple(z=16, y=512, x=512)  

# e.g. depends on data size
num_levels = None     

# e.g. will overwrite existing zarr
overwrite = True                                       

###############################################################################




if __name__ == "__main__":

    mesospim_to_omezarr(
        zarr_dir=zarr_dir,
        pattern=pattern,
        extension=extension,
        zarr_name=zarr_name,
        chunksize=chunk_sizes,
        metadata_file=metadata_file,
        channel_color_settings=channel_color_settings,
        num_levels=num_levels,
        overwrite=overwrite
    )