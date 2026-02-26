import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

from mesospim_fractal_tasks.tasks.mesospim_to_omezarr import mesospim_to_omezarr


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
channel_color_file = "default"            

# e.g. (32, 1024, 1024)
chunk_sizes = (16, 512, 512)  

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
        channel_color_file=channel_color_file,
        num_levels=num_levels,
        overwrite=overwrite
    )