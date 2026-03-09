import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

from mesospim_fractal_tasks.tasks.archive_or_dearchive_omezarr import archive_or_dearchive_omezarr 


###############################################################################
# Set the parameters of the task function




# e.g. "/data/zarr/sampleA"
zarr_url = "/data/zarr/sampleA" 

# e.g. "zstd" or "lz4" 
compressor_name= "zstd"

# e.g. integer
compression_level = 9

# e.g. True or False, whether to archive or dearchive
archive = True

# e.g. True or False, whether to output a downsampled version of the OME-Zarr for preview
output_preview = True
    
# e.g. True or False, whether to keep the TAR archive file or compressed OME-Zarr archive
keep_tar_archive = True

# e.g. True or False, whether to overwrite existing archive
overwrite = False    

###############################################################################




if __name__ == "__main__":

    archive_or_dearchive_omezarr(
        zarr_url=zarr_url,
        compressor_name=compressor_name,
        compression_level=compression_level,
        archive=archive,
        keep_tar_archive=keep_tar_archive,
        overwrite=overwrite,
    )