import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # macOS Accelerate framework
os.environ["NUMBA_NUM_THREADS"] = "1"

from mesospim_fractal_tasks.tasks.correct_flatfield_dask import correct_flatfield
from mesospim_fractal_tasks.utils.models import BaSiCPyModelParams




###############################################################################
# Set the parameters of the task function

# e.g. "data/zarr/sampleA/sampleA.zarr/raw_image"
zarr_url = "path/to/zarr/image" 

# e.g. min recommended is 50. Increase if using BaSiCPy to 200-300
n_zplanes = 200

# e.g. if using BaSiCPy, set to lowest otherwise set to 0
resolution_level = 4 

# e.g. depends on data                                   
z_levels = None     

# e.g. depends on data                                       
FOV_list= []

# e.g. if illumination profiles are available, give path to folder
models_folder = None

# e.g. True to save the computed illumination profiles
save_models = False

basicpy_params = { 
    "max_workers": 4,
    "get_darkfield": True,
    "smoothness_darkfield": 1,                         # Increase for smoother results
    "smoothness_flatfield": 1,                         # Increase for smoother results
    "working_size": [400]                                # Impacts memory usage
}

# e.g. True or False, whether to erase the source image after correction
erase_source_image = False


###############################################################################




if __name__ == "__main__":

    model_params = BaSiCPyModelParams(**basicpy_params)

    correct_flatfield(
        zarr_url=zarr_url,
        models_folder=models_folder,
        FOV_list=FOV_list,
        z_levels=z_levels,
        save_models=save_models,
        resolution_level=resolution_level,
        n_zplanes=n_zplanes,
        basicpy_model_params=model_params
    )