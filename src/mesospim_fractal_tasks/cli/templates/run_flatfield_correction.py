from mesospim_fractal_tasks.tasks.correct_flatfield import correct_flatfield
from mesospim_fractal_tasks.tasks.init_correct_flatfield import init_correct_flatfield
from mesospim_fractal_tasks.utils.models import BaSiCPyModelParams
import os
from concurrent.futures import ProcessPoolExecutor

def worker(args):
    return correct_flatfield(**args)

if __name__ == "__main__":

    N_WORKERS = os.cpu_count() or 1




    ###############################################################################
    # Set the parameters of the task function

    zarr_urls = ["path/to/zarr/image"]                    # e.g. ["data/zarr/sampleA/sampleA.zarr/raw_image"]
    zarr_dir = "path/to/zarr/directory"                   # e.g. "data/zarr/sampleA"
    n_images = 200                                        # e.g. min recommended is 50. Increase if using BaSiCPy to 200-300
    resolution_level = 4                                  # e.g. if using BaSiCPy, set to lowest otherwise set to 0
    max_z = None                                          # e.g. depends on data  
    FOV_list= []                                          # e.g. depends on data   

    basicpy_params = { 
        "max_workers": N_WORKERS,
        "get_darkfield": True,
        "smoothness_darkfield": 1,                         # Increase for smoother results
        "smoothness_flatfield": 1,                         # Increase for smoother results
        "working_size": 400                                # Impacts memory usage
    }


    ###############################################################################




    model_params = BaSiCPyModelParams(**basicpy_params)
    init_dict = init_correct_flatfield(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        FOV_list=FOV_list
    )

    print(f"Launching up to {N_WORKERS} parallel workers.")

    parallel_list = init_dict["parallelization_list"]
    new_list = []
    for element in parallel_list:
        param = {}
        param["zarr_url"] = element["zarr_url"]
        param["resolution_level"] = resolution_level
        param["n_zplanes"] = n_images
        param["advanced_basicpy_model_params"] = model_params
        param["init_args"] = element["init_args"]
        new_list.append(param)
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, new_list))
