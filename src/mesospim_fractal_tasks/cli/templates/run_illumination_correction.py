from mesospim_fractal_tasks.tasks.correct_illumination import correct_illumination
from mesospim_fractal_tasks.tasks.init_correct_illumination import init_correct_illumination
import os
from concurrent.futures import ProcessPoolExecutor


###############################################################################
# Set the parameters of the task function

# e.g. ["data/zarr/sampleA/sampleA.zarr/raw_image"]
zarr_urls = ["path/to/zarr/image"] 

# e.g. "data/zarr/sampleA"                   
zarr_dir = "path/to/zarr/directory"

# e.g. True or False                  
z_correction = False                                  

###############################################################################



def worker(args):
    return correct_illumination(**args)

if __name__ == "__main__":

    N_WORKERS = os.cpu_count() or 1
    init_dict = init_correct_illumination(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
    )

    print(f"Launching up to {N_WORKERS} parallel workers.")

    parallel_list = init_dict["parallelization_list"]
    new_list = []
    for element in parallel_list:
        param = {}
        param["zarr_url"] = element["zarr_url"]
        param["init_args"] = element["init_args"]
        param["z_correction"] = z_correction
        new_list.append(param)
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, new_list))
