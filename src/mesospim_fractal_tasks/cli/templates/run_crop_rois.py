from mesospim_fractal_tasks.tasks.crop_regions_of_interest import crop_regions_of_interest
from mesospim_fractal_tasks.tasks.init_crop_regions_of_interest import init_crop_regions_of_interest
import os
from concurrent.futures import ProcessPoolExecutor


###############################################################################
# Set the parameters of the task function

# e.g. ["data/zarr/sampleA/sampleA.zarr/raw_image"]
zarr_urls = ["path/to/zarr/image"]    

# e.g. "data/zarr/sampleA"
zarr_dir = "path/to/zarr/directory"

# e.g. number of pyramid levels to build         
num_levels = 2

# e.g. name of the table holding the ROI crop coordinates                     
roi_table_name = "roi_coords"                         

###############################################################################


def worker(args):
    return crop_regions_of_interest(**args)

if __name__ == "__main__":

    N_WORKERS = os.cpu_count() or 1
    init_dict = init_crop_regions_of_interest(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        roi_table_name=roi_table_name,
        num_levels=num_levels
    )

    print(f"Launching up to {N_WORKERS} parallel workers.")
    parallel_list = init_dict["parallelization_list"]
    new_list = []
    for element in parallel_list:
        param = {}
        param["zarr_url"] = element["zarr_url"]
        param["init_args"] = element["init_args"]
        new_list.append(param)
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(worker, new_list))