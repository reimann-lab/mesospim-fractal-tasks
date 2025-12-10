from mesospim_fractal_tasks.tasks.crop_regions_of_interest import crop_regions_of_interest
from mesospim_fractal_tasks.tasks.init_crop_regions_of_interest import init_crop_regions_of_interest

# Parameters to edit
zarr_urls = ["path/to/zarr/image"]
zarr_dir = "path/to/zarr/directory"
num_levels = 2
roi_table_name = "roi_coords"

# Init task
init_dict = init_crop_regions_of_interest(
    zarr_urls=zarr_urls,
    zarr_dir=zarr_dir,
    roi_table_name=roi_table_name,
    num_levels=num_levels
)

for i, element in enumerate(init_dict["parallelization_list"]):

    crop_regions_of_interest(
        zarr_url=element["zarr_url"],
        init_args=element["init_args"],
    )