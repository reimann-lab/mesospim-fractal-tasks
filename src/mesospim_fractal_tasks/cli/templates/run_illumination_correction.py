from mesospim_fractal_tasks.tasks.correct_illumination import correct_illumination
from mesospim_fractal_tasks.tasks.init_correct_illumination import init_correct_illumination

zarr_urls = ["path/to/zarr/image"]
zarr_dir = "path/to/zarr/directory"
z_correction = False

init_dict = init_correct_illumination(
    zarr_urls=zarr_urls,
    zarr_dir=zarr_dir,
)

parallelization_list = init_dict["parallelization_list"]
for element in parallelization_list:
    correct_illumination(
        zarr_url=element["zarr_url"],
        init_args=element["init_args"],
    )
