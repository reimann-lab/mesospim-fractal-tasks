from mesospim_fractal_tasks.tasks.correct_flatfield import correct_flatfield
from mesospim_fractal_tasks.tasks.init_correct_flatfield import init_correct_flatfield
from mesospim_fractal_tasks.utils.models import BaSiCPyModelParams

zarr_urls = ["path/to/zarr/image"]
zarr_dir = "path/to/zarr/directory"
n_images = 200
resolution_level = 4
max_z = None
FOV_list= []

basicpy_params = { 
    "max_workers": 4,
    "get_darkfield": True,
    "smoothness_darkfield": 1,
    "smoothness_flatfield": 1,
    "working_size": 400
}


model_params = BaSiCPyModelParams(**basicpy_params)


init_dict = init_correct_flatfield(
    zarr_urls=zarr_urls,
    zarr_dir=zarr_dir,
    FOV_list=FOV_list
)

parallelization_list = init_dict["parallelization_list"]
for element in parallelization_list:
    correct_flatfield(
        zarr_url=element["zarr_url"],
        init_args=element["init_args"],
        resolution_level=resolution_level,
        n_zplanes=n_images,
        advanced_basicpy_model_params=model_params,
    )
