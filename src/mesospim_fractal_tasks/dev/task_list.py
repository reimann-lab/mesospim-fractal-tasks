"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ConverterNonParallelTask,
    ParallelTask,
    CompoundTask,
)

AUTHORS = "Giliane Rochat"
DOCS_LINK = None
INPUT_MODELS = [["mesospim_fractal_tasks", "utils/models.py", "BaSiCPyModelParams"],]

TASK_LIST = [
    ConverterNonParallelTask(
        name="Convert mesoSPIM dataset to OME-ZARR",
        executable="tasks/mesospim_to_omezarr.py",
        meta={"cpus_per_task": 1, "mem": 8000, "time": "2-00:00:00"},
        category="Conversion",
        tags=["mesoSPIM", "Converter"],
        modality="lightsheet"
    ),
    ParallelTask(
        name="Perform Flatfield Correction",
        input_types=dict(flatfield_corrected=False),
        executable="tasks/correct_flatfield_parallel.py",
        output_types=dict(flatfield_corrected=True),
        meta={"cpus_per_task": 4, "mem": 16000},
        tags=["BaSiCPy", "Illumination", "Correction", "Flatfield", "Darkfield"],
        category="Image Processing",
        modality="lightsheet"
    ),
    CompoundTask(
        name="Perform Flatfield Correction (Parallel)",
        input_types=dict(flatfield_corrected=False),
        executable_init="tasks/init_correct_flatfield_parallel.py",
        executable="tasks/correct_flatfield_parallel.py",
        output_types=dict(flatfield_corrected=True),
        meta={"cpus_per_task": 1, "mem": 16000},
        tags=["BaSiCPy", "Illumination", "Correction", "Flatfield", "Darkfield"],
        category="Image Processing",
        modality="lightsheet"
    ),
    CompoundTask(
        name="Perform Global Illlumination Correction",
        input_types=dict(illumination_corrected=False),
        executable_init="tasks/init_correct_illumination.py",
        executable="tasks/correct_illumination.py",
        output_types=dict(illumination_corrected=True),
        meta={"cpus_per_task": 2, "mem": 4000},
        tags=["BaSiCPy", "Illumination", "Correction", "Global optimization",
              "Uneven illumination", "Z-correction"],
        category="Image Processing",
        modality="lightsheet"
    ),
    CompoundTask(
        name="Crop Regions of Interest",
        executable_init="tasks/init_crop_regions_of_interest.py",
        executable="tasks/crop_regions_of_interest.py",
        meta={"cpus_per_task": 1, "mem": 8000},
        tags=["ROI", "Cropping"],
        modality="lightsheet",
        category="Image Formatting",
    ),
    ParallelTask(
        name="Stitch with Multiview Stitcher",
        input_types=dict(stitched=False),
        executable="tasks/stitch_with_multiview_stitcher.py",
        output_types=dict(stitched=True),
        meta={"cpus_per_task": 4, "mem": 32000, "time": "3-00:00:00"},
        tags=["Stitching", "Multitile", "Multiview Stitcher"],
        category="Image Processing",
        modality="lightsheet"
    )
]
