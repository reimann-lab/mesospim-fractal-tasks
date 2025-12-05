from skinnervation3d_fractal_tasks.tasks.init_correct_flatfield import (
    init_correct_flatfield)
import shutil
from pathlib import Path
import zarr

def test_integration_creates_zarr_pyramid(tmp_dataset):
    example_zarr_path = Path("tests/data/ngff_example")
    shutil.copytree(example_zarr_path, tmp_dataset / "ngff_example")
    tmp_zarr = tmp_dataset / "ngff_example" / "my_image"

    out = init_correct_flatfield(
        zarr_urls=[str(tmp_zarr)],
        zarr_dir=str(tmp_dataset),
    )

    new_zarr = tmp_zarr.parent / "raw_image_flatfield_corr"
    level0 = new_zarr / "0"
    level1 = new_zarr / "1"

    shape0 = zarr.open(new_zarr)["0"].shape
    shape1 = zarr.open(new_zarr)["1"].shape

    # Check that the zarr groups exist
    assert (level0 / ".zarray").exists()
    assert (level1 / ".zarray").exists()
    assert shape0 == (1,2,540,1280)
    assert shape1 == (1,2,270,640)
