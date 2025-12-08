import numpy as np
import zarr
import shutil
from pathlib import Path

from mesospim_fractal_tasks.tasks.mesospim_to_omezarr import (
    mesospim_to_omezarr,
)

MODULE = "mesospim_fractal_tasks.tasks.mesospim_to_omezarr"

def test_mesospim_to_omezarr_writes_correct_data_h5(
    tmp_dataset, 
    mocker
):
    data_dir = Path("tests/data")
    for f in data_dir.glob("*example*"):
        if f.is_file():
            shutil.copy(f, tmp_dataset)

    mocker.patch(
        MODULE + ".build_pyramid"
    )
    mocker.patch(
        MODULE + "._determine_optimal_contrast",
        return_value={"0": {"start": 0, "end": 65535},
                      "1": {"start": 0, "end": 65535}},
    )
    mocker.patch(
        MODULE + ".load_channel_colors",
        return_value={
            "640": {
                "label": "ch0",
                "laser_wavelength": 640,
                "color": "00FF00",
            },
            "561": {
                "label": "ch1",
                "laser_wavelength": 561,
                "color": "00FF00",
            }
        },
    )
    tile = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

    # Run the full pipeline for H5 case
    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
        pattern="example",
        zarr_name="output_h5",
        num_levels=2,
        chunksize=(2, 3, 4),
        overwrite=True,
    )

    # Validate the output image data
    zarr_path = tmp_dataset / "output_h5.zarr" / "raw_image" / "0"
    store = zarr.storage.FSStore(zarr_path)

    arr = zarr.open(store, mode="r")
    assert arr.shape == (2, 2, 3, 4)
    np.testing.assert_array_equal(arr[0], tile)

    # Run the full pipeline for raw case
    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
        pattern="example",
        zarr_name="output_raw",
        extension="raw",
        num_levels=2,
        chunksize=(2, 3, 4),
        overwrite=True,
    )

    # Validate the output image data
    zarr_path = tmp_dataset / "output_raw.zarr" / "raw_image" / "0"
    store = zarr.storage.FSStore(zarr_path)

    arr = zarr.open(store, mode="r")
    assert arr.shape == (2, 2, 3, 4)
    np.testing.assert_array_equal(arr[0], tile)

    # Run the full pipeline for tiff case
    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
        pattern="example",
        zarr_name="output_tiff",
        extension="tiff",
        num_levels=2,
        chunksize=(2, 3, 4),
        overwrite=True,
    )

    # Validate the output image data
    zarr_path = tmp_dataset / "output_tiff.zarr" / "raw_image" / "0"
    store = zarr.storage.FSStore(zarr_path)

    arr = zarr.open(store, mode="r")
    assert arr.shape == (2, 2, 3, 4)
    np.testing.assert_array_equal(arr[0], tile)


