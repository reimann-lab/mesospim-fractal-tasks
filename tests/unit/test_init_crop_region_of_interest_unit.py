from mesospim_fractal_tasks.tasks.init_crop_regions_of_interest import (
    init_crop_regions_of_interest)
import pytest

MODULE = "mesospim_fractal_tasks.tasks.init_crop_regions_of_interest"

def test_multiple_zarr_urls_error(
    tmp_path,
    mock_init_crop_regions_env
):
    with pytest.raises(ValueError):
        init_crop_regions_of_interest(
            zarr_urls=["a.zarr", "b.zarr"],
            zarr_dir=str(tmp_path),
        )

def test_no_roi_table_found(
    tmp_path,
    mock_init_crop_regions_env
):
    zarr_dir = tmp_path / "img.zarr"
    zarr_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        init_crop_regions_of_interest(
            zarr_urls=[str(zarr_dir)],
            zarr_dir=str(tmp_path),
        )