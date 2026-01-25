import numpy as np
import shutil
from pathlib import Path
import anndata as ad
import dask.array as da
import zarr

from mesospim_fractal_tasks.tasks.correct_flatfield import (
    correct_flatfield,
    collect_fovs
)
from tests.conftest import mock_dask_distributed

MODULE = "mesospim_fractal_tasks.tasks.correct_flatfield"

def test_collect_fovs_default():

    example_zarr_path = Path("tests/data/ngff_example/my_image")
    df = ad.read_zarr(example_zarr_path / "tables" / "FOV_ROI_table").to_df()
    FOV_size = (round(df.iloc[0]["len_y_micrometer"] / 1.3), 
                round(df.iloc[0]["len_x_micrometer"] / 1.3))
    n_zplanes = 2

    result = collect_fovs(
        zarr_url=example_zarr_path,
        channel_index=0,
        FOV_list=None,
        resolution_level=1,
        pixel_sizes_yx=(1.3, 1.3),
        n_zplanes=n_zplanes,
        z_levels=None
    )

    # Should return exactly n_zplanes slices
    assert result.shape[0] == n_zplanes

    # Height/width match the computed ROI for FOV_0
    assert result.shape[1] == FOV_size[0]
    assert result.shape[2] == FOV_size[1]

def test_correct_flatfield_main_output(tmp_dataset, mocker):

    mock_dask_distributed(mocker, MODULE)

    example_zarr_path = Path("tests/data/ngff_example")
    shutil.copytree(example_zarr_path, tmp_dataset / "ngff_example")
    tmp_zarr = tmp_dataset / "ngff_example" / "my_image"

    out_tmp_zarr = tmp_dataset / "ngff_example" / "my_image_flatfield_corr"
    _ = zarr.open_group(out_tmp_zarr)
    shape = (1, 2, 540, 1280)
    for level in range(2):
        shape = (shape[0], shape[1],
                 shape[2] // 2**level, 
                 shape[3] // 2**level)
        _ = zarr.create(
            shape=shape,
            chunks=(1, 1) + shape[2:],
            store=zarr.storage.FSStore(Path(out_tmp_zarr, 
                                            str(level))),
            dtype=np.uint16,
            overwrite=True,
            dimension_separator="/",
        )

    mocker.patch(
        MODULE + ".collect_fovs",
    )
    mocker.patch(
        MODULE + ".compute_empty_fov_models",
    )
    fake_FOV = np.arange(2*540*640, dtype=np.uint16).reshape(1, 2, 540, 640)
    fake_dask = da.from_array(fake_FOV, 
                              chunks=(1, 1, 540, 640))
    mocker.patch(
        MODULE + ".correct",
        return_value=fake_dask,
    )
    mocker.patch(
        MODULE + ".resample_to_shape",
        return_value=fake_FOV[0,0,:,:],
    )

    out = correct_flatfield(
        zarr_url=str(tmp_zarr),
        init_args=dict(
            channel_name="DAPI",
            channel_index=0,
            FOV_list=[0],
            z_levels=None,
            saving_path=None,
        )
    )

    fov_table = out_tmp_zarr / "tables" / "FOV_ROI_table"
    assert fov_table.exists(), "ROI table folder should have been created"

    loaded_df = ad.read_zarr(fov_table).to_df()
    assert not loaded_df.empty
    assert set(["x_micrometer", 
                "y_micrometer",
                "len_x_micrometer",
                "len_y_micrometer"]).issubset(set(loaded_df.columns))
    
    attrs = zarr.open_group(out_tmp_zarr, mode="r").attrs.asdict()
    assert "multiscales" in attrs
    assert "fractal_tasks" in attrs
    assert "correct_flatfield" in attrs["fractal_tasks"]

    level0 = da.from_zarr(out_tmp_zarr / "0")
    readback = level0[0:1, 0:2, 0:540, 0:640].compute()

    assert np.array_equal(readback, fake_FOV)
