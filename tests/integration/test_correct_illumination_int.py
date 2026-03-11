import numpy as np
import shutil
from pathlib import Path
import anndata as ad
import dask.array as da
import zarr

from mesospim_fractal_tasks.tasks.correct_illumination import (
    correct_illumination,
)
from tests.conftest import mock_dask_distributed

MODULE = "mesospim_fractal_tasks.tasks.correct_illumination"

def test_correct_illumination_main_output(
    tmp_dataset, 
    mocker
):
    mock_dask_distributed(mocker, MODULE)
    example_zarr_path = Path("tests/data/ngff_example")
    shutil.copytree(example_zarr_path, tmp_dataset / "ngff_example")
    tmp_zarr = tmp_dataset / "ngff_example" / "my_image"

    out_tmp_zarr = tmp_dataset / "ngff_example" / "my_image_illum_corr"
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
    fake_gain_map = {"ROI_0": 1.0, "ROI_1": 1.0}
    mocker.patch(
        MODULE + ".compute_global_normalisation",
        return_value=fake_gain_map,
    )

    out = correct_illumination(
        zarr_url=str(tmp_zarr),
        init_args=dict(
            channel_name="DAPI",
            channel_index=0,
            n_FOVs=2,
            is_proxy=False,
            erase_source_image=False,
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
    assert "correct_illumination" in attrs["fractal_tasks"]

    level0 = da.from_zarr(out_tmp_zarr / "0")
    readback = level0[0:1, 0:2, 0:540, 0:640].compute()
    original = da.from_zarr(tmp_zarr / "0")[0:1, 0:2, 0:540, 0:640].compute()

    assert np.array_equal(readback, original)
