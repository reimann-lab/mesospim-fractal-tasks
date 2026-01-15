from mesospim_fractal_tasks.tasks.init_correct_flatfield_parallel import (
    init_correct_flatfield,
    group_by_channel)
import shutil

MODULE = "mesospim_fractal_tasks.tasks.init_correct_flatfield_parallel"

def test_group_by_channel(
    mocker, 
    mock_init_correct_flatfield_env
):
    mock_channel = mocker.Mock(label="Ch0", index=None)
    mock_channel2 = mocker.Mock(label="Ch1", index=5)

    mocker.patch(
        MODULE + ".get_omero_channel_list",
        return_value=[mock_channel, mock_channel2],
    )

    result = group_by_channel("fake.zarr")

    assert result == {
        "Ch0": {"zarr_url": "fake.zarr", "index": 0},
        "Ch1": {"zarr_url": "fake.zarr", "index": 5},
    }

def test_init_correct_flatfield_simple(
    mocker, 
    tmp_dataset, 
    mock_init_correct_flatfield_env
):
    
    out = init_correct_flatfield(
        zarr_urls=["fake.zarr"],
        zarr_dir=str(tmp_dataset),
    )

    # ---- assertions ----
    assert "parallelization_list" in out
    assert len(out["parallelization_list"]) == 1

    item = out["parallelization_list"][0]
    assert item["init_args"]["channel_name"] == "Ch0"
    assert item["init_args"]["channel_index"] == 0
    assert item["init_args"]["FOV_list"] is None
    assert item["init_args"]["z_levels"] is None

def test_init_correct_flatfield_max_z_logic(
    mocker, 
    mock_init_correct_flatfield_env
):
    out = init_correct_flatfield(
        zarr_urls=["fake.zarr"],
        zarr_dir="unused",
        z_levels=[5, 30],
    )

    args = out["parallelization_list"][0]["init_args"]

    assert sorted(args["FOV_list"]) == [0,2,6,8]
    assert args["z_levels"] == [5, 30]

def test_save_models_creates_correct_folder_name(
    mocker, 
    tmp_dataset, 
    mock_init_correct_flatfield_env
):
    fake_zarr = tmp_dataset / "fake.zarr" / "fake_image"
    fake_zarr.parent.mkdir(parents=True)
    fake_zarr.mkdir()

    init_correct_flatfield(
        zarr_urls=[str(fake_zarr)],
        zarr_dir=str(tmp_dataset),
        save_models=True,
        FOV_list=[0,1,2,3]
    )

    model_path = tmp_dataset / "fake.zarr" / "IllumModels" / "Ch0"
    assert model_path.exists()

    shutil.rmtree(tmp_dataset / "fake.zarr" / "IllumModels")
    init_correct_flatfield(
        zarr_urls=[str(fake_zarr)],
        zarr_dir=str(tmp_dataset),
        save_models=True,
        z_levels=[5, 10]
    )

    model_path = tmp_dataset / "fake.zarr" / "IllumModels" / "Ch0"
    assert model_path.exists()

    init_correct_flatfield(
        zarr_urls=[str(fake_zarr)],
        zarr_dir=str(tmp_dataset),
        save_models=True
    )

    model_path = tmp_dataset / "fake.zarr" / "BaSiCPyModels" / "Ch0"
    assert model_path.exists()