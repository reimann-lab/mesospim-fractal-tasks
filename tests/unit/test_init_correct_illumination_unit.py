from mesospim_fractal_tasks.tasks.init_correct_illumination import (
    init_correct_illumination, group_by_channel)

MODULE = "mesospim_fractal_tasks.tasks.init_correct_illumination"

def test_group_by_channel(
    mocker, 
    mock_init_correct_illumination_env
):
    mock_channel = mocker.Mock(label="Ch0", index=None)
    mock_channel2 = mocker.Mock(label="Ch1", index=5)

    mocker.patch(
        MODULE + ".get_omero_channel_list",
        return_value=[mock_channel, mock_channel2],
    )

    result = group_by_channel("fake.zarr")

    assert result == {
        "Ch0": {"zarr_url": "fake.zarr", "index": 0, "n_FOVs": 9},
        "Ch1": {"zarr_url": "fake.zarr", "index": 5, "n_FOVs": 9},
    }

def test_init_correct_illumination_simple(
    tmp_dataset, 
    mock_init_correct_illumination_env
):
    
    out = init_correct_illumination(
        zarr_urls=["fake.zarr"],
        zarr_dir=str(tmp_dataset),
    )

    # ---- assertions ----
    assert "parallelization_list" in out
    assert len(out["parallelization_list"]) == 1

    item = out["parallelization_list"][0]
    assert item["init_args"]["channel_name"] == "Ch0"
    assert item["init_args"]["channel_index"] == 0
    assert item["init_args"]["n_FOVs"] == 9
