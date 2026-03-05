import numpy as np
import dask.array as da
from mesospim_fractal_tasks.tasks.correct_illumination import (
    gain_residuals,
    compute_global_normalisation,
    correct_illumination)

MODULE = "mesospim_fractal_tasks.tasks.correct_illumination"

def test_gain_residuals_basic():
    gains = np.array([2.0, 4.0, 8.0])
    
    gain_graph = [
        ("R0", "R1", 2.0),   # log(2) - log(4) - log(2) = -log(4)
        ("R1", "R2", 2.0),   # log(4) - log(8) - log(2) = -log(4)
        ("R0", "R2", 4.0),   # log(2) - log(8) - log(4) = -log(16)
    ]
    
    indices = {"R0": 0, "R1": 1, "R2": 2}

    res = gain_residuals(gains, gain_graph, indices)

    expected = np.array([
        np.log(2) - np.log(4) - np.log(2),
        np.log(4) - np.log(8) - np.log(2),
        np.log(2) - np.log(8) - np.log(4),
    ])

    np.testing.assert_allclose(res, expected)

def test_gain_residuals_with_unity_gains():
    gains = np.array([1.0, 2.0])
    gain_graph = [("A", "B", 1.0)]
    indices = {"A": 0, "B": 1}
    
    res = gain_residuals(gains, gain_graph, indices)
    np.testing.assert_allclose(res, np.log(1) - np.log(2) - 0)

def test_gain_residuals_empty_graph():
    gains = np.array([1.0, 2.0, 3.0])
    graph = []
    idx = {"A": 0, "B": 1, "C": 2}

    res = gain_residuals(gains, graph, idx)

    assert res.size == 0

def test_gain_residuals_single_pair():
    gains = np.array([2.0, 8.0])
    graph = [("R0", "R1", 4.0)]
    idx = {"R0": 0, "R1": 1}

    res = gain_residuals(gains, graph, idx)

    expected = np.log(2) - np.log(8) - np.log(4)
    np.testing.assert_allclose(res, [expected])

def test_compute_global_normalisation_basic(
    mocker, 
    tmp_path,
    mock_correct_illumination_env
):
    
    fake_lsq = mocker.Mock()
    fake_lsq.x = np.array([2.0, 4.0])
    mock_lsq = mocker.patch(
        MODULE + ".least_squares",
        return_value=fake_lsq)
    image = da.ones((1, 1, 20, 20))
    mock_correct_illumination_env["from_zarr"] = mocker.patch(
        MODULE + ".da.from_zarr",
        return_value=image)
    
    z_profile = np.ones((1, 1, 1, 1))
    gain_map = compute_global_normalisation(
            zarr_path=tmp_path,
            channel_name="DAPI",
            channel_index=0,
            is_proxy=False,
        )

    mock_lsq.assert_called_once()
    called_func = mock_lsq.call_args[0][0]
    assert called_func == gain_residuals

    # Normalisation: max value becomes 1
    assert gain_map == {
        "ROI_0": 2/4,
        "ROI_1": 4/4,
    }

def test_compute_global_normalisation_no_tile_overlap(
    mocker, 
    tmp_path, 
    mock_correct_illumination_env
):

    # Fake indices: no overlap
    indices = [
        (0, 1, 0, 10, 0, 10),   # ROI0
        (0, 1, 20, 30, 20, 30), # ROI1 far away
    ]

    fake_indices = mock_correct_illumination_env["convert_ROI_table_to_indices"]
    fake_indices.side_effect = [indices, indices]

    z_profile = np.ones((1, 1, 1, 1))

    gain_map = compute_global_normalisation(
        zarr_path=tmp_path,
        channel_name="DAPI",
        channel_index=0,
        is_proxy=False,
    )

    # both gains are 1 → normalized to 1
    assert gain_map == {"ROI_0": 1., "ROI_1": 1.}

def test_compute_global_normalisation_insufficient_overlap(
    mocker, 
    tmp_path,
    mock_correct_illumination_env
):

    # Fake image with 99% zeros → masks too small
    fake_img = da.zeros((1, 1, 20, 20))
    mock_correct_illumination_env["from_zarr"].return_value = fake_img

    fake_lsq = mocker.Mock()
    fake_lsq.x = np.array([2.0, 4.0])
    mock_lsq = mocker.patch(
        MODULE + ".least_squares",
        return_value=fake_lsq)

    z_profile = np.ones((1, 1, 1, 1))

    gain_map = compute_global_normalisation(
        zarr_path=tmp_path,
        channel_name="DAPI",
        channel_index=0,
        is_proxy=False,
    )

    # gain_graph should contain one pair, with gain = 1
    args = mock_lsq.call_args.kwargs["args"][0]
    assert args[0][2] == 1

def test_compute_global_normalisation_single_roi(
    mocker, 
    tmp_path,
    mock_correct_illumination_env
):

    indices = [(0, 1, 0, 10, 0, 10)]  # Only 1 ROI

    mock_correct_illumination_env["convert_ROI_table_to_indices"].side_effect = [
        indices, indices]
    mocked_lsq = mocker.patch(
        MODULE + ".least_squares"
    )

    gain_map = compute_global_normalisation(
        zarr_path=tmp_path,
        channel_name="DAPI",
        channel_index=0,
        is_proxy=False,
    )

    mocked_lsq.assert_not_called()
    assert gain_map == {"ROI_0": 1.0}

def test_z_correction_call(
    mocker,
    tmp_dataset,
    mock_correct_illumination_env,
):
    mocker.patch(MODULE + ".compute_global_normalisation")

    mocker.patch(MODULE + ".da.from_array")
    mocker.patch(MODULE + ".da.ones")

    compute_z = mocker.patch(MODULE + ".compute_z_correction_profile")
    
    z_correction = True
    fake_original_zarr = tmp_dataset / "raw_image"
    fake_original_zarr.mkdir()
    fake_new_zarr = tmp_dataset / "raw_image_illum_corr"
    fake_new_zarr.mkdir()
    correct_illumination(
        zarr_url=str(fake_original_zarr),
        init_args=dict(
            channel_name="DAPI",
            channel_index=0,
            n_FOVs=9,
            is_proxy=False,
        ),
        z_correction=z_correction,
    )

    assert compute_z.call_count == 1
