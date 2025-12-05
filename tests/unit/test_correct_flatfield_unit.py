import numpy as np
import pytest
from pathlib import Path

from skinnervation3d_fractal_tasks.tasks.correct_flatfield import (
    correct_flatfield,
    correct,
    IlluminationModel,
    collect_fovs
)

def test_collect_fovs_with_fov_list(mock_flatfield_env):

    mock_flatfield_env["collect_fovs"] = collect_fovs
    n_zplanes = 2
    out = collect_fovs(
        zarr_url="fake.zarr",
        channel_index=0,
        FOV_list=[0],
        resolution_level=0,
        pixel_sizes_yx=(1, 1),
        n_zplanes=n_zplanes,
        max_z=None,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == n_zplanes
    assert out.shape == (2, 20, 20)

def test_collect_fovs_with_max_z(mock_flatfield_env):

    mock_flatfield_env["collect_fovs"] = collect_fovs
    n_zplanes = 2
    out = collect_fovs(
        zarr_url="fake_zarr",
        channel_index=0,
        FOV_list=[0],
        resolution_level=0,
        pixel_sizes_yx=(1., 1.),
        n_zplanes=n_zplanes,
        max_z=2,
    )

    # Should return exactly n_zplanes slices
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == n_zplanes
    assert out.shape == (2, 20, 20)

def test_collect_fovs_assert_fov_list_invalid():
    example_zarr_path = Path("tests/data/ngff_example/my_image")

    with pytest.raises(AssertionError):
        collect_fovs(
            zarr_url=str(example_zarr_path),
            channel_index=0,
            FOV_list=[5],   # invalid
            resolution_level=0,
            pixel_sizes_yx=(1.0, 1.0),
            n_zplanes=2,
            max_z=None
        )

def test_correct_flatfield_uses_empty_fov_models(mocker, mock_flatfield_env, tmp_path):
    mock_empty = mocker.patch("skinnervation3d_fractal_tasks.tasks.correct_flatfield."
                              "compute_empty_fov_models")
    mock_basicpy = mocker.patch(
        "skinnervation3d_fractal_tasks.tasks.correct_flatfield.compute_basicpy_models",
        return_value=mocker.MagicMock(),
    )

    init_args = dict(
        channel_name="test",
        channel_index=0,
        FOV_list=[0],       # << triggers compute_empty_fov_models
        max_z=None,
        saving_path=None,
    )

    correct_flatfield(
        zarr_url=str(tmp_path / "test.zarr"),
        init_args=init_args,
    )

    mock_empty.assert_called_once()
    mock_basicpy.assert_not_called()

def test_correct_flatfield_uses_basicpy_models(mocker, mock_flatfield_env, tmp_dataset):
    mock_empty = mocker.patch("skinnervation3d_fractal_tasks.tasks.correct_flatfield."
                              "compute_empty_fov_models")
    mock_basicpy = mocker.patch(
        "skinnervation3d_fractal_tasks.tasks.correct_flatfield.compute_basicpy_models",
        return_value=mocker.MagicMock(),
    )

    from skinnervation3d_fractal_tasks.tasks.correct_flatfield import correct_flatfield

    init_args = dict(
        channel_name="test",
        channel_index=0,
        FOV_list=None,     # << triggers compute_basicpy_models
        max_z=None,
        saving_path=None,
    )

    correct_flatfield(
        zarr_url=str(tmp_dataset / "test.zarr"),
        init_args=init_args,
    )

    mock_basicpy.assert_called_once()
    mock_empty.assert_not_called()

def test_correct_flatfield_loads_npz(mocker, mock_flatfield_env, tmp_path):
    mock_empty = mocker.patch("skinnervation3d_fractal_tasks.tasks.correct_flatfield."
                              "compute_empty_fov_models")
    mock_basicpy = mocker.patch(
        "skinnervation3d_fractal_tasks.tasks.correct_flatfield.compute_basicpy_models",
        return_value=mocker.MagicMock(),
    )

    # Prepare a fake illumination model file
    models_dir = tmp_path / "testchannel"
    models_dir.mkdir(parents=True)
    np.savez(
        models_dir / "profiles.npz",
        flatfield=np.ones((3, 3)),
        darkfield=np.zeros((3, 3)),
        baseline=5,
    )

    init_args = dict(
        channel_name="testchannel",
        channel_index=0,
        FOV_list=[0],
        max_z=None,
        saving_path=None,
    )

    correct_flatfield(
        zarr_url=str(tmp_path / "Z.zarr"),
        init_args=init_args,
        models_folder=str(tmp_path),
    )

    mock_basicpy.assert_not_called()
    mock_empty.assert_not_called()

def test_correct_flatfield_missing_flatfield_raises(mocker, mock_flatfield_env, tmp_path):
    
    # Prepare a fake illumination model file without flatfield
    models_dir = tmp_path / "testchannel"
    models_dir.mkdir(parents=True)
    np.savez(
        models_dir / "profiles.npz",
        darkfield=np.zeros((3, 3)),
        baseline=5,
    )

    init_args = dict(
        channel_name="testchannel",
        channel_index=0,
        FOV_list=[0],
        max_z=None,
        saving_path=None,
    )

    with pytest.raises(ValueError):
        correct_flatfield(
            zarr_url=str(tmp_path / "X.zarr"),
            init_args=init_args,
            models_folder=str(tmp_path),
        )

def test_correct_raises_for_wrong_shape():

    img = np.zeros((2, 3, 4, 5), dtype=np.uint16)  # bad: shape[0] != 1
    illum = IlluminationModel(flatfield=np.ones((4, 5)))

    with pytest.raises(ValueError):
        correct(img, illum)

def test_correct_flatfield_only():

    img = np.ones((1, 1, 2, 2), dtype=np.uint16) * 100
    flat = np.ones((2, 2)) * 10

    illum = IlluminationModel(flatfield=flat, darkfield=None, baseline=None)

    out = correct(img, illum)

    expected = (img.astype(np.float32) / (flat + 1e-6)).astype(np.uint16)
    assert np.array_equal(out, expected)





