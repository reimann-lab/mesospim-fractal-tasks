from skinnervation3d_fractal_tasks.tasks.mesospim_to_omezarr import (
    mesospim_to_omezarr,
    read_metadata,
    dispatcher,
    convert_h5_multitile,
    convert_tiff,
    convert_raw,
    load_channel_colors,
    write_ome_zarr_metadata
)
import zarr
import pytest
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tifffile import imwrite

def test_zarr_creation(
        tmp_dataset, h5_txt_metadata, mock_mesospim_env):
    
    # 1. Default Zarr creation
    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
    )

    expected_default = tmp_dataset / f"{tmp_dataset.name}.zarr"
    assert expected_default.exists()

    # check image group exists
    root = zarr.open_group(expected_default, mode="r")
    assert "raw_image" in root

    # 2. Custom Zarr creation
    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
        zarr_name="mydata",
        image_name="myimage",
    )

    expected_custom = tmp_dataset / "mydata.zarr"
    assert expected_custom.exists()

    # check image group exists
    root2 = zarr.open_group(expected_custom, mode="r")
    assert "myimage" in root2

def test_zarr_append_does_not_erase_existing(
        tmp_dataset, h5_txt_metadata, mock_mesospim_env):
    zarr_path = tmp_dataset / f"{tmp_dataset.name}.zarr"

    # 1. Pre-create a Zarr with content
    root = zarr.open_group(zarr_path, mode="w")
    root.create_group("existing_group")

    # 3. Call your function, which should NOT delete existing content
    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset)
    )

    # 4. Reload and verify existing content still exists
    root2 = zarr.open_group(zarr_path, mode="r")
    assert "existing_group" in root2

def test_metadata_file_provided_and_exists(
        tmp_dataset, h5_txt_metadata, mocker, mock_mesospim_env):

    # Mock dependencies
    mocker.patch("skinnervation3d_fractal_tasks.tasks.mesospim_to_omezarr.zarr.open_group")
    mock_read = mock_mesospim_env["read_metadata"]

    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
        metadata_file="image_0001.h5_meta.txt",
    )

    mock_read.assert_called_once_with(
        tmp_dataset / "image_0001.h5_meta.txt",
        []
    )

def test_metadata_file_provided_and_missing(tmp_dataset, mocker):
    mocker.patch("skinnervation3d_fractal_tasks.tasks.mesospim_to_omezarr.zarr.open_group")
    
    with pytest.raises(FileNotFoundError):
        mesospim_to_omezarr(
            zarr_dir=str(tmp_dataset),
            metadata_file="nope.txt",
        )

def test_metadata_autodiscovery_single_file(tmp_dataset, h5_txt_metadata, mock_mesospim_env):
    mock_read = mock_mesospim_env["read_metadata"]

    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
        pattern="image",
        metadata_file=None,
    )

    # Should have used the found metadata file
    used_path = mock_read.call_args[0][0]
    assert used_path.name == "image_0001.h5_meta.txt"

def test_metadata_autodiscovery_no_match(tmp_dataset):

    with pytest.raises(FileNotFoundError):
        mesospim_to_omezarr(
            zarr_dir=str(tmp_dataset),
            metadata_file=None,
        )

def test_metadata_autodiscovery_multiple_files(tmp_dataset, h5_txt_metadata):
    (tmp_dataset / "other_h5_txt_metadata.h5_meta.txt").write_text("1")

    with pytest.raises(FileNotFoundError):
        mesospim_to_omezarr(
            zarr_dir=str(tmp_dataset),
            metadata_file=None,
        )

def test_read_metadata_basic_structure():
    mfile = Path("tests", "data", "multitile_example.h5_meta.txt")

    df = read_metadata(mfile, exclusion_list=[])

    # expected rows
    assert len(df) == 2

    # expected columns
    required_cols = {
        "channel","filter",
        "x_scale","y_scale","z_scale","x_pos","y_pos",
        "x_n_pixels","y_n_pixels","z_n_pixels","ignore"
    }
    assert required_cols.issubset(df.columns)

    # dtypes
    if "intensity" in df.columns:
        assert df["intensity"].dtype == float
    assert df["x_n_pixels"].dtype == int
    assert df["ignore"].dtype == bool

    # no missing values in required fields
    assert not df[list(required_cols)].isna().any().any()

def test_empty_metadata_file(tmp_dataset, h5_txt_metadata):

    with pytest.raises(ValueError):
        read_metadata(h5_txt_metadata, exclusion_list=[])

def test_nan_in_metadata(tmp_dataset):
    source_txt = Path("tests", "data", "multitile_example.h5_meta.txt")
    new_txt = Path(tmp_dataset, "corrupted.h5_meta.txt")
    shutil.copy(source_txt, new_txt)

    with open(new_txt, "r") as f:
        lines = f.readlines()

    with open(new_txt, "w") as f:
        for line in lines:
            if "x_pixels" not in line:
                f.write(line)

    with pytest.raises(ValueError):
        read_metadata(new_txt, exclusion_list=[])

def test_incongruent_exclusion_list():
    mfile = Path("tests", "data", "multitile_example.h5_meta.txt")
    
    with pytest.raises(AssertionError):
        read_metadata(mfile, exclusion_list=[3, 4])

def test_dispatcher():
    func = dispatcher("h5")
    assert func is convert_h5_multitile

    func = dispatcher("tiff")
    assert func is convert_tiff

    func = dispatcher("tif")
    assert func is convert_tiff

    func = dispatcher("raw")
    assert func is convert_raw

def test_dispatcher_invalid():
    with pytest.raises(ValueError):
        dispatcher("none")

def test_convert_no_files(tmp_dataset, mocker):
    chunk_sizes = mocker.Mock(get_chunksize=mocker.Mock(return_value=(1,1,1,1)))
    image_group = {"raw_image": mocker.Mock()}
    meta_df = pd.DataFrame()

    with pytest.raises(FileNotFoundError):
        convert_raw(
            file_dir=str(tmp_dataset),
            basename="",
            image_group=image_group,
            image_path=str(tmp_dataset / "out"),
            meta_df=meta_df,
            chunk_sizes=chunk_sizes,
        )
    with pytest.raises(FileNotFoundError):
        convert_tiff(
            file_dir=str(tmp_dataset),
            basename="",
            image_group=image_group,
            image_path=str(tmp_dataset / "out"),
            meta_df=meta_df,
            chunk_sizes=chunk_sizes,
        )

def test_wrong_file_count(tmp_dataset, mocker):
    chunk_sizes = mocker.Mock(get_chunksize=mocker.Mock(return_value=(1,1,1,1)))
    image_group = {"raw_image": mocker.Mock()}

    # Create only ONE raw/TIFF file
    raw_file = tmp_dataset / "sample_ch0.raw"
    raw_file.write_bytes(b"\x00\x00" * (2*2*2))

    meta_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Number of raw files"):
        convert_raw(
            str(tmp_dataset),
            "sample",
            image_group,
            str(tmp_dataset / "out"),
            meta_df,
            chunk_sizes,
        )

    tiff_file = tmp_dataset / "sample_ch0.tiff"
    arr = np.zeros((2,2,2), dtype=np.uint16)
    imwrite(tiff_file, arr)
    with pytest.raises(ValueError, match="Number of tiff files"):
        convert_tiff(
            str(tmp_dataset),
            "sample",
            image_group,
            str(tmp_dataset / "out"),
            meta_df,
            chunk_sizes,
        )

def test_num_channel_mismatch(tmp_dataset, mocker):
    chunk_sizes = mocker.Mock(get_chunksize=mocker.Mock(return_value=(1,1,1,1)))
    image_group = {"raw_image": mocker.Mock()}

    # Wrong channel names in stems
    for name in ["wrong_channel", "640"]:
        p = tmp_dataset / f"sample_{name}.raw"
        p.write_bytes(b"\x00\x00" * (2*2*2))
    for name in ["wrong_channel", "640"]:
        p = tmp_dataset / f"sample_{name}.tiff"
        arr = np.zeros((2,2,2), dtype=np.uint16)
        imwrite(p, arr)

    meta_file = Path("tests", "data", "multitile_example.h5_meta.txt")
    meta_df = read_metadata(meta_file, exclusion_list=[])

    with pytest.raises(ValueError, match="No raw file found"):
        convert_raw(
            str(tmp_dataset),
            "sample",
            image_group,
            str(tmp_dataset / "out"),
            meta_df,
            chunk_sizes,
        )
    with pytest.raises(ValueError, match="No tiff file found"):
        convert_tiff(
            str(tmp_dataset),
            "sample",
            image_group,
            str(tmp_dataset / "out"),
            meta_df,
            chunk_sizes,
        )

def test_convert_h5_no_file(tmp_dataset, mocker):
    chunk_sizes = mocker.Mock(get_chunksize=mocker.Mock(return_value=(1,1,1,1)))
    image_group = {"raw_image": mocker.Mock()}

    meta_file = Path("tests", "data", "multitile_example.h5_meta.txt")
    meta_df = read_metadata(meta_file, exclusion_list=[])

    with pytest.raises(FileNotFoundError):
        convert_h5_multitile(
            file_dir=str(tmp_dataset),
            pattern="",
            image_group=image_group,
            image_path=str(tmp_dataset / "out"),
            meta_df=meta_df,
            chunk_sizes=chunk_sizes,
        )

def test_convert_h5_multiple_files(tmp_dataset, mocker):
    chunk_sizes = mocker.Mock(get_chunksize=mocker.Mock(return_value=(1,1,1,1)))
    image_group = {"raw_image": mocker.Mock()}

    meta_file = Path("tests", "data", "multitile_example.h5_meta.txt")
    meta_df = read_metadata(meta_file, exclusion_list=[])

    # Two matching .h5 files (empty; we never open them in this error path)
    (tmp_dataset / "sample_my_pattern_1.h5").touch()
    (tmp_dataset / "sample_my_pattern_2.h5").touch()

    with pytest.raises(FileNotFoundError):
        convert_h5_multitile(
            file_dir=str(tmp_dataset),
            pattern="my_pattern",
            image_group=image_group,
            image_path=str(tmp_dataset / "out"),
            meta_df=meta_df,
            chunk_sizes=chunk_sizes,
        )

def test_convert_h5_tile_channel_mismatch_raises(mocker):
    dataset_dir = Path("tests", "data")
    chunk_sizes = mocker.Mock(get_chunksize=mocker.Mock(return_value=(1,1,1,1)))
    image_group = {"raw_image": mocker.Mock()}

    meta_file = Path("tests", "data", "multitile_example.h5_meta.txt")
    meta_df = read_metadata(meta_file, exclusion_list=[])

    # Mock get_h5_structure so we control tile_names length
    # e.g. 3 tiles total, but 2 channels → 3 % 2 != 0 → ValueError
    mocker.patch(
        "skinnervation3d_fractal_tasks.tasks.mesospim_to_omezarr.get_h5_structure",
        return_value=["tile_0", "tile_1", "tile_2"],
    )

    chunk_sizes = mocker.Mock()
    chunk_sizes.get_chunksize.return_value = (1, 1, 1, 1)
    image_group = mocker.Mock()

    with pytest.raises(ValueError):
        convert_h5_multitile(
            file_dir=dataset_dir,
            pattern="example",
            image_group=image_group,
            image_path=str(dataset_dir / "out"),
            meta_df=meta_df,
            chunk_sizes=chunk_sizes,
        )

def test_load_channel_colors_from_path(tmp_dataset):
    json_file = tmp_dataset / "colors.json"
    json_file.write_text('{"ch0": {"label": "A", ' \
    '"laser_wavelength": 488, ' \
    '"color": "00FF00"}}')

    result = load_channel_colors(str(json_file))

    assert result == {"ch0": {"label": "A", "laser_wavelength": 488, "color": "00FF00"}}

def test_load_channel_colors_via_keyword(mocker, tmp_dataset):
    mocker.patch("importlib.resources.files", return_value=tmp_dataset
    )

    json_file = tmp_dataset / "default_colors.json"
    json_file.write_text('{"0": {"label": "CH0"}}')

    result = load_channel_colors("default")

    assert result == {"0": {"label": "CH0"}}

def test_load_channel_colors_keyword_missing(mocker, tmp_dataset):
    mocker.patch("importlib.resources.files", return_value=tmp_dataset)

    with pytest.raises(FileNotFoundError):
        load_channel_colors("unknown")

def test_load_channel_colors_keyword_multiple(mocker, tmp_dataset):
    (tmp_dataset / "default1.json").write_text("{}")
    (tmp_dataset / "default2.json").write_text("{}")

    mocker.patch("importlib.resources.files", return_value=tmp_dataset)

    with pytest.raises(FileNotFoundError):
        load_channel_colors("default")

def test_write_ome_zarr_metadata_basic(mocker, meta_df):
    fake_group = mocker.Mock()
    fake_group.name = "/raw"
    fake_group.attrs = {}

    # --- Mock channel colors ---
    mocker.patch(
        "skinnervation3d_fractal_tasks.tasks.mesospim_to_omezarr.load_channel_colors",
        return_value={
            "0": {"label": "Ch0", "laser_wavelength": 488, "color": "00FF00"},
            "1": {"label": "Ch1", "laser_wavelength": 561, "color": "FF0000"},
        }
    )

    # --- Mock NgffImageMeta so it doesn't validate deeply ---
    mocker.patch("skinnervation3d_fractal_tasks.tasks.mesospim_to_omezarr.NgffImageMeta")

    # --- Call function ---
    write_ome_zarr_metadata(
        zarr_group=fake_group,
        meta_df=meta_df,
        num_levels=2,
        coarsening_xy=2,
        user_channels_path="whatever",
        input_param={"foo": "bar"},
        contrast_limits={"0": {"start": 0, "end": 1000},
                         "1": {"start": 10, "end": 2000}}
    )

    # --- Assertions: ensure attrs were written ---
    # multiscales exists
    assert "multiscales" in fake_group.attrs
    ms = fake_group.attrs["multiscales"][0]
    assert ms["name"] == "raw"
    assert len(ms["datasets"]) == 2

    # omero exists
    assert "omero" in fake_group.attrs
    assert len(fake_group.attrs["omero"]["channels"]) == 2

    ch0 = fake_group.attrs["omero"]["channels"][0]
    assert ch0["label"] == "Ch0"
    assert ch0["window"]["start"] == 0
    assert ch0["window"]["end"] == 1000

    # acquisition metadata exists
    assert "acquisition_metadata" in fake_group.attrs
    acq = fake_group.attrs["acquisition_metadata"]["channels"]
    assert acq[0]["label"] == "Ch0"
    assert acq[1]["label"] == "Ch1"

    assert "fractal_tasks" in fake_group.attrs
    fractal_tasks = fake_group.attrs["fractal_tasks"]
    assert fractal_tasks["mesospim_to_omezarr"]["input_parameters"] == {"foo": "bar"}

    # And with default contrast limits
    fake_group.attrs = {}
    write_ome_zarr_metadata(
        zarr_group=fake_group,
        meta_df=meta_df,
        num_levels=2,
        coarsening_xy=2,
        user_channels_path="whatever",
        input_param={"foo": "bar"},
    )
    ch0 = fake_group.attrs["omero"]["channels"][0]
    assert ch0["window"]["start"] == 0
    assert ch0["window"]["end"] == (2**16 - 1)
