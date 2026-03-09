from mesospim_fractal_tasks.tasks.mesospim_to_omezarr import (
    mesospim_to_omezarr,
    read_metadata,
    dispatcher,
    convert_h5_multitile,
    convert_tiff,
    convert_raw,
    load_channel_colors,
    write_ome_zarr_metadata,
    find_metadata_file,
    find_raw_image_files
)
import zarr
import pytest
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tifffile import imwrite

MODULE = "mesospim_fractal_tasks.tasks.mesospim_to_omezarr"

def test_zarr_creation(
    tmp_dataset, 
    h5_txt_metadata, 
    mock_mesospim_env
):
    # 1. Default Zarr creation
    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
    )

    expected_default = tmp_dataset / f"{tmp_dataset.name}.zarr"
    assert expected_default.exists()

    # Check image group exists
    root = zarr.open_group(expected_default, mode="r")
    assert "raw_image" in root

    # 2. Custom Zarr creation (custom image name)
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
    tmp_dataset, 
    h5_txt_metadata, 
    mock_mesospim_env
):
    zarr_path = tmp_dataset / f"{tmp_dataset.name}.zarr"

    # 1. Pre-create Zarr with content
    root = zarr.open_group(zarr_path, mode="w")
    root.create_group("existing_group")

    # Function call should NOT delete existing content
    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset)
    )

    # Verify existing content still exists
    root2 = zarr.open_group(zarr_path, mode="r")
    assert "existing_group" in root2

def test_metadata_file_provided_and_exists(
    tmp_dataset, 
    h5_txt_metadata, 
    mocker, 
    mock_mesospim_env
):

    # Mock dependencies
    mocker.patch(MODULE + ".zarr.open_group")
    mock_read = mock_mesospim_env["read_metadata"]

    mesospim_to_omezarr(
        zarr_dir=str(tmp_dataset),
        metadata_file="image_0001.h5_meta.txt",
    )

    args, kwargs = mock_read.call_args
    assert args[0] == tmp_dataset / "image_0001.h5_meta.txt"
    assert kwargs["exclusion_list"] == []

def test_metadata_file_provided_and_missing(
    tmp_dataset, 
    mocker
):
    mocker.patch(MODULE + ".zarr.open_group")
    
    with pytest.raises(FileNotFoundError):
        mesospim_to_omezarr(
            zarr_dir=str(tmp_dataset),
            metadata_file="nope.txt",
        )

def test_raw_image_file_discovery(
    tmp_dataset
):
    h5_path = tmp_dataset / f"image_0001.h5"
    h5_path.touch()
    for i in range(2):
        tif_path = tmp_dataset / f"image_0001_ch{i}.tif"
        tif_path.touch()
    raw_path = tmp_dataset / f"image_0001.raw"
    raw_path.touch()

    # Test with h5 extension
    paths = find_raw_image_files(
        zarr_dir=str(tmp_dataset),
        pattern="image",
        extension="h5"
    )
    assert len(paths) == 1
    assert paths[0] == h5_path

    # Test with tif extension
    paths = find_raw_image_files(
        zarr_dir=str(tmp_dataset),
        pattern="image",
        extension="tif"
    )
    assert len(paths) == 2

    # Test with raw extension
    paths = find_raw_image_files(
        zarr_dir=str(tmp_dataset),
        pattern="image",
        extension="raw"
    )
    assert len(paths) == 1
    assert paths[0] == raw_path

    # Test with unsupported extension
    with pytest.raises(ValueError):
        find_raw_image_files(
            zarr_dir=str(tmp_dataset),
            pattern="image",
            extension="foo"
        )

def test_raw_image_multiple_h5(
    tmp_dataset
):
    h5_path = tmp_dataset / f"image_0001.h5"
    h5_path.touch()
    h5_path = tmp_dataset / f"image_0002.h5"
    h5_path.touch()
    
    with pytest.raises(FileNotFoundError):
        raw_paths = find_raw_image_files(
            zarr_dir=str(tmp_dataset),
            pattern="",
            extension="h5"
        )

def test_raw_image_no_catch(
    tmp_dataset
):

    with pytest.raises(FileNotFoundError):
        raw_paths = find_raw_image_files(
            zarr_dir=str(tmp_dataset),
            pattern="",
            extension="h5"
        )

def test_metadata_autodiscovery_single_file(
    tmp_dataset
):
    h5_path = tmp_dataset / f"image_0001.h5"
    h5_path_meta = tmp_dataset / f"image_0001.h5_meta.txt"
    h5_path_meta.touch()
    h5_path.touch()
    raw_image_paths = [h5_path]

    meta_path = find_metadata_file(
        zarr_dir=str(tmp_dataset),
        raw_image_paths=raw_image_paths,
        extension="h5"
    )

    assert meta_path.name == "image_0001.h5_meta.txt"

    raw_image_paths = []
    for i in range(2):
        tif_path = tmp_dataset / f"image_0001_Mag_ch{i}.tif"
        tif_path.touch()
        raw_image_paths.append(tif_path)
    tif_path_meta = tmp_dataset / f"image_0001.tif_meta.txt"
    tif_path_meta.touch()
    
    meta_path = find_metadata_file(
        zarr_dir=str(tmp_dataset),
        raw_image_paths=raw_image_paths,
        extension="tif"
    )
    assert meta_path.name == "image_0001.tif_meta.txt"

    raw_image_paths = []
    for i in range(2):
        raw_path = tmp_dataset / f"image_0001_Mag_ch{i}.raw"
        raw_path.touch()
        raw_image_paths.append(raw_path)
    raw_path_meta = tmp_dataset / f"image_0001.raw_meta.txt"
    raw_path_meta.touch()
    
    meta_path = find_metadata_file(
        zarr_dir=str(tmp_dataset),
        raw_image_paths=raw_image_paths,
        extension="raw"
    )
    assert meta_path.name == "image_0001.raw_meta.txt"

def test_metadata_autodiscovery_multiple_files(
    tmp_dataset
):
    h5_path = tmp_dataset / f"image_0001.h5"
    h5_path.touch()
    for i in range(2):
        h5_path_meta = tmp_dataset / f"image_0001.h5_s{i}-t0_meta.txt"
        h5_path_meta.touch()

    meta_path = find_metadata_file(
        zarr_dir=tmp_dataset,
        raw_image_paths=[h5_path],
        extension="h5"
    )
    assert meta_path.name == "image_0001.h5_meta.txt"

    raw_paths = []
    for i in range(2):
        tif_path = tmp_dataset / f"image_0001_Mag_ch{i}.tif"
        tif_path.touch()
        tif_path_meta = tmp_dataset / f"image_0001_Mag_ch{i}.tif_meta.txt"
        tif_path_meta.touch()
        raw_paths.append(tif_path)

    meta_path = find_metadata_file(
        zarr_dir=str(tmp_dataset),
        raw_image_paths=raw_paths,
        extension="tif"
    )
    assert meta_path.name == "image_0001.tif_meta.txt"

    raw_paths = []
    for i in range(2):
        raw_path = tmp_dataset / f"image_0001_Mag_ch{i}.raw"
        raw_path.touch()
        raw_path_meta = tmp_dataset / f"image_0001_Mag_ch{i}.raw_meta.txt"
        raw_path_meta.touch()
        raw_paths.append(raw_path)

    meta_path = find_metadata_file(
        zarr_dir=str(tmp_dataset),
        raw_image_paths=raw_paths,
        extension="raw"
    )

def test_metadata_autodiscovery_no_match(
    tmp_dataset
):
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

def test_empty_metadata_file(
    tmp_dataset, 
    h5_txt_metadata
):

    with pytest.raises(ValueError):
        read_metadata(h5_txt_metadata, exclusion_list=[])

def test_nan_in_metadata(
    tmp_dataset
):
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

def test_wrong_file_count(
    tmp_dataset, 
    mocker
):
    chunk_sizes = mocker.Mock(get_chunksize=mocker.Mock(return_value=(1,1,1,1)))
    image_group = {"raw_image": mocker.Mock()}

    # Create only ONE raw/TIFF file
    raw_file = tmp_dataset / "sample_ch0.raw"
    raw_file.write_bytes(b"\x00\x00" * (2*2*2))

    meta_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Number of raw files"):
        convert_raw(
            [raw_file],
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
            [tiff_file],
            image_group,
            str(tmp_dataset / "out"),
            meta_df,
            chunk_sizes,
        )

def test_num_channel_mismatch(
    tmp_dataset, 
    mocker
):
    chunk_sizes = mocker.Mock(get_chunksize=mocker.Mock(return_value=(1,1,1,1)))
    image_group = {"raw_image": mocker.Mock()}

    # Wrong channel names in stems
    image_paths_raw = []
    for name in ["wrong_channel", "640"]:
        p = tmp_dataset / f"sample_{name}.raw"
        image_paths_raw.append(p)
        p.write_bytes(b"\x00\x00" * (2*2*2))
    image_paths_tiff = []
    for name in ["wrong_channel", "640"]:
        p = tmp_dataset / f"sample_{name}.tiff"
        image_paths_tiff.append(p)
        arr = np.zeros((2,2,2), dtype=np.uint16)
        imwrite(p, arr)

    meta_file = Path("tests", "data", "multitile_example.h5_meta.txt")
    meta_df = read_metadata(meta_file, exclusion_list=[])

    with pytest.raises(ValueError, match="No raw file found"):
        convert_raw(
            image_paths_raw,
            image_group,
            str(tmp_dataset / "out"),
            meta_df,
            chunk_sizes,
        )
    with pytest.raises(ValueError, match="No tiff file found"):
        convert_tiff(
            image_paths_tiff,
            image_group,
            str(tmp_dataset / "out"),
            meta_df,
            chunk_sizes,
        )

def test_convert_h5_tile_channel_mismatch_raises(
    mocker,
    tmp_dataset
):
    h5_file = tmp_dataset / "example.h5"
    h5_file.touch()

    # Mock meta_df to have mismatch between channels and number of tiles
    meta_df = pd.DataFrame({"channel": ["ch0", "ch1", "ch1"], "value": [1, 2, 3]})

    chunk_sizes = mocker.Mock()
    chunk_sizes.get_chunksize.return_value = (1, 1, 1, 1)
    image_group = mocker.Mock()

    with pytest.raises(ValueError):
        convert_h5_multitile(
            [tmp_dataset / "example.h5"],
            image_group=image_group,
            image_path=str(tmp_dataset / "out"),
            meta_df=meta_df,
            chunk_sizes=chunk_sizes,
        )

def test_load_channel_colors_from_path(
    tmp_dataset
):
    json_file = tmp_dataset / "colors.json"
    json_file.write_text('{"ch0": {"label": "A", ' \
    '"laser_wavelength": 488, ' \
    '"color": "00FF00"}}')

    result = load_channel_colors(str(json_file))

    assert result == {"ch0": {"label": "A", "laser_wavelength": 488, "color": "00FF00"}}

def test_load_channel_colors_via_keyword(
    mocker, 
    tmp_dataset
):
    mocker.patch("mesospim_fractal_tasks.tasks.mesospim_to_omezarr.get_channel_settings_dir", 
                 return_value=tmp_dataset)
    

    json_file = tmp_dataset / "default_colors.json"
    json_file.write_text('{"0": {"label": "CH0"}}')

    result = load_channel_colors("default")

    assert result == {"0": {"label": "CH0"}}

def test_load_channel_colors_keyword_missing(
    mocker, 
    tmp_dataset
):
    mocker.patch("importlib.resources.files", return_value=tmp_dataset)

    with pytest.raises(FileNotFoundError):
        load_channel_colors("unknown")

def test_load_channel_colors_keyword_multiple(
    mocker, 
    tmp_dataset
):
    (tmp_dataset / "default1.json").write_text("{}")
    (tmp_dataset / "default2.json").write_text("{}")

    mocker.patch("importlib.resources.files", return_value=tmp_dataset)

    with pytest.raises(FileNotFoundError):
        load_channel_colors("default")

def test_write_ome_zarr_metadata_basic(
    mocker, 
    meta_df
):
    fake_group = mocker.Mock()
    fake_group.name = "/raw"
    fake_group.attrs = {}

    # --- Mock channel colors ---
    mocker.patch(
        MODULE + ".load_channel_colors",
        return_value={
            "0": {"label": "Ch0", "laser_wavelength": 488, "color": "00FF00"},
            "1": {"label": "Ch1", "laser_wavelength": 561, "color": "FF0000"},
        }
    )
    pyramid_dict = {
        "0": {"scale": (1.0, 1.0, 1.0), "coarsening_xy": 2, "coarsening_z": 1},
        "1": {"scale": (1.0, 2.0, 2.0), "coarsening_xy": 2, "coarsening_z": 2},
    }

    # --- Mock NgffImageMeta so it doesn't validate deeply ---
    mocker.patch(MODULE + ".NgffImageMeta")

    # --- Call function ---
    write_ome_zarr_metadata(
        zarr_group=fake_group,
        meta_df=meta_df,
        pyramid_dict=pyramid_dict,
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
        pyramid_dict=pyramid_dict,
        user_channels_path="whatever",
        input_param={"foo": "bar"},
    )
    ch0 = fake_group.attrs["omero"]["channels"][0]
    assert ch0["window"]["start"] == 0
    assert ch0["window"]["end"] == (2**16 - 1)