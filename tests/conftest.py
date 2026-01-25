import pytest
import pandas as pd
import numpy as np
import dask.array as da

class FakeDaskArray:
    def __init__(self, arr):
        self.arr = arr
        self.shape = arr.shape
        self.chunksize = arr.shape
        self.dtype = arr.dtype
        self.ndim = arr.ndim
        self.chunks = [[self.arr.shape[i]] for i in range(self.ndim)]

    def __getitem__(self, key):
        # Normalize list -> tuple (common bug source)
        if isinstance(key, list):
            key = tuple(key)

        # Allow int or slice directly (e.g. arr[0] or arr[:])
        if isinstance(key, (int, slice)):
            return FakeDaskArray(self.arr[key])

        # Allow tuple indexing (e.g. arr[(slice(...), slice(...), ...)])
        if isinstance(key, tuple):
            return FakeDaskArray(self.arr[key])

        raise TypeError(f"Invalid index type: {type(key)}")

    def compute(self):
        return self.arr

    def rechunk(self, *a, **k):
        return self

    def to_zarr(self, *a, **k):
        return None
    
def mock_dask_distributed(
    mocker,
    module: str
):
    mocks = {}

    # ---- Dask cluster CM
    cluster_cm = mocker.MagicMock(name="cluster_cm")
    fake_cluster = mocker.MagicMock(name="fake_cluster")
    cluster_cm.__enter__.return_value = fake_cluster
    cluster_cm.__exit__.return_value = None

    mocks["cluster"] = mocker.patch(
        module + "._set_dask_cluster",
        return_value=cluster_cm,
    )

    # ---- Dask client CM
    client_cm = mocker.MagicMock(name="client_cm")
    fake_client = mocker.MagicMock(name="fake_client")
    client_cm.__enter__.return_value = fake_client
    client_cm.__exit__.return_value = None

    mocks["client"] = mocker.patch(
        module + ".Client",
        return_value=client_cm,
    )
    return mocks

@pytest.fixture
def tmp_dataset(tmp_path):
    """
    Create an empty dataset directory that simulates the mesoSPIM input directory.
    """
    d = tmp_path / "dataset"
    d.mkdir()
    return d


@pytest.fixture
def h5_txt_metadata(tmp_dataset):
    """
    Create a single metadata TXT file that matches the pattern.
    """
    p = tmp_dataset / "image_0001.h5_meta.txt"
    p.write_text("dummy")
    return p

@pytest.fixture
def meta_df():
    return pd.DataFrame({
        "channel": ["0", "1"],
        "z_scale": [1.0, 1.0],
        "y_scale": [2.0, 2.0],
        "x_scale": [3.0, 3.0],
        "filter": [500, 600],
        "shutter": ["open", "open"],
        "zoom": [10, 10],
        "intensity": [20, 25],
    })

@pytest.fixture
def mock_mesospim_env(mocker):
    """Mock all heavy external dependencies of mesospim_to_omezarr."""

    module = "mesospim_fractal_tasks.tasks.mesospim_to_omezarr"
    mocks = {}

    mocks["dispatcher"] = mocker.patch(
        module + ".dispatcher")
    mocks["read_metadata"] = mocker.patch(
        module + ".read_metadata")
    mocks["build_pyramid"] = mocker.patch(
        module + ".build_pyramid")
    mocks["contrast"] = mocker.patch(
        module + "._determine_optimal_contrast")
    mocks["write_meta"] = mocker.patch(
        module + ".write_ome_zarr_metadata")

    mocks = mocks | mock_dask_distributed(mocker, module)

    return mocks

@pytest.fixture
def mock_init_correct_flatfield_env(mocker):

    module = "mesospim_fractal_tasks.tasks.init_correct_flatfield"
    
    mocks = {}
    df = pd.DataFrame({
        "y_micrometer": [0, 0, 0, 5, 5, 5, 10, 10, 10],
        "x_micrometer": [0, 5, 10, 0, 5, 10, 0, 5, 10],
    })
    mocks["read_zarr"]= mocker.patch(
        module + ".ad.read_zarr",
        return_value=mocker.Mock(to_df=lambda: df)
    )
    mocks["group_channel"] = mocker.patch(
        module + ".group_by_channel",
        return_value={"Ch0": {"zarr_url": "fake.zarr", "index": 0}},
    )
    mocks["load_meta"] = mocker.patch(
        module + ".load_NgffImageMeta",
        return_value=mocker.Mock(num_levels=1, coarsening_xy=2)
    )
    fake_array = mocker.Mock(shape=(1,1,10,10), chunks=(1,1,5,5))
    mocks["open"] = mocker.patch(
        module + ".zarr.open",
        return_value={"0": fake_array})
    mocks["create"] = mocker.patch(
        module + ".zarr.create")
    return mocks


@pytest.fixture
def mock_flatfield_env(
    mocker,
    tmp_path
):

    module = "mesospim_fractal_tasks.tasks.correct_flatfield"

    new_zarr = tmp_path / "fake_flatfield_corr"
    new_zarr.mkdir()

    mocks = {}
    mocks["collect_fovs"] = mocker.patch(
        module + ".collect_fovs", 
        return_value=np.ones((1, 5, 5)))

    # --- NGFF metadata ---
    class FakeNGFF:
        num_levels = 2
        coarsening_xy = 2
        def get_pixel_sizes_zyx(self, level):
            return (1, 1, 1)

    mocks["load_meta"] = mocker.patch(
        module + ".load_NgffImageMeta", 
        return_value=FakeNGFF())
    
    fake_df = pd.DataFrame(
        {
            "x_micrometer": [0],
            "y_micrometer": [0],
            "len_x_micrometer": [20],
            "len_y_micrometer": [20],
        },
        index=["FOV_0"],
    )
    mock_ad = mocker.Mock(to_df=lambda: fake_df)
    mocks["anndata_read"] = mocker.patch(
        module + ".ad.read_zarr",
        return_value=mock_ad)
    
    mocks["table_indices"] = mocker.patch(
        module + ".convert_ROI_table_to_indices", 
        return_value=[[0,5,0,20,0,20]])
    
    arr = np.arange(5*20*20).reshape(1, 5, 20, 20)
    fake_dask = FakeDaskArray(arr)
    mocks["from_zarr"] = mocker.patch(
        module + ".da.from_zarr",
        return_value=fake_dask)
    mocks["correct"] = mocker.patch(
        module + ".correct")
    mocks["open_group"] = mocker.patch(
        module + ".zarr.open_group")
    mocks["zarr_open"] = mocker.patch(
        module + ".zarr.open")
    mocks["zarr_open_array"] = mocker.patch(
        module + ".zarr.open_array")
    mocks["coarsen"] = mocker.patch(
        module + ".da.coarsen")
    mocks["copy_tables"] = mocker.patch(
        module + "._copy_tables_from_zarr_url")
    mocks["dask_to_zarr"] = mocker.patch(
        "dask.array.core.Array.to_zarr")
    mocks["dask_from_array"] = mocker.patch(
        "dask.array.from_array")
    mocks["resample"] = mocker.patch(
        module + ".resample_to_shape")
    mocks["concatenate"] = mocker.patch(
        module + ".da.concatenate",
    side_effect=lambda xs, axis=0: FakeDaskArray(
        np.concatenate(
            [x.arr if isinstance(x, FakeDaskArray) else x for x in xs],
            axis=axis
            )
        ),
    )

    mocks = mocks | mock_dask_distributed(mocker, module)

    return mocks

@pytest.fixture
def mock_init_correct_illumination_env(
    mocker
):
    
    module = "mesospim_fractal_tasks.tasks.init_correct_illumination"
    mocks = {}
    df = pd.DataFrame({
        "y_micrometer": [0, 0, 0, 5, 5, 5, 10, 10, 10],
        "x_micrometer": [0, 5, 10, 0, 5, 10, 0, 5, 10],
    })
    mocks["read_zarr"]= mocker.patch(
        module + ".ad.read_zarr",
        return_value=df
    )
    mocks["group_channel"] = mocker.patch(
        module + ".group_by_channel",
        return_value={"Ch0": {"zarr_url": "fake.zarr", "index": 0, "n_FOVs": 9}},
    )
    mocks["load_meta"] = mocker.patch(
        module + ".load_NgffImageMeta",
        return_value=mocker.Mock(num_levels=1, coarsening_xy=2)
    )
    fake_array = mocker.Mock(shape=(1,1,10,10), chunks=(1,1,5,5))
    mocks["open"] = mocker.patch(
        module + ".zarr.open",
        return_value={"0": fake_array})
    mocks["create"] = mocker.patch(
        module + ".zarr.create")
    
    return mocks

@pytest.fixture
def mock_correct_illumination_env(
    mocker
):
    module = "mesospim_fractal_tasks.tasks.correct_illumination"

    mocks = {}
    fake_meta = mocker.Mock()
    fake_meta.num_levels = 2
    fake_meta.coarsening_xy = 2
    fake_meta.get_pixel_sizes_zyx.return_value = (1, 1, 1)

    original_indices = [
        (0, 1, 0, 10, 0, 10),   # ROI 0
        (0, 1, 5, 15, 5, 15),   # ROI 1: overlaps by 5x5
    ]
    zarr_indices = original_indices
    mocks["meta"] = mocker.patch(
        module + ".load_NgffImageMeta",
        return_value=fake_meta)
    mocks["read_zarr"] = mocker.patch(
        module + ".ad.read_zarr")
    mocks["convert_ROI_table_to_indices"] = mocker.patch(
        module + ".convert_ROI_table_to_indices",
        side_effect=[original_indices, zarr_indices])
    mocks["load_meta"] = mocker.patch(
        module + ".load_NgffImageMeta",
        return_value=mocks["meta"])
    
    arr = np.arange(5*20*20).reshape(1, 5, 20, 20)
    fake_dask = FakeDaskArray(arr)
    mocks["from_zarr"] = mocker.patch(
        module + ".da.from_zarr",
        return_value=fake_dask)
    mocks["open_group"] = mocker.patch(
        module + ".zarr.open_group")
    mocks["zarr_open"] = mocker.patch(
        module + ".zarr.open")
    mocks["coarsen"] = mocker.patch(
        module + ".da.coarsen")
    mocks["copy_tables"] = mocker.patch(
        module + "._copy_tables_from_zarr_url")
    
    mocks = mocks | mock_dask_distributed(mocker, module)
    
    return mocks