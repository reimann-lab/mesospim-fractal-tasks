# mesoSPIM Fractal Tasks

[![PyPI version](https://img.shields.io/pypi/v/mesospim-fractal-tasks.svg)](https://pypi.org/project/mesospim-fractal-tasks/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://reimann-lab.github.io/mesospim-fractal-tasks)

This package contains a collection of [Fractal](https://fractal-analytics-platform.github.io/) tasks intended to convert and process [mesoSPIM](https://mesospim.org/) lightsheet image data into the [OME-Zarr v0.4](https://ngff.openmicroscopy.org/specifications/0.4/index.html) format for scalable downstream analysis and visualization.

---

## Background

**mesoSPIM** (mesoscale selective plane-illumination microscopy) is an open-hardware light-sheet microscope optimized for rapid, near-isotropic imaging of large (cm³-scale) cleared tissue samples.

**Fractal** is a bioimaging data processing framework developed at the BioVisionCenter, University of Zurich. It orchestrates scalable, reproducible analysis pipelines operating on OME-Zarr data, making large volumetric datasets tractable on both local machines and HPC clusters.

This package provides Fractal-compatible tasks that handle the specific file formats, metadata, and processing steps produced by mesoSPIM acquisitions.

---

## Available Tasks

| Task | Description |
|---|---|
| **Convert mesoSPIM to OME-Zarr** | Converts raw mesoSPIM image files to the OME-Zarr format with proper metadata and multi-resolution pyramids |
| **Prepare mesoSPIM OME-Zarr** | Prepare the OME-Zarr output by the mesoSPIM so that it is compatible with the rest of the mesospim-fractal-tasks pipeline |
| **Modify OME-Zarr Structure** | Modify post-creation the structure of an OME-Zarr such as channel labels, colors but also pyramid structure, chunksizes, etc... |
| **Correct Flatfield** | Perform flatfield correction for a multi-tile dataset |
| **Correct Illumination** | Perform a global illumination correction for a multi-tile dataset |
| **Stitch with multiview stitcher** | Stitch a multi-tile dataset using the multiview stitcher |
| **Crop Regions of Interest** | Extract regions of interest as a new image in the OME-Zarr |
| **Archive or dearchive an OME-Zarr** | Archive or dearchive an OME-Zarr by compressing or uncompressing all image data to a TAR file |


> For full task documentation including all parameters, refer to the full [documentation](https://reimann-lab.github.io/mesospim-fractal-tasks/tasks).

---

## Installation

### 1. Conda environment

A ready-to-use conda environment is provided:

```bash
conda env create -f environment.yml
conda activate mesospim-fractal-tasks
```

### 2. Package from source

```bash
git clone https://github.com/reimann-lab/mesospim-fractal-tasks.git
cd mesospim-fractal-tasks
pip install -e .
```

---

## Usage

### Running tasks via the Fractal server

1. You need a running Fractal server instance. See the [Fractal documentation](https://fractal-analytics-platform.github.io/) for details.
2. Collect the `mesospim-fractal-tasks` package on the server using the package name or a local wheel if not available in the task registry provided by Fractal.
3. Refer to the [Fractal documentation](https://fractal-analytics-platform.github.io/fractal/) for details on how to run tasks.

### Running tasks directly in Python

Tasks can also be called as plain Python functions, which can be useful for testing or local scripting. The run templates for each task can be copied in the current directory and modified to suit your needs:

```bash
cd /path/to/mesospim-fractal-tasks
copy-run-templates
```

Then, you can run the tasks directly in Python. See the [documentation](https://reimann-lab.github.io/mesospim-fractal-tasks/usage/local_run) for more information about the task parameters and the
run templates.

---

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes. 

---

## License

This project is licensed under the **BSD 3-Clause License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [mesoSPIM initiative](https://mesospim.org/) for the open-source microscope platform
- [Fractal Analytics Platform](https://fractal-analytics-platform.github.io/) (BioVisionCenter, University of Zurich) for the task framework
- [BioVisionCenter](https://www.biovisioncenter.uzh.ch/en.html)
- [Multiview Stitcher](https://github.com/mpicbg-csbd/multiview_stitching) for the multiview stitching implementation
- [NGFF OME-Zarr](https://ngff.openmicroscopy.org/index.html) for the OME-Zarr / NGFF specification