# MesoSPIM Fractal Tasks for Lightsheet Image Processing

This package provides a collection of [**Fractal-compatible tasks**](https://www.biovisioncenter.uzh.ch/en/fractal.html) to process
[**mesoSPIM 3D fluorescence microscopy data**](https://mesospim.org).

It currently offers tasks for:

- Converting mesoSPIM raw 3D stacks to [**OME-Zarr**](https://ngff.openmicroscopy.org/) (currently in TIFF, H5, RAW)
- Illumination correction:  
    - Flatfield
    - Z-intensity correction
    - Global intensity multi-tile optimization
- Multitile stitching
- ROI cropping from large 3D OME-Zarr datasets

These tasks are intended to run on the [**Fractal Analytics Platform**](https://fractal-analytics-platform.github.io/), where they can be integrated into scalable workflows and executed efficiently on a cluster environment. However, all tasks can also be executed **locally and individually** on a single machine using standard shell commands, without requiring a full Fractal deployment. Instructions for running tasks manually are provided in the [Usage](usage/index.md) section.
