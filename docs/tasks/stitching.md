# Stitching

This task performs a global illumination normalization across all tiles of a multitile dataset. It ensures that tile intensities match consistently in overlapping regions, reducing brightness differences that arise from acquisition variability.

The correction operates by analyzing all tile–tile overlaps, estimating multiplicative coefficients that bring tiles to a common illumination level, and applying these coefficients to the entire dataset. 

An optional Z-axis correction is available to compensate for Z band artifacts.

---
## Parameters

The task runs in parallel over the channels. There is a initialisation step before the illumination correction is applied to each channel. Below is a detailed explanation of all available parameters for the initialisation step and the correction step.

::: mesospim_fractal_tasks.tasks.stitch_with_multiview_stitcher.stitch_with_multiview_stitcher
    options:
        heading: "Stitching Task"
        toc_label: "Stitching Task"
        heading_level: 3
        show_docstring_type_parameters: false
        show_root_heading: true
        show_object_full_path: false
        show_docstring_description: false
        show_docstring_parameters: true
        show_source: false
        show_docstring_returns: false

---    

## Local Run Command

```bash
python run_stitching.py 
```

---