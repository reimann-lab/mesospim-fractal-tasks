# Crop Regions of Interest

This task crops regions of interest from a multi-channel OME-Zarr image and saves them as new images in the same OME-Zarr. The task takes as input a list of regions of interest and crops them from the full image. It reads an ROI coordinates table from the OME-Zarr 
directory and uses it to crop the image. To obtain the table, refer to the plugin [napari-crop-tool](https://github.com/girochat/napari-crop-tool/tree/main).

There can be two different types of region of interest:

1. ROI: The regions of interest are a sample of a given image intended for downstream analysis.
2. Crop: The region of interest here is simply a smaller version of the full image to remove empty space and reduce the size of the OME-Zarr.

---

## Parameters

Here is a detailed explanation of important parameters.

### Crop or ROI

The task can produce a crop of the full resolution image to remove empty space for example. With this parameter, you can specify whether the coordinates correspond to a crop or to one more ROIs. This matters mainly for file naming and task overhead.

### ROI Table Name

The task will look for a ROI coordinates table at the same location of the image that one wants to crop (e.g. if ROIs are taken from `raw_image_illum_corr`, the table should be in `raw_image_illum_corr`). By default it will look for a table named `roi_coords.csv`. If the name is different you can specify it with this parameter. 
When using the Napari plugin **Crop Tool** to crop the ROIs, it saves the output coordinates by default using the name `roi_coords.csv` in the image folder currently opened in Napari.

---

## Local Run Command

```bash
python run_crop_rois.py 
```
