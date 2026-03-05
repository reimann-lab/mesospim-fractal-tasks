# Crop Regions of Interest

This task crops regions of interest from a multi-channel OME-Zarr image and saves them as new images in the same OME-Zarr. The task takes as input a list of regions of interest and crops them from the full image. It reads an ROI coordinates table from the OME-Zarr 
directory and uses it to crop the image. To obtain the table, refer to the plugin [napari-crop-tool](https://github.com/girochat/napari-crop-tool/tree/main).

There can be two different types of region of interest:

1. ROI: The regions of interest are a sample of a given image intended for downstream analysis that contains structure(s) of interest (gland, muscle, DEJ...).
2. Crop: The region of interest here is simply a smaller version of the full image to remove empty space and reduce the size of the OME-Zarr.

---

## Parameters

Here is a detailed explanation of important parameters.

### Crop or ROI

The task can produce a crop of the full resolution image to remove empty space for example. With this parameter, you can specify whether the coordinates correspond to a crop or to one more ROIs. This matters mainly for file naming and task overhead.

### ROI Table Name

The task will look for a ROI coordinates table in the OME-Zarr folder of the image that one wants to crop (e.g. if ROIs are taken from `sample.zarr/raw_image_illum_corr`, the table should be in `sample.zarr`). By default it will look for a table named `roi_coords.csv`. If the name of the table is different you can specify it with this parameter.  

When using the Napari plugin **Crop Tool** to crop the ROIs, it saves the output coordinates by default using the name `roi_coords.csv` in the OME-Zarr folder currently opened in Napari.  

Each ROI that was drawn using the plugin will be attributed the name `<tag>_roi_<index>` where `<tag>` is a custom tag to help identify the type of ROI that was created (e.g. `muscle_roi_1`). The tag can be set in the plugin. The index of the ROI is automatically incremented for each ROI drawn.

### Chunksize

The OME-Zarr stores image data in chunks and at several resolution levels in a so-called pyramid of multi-resolution. The chunksize parameter allows you to set the size of the chunks in pixels. It is recommended to keep the size of the chunks small to avoid memory issues, but not too small to reduce the overhead of storing the data in OME-Zarr. The size should be around 100Mb-1Gb, see [this article](https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes) for more infos.  
You can compute the size of the chunksize by multiplying their values and by the byte size (for mesoSPIM image data it is typically 2, but prefer 4 to comply with analysis tasks).  Furthermore, the chunksize should be set to a value that is a power of 2, e.g. 64, 128, 512... The chunksize impacts how fast the OME-Zarr can be viewed but also the memory requirements to process it. There are three required numbers to provide, one for each axis in the order `Z`, `Y`, and `X`. The `Z` axis is often smaller than the other two, so it is recommended to provide a smaller chunksize for it. 

Here is a video with a good visual explanation about the OME-Zarr structure such as the pyramid of resolution and the chunks (first 15min): [Handling huge imaging data with OME-Zarr](https://www.youtube.com/watch?v=RlvJXqKjmek)

Typical values examples:
- 128x512x512 (~136Mb)
- 32x1024x1024 (~136Mb)
- 64x1024x1024 (~268Mb)

### Num Levels

The core idea of the OME-Zarr file format is to store the image data in a pyramid of multi-resolution. This allows to view large images without loading the entire dataset in memory. The pyramid is built progressively, starting from the highest resolution (raw data) and decreasing the resolution by a factor of two or more at each level. Often when viewing the full size image (raw data), we do not need a high level of details for the full image at once (the display cannot even handle so many pixels). Therefore, the pyramid allows us to display the image at different levels of detail, each one corresponding to a different resolution, depending on the zoom into the image. It is similar to the google maps experience.

This parameter allows you to specify explicitly how many pyramid levels to build. By default, the task will try to estimate the optimal number of levels based on the size of the image so that it can be viewed in 3D on a normal computer. If you have a more performant computer, you can decrease the number of levels to have a better 3D experience.

Here is a video with a good visual explanation about the OME-Zarr structure such as the pyramid of resolution and the chunks (first 15min): [Handling huge imaging data with OME-Zarr](https://www.youtube.com/watch?v=RlvJXqKjmek)

---      

## Local Run Command

```bash
python run_crop_rois.py 
```
