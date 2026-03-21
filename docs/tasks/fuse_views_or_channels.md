# Fuse Views or Channels

This task fuses multiple image sources (either channels acquired separately or multi-view acquisitions) into a single OME-Zarr image using [Multiview Stitcher](https://github.com/multiview-stitcher/multiview-stitcher).

Depending on the use case (channels or views fusion), fusion can be applied at different stages of the pipeline:

- Channels → typically fused early (right after OME-Zarr conversion, on the raw image)   
- Views → typically fused late (after stitching and preprocessing)   


---
## Parameters

In this section, we describe in more detail important parameters for the task.

### Channel

This parameter matters only if you want to fuse **Views**. It tells which channel should be used to compute the correct alignment of the views so they can be fused together into a single new image. The channel must be available in all the separate view images. It can be specified either by providing the `laser wavelength` or the `label` of the channel. All channels of the views will be fused according to the same registration parameters obtained using the specified channel by this parameter.

If you want to fuse **Channels**, you should leave this parameter empty. When this parameter is empty, the task will fuse all the separate channel images into a single new OME-Zarr image.

### Output Zarr Name

By default, the task will create a new OME-Zarr image with the common name of the input images. With this parameter, you can specify a custom name for the output OME-Zarr that will contain the fused image.

### Zarr Image Paths

By default, the task will search in the parent directory of the currently used OME-Zarr view or channel for other OME-Zarr images with the same common name. They will be selected as candidates for fusion. THe rule is to look for images with the same prefix and either a numbering suffix or a letter suffix. For example:

- common name = `IENFD25-6-1_ch...`, the task will look for `IENFD25-6-1_ch488`, `IENFD25-6-1_ch561`, etc.
- common name = `IENFD25-6-1_sh`, the task will look for `IENFD25-6-1_sh1`, `IENFD25-6-1_sh2`, etc.
- common name = `IENFD25-6-1_`, the task will look for `IENFD25-6-1_1`, `IENFD25-6-1_2`, or `IENFD25-6-1_a`, `IENFD25-6-1_b`, etc.

This automatic detection works only if the image names are consistent and differ only by a suffix. If the image names are not consistent, you can provide the explicit list of paths to the images you want to fuse using this parameter.


### Registration Resolution Level

The OME-Zarr image is typically stored with a pyramid of resolution levels. The highest resolution level is 0, the lowest resolution level is the number of levels minus 1. By default, the task will use the lowest resolution level for registration. You can use this parameter to set a different resolution level.


### Max Workers

The task can be run in parallel over all the chunks of the image. This parameter allows you to set the maximum number of workers to use for the registration. It cannot be higher than the number of cores of your machine. Increasing this number can also lead to memory overflow. There is a trade-off between the number of workers and the memory usage.

### Fusion Chunksize

The task fuses the tiles into a single image and do so per chunk. By default, it takes the chunksize of the original image. If this is too large, it can lead to memory overflow. You can set a smaller chunksize to reduce memory usage. However, this increases the time to fuse the tiles.

It is recommended to set a smaller chunksize than the original image chunksize. This improves the visualization experience of the fused image. For example:

- If the original image chunksize is (64, 1024, 1024), a chunksize of (32, 512, 512) or (32, 256, 256) can be used.

---    

## Local Run Command

```bash
python run_fuse_views_or_channels.py 
```

---