# Stitching

This task stitches FOVs of a multi-channel, multi-tile OME-Zarr image using [Multiview Stitcher](https://github.com/multiview-stitcher/multiview-stitcher). The coordinates of the tiles in the original metric space must be referenced in an `FOV_ROI_table` in the `tables` directory in the OME-Zarr directory. See the [Fractal documentation](https://fractal-analytics-platform.github.io/fractal-tasks-core/tables/#roi-tables) for more information on how to build such a table.

---
## Parameters

In this section, we describe in more detail important parameters for the task.

### Channel

The stitching will be performed using a specific channel to compute the new tile locations. The channel must be available in the OME-Zarr image. It can be specified either by providing the `laser wavelength` or the `label` of the channel. All channels will be stitched according to the same registration parameters obtained using the specified channel by this parameter.

### Registration Resolution Level

The OME-Zarr image is typically stored with a pyramid of resolution levels. The highest resolution level is 0, the lowest resolution level is the number of levels minus 1. By default, the task will use the lowest resolution level for registration. You can use this parameter to set a different resolution level.


### Max Workers

The task can be run in parallel over all the chunks of the image. This parameter allows you to set the maximum number of workers to use for the registration. It cannot be higher than the number of cores of your machine. Increasing this number can also lead to memory overflow. There is a trade-off between the number of workers and the memory usage.

### Fusion Chunksize

The task fuses the tiles into a single image and do so per chunk. By default, it takes the chunksize of the original image. If this is too large, it can lead to memory overflow. You can set a smaller chunksize to reduce memory usage. However, this increases the time to fuse the tiles.

---    

## Local Run Command

```bash
python run_stitching.py 
```

---