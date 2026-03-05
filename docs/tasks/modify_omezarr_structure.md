# Modify existing OME-Zarr structure

This task allows to modify the structure of an existing OME-Zarr image. If for some reason you need to change the structure of your OME-Zarr or the building of the pyramid failed but the full resolution image is still available, you can use this task to modify or recover the structure of the OME-Zarr.

The task can be used in two ways:

1. **On the Fractal platform** – as part of a workflow.
2. **Locally** – by running the task directly as a Python function or CLI command.

---


## Parameters

Below is a detailed explanation of important parameters.

### New Image Name

With this parameter, you can change the name of the image in the OME-Zarr structure.

For example if nothing is provided:

        Analysis Directory= `C:/path_to_directory/IENFD25-X-X`

        Image Name = `raw_image`

        New Image Name = `image`

        New OME-Zarr structure:

            IENFD25-X-X.zarr
                - image
                    - chunks of the image


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

### Channels List

With this parameter, you can modify the metadata of a channel. The metadata includes the channel label (.e.g PGP9.5), the color shown in Napari and the contrast limits. You can provide new settings for one or more channels. 
It is not mandatory to fill all the fields, but the laser wavelength and the label are mandatory to correctly identify the channel to be modified in the OME-Zarr.

Here is some information about the channel metadata:
- color:  
    it must be a hex color code (e.g. FF0000 for red, you can find the hex code of a color on several websites: https://www.color-hex.com/color-wheel/)
- contrast limits:  
    in Napari and other visualization software, the contrast limits of the image are adapted to the image intensity levels. For example, if most of the image is in the range of 0-100 and the contrast limits are set to 0-65535 (default, full range of possible intensities), then the image appears completely empty. By setting the start and end contrast limits to 0-100 for example, the image becomes visible.  
    When visualising your image, you can see what are the optimal contrast limits for your image by performing a right click on the contrast slider at the top left of Napari Layer panel.

---   

## Local Run Command

```bash
python run_modify_omezarr.py 
```

---
