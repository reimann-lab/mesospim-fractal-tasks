# Prepare mesoSPIM OME-Zarr

This task prepares a new OME-Zarr that will be used for the downstream analysis. To insure compatibility with the analysis pipeline and the OME-Zarrs output by mesoSPIM, it will create a tiny proxy raw image in this new OME-Zarr. This raw image is a fake OME-Zarr image that contains the same metadata as the original OME-Zarr and additional information to enable the analysis pipeline to find and correctly assemble the separate tiles output by the mesoSPIM. Once this task is completed, the analysis pipeline can be run on the proxy raw image normally.  
The task can be used in two ways:

1. **On the Fractal platform** – as part of a workflow.
2. **Locally** – by running the task directly as a Python function or CLI command.

---

## Overview

The task performs the following steps:

* Detects the main mesoSPIM OME-Zarr in `zarr_dir` matching `pattern` that contains all the OME-Zarr tiles of the multitile acquisition.
* Loads mesoSPIM metadata
* Assembles channels and tiles
* Writes a tiny proxy fake raw image OME-Zarr mirroring the separate tiles as one unique image.
* Assigns channel names and colors based on user-defined JSON file

---

## Parameters

Below is a detailed explanation of important parameters.

### Pattern

In case there are several OME-Zarrs in the same analysis directory, you can specify a pattern to select the specific OME-Zarr you want to prepare. This pattern needs to be uniquely present in the OME-Zarr name and its metadata files.

For example:

        Analysis Directory= `C:/path_to_directory/IENFD25-X-X`

        Files with the same suffix: `IENFD25-X-X.ome.zarr`, `IENFD25-X-X_restained.ome.zarr`, `IENFD25-X-X_lectin.ome.zarr`

        Pattern = `restained` if you want to convert only the `IENFD25-X-X_restained.ome.zarr` file.

### Zarr Name

It is recommended to set a name for the new OME-Zarr that will be created. This name will be used as the root of the new Zarr hierarchy. If nothing is provided, the task will use the name of the analysis directory.

For example if nothing is provided:

        Analysis Directory= `C:/path_to_directory/IENFD25-X-X`

        Zarr Name = `IENFD25-X-X.zarr`


### Channel Color Settings

There are default channel color settings saved in the package directory to define the color and naming of each channel based on the laser wavelength during acquisition. If you want to modify the available defaults and save your own channel color settings, you can provide a JSON file with the channel color definitions. The procedure is detailed [HERE](#channel-color-and-naming). You can select one of the default settings by providing its key in the parameters. You can also directly provide the path to compatible JSON file with your specific channel color definitions using this parameter.

For example:

        Channel Color File = `default` or `v0` or ...

        Channel Color File = `path_to/my_custom_channels.json`


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

## Channel Color and Naming

The task will assign channel names and colors based on a JSON file. There is a default template that is provided to set
custom channel settings. 

Here is the procedure to view the current available default settings:
    
1. View the available channel settings

    In a shell, in the package directory, run:

        get-channel-keys

    This will print the available channel keys, e.g. the identifiers for the channel color settings.

2. View a given channel setting

    In a shell, in the package directory, run:

        get-channel-setting <key>

    This will print the channel settings for the given channel key.

Here is the procedure to get the template and register your own settings as default in case of
recurring use.

1. Obtain the channel settings template  

    In a shell, in the package directory, run:  

        copy-channel-template

    This command copies the JSON template `channel_setting_template.json` into the package directory.

2. Edit the template  

    Open the copied file and modify it to describe your actual imaging channels.
    It is important to keep the elements and structure of the JSON file intact.  
    You can provide the path to this file to the task in the parameters to use these channel settings. In case of
    recurring use, you can register the settings as default in the package settings directory (see below).

3. Register the channel settings (Optional)

    To make your JSON file available to the conversion task as the default setting, run:  

        set-channel-setting my_custom_channels.json 
    
    or accessible with a specific name:  

        set-channel-setting my_custom_channels.json --setting-name myspecificname

    This will validate the JSON structure and store the file internally in the package settings directory. You can now
    access it using the key "default" or the name you provided.

## Local Run Command

```bash
python run_prepare_omezarr.py 
```

---

## Notes

* The task currently supports mesoSPIM H5/TIFF/RAW formats.
* Multiscale levels follow OME-NGFF conventions.

---
