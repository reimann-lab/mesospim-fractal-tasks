# Convert mesoSPIM Output to OME-Zarr

This task converts **mesoSPIM microscopy acquisitions** into an **OME-Zarr** dataset.
It supports mesoSPIM `.h5`, `.tiff`, and `.raw` files and generates a full-resolution OME-Zarr with multiscale levels, metadata integration, and channel labeling.

The task can be used in two ways:

1. **On the Fractal platform** – as part of a workflow.
2. **Locally** – by running the task directly as a Python function or CLI command.

---

## Overview

The converter performs the following steps:

* Detects all files in `zarr_dir` matching `pattern` and `extension`
* Loads mesoSPIM metadata (auto-detected or user-provided)
* Assembles channels and tiles depending on your dataset
* Writes an **OME-Zarr** dataset with multiscale pyramids
* Assigns channel names and colors based on user-defined JSON file


---

## Parameters

Below is a detailed explanation of important parameters.

### Pattern

In case there are several files with the same suffix (e.g. "h5") in the same analysis directory, you can specify a pattern to select the specific files you want to convert. This pattern needs to be uniquely present in the image file name and its metadata file.

For example:

        Analysis Directory= `C:/path_to_directory/IENFD25-X-X`

        Files with the same suffix: `IENFD25-X-X.h5`, `IENFD25-X-X_restained.h5`, `IENFD25-X-X_lectin.h5`

        Pattern = `restained` if you want to convert only the `IENFD25-X-X_restained.h5` file.

### Zarr Name

It is recommended to set a name for the new OME-Zarr that will be created. This name will be used as the root of the new Zarr hierarchy. If nothing is provided, the task will use the name of the analysis directory.

For example if nothing is provided:

        Analysis Directory= `C:/path_to_directory/IENFD25-X-X`

        Zarr Name = `IENFD25-X-X.zarr`

### Image Name

The OME-Zarr is a hierarchical structure. It can contain multiple related images, each with its own metadata. The task is converting the raw data to a single image in the OME-Zarr structure. By default, the task will give it the name `raw_image`. This parameter allows you to change the name of the image.

For example if nothing is provided:

        Analysis Directory= `C:/path_to_directory/IENFD25-X-X`

        Image Name = `raw_image`

        New OME-Zarr structure:

            IENFD25-X-X.zarr
                - raw_image
                    - chunks of the image

### Metadata File

The task will try to load the metadata based on the name of the source file (e.g. `\<source_filename\>_meta.txt`). In case the metadata filename has a different name than its source file, you can provide the path to the metadata file using this parameter.

For example:

        Analysis Directory= `C:/path_to_directory/IENFD25-X-X`
        
        Source File = `C:/path_to_directory/IENFD25-X-X/IENFD25-X-X.h5`

        Expected Metadata File = `C:/path_to_directory/IENFD25-X-X/IENFD25-X-X.h5_meta.txt`

### Channel Color File

There are default channel color settings saved in the package directory to define the color and naming of each channel based on the laser wavelength during acquisition. If you want to modify the available defaults and save your own channel color settings, you can provide a JSON file with the channel color definitions. The procedure is detailed [HERE](#channel-color-and-naming.md). You can select one of the default settings by providing its key in the parameters. You can also directly provide the path to compatible JSON file with your specific channel color definitions using this parameter.

For example:

        Channel Color File = `default` or `v0` or ...

        Channel Color File = `path_to/my_custom_channels.json`

### Exclusion List

This parameter allows you to exclude specific set of tiles from being converted to OME-Zarr. In case you have a multitile setup (e.g. h5 file) and a full
row and/or column of tiles are empty, you can provide a list of tile indices to exclude from the conversion. This reduces the size of the OME-Zarr and speeds up the conversion. 

!!! Note
    Typically, the mesoSPIM multitile output file has a tile ordering that follows an inverted `N` shape, e.g. the first tile is the top left tile, the second tile is the one below it, and so on. Once it reaches the bottom of the column, the next tile starts at the top of the next column, from left to right:

        tile 00 - tile 03 - tile 06
           |         |         |
        tile 01 - tile 04 - tile 07
           |         |         |
        tile 02 - tile 05 - tile 08

!!! Important
    The tile numbering starts at 0!

### Chunksize

The OME-Zarr stores image data in chunks and at several resolution levels in a so-called pyramid of multi-resolution. The chunksize parameter allows you to set the size of the chunks in pixels. It is recommended to keep the size of the chunks small to avoid memory issues, but not too small to reduce the overhead of storing the data in OME-Zarr. The size should be around 100Mb-1Gb, see [this article](https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes) for more infos. You can compute the size of the chunksize by multiplying their values and by the byte size (for mesoSPIM image data it is typically 2, but prefer 4 to comply with analysis tasks). Furthermore, the chunksize should be set to a value that is a power of 2, e.g. 64, 128, 512... The chunksize impacts how fast the OME-Zarr can be viewed but also the memory requirements to process it. There are three required numbers to provide, one for each axis in the order `Z`, `Y`, and `X`. The `Z` axis is often smaller than the other two, so it is recommended to provide a smaller chunksize for it. 

Typical values examples:
- 128x512x512 (~136Mb)
- 32x1024x1024 (~136Mb)
- 64x1024x1024 (~268Mb)
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
python run_convert_to_omezarr.py 
```

---

## Notes

* The task currently supports mesoSPIM H5/TIFF/RAW formats.
* Multiscale levels follow OME-NGFF conventions.

---
