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

Below is a detailed explanation of all available parameters.

---

::: mesospim_fractal_tasks.tasks.mesospim_to_omezarr.mesospim_to_omezarr
    options:
        heading: "Conversion Task"
        toc_label: "Conversion Task"
        heading_level: 3
        show_root_heading: true
        show_docstring_type_parameters: false
        show_object_full_path: false
        show_docstring_description: false
        show_docstring_parameters: true
        show_source: false
        show_docstring_returns: false

---      

## Channel Color and Naming

The task will assign channel names and colors based on a JSON file. There is a default template that is provided to set
custom channel settings. Here is the procedure to get the template and register your own settings as default in case of
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
