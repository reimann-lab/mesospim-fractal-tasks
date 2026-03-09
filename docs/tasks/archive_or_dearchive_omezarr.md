# Archive or Dearchive OME-Zarr

This task can be used to archive an OME-Zarr folder. All images contained in the folder will be first compressed to a higher compression level, the the full OME-Zarr will be archived to a unique TAR file for easier long-term archiving. The same task can also be used to later unarchive the OME-Zarr. 

The task can be used in two ways:

1. **On the Fractal platform** – as part of a workflow.
2. **Locally** – by running the task directly as a Python function or CLI command.

---


## Parameters

Below is a detailed explanation of important parameters.

### Zarr URL

In the archiving mode (see the [archive](#archive) parameter), you need to provide the path to any image stored in the OME-Zarr folder you wish to archive. The full OME-Zarr folder containing the image will then be archived. Although counterintuitive, this is to ensure compatibility with the Fractal platform, which requires the path to an OME-Zarr image to be provided.

In the unarchiving mode (see the [archive](#archive) parameter), you need to provide the path to the TAR archive file containing the OME-Zarr folder you wish to unarchive. It will then be extracted the OME-Zarr in the same directory as the directory containing the TAR file.


### Archive

This parameter can be used to run the task in archiving mode or in unarchiving mode:

```
    Archiving mode:  
        archive = True
    Unarchiving mode:  
        archive = False
```

### Keep TAR Archive

In the dearchiving mode, you can choose to still keep the TAR archive file after unarchiving using this parameter. By default, the archive is kept after unarchiving.

### Output Preview

By default, the task will create a downsampled version of the OME-Zarr folder called `preview`. This allows to still view the images contained in the OME-Zarr archive (which is unaccessible after archiving) at a much lower resolution than the original image, while drastically reducing the OME-Zarr folder size. With this parameter, you can choose to skip the creation of the preview.



---   

## Local Run Command

```bash
python run_archiving.py 
```

---
