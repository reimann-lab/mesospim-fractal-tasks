# Flat-Field Correction

This task corrects illumination non-uniformity by applying or estimating a flat-field image.  
It supports three different scenarios depending on what information is available.


## Scenarios
### 1. Flatfield image available (recommended)

If you already have a measured flat-field, you can provide it directly:  

- The flat-field must be stored in a `profiles.npz` named under the key `flatfield`.  
- The file `profiles.npz` must be located in a folder named after each channel label  
- You give the **path to the directory** containing all the channel folders (see [Parameters](#parameters)).

The task loads then the flatfield image and uses it to correct the raw data. This produces the most accurate correction.

---

### 2. Flatfield image from empty regions (recommended)

If your dataset contains **background tiles**, the task can estimate a flat-field from those.

There are two ways to specify empty data:  

#### **A. Whole empty tiles**
If one or more entire tiles are empty:  

- Set the **tile number** in the parameters.

If tile numbers are provided, the task extracts those tiles and uses them for the estimation.

#### **B. Only part of a tile is empty**
If there is no full tiles that contain only background but there is enough background at 
the **top or bottom** of tiles:  

- Provide the **start** and **end** Z-levels of the empty region.  
- Provide the **tile number** where these subvolumes must be extracted (optional).  

The task will then extract a subvolume up to the **Z start** and down to the **Z end**, and uses it for the estimation.

If no tile number is provided, the task automatically uses **the four corner tiles** to extract the top and bottom tile subvolume.

---

### 3. BaSiCPy Flatfield Model

If your dataset contains **no empty tiles or background regions**, the task computes a flat-field using a **statistical illumination model (BaSiCPy)**.

This method works even when the sample fills most of the field of view. It produces a robust estimation even when no explicit flat-field can be derived. it is however more computationally expensive and should be used only when no flat-field is available.
It is set when:  

- No tile numbers are provided.  
- No start/end Z-levels are provided.  
- No model folder is provided.  

---

### Summary

| Scenario                        | Flat-field source                    |
| ------------------------------- | ------------------------------------ |
| `profiles.npz` file provided    | Uses measured flat-field (best)      |
| Empty tiles or empty subvolumes | Estimates flat-field from background |
| No empty regions                | Computes flat-field via BaSiCPy      |


---
## Parameters

Here is a detailed explanation of important parameters.

### Models Folder

This parameter allows you to specify the path to a folder containing illumination profiles stored in `profiles.npz` files. The structure of the folder must satisfy the following rules:

- The folder must contain directory for each channel, named after the exact channel name as referenced in the channel settings. 
- Each channel folder must contain the file `profiles.npz`.
- The `profiles.npz` file must contain at least the `flatfield` key that stores the flatfield profile as a numpy array.


### FOV List

This parameter allows you to give the list of tiles to process that contain only empty space. Setting this parameter will skip the fitting of the BaSiCPy model and use the custom empty-tile flatfield modelling.

!!! Note
    Typically, the mesoSPIM multitile output file has a tile ordering that follows an inverted `N` shape, e.g. the first tile is the top left tile, the second tile is the one below it, and so on. Once it reaches the bottom of the column, the next tile starts at the top of the next column, from left to right:

        tile 00 - tile 03 - tile 06
           |         |         |
        tile 01 - tile 04 - tile 07
           |         |         |
        tile 02 - tile 05 - tile 08

!!! Important
    The tile numbering starts at 0!

### Z-Levels

This parameter allows you to specify the maximum number of z-levels to process at the top and bottom of the 3D tile stack. In case there is no tile that is completely empty, but some have sufficient empty space at the top or bottom of the stack, you can specify these of z-level limits. Z planes will then be sampled from the top and bottom of the stack, and the flatfield will be computed from these planes. Typically, if the sample has a round shape, there is often empty space in the corners of the image. This parameter allows you to use this empty space to compute the flatfield.

This parameter works in combination with the [FOV List](#fov-list) parameter. If both are provided, the z planes will be sampled from the provided FOV list (up to the limits provided), otherwise the z planes will be sampled from the four tiles at the corners of the image.

You need to provide two numbers for this parameter, the smallest one indicates the maximum height up to which z planes are sampled from the bottom of the stack, the biggest one indicates the maximum height down to which z planes are sampled from the top of the stack.

For example:

        Number of z planes in image = (0, 660) 

        z-levels = (50, 600)   =>   only z planes below 50 or above 600 can be sampled, supposed to be empty.

### Basicpy Model Parameters

This parameter allows you to specify the parameters used for the BaSiCPy model. See the [documentation](https://basicpy.readthedocs.io/en/latest/api.html) for more information. This task allows to modify only key parameters:

- autosegment : When True, it automatically tries to segment the image before fitting to remove sample information.
- autosegment_margin : Margin of the segmentation mask to the thresholded region.
- epsilon : Weight regularization term. Increasing it can lead to smoother models.
- get_darkfield : When True, will estimate the darkfield shading component.
- max_workers : Maximum number of threads used for processing. 
- smoothness_darkfield : Weight of the darkfield term in the Lagrangian. Increase to obtain smoother models.
- smoothness_flatfield : Weight of the flatfield term in the Lagrangian. Increase to obtain smoother models.
- working_size : Size for running computations. In particular, it allows to reduce the size of the image used 
  to fit the model. Can be useful to reduce memoryusage. None means no rescaling.

---    

## Local Run Command

```bash
python run_flatfield_correction.py 
```
