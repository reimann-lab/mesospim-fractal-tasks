# Flat-Field Correction

This task corrects illumination non-uniformity by applying or estimating a flat-field image.  
It supports three different scenarios depending on what information is available.


## Scenarios
### 1. Flatfield image available (recommended)
If you already have a measured flat-field, you can provide it directly:  
    - The flat-field must be stored in a `profiles.npz` named under the key `flatfield`.  
    - The file `profiles.npz` must be located in a folder named after each channel label  
    - You give the **path to the directory** containing all the channel folders (see Parameters).  

The task loads then the flatfield image and uses it to correct the raw data. This produces the most accurate correction.

---

### 2. Flatfield image from empty regions (recommended)

If your dataset contains **background tiles**, the task can estimate a flat-field from those.

There are two ways to specify empty data:  

#### **A. Whole empty tiles**
If one or more entire tiles are empty:  
    - Set the **tile index/indices** in the parameters.

If tile indices are provided, the task extracts those tiles and uses them for the estimation.

#### **B. Only part of a tile is empty**
If there is no full tiles that contain only background but there is enough background at 
the **top or bottom** of tiles:  
    - Provide the **start** and **end** Z-levels of the empty region.  
    - Provide the **tile index/indices** where these subvolumes must be extracted (optional).  

The task will then extract a subvolume up to the **Z start** and down to the **Z end**, and uses it for the estimation.

If no tile index is provided, the task automatically uses **the four corner tiles** to extract the top and bottom tile subvolume.

---

### 3. BaSiCPy Flatfield Model

If your dataset contains **no empty tiles or background regions**, the task computes a flat-field using a **statistical illumination model (BaSiCPy)**.

This method works even when the sample fills most of the field of view. It produces a robust estimation even when no explicit flat-field can be derived. it is however more computationally expensive and should be used only when no flat-field is available.
It is set when:  

- No tile index/indices are provided.  
- No start/end Z-levels are provided.  
- No flat-field image is provided.  

---

### Summary

| Scenario                        | Flat-field source                    |
| ------------------------------- | ------------------------------------ |
| `profiles.npz` file provided    | Uses measured flat-field (best)      |
| Empty tiles or empty subvolumes | Estimates flat-field from background |
| No empty regions                | Computes flat-field via BaSiCPy      |


---
## Parameters

The task runs in parallel over the channels. There is a initialisation step before the flat-field correction is applied each channel. Below is a detailed explanation of all available parameters for the initialisation step and the correction step.

::: mesospim_fractal_tasks.tasks.init_correct_flatfield.init_correct_flatfield
    options:
        heading: "Initialisation Task"
        toc_label: "Initialisation Task"
        heading_level: 3
        show_docstring_type_parameters: false
        show_root_heading: true
        show_object_full_path: false
        show_docstring_description: false
        show_docstring_parameters: true
        show_source: false
        show_docstring_returns: false

::: mesospim_fractal_tasks.tasks.correct_flatfield.correct_flatfield
    options:
        heading: "Correction Task"
        toc_label: "Correction Task"
        heading_level: 3
        show_docstring_type_parameters: false
        show_root_heading: true
        show_object_full_path: false
        show_docstring_description: false
        show_docstring_parameters: true
        show_source: false
        show_docstring_returns: false

---    

## Local Run Command

```bash
python run_flatfield_correction.py 
```
