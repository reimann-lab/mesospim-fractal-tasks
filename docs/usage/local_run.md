# Using Local Python Files

All tasks in this package can also be executed **outside of Fractal** using standard Python files. This is useful for:

- testing and debugging  
- running tasks locally  
- integrating tasks into custom scripts or pipelines  
- development and verification  

Local execution relies on small Python files that act as “task launchers,” each containing all parameters required for a given task.

---

# Procedure

Below is the recommended workflow for running tasks locally.

## 1. Copy the Python File Templates

Template files are stored in the repository and kept up to date with each release.  
Each template includes a complete list of parameters for a specific task.

To avoid accidentally editing the original files, you should **copy** the templates before modifying them.  
This only needs to be done once unless you want to refresh them later.

```bash
cd path/to/mesospim-fractal-tasks
copy-templates
```

```powershell
cd C:\path\to\mesospim-fractal-tasks
copy-templates
```

This will generate editable files in the current directory, following the naming pattern:
```
run_<taskname>.py
```

## Edit Template File

Before running the task, you need to edit the template file to set the parameters specific to your dataset. This 
includes the path to the OME-Zarr image to process for example.

## Run the task

Once the template file is edited, you can run the task using the following command:

```bash
python run_<taskname>.py
```

!!! tip 
    In case the list of parameters has been compromised, you can reiterate the copying step to retrieve the latest version of the template files.