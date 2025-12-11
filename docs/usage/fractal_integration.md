# Using Fractal Environment

This section explains how to run the mesoSPIM processing tasks within
the **Fractal platform** using workflows.

## Prerequisites

A working Fractal installation.  
If you are new to Fractal, there is a small wiki that describes the first steps to build a workflow:

👉 [**Fractal Wiki**](https://biovisioncenter.notion.site/?v=06a03711ab6c4c5a9763b0a6c078d278) 

This guide explains how to:

- create a project
- create a dataset
- assemble workflows
- add tasks
- run workflows
- visualise outputs

## Selecting MesoSPIM Tasks

Normally, the mesospim-fractal-tasks task package should be available in the list of collections of tasks provided by Fractal under the **Tasks** Tab. In case it is not present, the task package can be manually collected by providing a wheel file of the current package:

1. Download a wheel file of the package from the github repository (under releases) 
2. On Fractal website, go to the **Tasks** tab and click on **Manage Tasks**
3. Select **Local wh** for the **package type**
4. Provide the path to the wheel file of the mesospim-fractal-tasks package previously downloaded
5. Select the option **private task**
6. Run **Collect**

