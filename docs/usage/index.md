# Usage Overview

This package provides two ways of running the mesoSPIM processing tasks depending on the available resources:

1. [**Using Fractal Environment**](fractal_integration.md)  
   Provided that a full Fractal deployment is available, this is the most convenient approach when working with large datasets or when you want to leverage workflow automation and cluster execution. 👉 [**Fractal Execution**](fractal_integration.md)

2. [**Using local Python files**](local_run.md)  
   In the absence of the Fractal environment, each task can be run locally using python files. This is useful for testing, development, debugging, or processing smaller datasets on a local machine. 👉 [**Local Execution**](local_run.md)

Both methods use the same underlying task definitions. The main difference lies in how task execution is orchestrated.

