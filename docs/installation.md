# Installation

You can install the package either directly into a Conda/Mamba environment or in a Python virtual environment.

## Requirements

The package requires python version>=3.11, <3.13.

## Procedure

It is strongly recommended to install this package inside an isolated environment to avoid dependency conflicts.
You may use either Conda/Mamba or a Python virtual environment.
After the environment is created, the package is installed using pip.

1. Create a Conda or Mamba environment (recommended)

    An environment.yml file is provided to ensure all dependencies are installed with compatible versions. Additionally a environment_dev.yml is provided to install the package in development mode with dev dependencies.

    With Mamba/Conda:  
        ```bash
        mamba env create -f environment.yml
        mamba activate mesospim-tasks
        ```

2. Install the package in a Python virtual environment

    python -m venv venv
    source venv/bin/activate
    pip install mesospim-fractal-tasks
