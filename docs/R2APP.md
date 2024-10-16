### R Package converting to BioMANIA APP README

This README framework provides a step-by-step guide for integrating R packages into the BioMANIA app. The process is streamlined for R, leveraging its capabilities for direct package usage and documentation access. Python scripts are still used for data preparation and model training. 

#### Step 1: Prerequisites
Before starting, ensure that R and `rpy2` are installed. `rpy2` allows for running R within Python, bridging the two environments.

- **Install R**: Download and install the latest version of R from [CRAN](https://cran.r-project.org/).
- **Install rpy2**: In your Python environment, run 
```shell
pip install rpy2
```
- **Install your R lib**: In terminal, enter `R` to enter R environment, and install its package
```shell
R
> install.packages('Your Package Name')
> exit
```

#### Step 2: Generating API Initialization File
Use a Python script to generate the `./data/standard_process/{Your_R_Library_Name}/API_init.json` file. This file contains initial information about the APIs in the R package.

- Run the following command:
```bash
python -m src.R2APP.get_API_init_from_sourcecode_R --LIB [Your_R_Library_Name] # e.g.  Seurat, ggplot2
cp -r ./data/standard_process/{Your_R_Library_Name}/API_init.json /data/standard_process/{Your_R_Library_Name}/API_composite.json
```

#### Follow the [remaining steps](PyPI2APP) from preparing data synthesis on. Remember to add the information into `Lib_cheatsheet.py`

#### Key Differences Between R and Python Integration

- **Library Loading**: In R, use `library(LIB)` to load packages directly. 
- **Documentation Access**: R documentation can be accessed through `help()`, `??`, or the `.__doc__` attribute after converting R functions to Python via `rpy2`.
- **Arguments Information**: R documentation didn't always provide `type` information for parameters.
- **Simplified Process**: The process for R integration is more straightforward, focusing primarily on data preparation and model training, without the need to search for more documentation resources.

#### Final Notes
This framework outlines the key steps and differences for integrating an R package into BioMANIA. Adjust the Python commands and paths according to your package's specifics. If you have any questions or need assistance with specific steps, feel free to reach out!