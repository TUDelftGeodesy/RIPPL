# RIPPL

The [R]adar [I]nterferometric [P]arallel [P]rocessing [L]ab is a python package for processing SAR SLC data and NWP model data to create stacks of InSAR data including tropospheric delays derived from NWP model data. The name refers to the fringe pattern or ripples that we see in InSAR products. The main use case is the creation of stacks of interferograms and tropospheric delay maps for specific areas. To ease the use of this package, the search and download of Sentinel-1 data is supported by the package.

Common InSAR products covered by this package are:

- Complex interferograms;
- Absolute coherence values;
- Radiomatic calibrated amplitude data;
- Unwrapped interferograms (based on external package of snaphu);
- Download and creation of SRTM and TanDEM-X DEMs
- Common SAR geometry values as incidence angles, off-nadir angles and heading;
- Baselines, height to phase values;
- Projected and geocoded output grids of interferograms.

To create InSAR products, it includes functions to:

- Geocode SAR images;
- Coregister SAR images;
- Calculate and apply earth and topographic phase corrections;
- Resampling of SAR data using different kernels.

To derive tropospheric delay values and compare InSAR and NWP model data the following steps are implemented:

- Download of ERA5 and CERRA data from the climate data store
- Ray-tracing of the SAR signal through these NWP model datasets
- Derivation of tropospheric delays
- Creation of synthetic inteferograms based on tropospheric delays
- Use of time-shifts through advection to offset the timing of the original downloaded NWP model data

Current support sensors:

- Sentinel-1.

Support on more sensors is planned.

## Rationale

The rationale behind this software package is that we want to simplify the implementation of new processing steps as much as possible, while maintaining the processing speed needed for InSAR processing. This is also the reason why it is called the processing lab. At the same time, it includes all the necessary steps to estimate the tropospheric delays for individual interferograms based on NWP model data. We want to make the connection between the processing package with post-processing operation as easy as possbile. Therefore, the data is outputted as projected geotiff images and the stack is easily searchable for results of specific processing steps and can be loaded as numpy memmap files, which gives access to the full stack while the use of active memory is limited.

The implementation of your own processing steps within the package is encouraged. This can be added to later versions of the package after testing.

## Tutorial and examples

The RIPPL package includes two tutorials, one for the general InSAR processing and one including tropospheric delay calculations. Please try the tutorial jupyter notebook first to get familiar with the processing. You can find it under `.../tutorial/tutorial_Hawaii_earthquake_May_2018.ipynb`.

Note that before you can run these examples you should prepare your python setup and the RIPPL package setup. See the next section for details.

## Installation

### 1. Download RIPPL

```bash
git clone https://github.com/TUDelftGeodesy/RIPPL.git
```

### 2. Install Dependencies

To keep your system tidy, we recommend creating a virtual environment ([What is a virtual environment?](https://realpython.com/python-virtual-environments-a-primer/)). There are multiple ways of doing this, here we recommend  `conda`.

##### Step 1: Download and install anaconda or miniconda (https://www.anaconda.com/download)

##### Step 2: Install dependencies and activate environment (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

```bash
# create new virtual environment
RIPPL_PATH="$HOME/rippl"
# install packages defined in environment.yml inside the virtual environment.
conda env create -f $RIPPL_PATH/environment.yml
conda activate rippl
```

##### Step 3: Check if GDAL is installed correctly

`RIPPL` uses gdal ([what isother dependencies gdal?](https://gdal.org)) and it's [python binding](https://pypi.org/project/GDAL/), which can somtimes cause problems after or during installation. Please make sure that `gdal` is correctly installed on your system.

```bash
gdal-config --version
```

##### Step 4: Install Snaphu

If you want to be able to unwrap interferograms using RIPPL, you will have to install [Snaphu](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) on your system. If you cannot or not want to compile the code yourself, you can download the precompiled code [here](https://step.esa.int/main/snap-supported-plugins/snaphu/)

##### Step 5: Package setup

To start your first SAR processing using RIPPL you should also set your user and environment settings. You can find the installation code as a jupyter notebook .../rippl/user_setup.ipynb or as a regular python script .../rippl/user_setup.py. The latter can be run from the command line in the right folder:

> python user_setup.py

This will set your accounts to download Sentinel-1 data and external DEMs and create a folder structure for the SAR, orbit, DEM and GIS database as well as a place to store your radar data_stacks.

The output of the setup scripts will be saved as a .json file (.../rippl/user_settings.json), which you can edit later on if you want to change your account or data folder settings.
