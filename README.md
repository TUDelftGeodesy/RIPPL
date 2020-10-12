# RIPPL

The [R]adar [I]nterferometric [P]arallel [P]rocessing [L]ab is a python package to process SAR SLC data to create
individual or stacks of InSAR processing data. The name refers to the fringe pattern or ripples that we see
in InSAR products. The main user case is the creation of individual interferograms or stacks of interferograms
for specific areas. To ease the use of this package the search and download of Sentinel-1 data is covered by
the package. 

Common InSAR products covered by this package are: 
- complex interferograms
- absolute coherence values
- radiomatic calibrated amplitude data
- unwrapped interferograms (based on Snaphu)
- download and creation of SRTM and TanDEM-X DEMs
- common SAR geometry values as incidence angles, azimuth angles, off-nadir angles and satellite heading
- baselines and height to phase values
- creation of projected and geocoded output grids 

To create InSAR products it includes functions to:
- Geocode SAR images
- Coregister SAR images
- Calculate and apply earth and topographic phase corrections
- Resampling of SAR data using different kernels

### Rationale 

The rationale behind this software package is that we want to simplify the implementation of new processing
steps as much as possible, while maintaining the processing speed needed for InSAR processing. This is also
the reason why it is called the processing lab. At the same time we want to make the connection between 
the processing package with post-processing operation as easy as possbile. Therefore, the data is outputted
as projected geotiff images and the stack is easily searchable for results of specific processing steps
and can be loaded as numpy memmap files, which gives access to the full stack while the use of active
memory is limited.  

The implementation of your own processing steps within the package is encouraged and can be added to later
versions of the package after testing.

### Tutorial and examples

The RIPPL package includes both a tutorial and several examples. Please try the tutorial jupyter notebook first
to get familiar with the processing. You can find it under .../tutorial/tutorial_Hawaii_earthquake_May_2018.ipynb.

To show a wider set of applications you can also check the examples folder, where a few different example cases
are given. 

Note that before you can run these examples you should prepare your python setup and the RIPPL package setup. 
(see next sections for details)

### Python installation

If you downloaded RIPPL and did not setup your python environment yet, follow the next steps to do so:

1. Install Anaconda python 3 on your system. (https://www.anaconda.com/products/download)
2. After installation go to your terminal (or powershell on windows) and create a new conda environment:


    > conda create --name rippl_env 

3. After installation activate this and install the needed packages. These are:
- scipy
- numpy
- scikit-image
- requests
- lxml
- gdal
- fiona
- shapely
- matplotlib
- cartopy
- pyproj
- utm
- jupyter


    > conda activate rippl_env \
    > conda install -n rippl_env (package_name)

Possibly you will have to add the conda-forge channel first to install utm. You can add this channel using:

    > conda config --add channels https://conda.anaconda.org/conda-forge/

On some systems not all packages will not directly install and give a dependencies error. In these cases you can 
install the packages using PIP too:

    > conda activate rippl_env \
    > pip install (package_name)

Still, if possible install the packages with the conda command, as this guarantees a stable environment.

### Package setup

To start your first SAR processing using RIPPL you should also set your user and environment settings. You 
can find the installation code as a jupyter notebook .../rippl/user_setup.ipynb or as a regular python script
.../rippl/user_setup.py. The latter can be run from the command line in the right folder: 


    > python user_setup.py

This will set your accounts to download Sentinel-1 data and external DEMs and create
a folder structure for the SAR, orbit, DEM and GIS database as well as a place to store your radar data_stacks.

The output of the setup scripts will be saved as a .txt file (.../rippl/user_settings.txt), which you can edit
later on if you want to change your account or data folder settings.
