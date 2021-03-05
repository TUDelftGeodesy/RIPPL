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

## Rationale 

The rationale behind this software package is that we want to simplify the implementation of new processing
steps as much as possible, while maintaining the processing speed needed for InSAR processing. This is also
the reason why it is called the processing lab. At the same time we want to make the connection between 
the processing package with post-processing operation as easy as possbile. Therefore, the data is outputted
as projected geotiff images and the stack is easily searchable for results of specific processing steps
and can be loaded as numpy memmap files, which gives access to the full stack while the use of active
memory is limited.  

The implementation of your own processing steps within the package is encouraged and can be added to later
versions of the package after testing.

## Tutorial and examples

The RIPPL package includes both a tutorial and several examples. Please try the tutorial jupyter notebook first
to get familiar with the processing. You can find it under .../tutorial/tutorial_Hawaii_earthquake_May_2018.ipynb.

To show a wider set of applications you can also check the examples folder, where a few different example cases
are given. 

Note that before you can run these examples you should prepare your python setup and the RIPPL package setup. 
(see next sections for details)


## Installation

### 1. Download RIPPL

```bash
git clone git clone git@bitbucket.org:grsradartudelft/rippl.git $HOME/rippl
```

### 2. Install Dependencies

To keep your system tidy, we recommend creating a virtual environment ([What is a virtual environment?](https://realpython.com/python-virtual-environments-a-primer/)). There are multiple ways of doing this, here we recommend  `conda` and python `venv`.  

#### a). Conda (recommended)

##### Step 1: Download miniconda
Linux (and Windows with Ubuntu subsystem):

```bash
# download latest miniconda3 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
# install miniconda into your home directory
./Miniconda3-latest-Linux-x86_64 -b -p $HOME/miniconda
# conda initialization in bash
conda init bash
```

Mac OS
```zsh
# download latest miniconda3 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x Miniconda3-latest-MacOSX-x86_64.sh.sh
# install miniconda into your home directory
./Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda
# If you are on macOS Catalina or later, the new default shell is zsh. If you're still using bash, then use bash instead of zsh when init.
conda init zsh
```

##### Step 2: Install dependencies

```bash
# create new virtual environment
RIPPL_PATH="$HOME/rippl" 
conda env create -f $RIPPL_PATH/environment.yml
```

##### Step 3: Activate virtual environment

After the installation, all you need to do each time is to activate the virtual environment:
```
conda activate rippl
```

When you finished processing using `RIPPL`, you can deactivate the virtual environment by:

```bash
conda deactivate
```

#### b). Python `venv`

The steps should be identical for Mac OS, Linux and Windows (with ubuntu subsystem), as long as `python3` is installed on your system. If you're not sure, do:
```bash
which python3
```
to check if you have python3 on your system path. 

##### Step 1: Create virtual environment

Please make sure you have **python>=3.6**. Then do:
```bash
 python3 -m venv $HOME/.venv/rippl
 source $HOME/.venv/rippl/bin/activate
```

ℹ️ You can also use [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/) or other virtual environment manage tools if you're familiar with those tools, but we will not elaborate here. 

##### Step 2: Install gdal
⚠️ `RIPPL` uses gdal ([what is gdal?](https://gdal.org)) and it's [python binding](https://pypi.org/project/GDAL/). Although it comes in handy, `gdal` is rather infamous for its [complexity in installation](https://www.google.com/search?q=why+is+it+so+hard+to+install+gdal?). So before you start installing python dependencies, please make sure you have `gdal` installed on your system. 

Ubuntu:
```bash
# https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html
sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
```

Mac OS (homebrew)
```zsh
brew install gdal
```

**Check your gdal version:**
```bash
gdal-config --version
```

##### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -e $HOME/rippl
```

⚠️ By default `gdal` v2.2.3 is installed. If your `gdal` version is different, then update the `requirements.txt` with:
```
gdal==<YOUR GDAL VERSION>
```
to avoid running into installation error. 


## Package setup

To start your first SAR processing using RIPPL you should also set your user and environment settings. You 
can find the installation code as a jupyter notebook .../rippl/user_setup.ipynb or as a regular python script
.../rippl/user_setup.py. The latter can be run from the command line in the right folder: 


    > python user_setup.py

This will set your accounts to download Sentinel-1 data and external DEMs and create
a folder structure for the SAR, orbit, DEM and GIS database as well as a place to store your radar data_stacks.

The output of the setup scripts will be saved as a .txt file (.../rippl/user_settings.txt), which you can edit
later on if you want to change your account or data folder settings.
