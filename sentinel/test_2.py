#!/usr/bin/python
# Create a datastack

from stack import Stack
from sentinel.sentinel_stack import SentinelStack
from pipeline import Pipeline

track_no = 37
parallel = False

data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'

# t 37
if track_no == 37:
    start_date = '2017-11-02'
    end_date = '2017-11-20'
    master_date = '2017-11-09'

    database_folder = data_disk + 'radar_database/sentinel-1/dsc_t037'
    shapefile = data_disk + 'GIS/shapes/netherlands/zuid_holland.shp'
    orbit_folder = data_disk + 'orbits/sentinel-1'
    stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/dsc_t037_test'
    polarisation = 'VV'
    mode = 'IW'
    product_type = 'SLC'

# t 88
elif track_no == 88:
    start_date = '2015-07-01'
    end_date = '2018-09-12'
    master_date = '2017-02-21'

    database_folder = data_disk + 'radar_database/sentinel-1/asc_t088'
    shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
    orbit_folder = data_disk + 'orbits/sentinel-1'
    stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/asc_t088'
    polarisation = 'VV'
    mode = 'IW'
    product_type = 'SLC'

else:
    print('Non used track number')

n_jobs = 7
srtm_folder = data_disk + 'DEM/DEM_new'

# Download data and orbit
#download_data = DownloadSentinel(start_date, end_date, 'fjvanleijen', 'stevin01', shapefile, str(track_no), polarisation)
#download_data.sentinel_available()
#download_data.sentinel_download(destination_folder=os.path.dirname(database_folder))
#download_data.sentinel_check_validity(destination_folder=os.path.dirname(database_folder))

# Orbits
#precise_folder = os.path.join(orbit_folder, 'precise')
#download_orbit = DownloadSentinelOrbit(start_date, end_date, precise_folder)
#download_orbit.download_orbits()

# Number of cores
cores = 6

# Prepare processing
#self = SentinelStack(stack_folder)
#self.read_from_database(database_folder, shapefile, track_no, orbit_folder, start_date, end_date, master_date,
#                         mode, product_type, polarisation, cores=6)
# self = Stack(stack_folder)
# self.add_master_res_info()

# Read stack
self = Stack(stack_folder)
self.read_master_slice_list()
self.read_stack(start_date, end_date)

# create an SRTM DEM
password = 'Radar2016'
username = 'gertmulder'
self.create_SRTM_input_data(srtm_folder, username, password, srtm_type='SRTM1')

# Process stack
master_key = self.master_date
slave_keys = [key for key in self.images.keys() if key != master_key]
image_keys = self.images.keys()

# Check stack
for slave_key in slave_keys:
    self.images[slave_key].check_valid_burst_res()

# Run till the resample step.
for slave_key in slave_keys:




"""
ifgs = [self.interferograms[key] for key in self.interferograms.keys()]

import matplotlib.pyplot as plt
import numpy as np
import copy

for ifg in ifgs:
    if ifg.process_control['unwrap'] == '1':
        coh = copy.deepcopy(ifg.data_disk['coherence']['Data_ml_10_40'])
        pha = copy.deepcopy(ifg.data_disk['unwrap']['Data_ml_10_40'])

        pha[coh < 0.25] = np.nan

        plt.figure()
        plt.imshow(pha, cmap=plt.get_cmap('jet'))
        plt.colorbar()
"""
