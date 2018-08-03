#!/usr/bin/python
# Create a datastack

from stack import Stack
from sentinel.sentinel_stack import SentinelStack
from joblib import Parallel, delayed
import multiprocessing.dummy as mp
from parallel_functions import unwrap
from sentinel.sentinel_download import DownloadSentinel, DownloadSentinelOrbit
import os
from functools import partial

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
srtm_folder = data_disk + 'DEM/dem_new'

# Download data and orbit
#download_data = DownloadSentinel(start_date, end_date, 'fjvanleijen', 'stevin01', shapefile, str(track_no), polarisation)
#download_data.sentinel_available()
#download_data.sentinel_download(destination_folder=os.path.dirname(database_folder))
#download_data.sentinel_check_validity(destination_folder=os.path.dirname(database_folder))

# Orbits
#precise_folder = os.path.join(orbit_folder, 'precise')
#download_orbit = DownloadSentinelOrbit(start_date, end_date, precise_folder)
#download_orbit.download_orbits()

# Prepare processing
self = SentinelStack(stack_folder)
self.read_from_database(database_folder, shapefile, track_no, orbit_folder, start_date, end_date, master_date,
                          mode, product_type, polarisation, cores=7)

# Read stack
self = Stack(stack_folder)
self.read_master_slice_list()
self.read_stack(start_date, end_date)

# Process stack
master_key = self.master_slice_date[0][:4] + self.master_slice_date[0][5:7] + self.master_slice_date[0][8:10]
slave_keys = [key for key in self.images.keys() if key != master_key]
image_keys = self.images.keys()

# Check stack
for slave_key in slave_keys:
    self.images[slave_key].check_valid_burst_res()

# Geocode
self.images[master_key].geocoding(n_jobs=n_jobs, dem_folder=srtm_folder, parallel=parallel)

# Geocoding on sparse grid for full image.
self.images[master_key].sparse_geocoding(n_jobs=n_jobs, dem_folder=srtm_folder, parallel=parallel,
                                         multilook_coarse=[64, 256], multilook_fine=[16, 64])

# Resample
for slave_key in slave_keys:
    self.images[slave_key].resample(n_jobs=n_jobs, master=self.images[master_key], parallel=parallel)

# Create ifg with different multilooking factors
ml_s = [[16, 64], [32, 128], [64, 256]]
ovr_s = [[1, 1], [2, 2], [4, 4]]
# The offsets will be the same everywhere, as we assume that the non-usefull areas

for image_1 in image_keys:
    for image_2 in image_keys:
        if int(image_1) > int(image_2):

            month_diff = float(int(image_1[:4]) * 12 - int(image_2[:4]) * 12 + int(image_1[4:6]) - int(image_2[4:6])) + \
                         float(image_1[6:]) / 30.0 - float(image_2[6:]) / 30.0

            if int(image_1) > int(image_2) and month_diff < 12:  # All ifgs within 12 months are created.
                ifg = self.images[image_1].interferogram(slave=self.images[image_2], multilook=ml_s, offset_burst=[10, 50],
                                                         oversampling=ovr_s, n_jobs=n_jobs, parallel=parallel)
                if ifg:
                    self.interferograms[str(image_1) + '_' + str(image_2)] = ifg


# Unwrap
keys = self.interferograms.keys()
meta_data = [self.interferograms[key] for key in keys]

for ml, ovr in zip(ml_s, ovr_s):
    if parallel:
        pool = mp.Pool(n_jobs)
        ifgs = pool.map(partial(unwrap, multilook=ml, oversample=ovr), meta_data)

        #ifgs = Parallel(n_jobs=n_jobs)(delayed(unwrap)(ifg, multilook=multilook)
        #                               for ifg in meta_data)
    else:
        for ifg in meta_data:
            unwrap(ifg, multilook=ml, oversample=ovr)
        ifgs = meta_data

    for ifg in ifgs:
        ifg.write()
        ifg.read_data()


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
