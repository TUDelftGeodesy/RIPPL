#!/usr/bin/python
# Create a datastack

from stack import Stack
from sentinel.sentinel_stack import SentinelStack
from coordinate_system import CoordinateSystem

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
self = Stack(stack_folder)
self.read_master_slice_list()
self.read_stack(start_date, end_date)
self.add_master_res_info()
self.create_network_ifgs(temp_baseline=30)

# create an SRTM DEM
password = 'Radar2016'
username = 'gertmulder'
self.create_SRTM_input_data(srtm_folder, username, password, srtm_type='SRTM1')

# Prepare settings by loading on of the settings for the DEM input
settings = dict()
settings['radar_DEM'] = dict([('full', dict())])
slice_names = self.images[self.master_date].slice_names
for slice_name in slice_names:
    settings['radar_DEM'][slice_name] = dict()
    settings['radar_DEM'][slice_name]['coor_in'] = self.images[self.master_date].slices[slice_name].read_res_coordinates('import_DEM')[0]
settings['radar_DEM']['full']['coor_in'] = self.images[self.master_date].res_data.read_res_coordinates('import_DEM')[0]

parallel = False

# Run the geocoding for the slices.
coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
coordinates.slice = True
self('radar_DEM', settings, coordinates, 'cmaster', file_type='radar_DEM', parallel=parallel)
# Run the geocoding for the slices.
self('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel)
# Get the image orientation
self('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)


coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
coordinates.slice = False
# Get the image orientation
self('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)
# Run till the earth_topo_phase step.
self(['earth_topo_phase', 'height_to_phase'], settings, coordinates, 'slave', file_type=[['earth_topo_phase'], ['height_to_phase']], parallel=parallel)

# Get the multilooked square amplitudes
for multilook in [[5, 20], [10, 40], [20, 80]]:
    coordinates = CoordinateSystem()
    coordinates.create_radar_coordinates(multilook=multilook, offset=[0, 0], oversample=[1, 1])
    coordinates.slice = False
    self('square_amplitude', settings, coordinates, 'slave', file_type='square_amplitude', parallel=parallel)
    # Get the harmonie (h38) and ECMWF (ERA5) data
    self('harmonie_aps', settings, coordinates, 'slave', file_type=['harmonie_h38_aps'])
    self('ecmwf_aps', settings, coordinates, 'slave', file_type=['ecmwf_ERA5_aps'])

    # Create ifgs / coherence for daisy chain.
    self('interferogram', settings, coordinates, 'ifg', file_type='interferogram', parallel=parallel)
    self('coherence', settings, coordinates, 'ifg', file_type='coherence', parallel=parallel)

    self('radar_DEM', settings, coordinates, 'cmaster', file_type='radar_DEM', parallel=parallel)
    # Run the geocoding for the slices.
    self('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel)
    self('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)


# Finally do the unwrapping (not implemented yet...)
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
