#!/usr/bin/python
# Create a datastack

from stack import Stack
from sentinel.sentinel_stack import SentinelStack
import os
from sentinel.sentinel_download import DownloadSentinelOrbit
from coordinate_system import CoordinateSystem

track_no = 37
parallel = False

data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'

# t 37
start_date = '2016-01-01'
end_date = '2018-10-30'
master_date = '2017-11-09'

database_folder = data_disk + 'radar_database/sentinel-1/dsc_t037'
shapefile = data_disk + 'GIS/shapes/groningen/nam_stochastics_project/nam_stochastics_project.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/dsc_t037_depsi_groningen'
polarisation = 'VV'
mode = 'IW'
product_type = 'SLC'

n_jobs = 7
srtm_folder = data_disk + 'DEM/DEM_new'

# Download data and orbit
#download_data = DownloadSentinel(start_date, end_date, 'fjvanleijen', 'stevin01', shapefile, str(track_no), polarisation)
#download_data.sentinel_available()
#download_data.sentinel_download(destination_folder=os.path.dirname(database_folder))
#download_data.sentinel_check_validity(destination_folder=os.path.dirname(database_folder))

# Orbits
# precise_folder = os.path.join(orbit_folder, 'precise')
# download_orbit = DownloadSentinelOrbit(start_date, end_date, precise_folder)
# download_orbit.download_orbits()

# Number of cores
cores = 6

# Prepare processing
self = SentinelStack(stack_folder)
self.read_from_database(database_folder, shapefile, track_no, orbit_folder, start_date, end_date, master_date,
                         mode, product_type, polarisation, cores=6)

self = Stack(stack_folder)
self.read_master_slice_list()
self.read_stack(start_date, end_date)
self.add_master_res_info()
self.create_network_ifgs(network_type='single_master')

# create an SRTM DEM
#password = 'Radar2016'
#username = 'gertmulder'
#self.create_SRTM_input_data(srtm_folder, username, password, srtm_type='SRTM3')

# Prepare settings by loading on of the settings for the DEM input
settings = dict()
settings['radar_DEM'] = dict([('full', dict())])
slice_names = self.images[self.master_date].slice_names
for slice_name in slice_names:
    settings['radar_DEM'][slice_name] = dict()
    settings['radar_DEM'][slice_name]['coor_in'] = self.images[self.master_date].slices[slice_name].read_res_coordinates('import_DEM')[0]
settings['radar_DEM']['full']['coor_in'] = self.images[self.master_date].res_data.read_res_coordinates('import_DEM')[0]

parallel = True

# Run the geocoding for the slices.
#coordinates = CoordinateSystem()
#coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
#coordinates.slice = True
#self('radar_DEM', settings, coordinates, 'cmaster', file_type='radar_DEM', parallel=parallel)
# Run the geocoding for the slices.
#self('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel)
# Get the image orientation
#self('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)

# Run azimuth elevation angles for the full image
coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
coordinates.slice = False
# Run till the earth_topo_phase step.
self(['earth_topo_phase', 'height_to_phase'], settings, coordinates, 'slave', file_type=[['earth_topo_phase'], ['height_to_phase']], parallel=parallel)
# Create amplitude images
self(['amplitude'], settings, coordinates, 'slave', file_type=['amplitude'], parallel=parallel)
# Get the image orientation
#self('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)

# Create ifgs for single master.
self('interferogram', settings, coordinates, 'ifg', file_type='interferogram', parallel=parallel)
