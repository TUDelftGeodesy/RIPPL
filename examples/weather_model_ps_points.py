#!/usr/bin/python
# Create a datastack

from rippl.stack import Stack
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.coordinate_system import CoordinateSystem
import os
import numpy as np
from rippl.processing_steps.create_point_data import CreatePointData
from rippl.processing_steps.sparse_data import SparseData

data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'


polarisation = 'VV'
mode = 'IW'
product_type = 'SLC'
srtm_folder = data_disk + 'DEM/DEM_new'
orbit_folder = data_disk + 'orbits/sentinel-1'

track_no = 37
if track_no == 37:
    start_date = '2014-10-15'
    end_date = '2019-01-01'
    master_date = '2017-11-15'

    database_folder = data_disk + 'radar_database/sentinel-1/dsc_t037'
    shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
    stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/dsc_t037'

elif track_no == 88:
    start_date = '2014-02-12'
    end_date = '2019-01-01'
    master_date = '2017-02-21'

    database_folder = data_disk + 'radar_database/sentinel-1/asc_t088'
    shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
    orbit_folder = data_disk + 'orbits/sentinel-1'
    stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/asc_t088'


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
cores = 3

# Prepare processing
s1_stack = Stack(stack_folder)
s1_stack.read_master_slice_list()
s1_stack.read_stack(start_date, end_date)

# Create dummy point data

x = np.array([2000, 2000, 4000, 5000, 5460, 8880, 10000, 20000])
y = np.array([1000, 4000, 5000, 6000, 7500, 8980, 10000, 20000])

master_date = s1_stack.master_date
coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
coordinates.add_res_info(s1_stack.images[master_date].res_data)
coordinates.shape = (1, len(x))
coordinates.slice = False

point_data = CreatePointData(s1_stack.images[master_date].res_data, coordinates, point_data_name='ps_points_1',
                             points=np.concatenate((x[:, None], y[:, None]), 1))
point_data()
coordinates = point_data.coordinates

point_data.create_output_files(s1_stack.images[master_date].res_data, ['line', 'pixel'], coordinates)
point_data.save_to_disk(s1_stack.images[master_date].res_data, ['line', 'pixel'], coordinates)
point_data.clear_memory(s1_stack.images[master_date].res_data, ['line', 'pixel'], coordinates)
s1_stack.images[master_date].res_data.write()

coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1], sparse_name='ps_points_1')
coordinates.slice = False
coor_in = CoordinateSystem()
coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
coor_in.slice = False
meta = s1_stack.images[s1_stack.master_date].res_data

sparse_dem = SparseData(meta, meta, 'radar_DEM', 'DEM', coordinates, coor_in)
sparse_dem()
coordinates = sparse_dem.coor_out
sparse_dem.create_output_files(meta, 'radar_DEM', 'DEM', coordinates)
sparse_dem.save_to_disk(meta, 'radar_DEM', 'DEM', coordinates)
sparse_dem.clear_memory(meta, 'radar_DEM', 'DEM', coordinates)
meta.write()

# Now do the processing for the different steps for weather model processing.
ps_name = 'PS_points_1'
coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1], sparse_name=ps_name)
coordinates.shape = s1_stack.images[s1_stack.master_date].res_data.data_sizes['point_data']['line' + coordinates.sample]
coordinates.slice = False

settings = dict()
settings['radar_DEM'] = dict([('full', dict())])
settings['coor_DEM'] = dict([('full', dict())])
slice_names = s1_stack.images[s1_stack.master_date].slice_names
for slice_name in slice_names:
    settings['radar_DEM'][slice_name] = dict()
    settings['radar_DEM'][slice_name]['coor_in'] = s1_stack.images[s1_stack.master_date].slices[slice_name].read_res_coordinates('import_DEM')[0]
    settings['coor_DEM'][slice_name] = dict()
    settings['coor_DEM'][slice_name]['coor_in'] = s1_stack.images[s1_stack.master_date].slices[slice_name].read_res_coordinates('import_DEM')[0]
settings['radar_DEM']['full']['coor_in'] = s1_stack.images[s1_stack.master_date].res_data.read_res_coordinates('import_DEM')[0]
settings['coor_DEM']['full']['coor_in'] = s1_stack.images[s1_stack.master_date].res_data.read_res_coordinates('import_DEM')[0]

parallel = False
# Run the geocoding for the slices.
s1_stack('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel, cores=cores)
# Get the image orientation
s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)

# Download ECMWF data.
ecmwf_folder = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/weather_models/ecmwf_data'
processes = 8
s1_stack.download_ECMWF_data(dat_type='oper', ecmwf_data_folder=ecmwf_folder, processes=processes)
s1_stack.download_ECMWF_data(dat_type='era5', ecmwf_data_folder=ecmwf_folder, processes=processes)

# Finally process the different weather models
s1_stack('harmonie_aps', settings, coordinates, 'slave', file_type=['harmonie_aps'], parallel=parallel)
s1_stack('ecmwf_oper_aps', settings, coordinates, 'slave', file_type=['ecmwf_oper_aps'], parallel=parallel)
# s1_stack('ecmwf_era5_aps', settings, coordinates, 'slave', file_type=['ecmwf_era5_aps'], parallel=parallel)
