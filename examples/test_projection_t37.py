#!/usr/bin/python
# Create a datastack

from rippl.stack import Stack
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.coordinate_system import CoordinateSystem
import os
import numpy as np

data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'

start_date = '2014-10-15'
end_date = '2018-01-22'
master_date = '2017-11-15'

database_folder = data_disk + 'radar_database/sentinel-1/dsc_t037'
shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/dsc_t037'
polarisation = 'VV'
mode = 'IW'
product_type = 'SLC'
track_no = 37

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
cores = 3

# Prepare processing
if not os.listdir(stack_folder):
    s1_stack = SentinelStack(stack_folder)
    s1_stack.read_from_database(database_folder, shapefile, track_no, orbit_folder, start_date, end_date, master_date,
                             mode, product_type, polarisation, cores=cores)
    s1_stack = Stack(stack_folder)
    s1_stack.read_master_slice_list()
    s1_stack.read_stack(start_date, end_date)
    s1_stack.add_master_res_info()
    s1_stack.create_network_ifgs(temp_baseline=60)
else:
    s1_stack = Stack(stack_folder)
    s1_stack.read_master_slice_list()
    s1_stack.read_stack(start_date, end_date)

# create an SRTM DEM
password = 'Radar2016'
username = 'gertmulder'
s1_stack.create_SRTM_input_data(srtm_folder, username, password, srtm_type='SRTM3', buf=3, parallel=True)

# Prepare settings by loading on of the settings for the DEM input
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

parallel = True
single_images = False

# Run the geocoding for the slices.
coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
coordinates.slice = True
# s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='DEM', parallel=parallel, cores=cores)
# Run the geocoding for the slices.
# s1_stack('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel, cores=cores)
# Get the image orientation
# s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)
# Run processing
# s1_stack(['earth_topo_phase'], settings, coordinates, 'slave', file_type=['earth_topo_phase'], parallel=parallel, cores=cores)

coordinates.slice = False
# Full radar DEM grid
# s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='DEM', parallel=parallel, cores=cores)
# Get the image orientation
# s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel, cores=cores)
# Get full image 
# s1_stack('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel, cores=cores)

# Create DEM for harmonie aps
dem_coordinates = CoordinateSystem()
dem_coordinates.create_radar_coordinates(multilook=[50, 200], offset=[-100, -400])
# s1_stack('radar_DEM', settings, dem_coordinates, 'cmaster', file_type='DEM', parallel=parallel, cores=cores)
# s1_stack('geocode', settings, dem_coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel, cores=cores)
# s1_stack('azimuth_elevation_angle', settings, dem_coordinates, 'cmaster',
#         file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel, cores=cores)

for resolution in [0.5]: #
    # Now we create the projected grid. The coordinates are chosen similar to the rainfall radar product from the KNMI
    lat_offset = [200, 115]
    lon_offset = [200, 50]
    shape = np.array([765 - np.sum(lat_offset), 700 - np.sum(lon_offset)]) * np.int(1 / resolution)
    x0 = 0 + resolution / 2 + lon_offset[0]
    y0 =-3650 - resolution / 2 - lat_offset[0]

    coordinates = CoordinateSystem()
    coordinates.create_projection(resolution, -resolution, projection_type='rainfall_NL',
                                  proj4_str="+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0",
                                  shape=shape, x0=x0, y0=y0)
    coordinates.slice = False

    # Get the image orientation
    s1_stack('projection_coor', settings, coordinates, 'cmaster', file_type=['lat', 'lon'], parallel=parallel, cores=cores)
    # Full radar DEM grid
    s1_stack('coor_DEM', settings, coordinates, 'cmaster', file_type='DEM', parallel=parallel, cores=cores)
    # Do the geocoding
    s1_stack('coor_geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'line', 'pixel'], parallel=parallel, cores=cores)
    # azimuth elevation angle
    s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel, cores=cores)

    # After geocoding of the projected image we run the weather model for different types
    # s1_stack('harmonie_aps', settings, coordinates, 'slave', file_type=['harmonie_aps'], parallel=parallel)
    # s1_stack('ecmwf_era5_aps', settings, coordinates, 'slave', file_type=['ecmwf_era5_aps'], parallel=parallel)
    # s1_stack('ecmwf_oper_aps', settings, coordinates, 'slave', file_type=['ecmwf_oper_aps'], parallel=parallel)

    # Create the conversion grids (This should work automatically in the future, but we do it like this for now.)
    coordinates.slice = True
    s1_stack('conversion_grid', settings, coordinates, 'cmaster', file_type=['sum_ids', 'sort_ids', 'looks', 'output_ids'], parallel=parallel, cores=cores)

    # And create the interferograms for these images.
    coordinates.slice = False
    # s1_stack('harmonie_interferogram', settings, coordinates, 'ifg', file_type=['harmonie_interferogram'], parallel=parallel)

    # Then do the multilooking for the ifg/amplitude/coherence images
    # s1_stack('interferogram', settings, coordinates, 'ifg', file_type='interferogram', parallel=parallel, cores=cores)
    s1_stack('square_amplitude', settings, coordinates, 'slave', file_type='square_amplitude', parallel=parallel, cores=cores)
    s1_stack('coherence', settings, coordinates, 'ifg', file_type='coherence', parallel=parallel, cores=cores)

for resolution in [0.5]:  #
    # Now we create the projected grid. The coordinates are chosen similar to the rainfall radar product from the KNMI

    lat_offset = [200, 115]
    lon_offset = [200, 50]
    shape = np.array([765 - np.sum(lat_offset), 700 - np.sum(lon_offset)]) * np.int(1 / resolution)
    x0 = 0 + resolution / 2 + lon_offset[0]
    y0 =-3650 - resolution / 2 - lat_offset[0]

    coordinates = CoordinateSystem()
    coordinates.create_projection(resolution, -resolution, projection_type='rainfall_NL',
                                  proj4_str="+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0",
                                  shape=shape, x0=x0, y0=y0)
    coordinates.slice = False
    # Finally do the unwrapping.
    s1_stack('unwrap', settings, coordinates, 'ifg', file_type='unwrap', parallel=parallel, cores=cores)

exp_strs = ['_rainfall_NL_stp_1_-1', '_rainfall_NL_stp_0.5_-0.5', '_rainfall_NL_stp_0.25_-0.25']

# Export outputs as geotiff
for exp_str in exp_strs:
    s1_stack.export_to_geotiff(['square_amplitude'], ['square_amplitude' + exp_str], interferogram=False)
    s1_stack.export_to_geotiff(['coherence'], ['coherence' + exp_str])
    s1_stack.export_to_geotiff(['interferogram'], ['interferogram'  + exp_str])

    s1_stack.export_to_geotiff(['unwrap'], ['unwrap' + exp_str])
    s1_stack.export_to_geotiff('harmonie_interferogram', 'harmonie_interferogram' + exp_str)
