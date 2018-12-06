#!/usr/bin/python
# Create a datastack

from stack import Stack
from sentinel.sentinel_stack import SentinelStack
from coordinate_system import CoordinateSystem
import os

track_no = 88

data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'

# t 37
if track_no == 37:
    start_date = '2017-11-01'
    end_date = '2017-11-25'
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
    start_date = '2017-02-10'
    end_date = '2017-03-01'
    master_date = '2017-02-21'

    database_folder = data_disk + 'radar_database/sentinel-1/asc_t088'
    shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
    orbit_folder = data_disk + 'orbits/sentinel-1'
    stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/asc_t088_test'
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
cores = 8

# Prepare processing
if not os.listdir(stack_folder):
    s1_stack = SentinelStack(stack_folder)
    s1_stack.read_from_database(database_folder, shapefile, track_no, orbit_folder, start_date, end_date, master_date,
                             mode, product_type, polarisation, cores=6)
    s1_stack = Stack(stack_folder)
    s1_stack.read_master_slice_list()
    s1_stack.read_stack(start_date, end_date)
    s1_stack.add_master_res_info()
    s1_stack.create_network_ifgs(temp_baseline=30)
else:
    s1_stack = Stack(stack_folder)
    s1_stack.read_master_slice_list()
    s1_stack.read_stack(start_date, end_date)

# create an SRTM DEM
password = 'Radar2016'
username = 'gertmulder'
s1_stack.create_SRTM_input_data(srtm_folder, username, password, srtm_type='SRTM3')

# Prepare settings by loading on of the settings for the DEM input
settings = dict()
settings['radar_DEM'] = dict([('full', dict())])
slice_names = s1_stack.images[s1_stack.master_date].slice_names
for slice_name in slice_names:
    settings['radar_DEM'][slice_name] = dict()
    settings['radar_DEM'][slice_name]['coor_in'] = s1_stack.images[s1_stack.master_date].slices[slice_name].read_res_coordinates('import_DEM')[0]
settings['radar_DEM']['full']['coor_in'] = s1_stack.images[s1_stack.master_date].res_data.read_res_coordinates('import_DEM')[0]

parallel = True
# Run the geocoding for the slices.
coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
coordinates.slice = True
s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='radar_DEM', parallel=parallel)
# Run the geocoding for the slices.
s1_stack('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel)
# Get the image orientation
s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)

coordinates.slice = False
# Full radar DEM grid
s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='radar_DEM', parallel=parallel)
# Get the image orientation
s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)
# Run till the earth_topo_phase step.
coordinates.slice = True
s1_stack(['earth_topo_phase'], settings, coordinates, 'slave', file_type=['earth_topo_phase'], parallel=parallel)

# Get the multilooked square amplitudes
for multilook in [[5, 20], [10, 40], [20, 80]]:
    coordinates = CoordinateSystem()
    coordinates.create_radar_coordinates(multilook=multilook, offset=[0, 0], oversample=[1, 1])
    coordinates.slice = False
    s1_stack('square_amplitude', settings, coordinates, 'slave', file_type='square_amplitude', parallel=parallel)
    # Create the DEM first
    s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='radar_DEM', parallel=parallel)

    # Create ifgs / coherence for daisy chain.
    s1_stack('interferogram', settings, coordinates, 'ifg', file_type='interferogram', parallel=parallel)
    s1_stack('coherence', settings, coordinates, 'ifg', file_type='coherence', parallel=parallel)

    # Run the geocoding for the slices.
    s1_stack('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel)
    s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)

    # Get the harmonie (h38) and ECMWF (ERA5) data
    s1_stack('harmonie_aps', settings, coordinates, 'slave', file_type=['harmonie_h38_aps'], parallel=parallel)
    #s1_stack('ecmwf_aps', settings, coordinates, 'slave', file_type=['ecmwf_ERA5_aps'], parallel=parallel)


# Get the multilooked square amplitudes
# Finally do the unwrapping (not implemented yet...)
for multilook in [[5, 20], [10, 40], [20, 80]]:
    coordinates = CoordinateSystem()
    coordinates.create_radar_coordinates(multilook=multilook, offset=[0, 0], oversample=[1, 1])
    coordinates.slice = False

    s1_stack('unwrap', settings, coordinates, 'ifg', file_type='unwrap', parallel=parallel)
