#!/usr/bin/python
# Create a datastack

from rippl.meta_data.stack import Stack
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
import os

track_no = 37

data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'

# t 37
if track_no == 37:
    start_date = '2017-11-08'
    end_date = '2017-11-22'
    master_date = '2017-11-15'

    database_folder = data_disk + 'radar_database/sentinel-1/dsc_t037'
    shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
    orbit_folder = data_disk + 'orbits/sentinel-1'
    stack_folder = datastack_disk + 'radar_datastacks/sentinel-1/dsc_t037_test'
    polarisation = 'VH'
    mode = 'IW'
    product_type = 'SLC'

# t 88
elif track_no == 88:
    start_date = '2015-02-01'
    end_date = '2019-02-28'
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
# precise_folder = os.path.join(orbit_folder, 'precise')
# download_orbit = DownloadSentinelOrbit(start_date, end_date, precise_folder)
# download_orbit.download_orbits()

# Number of cores
cores = 6

# Prepare processing
if not os.listdir(stack_folder):
    s1_stack = SentinelStack(stack_folder)
    s1_stack.read_from_database(database_folder, shapefile, track_no, orbit_folder, start_date, end_date, master_date,
                             mode, product_type, polarisation, cores=6)
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
single_images = False

# Run the geocoding for the slices.
coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
coordinates.slice = True
s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='DEM', parallel=parallel)
# Run the geocoding for the slices.
s1_stack('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel)
# Get the image orientation
s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)

coordinates.slice = False
# Full radar DEM grid
s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='DEM', parallel=parallel)
# Get the image orientation
s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)

coordinates = CoordinateSystem()
coordinates.create_projection(1, 1, projection_type='rainfall_NL', proj4_str="+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0")
coordinates.slice = False
# Full radar DEM grid

s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='DEM', parallel=parallel)
# Get the image orientation
s1_stack('projection_coor', settings, coordinates, 'cmaster', file_type=['lat', 'lon'], parallel=parallel)
s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)
# Do the geocoding
s1_stack('coor_geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z'], parallel=parallel)







# Run till the earth_topo_phase step.
coordinates.slice = True
s1_stack(['earth_topo_phase'], settings, coordinates, 'slave', file_type=['earth_topo_phase'], parallel=parallel)

coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[2, 6], offset=[0, 0], oversample=[1, 1])
coordinates.slice = False

s1_stack('interferogram', settings, coordinates, 'ifg', file_type='interferogram', parallel=parallel)
s1_stack('square_amplitude', settings, coordinates, 'slave', file_type='square_amplitude', parallel=parallel)
s1_stack('coherence', settings, coordinates, 'ifg', file_type='coherence', parallel=parallel)

# Get the multilooked square amplitudes
for multilook in [[20, 80]]:

    #
    # Create the DEM first
    # s1_stack('radar_DEM', settings, coordinates, 'cmaster', file_type='DEM', parallel=parallel)

    # Create ifgs / coherence for daisy chain.
    #
    #

    # Run the geocoding for the slices.
    # s1_stack('geocode', settings, coordinates, 'cmaster', file_type=['X', 'Y', 'Z', 'lat', 'lon'], parallel=parallel)
    s1_stack('azimuth_elevation_angle', settings, coordinates, 'cmaster', file_type=['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], parallel=parallel)
    s1_stack('height_to_phase', settings, coordinates, 'slave', file_type=['height_to_phase'], parallel=parallel)

    # Get the harmonie and ECMWF (ERA5) data
    s1_stack('harmonie_aps', settings, coordinates, 'slave', file_type=['harmonie_aps'], parallel=parallel)
    # s1_stack('harmonie_interferogram', settings, coordinates, 'ifg', file_type=['harmonie_interferogram'], parallel=parallel)
    # s1_stack('ecmwf_era5_aps', settings, coordinates, 'slave', file_type=['ecmwf_era5_aps'], parallel=parallel)
    # s1_stack('ecmwf_oper_aps', settings, coordinates, 'slave', file_type=['ecmwf_oper_aps'], parallel=parallel)
    # s1_stack('ecmwf_interim_aps', settings, coordinates, 'slave', file_type=['ecmwf_interim_aps'], parallel=parallel)


# Get the multilooked square amplitudes
# Finally do the unwrapping (not implemented yet...)
for multilook in [[20, 80]]:
    coordinates = CoordinateSystem()
    coordinates.create_radar_coordinates(multilook=multilook, offset=[0, 0], oversample=[1, 1])
    coordinates.slice = False

    s1_stack('unwrap', settings, coordinates, 'ifg', file_type='unwrap', parallel=parallel)

# Export outputs as geotiff
s1_stack.export_to_geotiff(['interferogram', 'unwrap', 'harmonie_interferogram', 'coherence'],
                           ['interferogram_ml_10_40', 'unwrap_ml_10_40', 'harmonie_interferogram_ml_10_40', 'coherence_ml_10_40'])
s1_stack.export_to_geotiff(['interferogram', 'unwrap', 'harmonie_interferogram', 'coherence'],
                           ['interferogram_ml_20_80', 'unwrap_ml_20_80', 'harmonie_interferogram_ml_20_80', 'coherence_ml_20_80'])
s1_stack.export_to_geotiff(['interferogram'], ['interferogram_ml_20_80'])

s1_stack.export_to_geotiff(['unwrap'], ['unwrap_ml_10_40'])
s1_stack.export_to_geotiff(['unwrap'], ['unwrap_ml_20_80'])

s1_stack.export_to_geotiff('harmonie_interferogram', 'harmonie_interferogram_ml_10_40')
s1_stack.export_to_geotiff('harmonie_interferogram', 'harmonie_interferogram_ml_20_80')

# Plotting of stack
import matplotlib.pyplot as plt
import copy
import numpy as np

for ifg_key in s1_stack.interferograms.keys():

    if np.abs(int(ifg_key[:8]) - int(ifg_key[9:])) > 20:
        continue

    coherence = copy.copy(s1_stack.interferograms[ifg_key].res_data.data_disk['coherence']['coherence_ml_20_80'])
    harmonie = copy.copy(- s1_stack.interferograms[ifg_key].res_data.data_disk['harmonie_interferogram']['harmonie_interferogram_ml_20_80'])
    unwrap = copy.copy(s1_stack.interferograms[ifg_key].res_data.data_disk['unwrap']['unwrap_ml_20_80'])
    wrapped = copy.copy(s1_stack.interferograms[ifg_key].res_data.data_disk['interferogram']['interferogram_ml_20_80'])

    if np.abs(np.mean(harmonie)) > 1:
        continue
    if np.mean(harmonie) == 0:
        continue

    unwrap *= (0.014 / np.pi)
    wrapped = np.angle(wrapped) * (0.014 / np.pi)

    unwrap[coherence < 0.05] = 0
    wrapped[coherence < 0.05] = 0
    harmonie[coherence < 0.05] = 0

    im_diff = np.mean(unwrap[coherence > 0.15]) - np.mean(harmonie[coherence > 0.15])

    limits = [np.min(harmonie[coherence > 0.15]), np.max(harmonie[coherence > 0.15])]

    a = plt.figure(figsize=(10,12))
    a.tight_layout()
    #st = a.suptitle('Harmonie ifg comparison ' + ifg_key)

    plt.subplot(2,2,1)
    plt.imshow(harmonie, vmin=limits[0], vmax=limits[1])
    plt.colorbar()
    plt.title('Harmonie ifg')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    plt.subplot(2,2,2)
    plt.imshow(unwrap - im_diff, vmin=limits[0], vmax=limits[1])
    plt.colorbar()
    plt.title('Unwrapped ifg')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    plt.subplot(2,2,3)
    plt.imshow(coherence)
    plt.colorbar()
    plt.title('Coherence')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    plt.subplot(2,2,4)
    plt.imshow(wrapped)
    plt.colorbar()
    plt.title('Wrapped ifg')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    plt.savefig(os.path.join(stack_folder, 'harmonie_comparison_' + ifg_key + '.png'))
    print('Saved ' + os.path.join(stack_folder, 'harmonie_comparison_' + ifg_key + '.png'))
    plt.close()
