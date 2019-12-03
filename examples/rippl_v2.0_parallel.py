
from rippl.meta_data.stack import Stack
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinelOrbit, DownloadSentinel
import os
import numpy as np

data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'

start_date = '2015-02-07'
end_date = '2016-02-25'
master_date = '2017-02-21'

database_folder = data_disk + 'radar_database/sentinel-1/'
shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = '/home/gert/fast_datastacks/Netherlands/s1_t88'
polarisation = 'VV'
mode = 'IW'
product_type = 'SLC'

n_jobs = 7
srtm_folder = data_disk + 'DEM/SRTM'

download = False
track = 37

if download:
    # Download data and orbit
    download_data = DownloadSentinel(start_date, end_date, shapefile, track, polarisation=polarisation)
    download_data.sentinel_available('fjvanleijen', 'stevin01')
    download_data.sentinel_download_ASF(database_folder, 'gertmulder', 'Radar2016')
    download_data.sentinel_check_validity(database_folder, 'fjvanleijen', 'stevin01')

    # Orbits
    precise_folder = os.path.join(orbit_folder, 'precise')
    download_orbit = DownloadSentinelOrbit(start_date, end_date, precise_folder)
    download_orbit.download_orbits()

# Number of cores
cores = 6

# Prepare processing
if not os.listdir(stack_folder):
    s1_stack = SentinelStack(stack_folder)
    s1_stack.read_from_database(database_folder, shapefile, 88, orbit_folder, start_date, end_date, master_date,
                         mode, product_type, polarisation, cores=cores)
s1_stack = Stack(stack_folder)
s1_stack.read_master_slice_list()
s1_stack.read_stack(start_date, end_date)
s1_stack.create_interferogram_network(network_type='daisy_chain', temporal_no=5)
password = 'Radar2016'
username = 'gertmulder'
s1_stack.download_SRTM_dem(srtm_folder, username, password, buffer=2, rounding=1, srtm_type='SRTM1')
s1_stack.download_SRTM_dem(srtm_folder, username, password, buffer=2, rounding=1, srtm_type='SRTM3')

from rippl.processing_steps.import_dem import ImportDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.resample_dem import ResampleDem
from rippl.processing_steps.geocode import Geocode
from rippl.processing_steps.geometric_coregistration import GeometricCoregistration
from rippl.processing_steps.earth_topo_phase import EarthTopoPhase
from rippl.processing_steps.square_amplitude import SquareAmplitude
from rippl.processing_steps.reramp import Reramp
from rippl.processing_steps.radar_ray_angles import RadarRayAngles
from rippl.processing_steps.baseline import Baseline
from rippl.processing_steps.coherence import Coherence
from rippl.processing_steps.deramp_resample_radar_grid import DerampResampleRadarGrid
from rippl.processing_steps.interferogram_multilook import InterferogramMultilook
from rippl.processing_steps.multilook_prepare import MultilookPrepare
from rippl.processing_steps.resample_prepare import ResamplePrepare
from rippl.processing_steps.unwrap import Unwrap

from rippl.processing_steps.multilook import Multilook
from rippl.pipeline import Pipeline

master_date = s1_stack.master_date
slice_names = np.sort(s1_stack.slcs[master_date].slice_names)

##################################################################################################################
# DEM creation and geocoding
# Import the DEM and do the inverse geocode
#
# Create coordinate systems.
radar_coor = CoordinateSystem()
radar_coor.create_radar_coordinates()
dem_coor = ImportDem.create_dem_coor('SRTM3')

# Create the first multiprocessing pipeline.
coreg_slices = [s1_stack.slcs[master_date].slice_data[slice] for slice in slice_names]
dem_pipeline = Pipeline(pixel_no=5000000, processes=8)
dem_pipeline.add_processing_data(coreg_slices, 'coreg_master')
dem_pipeline.add_processing_step(ImportDem(in_coor=radar_coor, coreg_master='coreg_master', dem_type='SRTM3', dem_folder=srtm_folder, buffer=1, rounding=1), True)
dem_pipeline.add_processing_step(InverseGeocode(in_coor=radar_coor, out_coor=dem_coor, coreg_master='coreg_master', dem_type='SRTM3'), True)
dem_pipeline()
dem_pipeline.save_processing_results()

# Then create the radar DEM, geocoding, incidence angles for the master grid
s1_stack.read_stack(start_date, end_date)
coreg_slices = [s1_stack.slcs[master_date].slice_data[slice] for slice in slice_names]
geocode_pipeline = Pipeline(pixel_no=3000000, processes=6)
geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
geocode_pipeline.add_processing_step(ResampleDem(out_coor=radar_coor, in_coor=dem_coor, coreg_master='coreg_master'), True)
geocode_pipeline.add_processing_step(Geocode(out_coor=radar_coor, coreg_master='coreg_master'), True)
geocode_pipeline.add_processing_step(RadarRayAngles(out_coor=radar_coor, coreg_master='coreg_master'), True)
geocode_pipeline.add_processing_step(SquareAmplitude(out_coor=radar_coor, slave='coreg_master', master_image=True, polarisation='VV'), True)
geocode_pipeline()
geocode_pipeline.save_processing_results()

#########################################################################################################
# Resampling of data.

# Then do the resampling for the slave images
# To show the ability of the parallel processing we add all the images at once. In this way we will not lose any of the
# possible multicore availability.
s1_stack.read_stack(start_date, end_date)

slave_keys = [key for key in s1_stack.slcs.keys() if key != master_date]
slave_slices = []
coreg_slices = []
for slave_key in slave_keys:
    for slice_name in slice_names:
        slave_slices.append(s1_stack.slcs[slave_key].slice_data[slice_name])
        coreg_slices.append(s1_stack.slcs[master_date].slice_data[slice_name])

resampling_pipeline = Pipeline(pixel_no=5000000, processes=6)
resampling_pipeline.add_processing_data(coreg_slices, 'coreg_master')
resampling_pipeline.add_processing_data(slave_slices, 'slave')
resampling_pipeline.add_processing_step(GeometricCoregistration(out_coor=radar_coor, in_coor=radar_coor, coreg_master='coreg_master', slave='slave'), False)
resampling_pipeline.add_processing_step(DerampResampleRadarGrid(out_coor=radar_coor, in_coor=radar_coor, polarisation='VV', slave='slave'), False)
resampling_pipeline.add_processing_step(Reramp(out_coor=radar_coor, polarisation='VV', slave='slave'), False)
resampling_pipeline.add_processing_step(EarthTopoPhase(out_coor=radar_coor, polarisation='VV', slave='slave'), True)
resampling_pipeline.add_processing_step(SquareAmplitude(out_coor=radar_coor, polarisation='VV', slave='slave'), True)
resampling_pipeline()
resampling_pipeline.save_processing_results()

##########################################################################################################
# Concatenation of stack images.
# Concatenate slave images. This is done using one core processing only because it mainly involves reading and writing
# to and from disk, which will not speed up.
slave_keys = [key for key in s1_stack.slcs.keys() if key != master_date]
for slave_key in slave_keys:
    slave = s1_stack.slcs[slave_key]
    slave.create_concatenate_image(process='square_amplitude', file_type='square_amplitude', coor=radar_coor, transition_type='cut_off')
    slave.create_concatenate_image(process='earth_topo_phase', file_type='earth_topo_phase_corrected', coor=radar_coor, transition_type='cut_off')

# Concatenate to full images
coreg_image = s1_stack.slcs[master_date]
dem_coor = ImportDem.create_dem_coor('SRTM3')

# These coordinate systems are needed for concatenation
coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=dem_coor, replace=True)
coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='geocode', file_type='lat', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='geocode', file_type='lon', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='crop', file_type='crop', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='square_amplitude', file_type='square_amplitude', coor=radar_coor, transition_type='cut_off')

###########################################################################################################
#
# Multilooking and creation of interferograms
# Define the coordinate system we want to create the new grid in.
s1_stack.read_stack(start_date, end_date)
ml_coor = CoordinateSystem()
ml_coor.create_geographic(0.005, 0.005, lat0=50, lon0=0)

# Multilook to regular lat/lon grids.
coreg_image = s1_stack.slcs[master_date]
slave_keys = [key for key in s1_stack.slcs.keys()]
slaves = [s1_stack.slcs[slave_key].data for slave_key in slave_keys]
coreg_images = [coreg_image.data for slave_key in slave_keys]

create_multilooking_grid = Pipeline(pixel_no=5000000, processes=6)
create_multilooking_grid.add_processing_data(coreg_images, 'coreg_master')
create_multilooking_grid.add_processing_data(slaves, 'slave')
create_multilooking_grid.add_processing_step(MultilookPrepare(in_polarisation='VV', in_coor=radar_coor, out_coor=ml_coor,
                                                              in_file_type='square_amplitude', in_process='square_amplitude',
                                                              slave='coreg_master', coreg_master='coreg_master'), True)
create_multilooking_grid()
create_multilooking_grid.save_processing_results()

# If the image to be multilooked is large, the only way to do the processing
s1_stack.read_stack(start_date, end_date)
coreg_image = s1_stack.slcs[master_date]
slaves = [s1_stack.slcs[slave_key].data for slave_key in slave_keys]
for slave in slaves:
    multilook = Multilook(in_polarisation='VV', in_coor=radar_coor, out_coor=ml_coor,
                          in_file_type='square_amplitude', in_process='square_amplitude',
                          slave=slave, coreg_master=coreg_image.data, batch_size=5000000)
    multilook()

ifg_keys = list(s1_stack.ifgs.keys())
masters = [s1_stack.slcs[ifg_key[:8]].data for ifg_key in ifg_keys]
slaves = [s1_stack.slcs[ifg_key[9:]].data for ifg_key in ifg_keys]
ifgs = [s1_stack.ifgs[ifg_key].data for ifg_key in ifg_keys]

for slave, master, ifg in zip(slaves, masters, ifgs):
    ifg_multilook = InterferogramMultilook(polarisation='VV', in_coor=radar_coor, out_coor=ml_coor,
                           slave=slave, coreg_master=coreg_image.data, ifg=ifg, master=master)
    ifg_multilook()

import matplotlib.pyplot as plt

for ifg in ifgs:
    data = ifg.data_disk['interferogram']['interferogram_#coor#_geo_WGS84_18_18_#in_coor#_radar_#pol#_VV']['interferogram']['data']
    plt.figure()
    plt.imshow(np.angle(data))
    plt.show()

# Create coherence.
coreg_images = [coreg_image.data for ifg_key in ifg_keys]
create_multilooking_grid = Pipeline(pixel_no=5000000, processes=6)
create_multilooking_grid.add_processing_data(coreg_images, 'coreg_master')
create_multilooking_grid.add_processing_data(slaves, 'slave')
create_multilooking_grid.add_processing_data(masters, 'master')
create_multilooking_grid.add_processing_data(ifgs, 'ifg')
create_multilooking_grid.add_processing_step(Coherence(polarisation='VV', out_coor=ml_coor, ifg='ifg', master='master', slave='slave'), True)
create_multilooking_grid()
create_multilooking_grid.save_processing_results()

# Unwrapping of multilooked ifgs.
s1_stack.read_stack(start_date, end_date)
ifgs = [s1_stack.ifgs[ifg_key].data for ifg_key in ifg_keys]
create_multilooking_grid = Pipeline(pixel_no=0, processes=1)
create_multilooking_grid.add_processing_data(ifgs, 'ifg')
create_multilooking_grid.add_processing_step(Unwrap(polarisation='VV', out_coor=ml_coor, ifg='ifg'), True)
create_multilooking_grid()
create_multilooking_grid.save_processing_results()

s1_stack.read_stack(start_date, end_date)
ifgs = [s1_stack.ifgs[ifg_key].data for ifg_key in ifg_keys]
for ifg in ifgs:
    data = ifg.data_disk['unwrap']['unwrap_#coor#_geo_WGS84_18_18_#pol#_VV']['unwrapped']['data']
    plt.figure()
    plt.imshow(data)
    plt.show()


#########################################################################################################
# Creation of lat/lon grids and ray tracing of weather models.
#
# Create coordinate systems for these lat/lon grids.
ml_coordinates = Pipeline(pixel_no=5000000, processes=1)
ml_coordinates.add_processing_data([coreg_image], 'coreg_master')
create_multilooking_grid.add_processing_step(ResamplePrepare(in_coor=dem_coor, out_coor=ml_coor,
                                                             in_file_type='dem', in_process='dem',
                                                             out_file_type='crop', out_process='crop', out_polarisation='VV',
                                                             slave='coreg_image', coreg_master='coreg_image'), True)
create_multilooking_grid.add_processing_step(ResampleDem(in_coor=dem_coor, out_coor=ml_coor,
                                                             in_file_type='dem', in_process='dem',
                                                             slave='coreg_image', coreg_master='coreg_image'), True)
ml_coordinates.add_processing_step(Geocode(out_coor=ml_coor, coreg_master='coreg_master'), True)
ml_coordinates.add_processing_step(RadarRayAngles(out_coor=ml_coor, coreg_master='coreg_master'), True)
ml_coordinates()
ml_coordinates.save_processing_results()
