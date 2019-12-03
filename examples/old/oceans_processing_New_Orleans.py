"""
For the ocean processing we apply the following steps:

1. Download the needed data
2. Do a geocoding to retrieve the lat/lon values
3. Calculate the incidence angles
4. Apply a calibration to the data of individual bursts
5. Add the amplitude images together to one image (for all slave images)
"""

from rippl.meta_data.stack import Stack
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinelOrbit, DownloadSentinel
import os
from rippl.meta_data.image_data import ImageData
import numpy as np

# Period of ingestion of dataset
start_date = '2018-01-01'
end_date = '2019-03-01'
master_date = '2019-02-22'

# Get the folders for the dataset processing
data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'
database_folder = data_disk + 'radar_database/sentinel-1'
shapefile = data_disk + 'GIS/shapes/Stereoid_cases/gulf_of_mexico_small.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/RIPPL_v2.0/Sentinel_1/gulf_of_mexico'
srtm_folder = data_disk + 'DEM/SRTM'

track = 63
imaging_type = 'IW'
processing_type = 'SLC'

download = False
if download:
    # Download data and orbit
    download_data = DownloadSentinel(start_date, end_date, shapefile, track, polarisation='VV')
    download_data.sentinel_available('fjvanleijen', 'stevin01')
    download_data.sentinel_download_ASF(database_folder, 'gertmulder', 'Radar2016')
    download_data.sentinel_check_validity(database_folder, 'fjvanleijen', 'stevin01')

    # Orbits
    precise_folder = os.path.join(orbit_folder, 'precise')
    download_orbit = DownloadSentinelOrbit(start_date, end_date, precise_folder)
    download_orbit.download_orbits()

# Create stack.
cores = 6
if not os.listdir(stack_folder):
    for pol in ['VV', 'VH']:
        s1_stack = SentinelStack(stack_folder)
        s1_stack.read_from_database(database_folder, shapefile, track, orbit_folder, start_date, end_date, master_date,
                             imaging_type, processing_type, pol, cores=cores)

s1_stack = Stack(stack_folder)
s1_stack.read_master_slice_list()
s1_stack.read_stack(start_date, end_date)
password = 'Radar2016'
username = 'gertmulder'
s1_stack.download_SRTM_dem(srtm_folder, username, password, buffer=1, rounding=1, srtm_type='SRTM1')
s1_stack.download_SRTM_dem(srtm_folder, username, password, buffer=1, rounding=1, srtm_type='SRTM3')

# Import processing steps
from rippl.processing_steps.import_dem import ImportDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.resample_dem import ResampleDem
from rippl.processing_steps.geocode import Geocode
from rippl.processing_steps.geometric_coregistration import GeometricCoregistration
from rippl.processing_steps.earth_topo_phase import EarthTopoPhase
from rippl.processing_steps.square_amplitude import SquareAmplitude
from rippl.processing_steps.reramp import Reramp
from rippl.processing_steps.radar_ray_angles import RadarRayAngles
from rippl.processing_steps.deramp_resample_radar_grid import DerampResampleRadarGrid
from rippl.processing_steps.multilook_prepare import MultilookPrepare
from rippl.processing_steps.resample_prepare import ResamplePrepare
from rippl.processing_steps.calibrated_amplitude import CalibratedAmplitude

from rippl.pipeline import Pipeline
from rippl.processing_steps.multilook import Multilook

master_date = s1_stack.master_date
slice_names = np.sort(s1_stack.slcs[master_date].slice_names)

# Define coordinate system.
radar_coor = CoordinateSystem()
radar_coor.create_radar_coordinates()
dem_coor = ImportDem.create_dem_coor('SRTM3')

# Find the slices of the stack.
master_key = s1_stack.master_date
slices = sorted(list(s1_stack.slcs[master_key].slice_names))
images = list(s1_stack.slcs.keys())

# Run the processing for this stack.
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
geocode_pipeline = Pipeline(pixel_no=3000000, processes=8)
geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
geocode_pipeline.add_processing_step(ResampleDem(out_coor=radar_coor, in_coor=dem_coor, coreg_master='coreg_master'), True)
geocode_pipeline.add_processing_step(Geocode(out_coor=radar_coor, coreg_master='coreg_master'), True)
geocode_pipeline.add_processing_step(RadarRayAngles(out_coor=radar_coor, coreg_master='coreg_master'), True)
geocode_pipeline()
geocode_pipeline.save_processing_results()


for pol in ['VV', 'VH']:

    s1_stack.read_stack(start_date, end_date)
    coreg_slices = [s1_stack.slcs[master_date].slice_data[slice] for slice in slice_names]
    geocode_pipeline = Pipeline(pixel_no=3000000, processes=8)
    geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
    geocode_pipeline.add_processing_step(CalibratedAmplitude(out_coor=radar_coor, slave='coreg_master', master_image=True, polarisation=pol), True)
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

    resampling_pipeline = Pipeline(pixel_no=5000000, processes=8)
    resampling_pipeline.add_processing_data(coreg_slices, 'coreg_master')
    resampling_pipeline.add_processing_data(slave_slices, 'slave')
    resampling_pipeline.add_processing_step(GeometricCoregistration(out_coor=radar_coor, in_coor=radar_coor, coreg_master='coreg_master', slave='slave'), False)
    resampling_pipeline.add_processing_step(DerampResampleRadarGrid(out_coor=radar_coor, in_coor=radar_coor, polarisation=pol, slave='slave'), False)
    resampling_pipeline.add_processing_step(Reramp(out_coor=radar_coor, polarisation=pol, slave='slave'), False)
    resampling_pipeline.add_processing_step(EarthTopoPhase(out_coor=radar_coor, polarisation=pol, slave='slave'), False)
    resampling_pipeline.add_processing_step(CalibratedAmplitude(out_coor=radar_coor, polarisation=pol, slave='slave'), True)
    resampling_pipeline()
    resampling_pipeline.save_processing_results()

# Concatenate the slices
coreg_image = s1_stack.slcs[master_key]
coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='geocode', file_type='lat', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='radar_ray_angles', file_type='incidence_angle', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='geocode', file_type='lon', coor=radar_coor, transition_type='cut_off')

# Create concatenated images for calibration.
for image_key in images:
    image = s1_stack.slcs[image_key]

    for pol in ['VV', 'VH']:
        for process_type in ['calibrated_amplitude_db']:
            image.create_concatenate_image(process='calibrated_amplitude', file_type=process_type,
                                           coor=radar_coor, polarisation=pol, transition_type='cut_off')

# Apply multilooking for the calibrated amplitude values
s1_stack.read_stack(start_date, end_date)
ml_coor = CoordinateSystem()
ml_coor.create_geographic(0.001, 0.001, lat0=20, lon0=-90)

# Multilook to regular lat/lon grids.
coreg_image = s1_stack.slcs[master_date]

create_multilooking_grid = Pipeline(pixel_no=5000000, processes=8)
create_multilooking_grid.add_processing_data([coreg_image.data], 'coreg_master')
create_multilooking_grid.add_processing_step(MultilookPrepare(in_polarisation='VV', in_coor=radar_coor, out_coor=ml_coor,
                                                              in_file_type='calibrated_amplitude', in_process='calibrated_amplitude',
                                                              slave='coreg_master', coreg_master='coreg_master'), True)
create_multilooking_grid()
create_multilooking_grid.save_processing_results()

# If the image to be multilooked is large, the only way to do the processing
s1_stack.read_stack(start_date, end_date)
coreg_image = s1_stack.slcs[master_date]
slave_keys = [key for key in s1_stack.slcs.keys()]
slaves = [s1_stack.slcs[slave_key].data for slave_key in slave_keys]
for pol in ['VH', 'VV']:
    for slave in slaves:
        multilook = Multilook(in_polarisation=pol, in_coor=radar_coor, out_coor=ml_coor,
                              in_file_type='calibrated_amplitude_db', in_process='calibrated_amplitude',
                              slave=slave, coreg_master=coreg_image.data, batch_size=20000000)
        multilook()

multilook = Multilook(in_coor=radar_coor, out_coor=ml_coor,
                      in_file_type=['lat', 'lon'], in_process='geocode',
                      slave=coreg_image.data, coreg_master=coreg_image.data, batch_size=5000000)
multilook()
multilook = Multilook(in_coor=radar_coor, out_coor=ml_coor,
                      in_file_type='incidence_angle', in_process='radar_ray_angles',
                      slave=coreg_image.data, coreg_master=coreg_image.data, batch_size=5000000)
multilook()

# Generate output geotiff files.
calibrated_amplitudes = s1_stack.stack_data_iterator(['calibrated_amplitude'], [ml_coor], ifg=False)[-1]
for calibrated_amplitude in calibrated_amplitudes:          # type: ImageData
    calibrated_amplitude.save_tiff(main_folder=True)

geometry_datasets = s1_stack.stack_data_iterator(['radar_ray_angles', 'geocode'], coordinates=[ml_coor], process_types=['lat', 'lon', 'incidence_angle'])[-1]
for geometry_dataset in geometry_datasets:                  # type: ImageData
    geometry_dataset.save_tiff(main_folder=True)
