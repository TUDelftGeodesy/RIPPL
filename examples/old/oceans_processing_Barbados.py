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

# Period of ingestion of dataset
start_date = '2019-02-01'
end_date = '2019-06-01'
master_date = '2019-05-06'

# Get the folders for the dataset processing
data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'
database_folder = data_disk + 'radar_database/sentinel-1'
shapefile = data_disk + 'GIS/shapes/Stereoid_cases/Barbados.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = datastack_disk + 'radar_datastacks/RIPPL_v2.0/Sentinel_1/barbados'
srtm_folder = data_disk + 'DEM/DEM_new'

track = 83
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

# Create stack
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
from rippl.processing_steps.radar_ray_angles import RadarRayAngles
from rippl.processing_steps.calibrated_amplitude import CalibratedAmplitude
from rippl.processing_steps.deramp import Deramp
from rippl.processing_steps.import_dem import ImportDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.resample_dem import ResampleDem
from rippl.processing_steps.geocode import Geocode
from rippl.processing_steps.resample_radar_grid import ResampleRadarGrid
from rippl.processing_steps.geometric_coregistration import GeometricCoregistration

# Define coordinate system.
radar_coor = CoordinateSystem()
radar_coor.create_radar_coordinates()
dem_coor = ImportDem.create_dem_coor('SRTM3')

# Find the slices of the stack.
master_key = s1_stack.master_date
slices = sorted(list(s1_stack.slcs[master_key].slice_names))
images = list(s1_stack.slcs.keys())

# Run the processing for this stack.
for slice in slices:
    coreg_image = s1_stack.slcs[master_key].slice_data[slice]

    # Creat DEM and geocode
    srtm = ImportDem(in_coor=radar_coor, coreg_master=coreg_image, dem_type='SRTM3', dem_folder=srtm_folder,
                     overwrite=False)
    srtm()
    inverse_geocode = InverseGeocode(out_coor=dem_coor, in_coor=radar_coor, coreg_master=coreg_image, overwrite=False)
    inverse_geocode()
    resample_dem = ResampleDem(in_coor=dem_coor, out_coor=radar_coor, coreg_master=coreg_image, overwrite=False)
    resample_dem()
    geocode = Geocode(out_coor=radar_coor, coreg_master=coreg_image, overwrite=False)
    geocode()
    radar_ray_angles = RadarRayAngles(out_coor=radar_coor, coreg_master=coreg_image, overwrite=False)
    radar_ray_angles()

    # Now apply the calibration for the different slices.
    for image_key in images:
        image = s1_stack.slcs[image_key].slice_data[slice]

        geometric_coreg = GeometricCoregistration(in_coor=radar_coor, out_coor=radar_coor, slave=image, coreg_master=coreg_image, overwrite=False)
        geometric_coreg()
        # Correct amplitudes
        for pol in ['VV', 'VH']:
            resample = ResampleRadarGrid(polarisation=pol, in_coor=radar_coor, out_coor=radar_coor, slave=image, overwrite=False)
            resample.input_info['process_types'][0] = 'crop'
            resample.input_info['file_types'][0] = 'crop'
            resample()
            calibrate_amplitude = CalibratedAmplitude(polarisation=pol, out_coor=radar_coor, slave=image, coreg_master=coreg_image, resampled=False)
            calibrate_amplitude.input_info['process_types'][0] = 'resample'
            calibrate_amplitude.input_info['file_types'][0] = 'resampled'
            calibrate_amplitude()

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
        image.create_concatenate_image(process='calibrated_amplitude', file_type='calibrated_amplitude', coor=radar_coor, polarisation=pol, transition_type='cut_off')
