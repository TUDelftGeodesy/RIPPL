
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

start_date = '2017-02-07'
end_date = '2017-02-25'
master_date = '2017-02-21'

database_folder = data_disk + 'radar_database/sentinel-1/asc_t088_test'
shapefile = data_disk + 'GIS/shapes/netherlands/zuid_holland.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = datastack_disk + 'radar_datastacks/RIPPL_v2.0/Sentinel_1/Netherlands/asc_t088_test'
polarisation = 'VH'
mode = 'IW'
product_type = 'SLC'

n_jobs = 7
srtm_folder = data_disk + 'DEM/DEM_new'

download = False

if download:
    # Download data and orbit
    download_data = DownloadSentinel(start_date, end_date, 'fjvanleijen', 'stevin01', shapefile, '88', polarisation)
    download_data.sentinel_available()
    download_data.sentinel_download(destination_folder=os.path.dirname(database_folder))
    download_data.sentinel_check_validity(destination_folder=os.path.dirname(database_folder))

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
s1_stack.download_SRTM_dem(srtm_folder, username, password, buffer=1, rounding=1, srtm_type='SRTM1')
s1_stack.download_SRTM_dem(srtm_folder, username, password, buffer=1, rounding=1, srtm_type='SRTM3')

# Import processing steps
from rippl.processing_steps.deramp import Deramp
from rippl.processing_steps.create_dem import CreateDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.radar_dem import RadarDem
from rippl.processing_steps.geocode import Geocode
from rippl.processing_steps.resample_radar_grid import ResampleRadarGrid
from rippl.processing_steps.geometric_coregistration import GeometricCoregistration
from rippl.processing_steps.earth_topo_phase import EarthTopoPhase
from rippl.processing_steps.square_amplitude import SquareAmplitude
from rippl.processing_steps.reramp import Reramp
from rippl.processing_steps.radar_ray_angles import RadarRayAngles
from rippl.processing_steps.baseline import Baseline
from rippl.processing_steps.interferogram import Interferogram
from rippl.processing_steps.coherence import Coherence

# Test processing for full burst with processes only.
slices = np.sort(s1_stack.slcs['20170221'].slice_names)

for slice in [slices[0]]:

    s1_stack.slcs['20170221'].load_slice_memmap()
    coreg_image = s1_stack.slcs['20170221'].slice_data[slice]
    coreg_coor = coreg_image.processes['crop']['crop_#coor#__#id#_none_#pol#_VH'].coordinates

    # Creat DEM and geocode
    srtm = CreateDem(coor_in=coreg_coor, coreg_master=coreg_image, dem_type='SRTM3', dem_folder=srtm_folder,
                     overwrite=False)
    srtm()
    dem_coor = coreg_image.processing_image_data_iterator(processes=['create_dem'], file_types=['dem'])[-1][
        0].coordinates
    inverse_geocode = InverseGeocode(coor_in=dem_coor, coreg_master=coreg_image, overwrite=False)
    inverse_geocode()
    radar_dem = RadarDem(coor_in=dem_coor, coor_out=coreg_coor, coreg_master=coreg_image, overwrite=False)
    radar_dem()
    geocode = Geocode(coor_in=coreg_coor, coreg_master=coreg_image, overwrite=False)
    geocode()
    radar_ray_angles = RadarRayAngles(coor_in=coreg_coor, coreg_master=coreg_image, overwrite=False)
    radar_ray_angles()
    square_amplitude_master = SquareAmplitude(polarisation='VH', coor_in=coreg_coor, slave=coreg_image,
                                              in_processes=['crop'], in_file_types=['crop'], overwrite=False)
    square_amplitude_master()

    for slave_date in ['20170215', '20170209']:

        s1_stack.slcs[slave_date].load_slice_memmap()
        image = s1_stack.slcs[slave_date].slice_data[slice]
        coor = image.processes['crop']['crop_#coor#__#id#_none_#pol#_VH'].coordinates

        # Resample and correct images
        deramp = Deramp(polarisation='VH', coor_in=coor, slave=image, overwrite=False)
        deramp()
        geometric_coreg = GeometricCoregistration(coor_in=coor, coor_out=coreg_coor, slave=image, coreg_master=coreg_image, overwrite=False)
        geometric_coreg()
        resample = ResampleRadarGrid(polarisation='VH', coor_in=coor, coor_out=coreg_coor, slave=image, overwrite=False)
        resample()
        reramp = Reramp(polarisation='VH', coor_in=coreg_coor, slave=image, overwrite=False)
        reramp()
        earth_topo_phase = EarthTopoPhase(polarisation='VH', coor_in=coreg_coor, slave=image, overwrite=False)
        earth_topo_phase()

        square_amplitude_slave = SquareAmplitude(polarisation='VH', coor_in=coreg_coor, slave=image, overwrite=False)
        square_amplitude_slave()
        baseline = Baseline(coor_in=coreg_coor, coreg_master=coreg_image, slave=image, overwrite=True)
        baseline()


s1_stack.ifgs['20170209_20170221'].load_slice_memmap()
ifg_image = s1_stack.ifgs['20170209_20170221'].slice_data[slice]
s1_stack.ifgs['20170209_20170215'].load_slice_memmap()
ifg_image = s1_stack.ifgs['20170209_20170215'].slice_data[slice]

# Run interferogram, coherence
interferogram = Interferogram(polarisation='VH', coor_in=coreg_coor, master=coreg_image, slave=image, coreg_master=coreg_image, ifg=ifg_image, overwrite=True)
interferogram()
coherence = Coherence(polarisation='VH', coor_in=coreg_coor, master=coreg_image, slave=image, coreg_master=coreg_image, ifg=ifg_image, overwrite=True)
coherence()

# Run approximation tropospheric and ionospheric delay


# Apply multilooking and run unwrapping


