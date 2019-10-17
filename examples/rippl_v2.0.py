
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


"""
import rippl.processing_steps as processing_steps
import os

dir = os.path.dirname(os.path.abspath(processing_steps.__file__))
for file in os.listdir(dir):
    if file.endswith('.py'):
        name = file.split('.')[0]
        # add package prefix to name, if required
        module = exec('from rippl.processing_steps.' + name + ' import *')
"""


# Import processing steps
from rippl.processing_steps.deramp import Deramp
from rippl.processing_steps.import_dem import ImportDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.resample_dem import ResampleDem
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

from rippl.processing_steps.reproject import Reproject
from rippl.processing_steps.multilook import Multilook

# Test processing for full burst with processes only.
slices = np.sort(s1_stack.slcs['20170221'].slice_names)


for slice in slices:

    ml_coor = CoordinateSystem()
    ml_coor.create_geographic(0.005, 0.005, lat0=50, lon0=0)

    s1_stack.slcs['20170221'].load_slice_memmap()
    coreg_image = s1_stack.slcs['20170221'].slice_data[slice]
    coreg_image.load_memmap_files()
    coreg_coor = coreg_image.processes['crop']['crop_#coor#_radar_20170221_#pol#_VH'].coordinates
    orbit = coreg_image.find_best_orbit()
    coreg_coor.load_orbit(orbit)

    # Creat DEM and geocode
    srtm = ImportDem(in_coor=coreg_coor, coreg_master=coreg_image, dem_type='SRTM3', dem_folder=srtm_folder,
                     overwrite=False)
    srtm()
    dem_coor = coreg_image.processing_image_data_iterator(processes=['dem'], file_types=['dem'])[-1][0].coordinates
    inverse_geocode = InverseGeocode(coordinates=dem_coor, coor_ref=coreg_coor, coreg_master=coreg_image, overwrite=False)
    inverse_geocode()
    resample_dem = ResampleDem(in_coor=dem_coor, out_coor=coreg_coor, coreg_master=coreg_image, overwrite=False)
    resample_dem()
    geocode = Geocode(coordinates=coreg_coor, coreg_master=coreg_image, overwrite=False)
    geocode()
    radar_ray_angles = RadarRayAngles(coordinates=coreg_coor, coreg_master=coreg_image, overwrite=False)
    radar_ray_angles()
    square_amplitude_master = SquareAmplitude(polarisation='VH', coordinates=coreg_coor, slave=coreg_image,
                                              master_image=True, overwrite=False)
    square_amplitude_master()

    # Multilook square amplitude
    reproject = Reproject(in_coor=coreg_coor, out_coor=ml_coor, coreg_master=coreg_image, overwrite=False)
    reproject()
    ml_square_amplitude = Multilook(in_polarisation='VH', in_coor=coreg_coor, out_coor=ml_coor,
                                    in_file_type='square_amplitude', in_process='square_amplitude', in_data_type='real4',
                                    slave=coreg_image, coreg_master=coreg_image, overwrite=False)
    ml_square_amplitude()

    for slave_date in ['20170215', '20170209']:

        s1_stack.slcs[slave_date].load_slice_memmap()
        image = s1_stack.slcs[slave_date].slice_data[slice]
        coor = image.processes['crop']['crop_#coor#_radar_' + slave_date + '_#pol#_VH'].coordinates
        orbit = image.find_best_orbit()
        coor.load_orbit(orbit)

        # Resample and correct images
        deramp = Deramp(polarisation='VH', coordinates=coor, slave=image, overwrite=False)
        deramp()
        geometric_coreg = GeometricCoregistration(in_coor=coor, out_coor=coreg_coor, slave=image, coreg_master=coreg_image, overwrite=False)
        geometric_coreg()
        resample = ResampleRadarGrid(polarisation='VH', in_coor=coor, out_coor=coreg_coor, slave=image, overwrite=False)
        resample()
        reramp = Reramp(polarisation='VH', coordinates=coreg_coor, slave=image, overwrite=False)
        reramp()
        earth_topo_phase = EarthTopoPhase(polarisation='VH', coordinates=coreg_coor, slave=image, overwrite=False)
        earth_topo_phase()

        square_amplitude_slave = SquareAmplitude(polarisation='VH', coordinates=coreg_coor, slave=image, overwrite=False)
        square_amplitude_slave()
        ml_square_amplitude = Multilook(in_polarisation='VH', in_coor=coreg_coor, out_coor=ml_coor,
                                        in_file_type='square_amplitude', in_process='square_amplitude', in_data_type='real4',
                                        slave=image, coreg_master=coreg_image, overwrite=False)
        ml_square_amplitude()

        baseline = Baseline(coordinates=coreg_coor, coreg_master=coreg_image, slave=image, overwrite=False)
        baseline()

    for ifg in s1_stack.ifgs.keys():
        s1_stack.ifgs[ifg].load_slice_memmap()

        coreg_image = s1_stack.slcs['20170221'].slice_data[slice]
        coreg_image.load_memmap_files()
        master = s1_stack.slcs[ifg[:8]].slice_data[slice]
        master.load_memmap_files()
        slave = s1_stack.slcs[ifg[9:]].slice_data[slice]
        slave.load_memmap_files()
        ifg_image = s1_stack.ifgs[ifg].slice_data[slice]
        ifg_image.load_memmap_files()

        # Run interferogram, coherence
        interferogram = Interferogram(polarisation='VH', coordinates=coreg_coor, master=master, slave=slave, coreg_master=coreg_image, ifg=ifg_image, overwrite=False)
        interferogram()
        ml_interferogram = Multilook(in_polarisation='VH', in_coor=coreg_coor, out_coor=ml_coor,
                                        in_file_type='interferogram', in_process='interferogram', in_data_type='complex_real4',
                                        slave=ifg_image, coreg_master=coreg_image, overwrite=False)
        ml_interferogram()

        coherence = Coherence(polarisation='VH', coordinates=ml_coor, master=master, slave=slave, ifg=ifg_image, overwrite=False)
        coherence()

# Concatenate the slices.
ml_coor = CoordinateSystem()
ml_coor.create_geographic(0.005, 0.005, lat0=50, lon0=0)
ml_coor2 = CoordinateSystem()
ml_coor2.create_geographic(0.001, 0.001, lat0=50, lon0=0)
radar_coor = CoordinateSystem()
radar_coor.create_radar_coordinates()
radar_coor.date = '2017-02-21'

coreg_image = s1_stack.slcs['20170221']
coreg_image.load_full_memmap()
coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='geocode', file_type='lat', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='geocode', file_type='lon', coor=radar_coor, transition_type='cut_off')


for ifg in s1_stack.ifgs.keys():
    s1_stack.ifgs[ifg].load_slice_memmap()
    s1_stack.ifgs[ifg].load_full_memmap()
    ifg_image = s1_stack.ifgs[ifg]


    ifg_image.create_concatenate_image(process='interferogram', file_type='interferogram', coor=ml_coor, polarisation='VH')
    ifg_image.create_concatenate_image(process='interferogram', file_type='interferogram', coor=radar_coor, polarisation='VH', transition_type='cut_off')

    radar_coor = ifg_image.concat_image_data_iterator(processes=['interferogram'], coordinates=[radar_coor], slices=False)[3][0]
    orbit = ifg_image.data.orbits['coreg_#type#_precise']
    radar_coor.load_orbit(orbit)

    reproject = Reproject(in_coor=radar_coor, out_coor=ml_coor2, coreg_master=coreg_image.data, overwrite=False)
    reproject()
    ml_interferogram = Multilook(in_polarisation='VH', in_coor=radar_coor, out_coor=ml_coor2,
                                 in_file_type='interferogram', in_process='interferogram', in_data_type='complex_real4',
                                 slave=ifg_image.data, coreg_master=coreg_image.data, overwrite=False)
    ml_interferogram()
    image = ifg_image.concat_image_data_iterator(processes=['interferogram'], coordinates=[ml_coor2], slices=False)[-1][0]
    image.save_tiff()

    # image.save_tiff()
    image = ifg_image.concat_image_data_iterator(processes=['interferogram'], coordinates=[ml_coor], slices=False)[-1][0]
    image.save_tiff()

# Run approximation tropospheric and ionospheric delay


# Apply multilooking and run unwrapping


