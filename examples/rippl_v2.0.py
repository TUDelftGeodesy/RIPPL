
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

database_folder = data_disk + 'radar_database/sentinel-1/'
shapefile = data_disk + 'GIS/shapes/netherlands/zuid-holland.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = '/home/gert/fast_datastacks/Netherlands/s1_t88_single_core'
polarisation = 'VV'
mode = 'IW'
product_type = 'SLC'

n_jobs = 7
srtm_folder = data_disk + 'DEM/SRTM'

download = False
track = 88

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
from rippl.processing_steps.interferogram_multilook import InterferogramMultilook
from rippl.processing_steps.interferogram import Interferogram
from rippl.processing_steps.coherence import Coherence

from rippl.processing_steps.multilook_prepare import MultilookPrepare
from rippl.processing_steps.multilook import Multilook

# Test processing for full burst with processes only.
slices = np.sort(s1_stack.slcs['20170221'].slice_names)

ml_coor = CoordinateSystem()
ml_coor.create_geographic(0.005, 0.005, lat0=50, lon0=0)
radar_coor = CoordinateSystem()
radar_coor.create_radar_coordinates()
dem_coor = ImportDem.create_dem_coor('SRTM3')

for slice in slices:
    coreg_image = s1_stack.slcs['20170221'].slice_data[slice]

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
    square_amplitude_master = SquareAmplitude(polarisation='VV', out_coor=radar_coor, slave=coreg_image,
                                              master_image=True, overwrite=False)
    square_amplitude_master()

    for slave_date in ['20170215', '20170209']:

        image = s1_stack.slcs[slave_date].slice_data[slice]

        # Resample and correct images
        deramp = Deramp(polarisation='VV', out_coor=radar_coor, slave=image, overwrite=False)
        deramp()
        geometric_coreg = GeometricCoregistration(in_coor=radar_coor, out_coor=radar_coor, slave=image, coreg_master=coreg_image, overwrite=False)
        geometric_coreg()
        resample = ResampleRadarGrid(polarisation='VV', in_coor=radar_coor, out_coor=radar_coor, slave=image, overwrite=False)
        resample()
        reramp = Reramp(polarisation='VV', out_coor=radar_coor, slave=image, overwrite=False)
        reramp()
        earth_topo_phase = EarthTopoPhase(polarisation='VV', out_coor=radar_coor, slave=image, overwrite=False)
        earth_topo_phase()

        square_amplitude_slave = SquareAmplitude(polarisation='VV', out_coor=radar_coor, slave=image, overwrite=False)
        square_amplitude_slave()

        baseline = Baseline(out_coor=radar_coor, coreg_master=coreg_image, slave=image, overwrite=False)
        baseline()

# Concatenate the slices.
coreg_image = s1_stack.slcs['20170221']
dem_coor = ImportDem.create_dem_coor('SRTM3')
coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=dem_coor, replace=True)
coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='geocode', file_type='lat', coor=radar_coor, transition_type='cut_off')
coreg_image.create_concatenate_image(process='geocode', file_type='lon', coor=radar_coor, transition_type='cut_off')

for ifg in s1_stack.ifgs.keys():
    ifg_image = s1_stack.ifgs[ifg]
    ifg_image.create_concatenate_image(process='interferogram', file_type='interferogram', coor=ml_coor, polarisation='VV')
    ifg_image.create_concatenate_image(process='interferogram', file_type='interferogram', coor=radar_coor, polarisation='VV', transition_type='cut_off')

    ml_prepare = MultilookPrepare(in_polarisation='VV', in_coor=radar_coor, out_coor=ml_coor,
                                 in_file_type='interferogram', in_process='interferogram',
                                 slave=ifg_image.data, coreg_master=coreg_image.data, overwrite=False)
    ml_prepare()
    ml_interferogram = Multilook(in_polarisation='VV', in_coor=radar_coor, out_coor=ml_coor,
                                 in_file_type='interferogram', in_process='interferogram',
                                 slave=ifg_image.data, coreg_master=coreg_image.data, overwrite=False)
    ml_interferogram()

    image = ifg_image.concat_image_data_iterator(processes=['interferogram'], coordinates=[ml_coor], slices=False)[-1][0]
    image.save_tiff()

    # image.save_tiff()
    image = ifg_image.concat_image_data_iterator(processes=['interferogram'], coordinates=[ml_coor], slices=False)[-1][0]
    image.save_tiff()

# Run approximation tropospheric and ionospheric delay


# Apply multilooking and run unwrapping


