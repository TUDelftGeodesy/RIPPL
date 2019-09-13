
from rippl.meta_data.stack import Stack
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinelOrbit, DownloadSentinel
import os

data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'

start_date = '2017-02-01'
end_date = '2017-03-01'
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

# Test processing for full burst with processes only.
s1_stack.slcs['20170221'].load_slice_memmap()
image = s1_stack.slcs['20170215'].slice_data['slice_500_swath_1']
coreg_image = s1_stack.slcs['20170221'].slice_data['slice_500_swath_1']

coor = image.processes['crop']['crop_#coor#__#id#_none_#pol#_VH'].coordinates
coreg_coor = coreg_image.processes['crop']['crop_#coor#__#id#_none_#pol#_VH'].coordinates

# Creat DEM and geocode
srtm = CreateDem(coor_in=coreg_coor, coreg_master=coreg_image, dem_type='SRTM3', dem_folder=srtm_folder)
srtm()
dem_coor = srtm.coor_out
inverse_geocode = InverseGeocode(coor_in=dem_coor, coreg_master=coreg_image)
inverse_geocode()
radar_dem = RadarDem(coor_in=dem_coor, coor_out=coreg_coor, coreg_master=coreg_image)
radar_dem()
geocode = Geocode(coor_in=coreg_coor, coreg_master=coreg_image)

# Resample and correct images
deramp = Deramp(polarisation='VH', coor_in=coor, slave=image)
deramp()
geometric_coreg = GeometricCoregistration(coor_in=coor, coor_out=coreg_coor, slave=image, coreg_master=coreg_image)
geometric_coreg()
resample = ResampleRadarGrid(polarisation='VH', coor_in=coor, coor_out=coreg_coor, slave=image)
resample()
reramp = Reramp(polarisation='VH', coor_in=coreg_coor, slave=image)
reramp()
earth_topo_phase = EarthTopoPhase(polarisation='VH', coor_in=coreg_coor, slave=image)
earth_topo_phase()
square_amplitude_master = SquareAmplitude(polarisation='VH', coor_in=coreg_coor, slave=coreg_image, in_processes=['crop'], in_file_types=['crop'])
square_amplitude_master()
square_amplitude_slave = SquareAmplitude(polarisation='VH', coor_in=coreg_coor, slave=image)
square_amplitude_slave()

# Create interferogram


# Run interferogram, coherence and unwrapping
interferogram =


# Apply multilooking and run unwrapping


