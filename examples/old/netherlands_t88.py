from rippl.processing_templates.general_sentinel_1 import GeneralPipelines
from rippl.processing_templates.troposphere import Troposphere

# Settings where the data is stored
data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'
database_folder = data_disk + 'radar_database/sentinel-1/'
shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/RIPPL_v2.0/Sentinel_1/netherlands_t88'
dem_folder = data_disk + 'DEM/SRTM'

harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'

# Track and data type of Sentinel data
track = 88
mode = 'IW'
product_type = 'SLC'
polarisation = ['VV']

# Start, master and end date of processing
start_date = '2016-01-01'
end_date = '2018-12-31'
master_date = '2017-02-21'

# Passwords for data and DEM download
SRTM_username = 'gertmulder'
SRTM_password = 'Radar2016'
ASF_username = 'gertmulder'
ASF_password = 'Radar2016'
ESA_username = 'fjvanleijen'
ESA_password = 'stevin01'

# DEM type
dem_type = 'SRTM3'
dem_buffer = 1
dem_rounding = 1

# Multilooking coordinates
dlat = 0.01
dlon = 0.01
lat0 = 45
lon0 = 0

processes = 8
troposphere_processing = Troposphere(processes=processes)

# Download and create the dataset
# troposphere_processing.download_sentinel_data(start_date, end_date, track, polarisation, shapefile, database_folder,
#                                        orbit_folder, ESA_username, ESA_password, ASF_username, ASF_password)
troposphere_processing.create_sentinel_stack(start_date, end_date, master_date, track, polarisation, shapefile,
                                       database_folder, orbit_folder, stack_folder, mode, product_type)
troposphere_processing.read_stack(stack_folder, start_date, end_date)

# Coordinate systems
troposphere_processing.create_radar_coordinates()
troposphere_processing.create_dem_coordinates(dem_type)
troposphere_processing.calc_radar_multilooked_geometry([50, 200], dem_folder, dem_type, dem_buffer, dem_rounding)

# Data processing
troposphere_processing.create_ifg_network(network_type='temp_baseline', temporal_baseline=60)
troposphere_processing.download_external_dem(dem_folder, dem_type, ASF_username, ASF_password, buffer=2, rounding=1)
troposphere_processing.geocoding(dem_folder, dem_type, dem_buffer, dem_rounding)
troposphere_processing.geometric_coregistration_resampling(polarisation)

for dlat, dlon in zip([0.01, 0.005, 0.0025], [0.01, 0.005, 0.0025]):
    troposphere_processing.create_ml_coordinates(coor_type='geographic', dlat=dlat, dlon=dlon, lat0=lat0, lon0=lon0)
    troposphere_processing.prepare_multilooking_grid(polarisation[0])
    troposphere_processing.create_calibrated_amplitude_multilooked(polarisation)
    troposphere_processing.create_interferogram_multilooked(polarisation)
    troposphere_processing.create_coherence_multilooked(polarisation)
    troposphere_processing.create_unwrapped_images(polarisation)
    troposphere_processing.create_geometry_mulitlooked(dem_folder, dem_type, dem_buffer, dem_rounding)

    # Create the geotiffs
    troposphere_processing.create_output_tiffs_coherence_unwrap()
    troposphere_processing.create_output_tiffs_geometry()

"""
Remove processing files bursts afterwards
find /mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/RIPPL_v2.0/Sentinel_1/netherlands_t88 -type f -size -500M -name "*crop_VV@radar.raw" -exec rm -f {} \;
find /mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/RIPPL_v2.0/Sentinel_1/netherlands_t88 -type f -size -500M -name "*earth_topo_phase_corrected_VV@radar.raw" -exec rm -f {} \;
"""


