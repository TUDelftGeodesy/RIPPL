from rippl.processing_templates.general import GeneralPipelines
from rippl.processing_templates.oceans import Oceans

# Settings where the data is stored
data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'
database_folder = data_disk + 'radar_database/sentinel-1'
shapefile = data_disk + 'GIS/shapes/Stereoid_cases/gulf_of_mexico_small.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/RIPPL_v2.0/Sentinel_1/gulf_of_mexico_new'
dem_folder = data_disk + 'DEM/SRTM'

# Track and data type of Sentinel data
track = 63
mode = 'IW'
product_type = 'SLC'
polarisation = ['VV', 'VH']

# Start, master and end date of processing
start_date = '2019-02-01'
end_date = '2019-03-01'
master_date = '2019-02-22'

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
dlat = 0.001
dlon = 0.001
lat0 = 20
lon0 = -100

ocean_processing = Oceans(processes=8)

# Download and create the dataset
ocean_processing.download_sentinel_data(start_date, end_date, track, polarisation, shapefile, database_folder,
                                        orbit_folder, ESA_username, ESA_password, ASF_username, ASF_password)
ocean_processing.create_sentinel_stack(start_date, end_date, master_date, track, polarisation, shapefile,
                                       database_folder, orbit_folder, stack_folder, mode, product_type)
ocean_processing.read_stack(stack_folder, start_date, end_date)

# Coordinate systems
ocean_processing.create_dem_coordinates(dem_type)
ocean_processing.create_ml_coordinates(coor_type='geographic', dlat=dlat, dlon=dlon, lat0=lat0, lon0=lon0)

# Data processing
ocean_processing.download_external_dem(dem_folder, dem_type, ASF_username, ASF_password)
ocean_processing.geocoding(dem_folder, dem_type, dem_buffer, dem_rounding)
ocean_processing.geometric_coregistration_resampling(polarisation)
ocean_processing.prepare_multilooking_grid(polarisation[0])
ocean_processing.create_calibrated_amplitude_multilooked(polarisation)

# Create the geotiffs
ocean_processing.create_output_tiffs()
