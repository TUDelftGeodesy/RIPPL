from rippl.processing_templates.general import GeneralPipelines
from rippl.processing_templates.land_ice import LandIce

# Settings where the data is stored
data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
datastack_disk = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/'
database_folder = data_disk + 'radar_database/sentinel-1'
shapefile = data_disk + 'GIS/shapes/Stereoid_cases/East_greenland_small.shp'
orbit_folder = data_disk + 'orbits/sentinel-1'
stack_folder = datastack_disk + 'radar_datastacks/RIPPL_v2.0/Sentinel_1/east_greenland_small'
dem_folder = data_disk + 'DEM/Tandem_X'

# Track and data type of Sentinel data
track = 141
mode = 'IW'
product_type = 'SLC'
polarisation = ['HH']       # Possibly add HV later on

# Start, master and end date of processing
start_date = '2019-03-24'
end_date = '2019-10-10'
master_date = '2019-10-01'

# Passwords for data and DEM download
SRTM_username = 'gertmulder'
SRTM_password = 'Radar2016'
ASF_username = 'gertmulder'
ASF_password = 'Radar2016'
ESA_username = 'fjvanleijen'
ESA_password = 'stevin01'

# DEM type
dem_type = 'Tandem_X'
dem_buffer = 1
dem_rounding = 1
lon_resolution = 6

# Multilooking coordinates
dlat = 0.001
dlon = 0.001
lat0 = 60
lon0 = -50

land_ice_processing = LandIce(processes=8)

# Download and create the dataset
land_ice_processing.download_sentinel_data(start_date, end_date, track, polarisation, shapefile, database_folder,
                                        orbit_folder, ESA_username, ESA_password, ASF_username, ASF_password)
land_ice_processing.create_sentinel_stack(start_date, end_date, master_date, track, polarisation, shapefile,
                                       database_folder, orbit_folder, stack_folder, mode, product_type)
land_ice_processing.read_stack(stack_folder, start_date, end_date)
land_ice_processing.create_ifg_network(network_type='temp_baseline', temporal_baseline=15)

# Coordinate systems
land_ice_processing.create_radar_coordinates()
land_ice_processing.create_dem_coordinates(dem_type, lon_resolution=6)
land_ice_processing.create_ml_coordinates(coor_type='geographic', dlat=dlat, dlon=dlon, lat0=lat0, lon0=lon0)

# Data processing
land_ice_processing.download_external_dem(dem_folder, dem_type, ASF_username, ASF_password)
land_ice_processing.geocoding(dem_folder, dem_type, dem_buffer, dem_rounding)
land_ice_processing.geometric_coregistration_resampling(polarisation)
land_ice_processing.prepare_multilooking_grid(polarisation[0])
land_ice_processing.create_calibrated_amplitude_multilooked(polarisation)
land_ice_processing.create_interferogram_multilooked(polarisation)
land_ice_processing.create_coherence_multilooked(polarisation)
land_ice_processing.create_unwrapped_images(polarisation)

# Create the geotiffs
land_ice_processing.create_output_tiffs_amplitude()
land_ice_processing.create_output_tiffs_coherence_unwrap()
