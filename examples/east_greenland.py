from rippl.processing_templates.general_sentinel_1 import GeneralPipelines
import datetime

# Settings where the data is stored
shapefile = 'East_Greenland_Mini.shp'
stack_name = 'east_greenland_mini'

# Track and data type of Sentinel data
track = 141
mode = 'IW'
product_type = 'SLC'
polarisation = ['HH', 'HV']       # Possibly add HV later on

# Start, master and end date of processing
start_date = datetime.datetime(year=2018, month=2, day=1)
end_date = datetime.datetime(year=2018, month=2, day=10)
master_date = datetime.datetime(year=2018, month=2, day=8)

# DEM type
dem_type = 'TanDEM-X'
dem_buffer = 1
dem_rounding = 1
lon_resolution = 6

# Multilooking coordinates
dx = 50
dy = 50

land_ice_processing = GeneralPipelines(processes=4)

# Download and create the dataset
land_ice_processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track,
                                           polarisation=polarisation, shapefile=shapefile, data=True)
land_ice_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, master_date=master_date,
                                          track=track,stack_name=stack_name, polarisation=polarisation,
                                          shapefile=shapefile, mode=mode, product_type=product_type)

# Load stack
land_ice_processing.read_stack(start_date=start_date, end_date=end_date, stack_name=stack_name)
land_ice_processing.create_ifg_network(network_type='temp_baseline', temporal_baseline=30)

# Coordinate systems
land_ice_processing.create_radar_coordinates()
land_ice_processing.create_dem_coordinates(dem_type=dem_type, lon_resolution=6)
land_ice_processing.create_ml_coordinates(standard_type='oblique_mercator', dx=dx, dy=dy, buffer=0, rounding=0)

# Data processing
land_ice_processing.download_external_dem(dem_type=dem_type, lon_resolution=lon_resolution)
land_ice_processing.geocoding(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)
land_ice_processing.geometric_coregistration_resampling(polarisation)

# Multilooking
land_ice_processing.prepare_multilooking_grid(polarisation)
land_ice_processing.create_calibrated_amplitude_multilooked(polarisation)
land_ice_processing.create_interferogram_multilooked(polarisation)
land_ice_processing.create_coherence_multilooked(polarisation)

# Calculate geometry
land_ice_processing.create_geometry_mulitlooked(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)

# Create the geotiffs
land_ice_processing.create_output_tiffs_amplitude()
land_ice_processing.create_output_tiffs_coherence_ifg()
land_ice_processing.create_output_tiffs_geometry()
