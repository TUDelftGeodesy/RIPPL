from rippl.processing_templates.general_sentinel_1 import GeneralPipelines
from rippl.processing_templates.troposphere import Troposphere

# Settings where the data is stored
data_disk = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/'
shapefile = data_disk + 'GIS/shapes/netherlands/netherland.shp'
harmonie_data = data_disk + 'weather_models/harmonie_data'
ecmwf_data = data_disk + 'weather_models/ecmwf_data'
stack_name = 'Netherlands_t37'

# Track and data type of Sentinel data
track = 37
mode = 'IW'
product_type = 'SLC'
polarisation = ['VV']

# Start, master and end date of processing
start_date = '2016-01-01'
end_date = '2018-12-31'
master_date = '2017-11-15'

# DEM type
dem_type = 'SRTM3'
dem_buffer = 1
dem_rounding = 1

# Multilooking coordinates
dlat = 0.01
dlon = 0.01
lat0 = 45
lon0 = 0

processes = 4
troposphere_processing = Troposphere(processes=processes)

# Download and create the dataset
troposphere_processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track,
                                           polarisation=polarisation, shapefile=shapefile, data=True)
troposphere_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, master_date=master_date,
                                          track=track,stack_name='east_greenland_mini', polarisation=polarisation,
                                          shapefile=shapefile, mode=mode, product_type=product_type)

# Load stack
troposphere_processing.read_stack(start_date=start_date, end_date=end_date, stack_name=stack_name)
troposphere_processing.create_ifg_network(network_type='temp_baseline', temporal_baseline=30)

# Coordinate systems
troposphere_processing.create_radar_coordinates()
troposphere_processing.create_dem_coordinates(dem_type=dem_type, lon_resolution=6)

# Data processing
troposphere_processing.download_external_dem(dem_type=dem_type)
troposphere_processing.geocoding(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)
troposphere_processing.geometric_coregistration_resampling(polarisation)

# Multilooking
land_ice_processing.prepare_multilooking_grid(polarisation[0])
land_ice_processing.create_calibrated_amplitude_multilooked(polarisation)
land_ice_processing.create_interferogram_multilooked(polarisation)
land_ice_processing.create_coherence_multilooked(polarisation)
# land_ice_processing.create_unwrapped_images(polarisation)

# AASR calculation
land_ice_processing.calc_AASR_amplitude_multilooked(polarisation, amb_no=2, gaussian_spread=1, kernel_size=5)
land_ice_processing.create_output_tiffs_AASR()

# Calculate geometry
land_ice_processing.create_geometry_mulitlooked(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)

# Create the geotiffs
land_ice_processing.create_output_tiffs_amplitude()
land_ice_processing.create_output_tiffs_coherence_ifg()
land_ice_processing.create_output_tiffs_geometry()

for dlat, dlon in zip([0.01, 0.005, 0.0025], [0.01, 0.005, 0.0025]):
    troposphere_processing.create_ml_coordinates(coor_type='geographic', dlat=dlat, dlon=dlon, lat0=lat0, lon0=lon0)
    troposphere_processing.prepare_multilooking_grid(polarisation[0])
    troposphere_processing.create_calibrated_amplitude_multilooked(polarisation)
    troposphere_processing.create_interferogram_multilooked(polarisation)
    troposphere_processing.create_coherence_multilooked(polarisation)
    troposphere_processing.create_unwrapped_images(polarisation)
    troposphere_processing.create_geometry_mulitlooked(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding))

    # Create the geotiffs
    troposphere_processing.create_output_tiffs_coherence_ifg()
    troposphere_processing.create_output_tiffs_geometry()

