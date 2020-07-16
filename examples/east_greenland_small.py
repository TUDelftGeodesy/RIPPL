from rippl.processing_templates.general_sentinel_1 import GeneralPipelines
from rippl.processing_templates.land_ice import LandIce

# Settings where the data is stored
shapefile = 'East_Greenland_Mini.shp'
stack_name = 'east_greenland_mini'

# Track and data type of Sentinel data
track = 141
mode = 'IW'
product_type = 'SLC'
polarisation = ['HH', 'HV']       # Possibly add HV later on

# Start, master and end date of processing
start_date = '2018-02-01'
end_date = '2018-02-10'
master_date = '2018-02-08'

# DEM type
dem_type = 'TanDEM-X'
dem_buffer = 1
dem_rounding = 1
lon_resolution = 6

# Multilooking coordinates
dlat = 0.001
dlon = 0.001

land_ice_processing = LandIce(processes=1)

create_stack = False
load_stack = True
coreg = False
multilook = False
AASR = False
geotiff = False
plot = True

# Download and create the dataset
if create_stack:
    land_ice_processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track,
                                               polarisation=polarisation, shapefile=shapefile, data=False)
    land_ice_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, master_date=master_date,
                                              track=track,stack_name=stack_name, polarisation=polarisation,
                                              shapefile=shapefile, mode=mode, product_type=product_type)

# Load stack
if load_stack:
    land_ice_processing.read_stack(start_date=start_date, end_date=end_date, stack_name=stack_name)
    land_ice_processing.create_ifg_network(network_type='temp_baseline', temporal_baseline=30)

    # Coordinate systems
    land_ice_processing.create_radar_coordinates()
    land_ice_processing.create_dem_coordinates(dem_type=dem_type, lon_resolution=6)
    land_ice_processing.create_ml_coordinates(coor_type='geographic', dlat=dlat, dlon=dlon)

# Data processing
if coreg:
    land_ice_processing.download_external_dem(dem_type=dem_type, lon_resolution=lon_resolution)
    land_ice_processing.geocoding(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)
    land_ice_processing.geometric_coregistration_resampling(polarisation)

# Multilooking
if multilook:
    land_ice_processing.prepare_multilooking_grid(polarisation[0])
    land_ice_processing.create_calibrated_amplitude_multilooked(polarisation)
    land_ice_processing.create_interferogram_multilooked(polarisation)
    land_ice_processing.create_coherence_multilooked(polarisation)
    # land_ice_processing.create_unwrapped_images(polarisation)

    # Calculate geometry
    land_ice_processing.create_geometry_mulitlooked(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)

# AASR calculation
if AASR:
    land_ice_processing.calc_AASR_multilooked(polarisation, amb_no=2, gaussian_spread=1, kernel_size=5)


# Create the geotiffs
if geotiff:
    land_ice_processing.create_output_tiffs_AASR()
    land_ice_processing.create_output_tiffs_amplitude()
    land_ice_processing.create_output_tiffs_coherence_ifg()
    land_ice_processing.create_output_tiffs_geometry()

if plot:
    land_ice_processing.create_plots_AASR(True)
    land_ice_processing.create_plots_amplitude(False)
    land_ice_processing.create_plots_coherence(False)
    land_ice_processing.create_plots_ifg(False)
