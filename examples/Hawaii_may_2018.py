# This example will show the InSAR images for the Halabjah earthquake in 2017, by selecting downloading and processing
# the data for different stacks.

import datetime

from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinel
from rippl.processing_templates.general_sentinel_1 import GeneralPipelines

# First define the area of interest
Hawaii_shape = [(-155.75, 18.90), (-155.75, 20.3), (-154.75, 19.50), (-155.75, 18.90)]
study_area = ReadWriteShapes()
study_area(Hawaii_shape)

# Track and data type of Sentinel data
mode = 'IW'
product_type = 'SLC'
polarisation = 'VV'

# First we check using a time window
earthquake_date = datetime.datetime(year=2018, month=5, day=4, hour=22)
time_window = datetime.timedelta(days=12)

find_track = DownloadSentinel(date=earthquake_date, time_window=time_window, shape=study_area.shape, sensor_mode=mode,
                              polarisation=polarisation)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

# But using a start and end date will result in a similar selection
start_date = datetime.datetime(year=2018, month=4, day=22)
end_date = datetime.datetime(year=2018, month=5, day=8)

find_track = DownloadSentinel(shape=study_area.shape, sensor_mode=mode,
                              polarisation=polarisation, start_date=start_date, end_date=end_date)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

# From that we can select and check for a longer time-series for one track
# In this case we search for a few image half a year before and half a year afterwards, to evaluate rebound effects.
dates = [datetime.datetime(year=2018, month=5, day=4, hour=22),
         datetime.datetime(year=2018, month=11, day=4, hour=22),
         datetime.datetime(year=2017, month=5, day=4, hour=22)]
time_window = datetime.timedelta(days=6)

find_track = DownloadSentinel(dates=dates, time_window=time_window, shape=study_area.shape, sensor_mode=mode,
                              polarisation=polarisation)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

# Same thing but now different start and end dates.
start_dates = [datetime.datetime(year=2018, month=4, day=28, hour=22),
               datetime.datetime(year=2018, month=10, day=28, hour=22),
               datetime.datetime(year=2017, month=4, day=28, hour=22)]
end_dates = [datetime.datetime(year=2018, month=5, day=10, hour=22),
             datetime.datetime(year=2018, month=11, day=10, hour=22),
             datetime.datetime(year=2017, month=5, day=10, hour=22)]

find_track = DownloadSentinel(start_dates=start_dates, end_dates=end_dates, time_window=time_window,
                              shape=study_area.shape, sensor_mode=mode, polarisation=polarisation)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

###############################################################################
# Now we process a number of tracks with just one image before and after the earthquake. (We assume that rebound effects
# are relatively small). But if we would take larger stacks we could study those.

# Create the list of the 4 different stacks.
track_no = 87
stack_name = 'Hawaii_may_2018_descending'
# For every track we have to select a master date. This is based on the search results earlier.
# Choose the date with the lowest coverage to create an image with only the overlapping parts.
start_date = datetime.datetime(year=2018, month=4, day=22)
end_date = datetime.datetime(year=2018, month=5, day=8)
master_date = datetime.datetime(year=2018, month=5, day=5)

s1_processing = GeneralPipelines(processes=4)

# Download and create the dataset
s1_processing.download_sentinel_data(date=earthquake_date, time_window=time_window, track=track_no,
                                     start_date=start_date, end_date=end_date,
                                     polarisation=polarisation, shapefile=study_area.shape, data=True)


s1_processing = GeneralPipelines(processes=4)

# Create stack from downloaded data
s1_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, master_date=master_date,
                                    track=track_no, stack_name=stack_name, polarisation=polarisation,
                                    shapefile=study_area.shape, mode=mode, product_type=product_type)

# Load stack
s1_processing.read_stack(start_date=start_date, end_date=end_date, stack_name=stack_name)
s1_processing.create_ifg_network(network_type='temp_baseline', temporal_baseline=200)

# Coordinate systems
s1_processing.create_radar_coordinates()

# DEM type (We choose SRTM as we are within 60 latitude boundaries
dem_type = 'SRTM3'
dem_buffer = 1
dem_rounding = 1
s1_processing.create_dem_coordinates(dem_type=dem_type)

# Data processing
s1_processing.download_external_dem(dem_type=dem_type)
s1_processing.geocoding(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)
s1_processing.geometric_coregistration_resampling(polarisation)

for geographic in [True, False]:

    # Define multilooked image
    if geographic:
        # Resolution of output georeferenced grid
        dlat = 0.001
        dlon = 0.001
        s1_processing.create_ml_coordinates(standard_type='geographic', dlat=dlat, dlon=dlon, buffer=0, rounding=0)
    else:
        # Otherwise we create an oblique mercator grid which respects the azimuth and range directions and generates a grid
        # with equal pixel size in square meters (for degrees this can vary)
        dx = 100
        dy = 100
        s1_processing.create_ml_coordinates(standard_type='oblique_mercator', dx=dx, dy=dy, buffer=0, rounding=0)

    # Multilooking
    s1_processing.prepare_multilooking_grid(polarisation)
    s1_processing.create_calibrated_amplitude_multilooked(polarisation)

    # Interferogram, coherence and unwrapping
    s1_processing.create_interferogram_multilooked(polarisation)
    s1_processing.create_coherence_multilooked(polarisation)

    # Calculate geometry
    s1_processing.create_geometry_mulitlooked(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)

    # Create the geotiffs
    s1_processing.create_output_tiffs_amplitude()
    s1_processing.create_output_tiffs_coherence_ifg()
    s1_processing.create_output_tiffs_geometry()

    # Finally do the unwrapping and save as geotiff
    s1_processing.create_unwrapped_images(polarisation)
    s1_processing.create_output_tiffs_unwrap()
