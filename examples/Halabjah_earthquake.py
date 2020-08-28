# This example will show the InSAR images for the Halabjah earthquake in 2017, by selecting downloading and processing
# the data for different stacks.

import datetime

from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinel
from rippl.processing_templates.general_sentinel_1 import GeneralPipelines

# First define the area of interest
Halabja_shape = [(45.2, 34.2), (45.2, 35.4), (46.5, 35.4), (46.5, 34.2), (45.2, 34.2)]
study_area = ReadWriteShapes()
study_area(Halabja_shape)

# Track and data type of Sentinel data
mode = 'IW'
product_type = 'SLC'
polarisation = 'VV'

# First we check using a time window
earthquake_date = datetime.datetime(year=2017, month=11, day=12, hour=18)
time_window = datetime.timedelta(days=8)

find_track = DownloadSentinel(date=earthquake_date, time_window=time_window, shape=study_area.shape, sensor_mode=mode, polarisation=polarisation)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

# But using a start and end date will result in a similar selection
start_date = datetime.datetime(year=2017, month=11, day=6)
end_date = datetime.datetime(year=2017, month=11, day=18)

find_track = DownloadSentinel(date=earthquake_date, time_window=time_window, shape=study_area.shape, sensor_mode=mode, polarisation=polarisation)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

# From that we can select and check for a longer time-series for one track
# In this case we search for a few image half a year before and half a year afterwards, to evaluate rebound effects.
dates = [datetime.datetime(year=2017, month=5, day=12, hour=18),
         datetime.datetime(year=2017, month=11, day=12, hour=18),
         datetime.datetime(year=2018, month=5, day=12, hour=18)]
time_window = datetime.timedelta(days=6)

find_track = DownloadSentinel(dates=dates, time_window=time_window, shape=study_area.shape, sensor_mode=mode, polarisation=polarisation)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

# Same thing but now different start and end dates.
start_dates = [datetime.datetime(year=2017, month=5, day=6, hour=18),
               datetime.datetime(year=2017, month=11, day=6, hour=18),
               datetime.datetime(year=2018, month=5, day=6, hour=18)]
end_dates = [datetime.datetime(year=2017, month=5, day=18, hour=18),
             datetime.datetime(year=2017, month=11, day=18, hour=18),
             datetime.datetime(year=2018, month=5, day=18, hour=18)]

find_track = DownloadSentinel(start_dates=start_dates, end_dates=end_dates, time_window=time_window,
                              shape=study_area.shape, sensor_mode=mode, polarisation=polarisation)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

###############################################################################
# Now we process a number of tracks with just one image before and after the earthquake. (We assume that rebound effects
# are relatively small). But if we would take larger stacks we could study those.

# Create the list of the 4 different stacks.
track_nos = [6, 72, 79, 174]
stack_names = ['Halabjah_track_6', 'Halabjah_track_72', 'Halabjah_track_79', 'Halabjah_track_174']
# For every track we have to select a master date. This is based on the search results earlier.
# Choose the date with the lowest coverage to create an image with only the overlapping parts.
master_dates = [datetime.datetime(year=2017, month=11, day=7),
                datetime.datetime(year=2017, month=11, day=11),
                datetime.datetime(year=2017, month=11, day=12),
                datetime.datetime(year=2017, month=11, day=6)]

# We start by downloading.
for track_no, stack_name, master_date in zip(track_nos, stack_names, master_dates):

    s1_processing = GeneralPipelines(processes=4)

    # Download and create the dataset
    s1_processing.download_sentinel_data(date=earthquake_date, time_window=time_window, track=track_no,
                                               polarisation=polarisation, shapefile=study_area.shape, data=True)

for track_no, stack_name, master_date in zip(track_nos, stack_names, master_dates):

    s1_processing = GeneralPipelines(processes=4)

    # Create stack from downloaded data
    s1_processing.create_sentinel_stack(date=earthquake_date, time_window=time_window, master_date=master_date,
                                              track=track_no,stack_name=stack_name, polarisation=polarisation,
                                              shapefile=study_area.shape, mode=mode, product_type=product_type)

    # Load stack
    s1_processing.read_stack(date=earthquake_date, time_window=time_window, stack_name=stack_name)
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
        s1_processing.create_unwrapped_images(polarisation)

        # Calculate geometry
        s1_processing.create_geometry_mulitlooked(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)

        # Create the geotiffs
        s1_processing.create_output_tiffs_amplitude()
        s1_processing.create_output_tiffs_coherence_ifg()
        s1_processing.create_output_tiffs_geometry()
        s1_processing.create_output_tiffs_unwrap()

