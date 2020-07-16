# Example to run a basic script to obtain multilooked calibrated amplitude images over oceans.
import datetime

from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinel
from rippl.processing_templates.general_sentinel_1 import GeneralPipelines

# Create list of dates. For the dates we search for Sentinel images in a time window around that date.
dates = [datetime.datetime(year=2019, month=6, day=14, hour=7, minute=22, second=36),
         datetime.datetime(year=2019, month=6, day=9, hour=7, minute=57, second=35)]
time_window = datetime.timedelta(hours=12)

# Select the region of interes (last pair of coordinates should be the same as the first)
Malvina_shape = [(19.5, -33), (19.5, -37), (27, -37), (27, -33), (19.5, -33)]
study_area = ReadWriteShapes()
study_area(Malvina_shape)

# Search for Sentinel images
polarisation = ['VV', 'VH']
mode = 'IW'
product_type = 'SLC'

find_track = DownloadSentinel(dates=dates, time_window=time_window, shape=study_area.shape, sensor_mode=mode, polarisation=polarisation)
find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=False, buffer=2)

################################################################################################################
# Based on the results from the line before you can now select the stack results.
selected_tracks = [58]
selected_master_dates = [datetime.datetime(year=2019, month=6, day=9)]
stack_names = ['Malvina_track_58']

geographic = True

# Initialize stacks
for track_no, master_date, stack_name in zip(selected_tracks, selected_master_dates, stack_names):

    # Number of processes for parallel processing. Make sure that for every process at least 3GB of RAM is available
    no_processes = 1

    s1_processing = GeneralPipelines(processes=no_processes)
    s1_processing.download_sentinel_data(dates=dates, time_window=time_window, track=track_no,
                                         polarisation=polarisation, shapefile=study_area.shape, data=True)
    s1_processing.create_sentinel_stack(dates=dates, time_window=time_window, master_date=master_date,
                                        track=track_no, stack_name=stack_name, polarisation=polarisation,
                                        shapefile=study_area.shape, mode=mode, product_type=product_type)

    # Finally load the stack itself. If you want to skip the download step later, run this line before other steps!
    s1_processing.read_stack(dates=dates, time_window=time_window, stack_name=stack_name)

    # Create radar coordinates for the full stack.
    s1_processing.create_radar_coordinates()

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

    # Run images for stacks of interest
    s1_processing.create_calibrated_amplitude_approx_multilooked(polarisation=polarisation)
    s1_processing.create_output_tiffs_amplitude()

    # Create lat/lon/incidence angle/DEM for multilooked grid.
    s1_processing.create_dem_coordinates(dem_type='SRTM3')
    s1_processing.download_external_dem(buffer=1, rounding=1)
    s1_processing.create_geometry_mulitlooked(lon_resolution=3, dem_buffer=0.1, dem_rounding=0.1, dem_type='SRTM3')
    s1_processing.create_output_tiffs_geometry()
