# Example to run a basic script to obtain multilooked calibrated amplitude images over oceans.
import datetime

from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinel
from rippl.processing_templates.general_sentinel_1 import GeneralPipelines

# Create list of dates. For the dates we search for Sentinel images in a time window around that date.
dates = [datetime.datetime(year=2020, month=6, day=19, hour=8, minute=5, second=19),
         datetime.datetime(year=2020, month=4, day=30, hour=8, minute=1, second=28),
         datetime.datetime(year=2020, month=4, day=18, hour=8, minute=12, second=43),
         datetime.datetime(year=2020, month=4, day=4, hour=7, minute=35, second=5),
         datetime.datetime(year=2020, month=4, day=3, hour=8, minute=1, second=24),
         datetime.datetime(year=2020, month=3, day=30, hour=8, minute=5, second=10),
         datetime.datetime(year=2020, month=3, day=19, hour=7, minute=50, second=7),
         datetime.datetime(year=2020, month=3, day=4, hour=7, minute=38, second=49),
         datetime.datetime(year=2020, month=3, day=3, hour=8, minute=5, second=9),
         datetime.datetime(year=2020, month=2, day=25, hour=7, minute=46, second=20),
         datetime.datetime(year=2020, month=2, day=21, hour=7, minute=50, second=5),
         datetime.datetime(year=2020, month=2, day=19, hour=7, minute=41, second=15),
         datetime.datetime(year=2020, month=2, day=12, hour=7, minute=22, second=33),
         datetime.datetime(year=2020, month=1, day=4, hour=7, minute=33, second=45),
         datetime.datetime(year=2019, month=12, day=22, hour=7, minute=31, second=18),
         datetime.datetime(year=2019, month=12, day=20, hour=7, minute=22, second=30),
         datetime.datetime(year=2019, month=11, day=28, hour=7, minute=53, second=53),
         datetime.datetime(year=2019, month=11, day=2, hour=7, minute=27, second=31),
         datetime.datetime(year=2019, month=9, day=26, hour=21, minute=24, second=12),
         datetime.datetime(year=2019, month=9, day=26, hour=7, minute=26, second=18),
         datetime.datetime(year=2019, month=9, day=8, hour=7, minute=53, second=51),
         datetime.datetime(year=2019, month=9, day=7, hour=7, minute=18, second=47),
         datetime.datetime(year=2019, month=9, day=3, hour=7, minute=22, second=31),
         datetime.datetime(year=2019, month=8, day=21, hour=7, minute=59, second=56),
         datetime.datetime(year=2019, month=8, day=15, hour=8, minute=16, second=25),
         datetime.datetime(year=2019, month=8, day=10, hour=7, minute=45, second=0),
         datetime.datetime(year=2019, month=8, day=7, hour=7, minute=22, second=34),
         datetime.datetime(year=2019, month=7, day=11, hour=7, minute=22, second=36),
         datetime.datetime(year=2019, month=6, day=17, hour=7, minute=45, second=2),
         datetime.datetime(year=2019, month=6, day=15, hour=7, minute=57, second=35),
         datetime.datetime(year=2019, month=6, day=14, hour=7, minute=22, second=36)]
time_window = datetime.timedelta(hours=12)

# Select the region of interes (last pair of coordinates should be the same as the first)
Malvina_shape = [(-97, 30.5), (-97, 26), (-85, 26), (-85, 30.5), (-97, 30.5)]
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
selected_tracks = [63]
selected_master_dates = [datetime.datetime(year=2019, month=8, day=21)]
stack_names = ['GoM_track_63']

geographic = False

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
