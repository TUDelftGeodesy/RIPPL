
import datetime
import sys
import shutil
import numpy as np
import os
import argparse
sys.path.extend(['/home/gert/Software/rippl_main'])

from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.processing_templates.general_sentinel_1 import GeneralPipelines


if __name__ == '__main__':
    # First we read the input start and end date for the processing
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", help="Start date of processing yyyymmdd")
    parser.add_argument("-e", "--end_date", help="End date of processing yyyymmdd")
    parser.add_argument("-c", "--cores", help="Number of cores processing")
    parser.add_argument("-t", "--temp", help="Temp directory location", default='')
    parser.add_argument("-r", "--resampling_temp",
                        help="Temp directory used for master image coordinates for resampling", default='')
    parser.add_argument("-m", "--multilooking_temp",
                        help="Temp directory used for master image coordinates for multilooking", default='')

    args = parser.parse_args()
    start_date = datetime.datetime.strptime(args.start_date, '%Y%m%d')
    print('start date is ' + str(start_date.date()))
    end_date = datetime.datetime.strptime(args.end_date, '%Y%m%d')
    print('start date is ' + str(end_date.date()))
    no_processes = int(args.cores)
    print('running code with ' + str(no_processes) + ' cores.')

    # Define temporary directories
    tmp_directory = args.temp
    resampling_tmp_directory = args.resampling_temp
    if resampling_tmp_directory == '':
        resampling_tmp_directory = tmp_directory
    ml_grid_tmp_directory = args.multilooking_temp
    if ml_grid_tmp_directory == '':
        ml_grid_tmp_directory = tmp_directory

    if not os.path.exists(tmp_directory):
        os.mkdir(tmp_directory)
    if not os.path.exists(resampling_tmp_directory):
        os.mkdir(resampling_tmp_directory)
    if not os.path.exists(ml_grid_tmp_directory):
        os.mkdir(ml_grid_tmp_directory)

    glacier_shape = [[-62.84179687499999, -66.89559561140706],
                     [-59.45800781249999, -64.68031793418314],
                     [-59.94140624999999, -63.84066844285507],
                     [-64.072265625, -64.99793920061401],
                     [-66.357421875, -66.40795547978848],
                     [-67.1484375, -67.32292427995345],
                     [-67.587890625, -68.57644086491784],
                     [-66.97265625, -69.24057862839376],
                     [-64.4677734375, -70.3335333600625],
                     [-63.58886718749999, -71.60828252210263],
                     [-63.6328125, -73.15043991163012],
                     [-65.654296875, -74.09197431391085],
                     [-66.62109375, -75.05035357407698],
                     [-62.57812500000001, -75.20824498631067],
                     [-60.29296874999999, -73.92246884621463],
                     [-59.501953125, -73.09941313082074],
                     [-59.58984374999999, -72.19524581118353],
                     [-60.380859375, -70.80136623397622],
                     [-61.34765625, -69.62651016802958],
                     [-62.22656249999999, -68.73638345287264],
                     [-63.6328125, -67.89208614070753],
                     [-62.84179687499999, -66.89559561140706]]
    study_area = ReadWriteShapes()
    study_area(glacier_shape)

    """
    After selection of the right track we can start the actual download of the images. In our case we use track 88.

    This will download our data automatically to our radar database. Additionally, it will download the precise orbit files.
    These files are created within a few weeks after the data acquisition and define the satellite orbit within a few cm
    accuracy. These orbits are necessary to accurately define the positions of the radar pixels on the ground later on
    in the processing.
    """

    # Track and data type of Sentinel data
    mode = 'IW'
    product_type = 'SLC'
    polarisation = 'HH'

    # Create the list of the 4 different stacks.
    track_no = 38
    stack_name = 'glaciers_Antarctic_Peninsula'
    # For every track we have to select a master date. This is based on the search results earlier.
    # Choose the date with the lowest coverage to create an image with only the overlapping parts.
    master_date = datetime.datetime(year=2018, month=4, day=26)

    # Stack test
    start_date = datetime.datetime(year=2017, month=10, day=1)
    end_date = datetime.datetime(year=2018, month=10, day=1)

    # Number of processes for parallel processing. Make sure that for every process at least 2GB of RAM is available
    # no_processes = 4

    s1_processing = GeneralPipelines(processes=no_processes)
    s1_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, master_date=master_date, cores=no_processes,
                                             track=track_no,stack_name=stack_name, polarisation=polarisation,
                                             shapefile=study_area.shape, mode=mode, product_type=product_type)

    # Finally load the stack itself. If you want to skip the download step later, run this line before other steps!
    s1_processing.read_stack(start_date=start_date, end_date=end_date, stack_name=stack_name)

    """
    To define the location of the radar pixels on the ground we need the terrain elevation. Although it is possible to 
    derive terrain elevation from InSAR data, our used Sentinel-1 dataset is not suitable for this purpose. Therefore, we
    download data from an external source to create a digital elevation model (DEM). In our case we use SRTM data. 

    However, to find the elevation of the SAR data grid, we have to resample the data to the radar grid first to make it
    usable. This is done in the next steps.
    """

    # Some basic settings for DEM creation.
    dem_buffer = 0.1  # Buffer around radar image where DEM data is downloaded
    dem_rounding = 0.1  # Rounding of DEM size in degrees
    dem_type = 'TanDEM-X' # DEM type of data we download (SRTM1, SRTM3 and TanDEM-X are supported)

    # Define both the coordinate system of the full radar image and imported DEM
    s1_processing.create_radar_coordinates()
    s1_processing.create_dem_coordinates(dem_type=dem_type)

    # Download external DEM
    s1_processing.download_external_dem(dem_type=dem_type, buffer=dem_buffer, rounding=dem_rounding,
                                        n_processes=no_processes)

    """

    Using the obtained elevation model the exact location of the radar pixels in cartesian (X,Y,Z) and geographic (Lat/Lon)
    can be derived. This is only done for the master or reference image. This process is referred to as geocoding.

    """

    # Geocoding of image
    s1_processing.geocoding()

    """
    The information from the geocoding can directly be used to find the location of the master grid pixels in the slave
    grid images. This process is called coregistration. Because the orbits are not exactly the same with every satellite 
    overpass but differ hundreds to a few thousand meters every overpass, the grids are slightly shifted with respect to 
    each other. These shift are referred to as the spatial baseline of the images. To correctly overlay the master and slave
    images the software coregisters and resamples to the master grid.

    To do so the following steps are done:
    1. Coregistration of slave to master image
    2. Deramping the doppler effects due to TOPs mode of Sentinel-1 satellite
    3. Resampling of slave image
    4. Reramping resampled slave image.

    Due to the different orbits of the master and slave image, the phase of the radar signal is also shifted. We do not 
    know the exact shift of the two image, but using the geometry of the two images we can estimate the shift of the phase
    between different pixels. Often this shift is split in two contributions:
    1. The flat earth phase. This phase is the shift in the case the earth was a perfect ellipsoid
    2. The topographic phase. This is the phase shift due to the topography on the ground.
    In our processing these two corrections are done in one go.
    """

    # Next step applies resampling and phase correction in one step.
    # Polarisation

    # Because with the geometric coregistrtation we load the X,Y,Z files of the main image for every calculation it can
    # be beneficial to load them to a fast temporary disk. (If enough space you should load them to memory disk)
    s1_processing.geometric_coregistration_resampling(polarisation=polarisation, output_phase_correction=True,
                                                      coreg_tmp_directory=resampling_tmp_directory,
                                                      tmp_directory=tmp_directory, baselines=False,
                                                      height_to_phase=True)
    shutil.rmtree(resampling_tmp_directory)
    os.mkdir(resampling_tmp_directory)
    shutil.rmtree(tmp_directory)
    os.mkdir(tmp_directory)

    """
    Now we can create calibrated amplitudes, interferograms and coherences.
    """

    # Load the images in blocks to temporary disk (or not if only coreg data is loaded to temp disk)
    temporal_baseline = 15
    min_timespan = temporal_baseline * 2 * 200
    # Every process can only run 1 multilooking job. Therefore, in the case of amplitude calculation the number of processes
    # is limited too the number of images loaded.
    amp_processing_efficiency = 0.5
    effective_timespan = np.maximum(no_processes * 6 * amp_processing_efficiency, min_timespan)

    no_days = datetime.timedelta(days=int(effective_timespan / 2))
    if no_days < (end_date - start_date):
        step_date = start_date
        step_dates = []
        while step_date < end_date:
            step_dates.append(step_date)
            step_date += no_days
        step_dates.append(end_date)

        start_dates = step_dates[:-2]
        end_dates = step_dates[2:]
    else:
        end_dates = [end_date]
        start_dates = [start_date]

    if not isinstance(polarisation, list):
        pol = [polarisation]
    else:
        pol = polarisation

    for start_date, end_date in zip(start_dates, end_dates):
        s1_processing.read_stack(start_date=start_date, end_date=end_date, stack_name=stack_name)
        # We split the different polarisation to limit the number of files in the temporary folder.
        for p in pol:
            for dx, dy in zip([50, 100, 200, 500, 1000], [50, 100, 200, 500, 1000]):
                # The actual creation of the calibrated amplitude images
                s1_processing.create_ml_coordinates(standard_type='oblique_mercator', dx=dx, dy=dy, buffer=0,
                                                    rounding=0)
                s1_processing.prepare_multilooking_grid(polarisation)
                s1_processing.create_calibrated_amplitude_multilooked(polarisation,
                                                                      coreg_tmp_directory=ml_grid_tmp_directory,
                                                                      tmp_directory=tmp_directory)
                s1_processing.create_output_tiffs_amplitude()

                s1_processing.create_ifg_network(temporal_baseline=temporal_baseline)
                s1_processing.create_interferogram_multilooked(polarisation,
                                                               coreg_tmp_directory=ml_grid_tmp_directory,
                                                               tmp_directory=tmp_directory)
                s1_processing.create_coherence_multilooked(polarisation, coreg_tmp_directory=ml_grid_tmp_directory,
                                                           tmp_directory=tmp_directory)

                # Create output geotiffs
                s1_processing.create_output_tiffs_coherence_ifg()

                # Create lat/lon/incidence angle/DEM for multilooked grid.
                s1_processing.create_geometry_mulitlooked(baselines=True, height_to_phase=True)
                s1_processing.create_output_tiffs_geometry()

                # The coreg temp directory will only contain the loaded input lines/pixels to do the multilooking. These
                # files will be called by every process so it can be usefull to load them in memory the whole time.
                # If not given, these files will be loaded in the regular tmp folder.
                if resampling_tmp_directory:
                    if os.path.exists(ml_grid_tmp_directory):
                        shutil.rmtree(ml_grid_tmp_directory)
                        os.mkdir(ml_grid_tmp_directory)

            if tmp_directory:
                if os.path.exists(tmp_directory):
                    shutil.rmtree(tmp_directory)
                    os.mkdir(tmp_directory)
