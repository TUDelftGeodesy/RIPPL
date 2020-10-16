# Import script to embed geolocation.
import datetime
import sys
import shutil
import numpy as np
import os

sys.path.extend(['/Users/gertmulder/software/rippl_main'])
import getopt
import argparse
import rippl

from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinel
from rippl.processing_templates.general_sentinel_1 import GeneralPipelines

if __name__ == '__main__':
    # First we read the input start and end date for the processing
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", help="Start date of processing yyyymmdd")
    parser.add_argument("-e", "--end_date", help="End date of processing yyyymmdd")
    parser.add_argument("-c", "--cores", help="Number of cores processing")

    args = parser.parse_args()
    start_date = datetime.datetime.strptime(args.start_date, '%Y%m%d')
    print('start date is ' + str(start_date.date()))
    end_date = datetime.datetime.strptime(args.end_date, '%Y%m%d')
    print('start date is ' + str(end_date.date()))
    no_processes = int(args.cores)
    print('running code with ' + str(no_processes) + ' cores.')

    Benelux_shape = [[7.218017578125001, 53.27178347923819],
                     [7.00927734375, 53.45534913802113],
                     [6.932373046875, 53.72921671251272],
                     [6.756591796875, 53.68369534495075],
                     [6.1962890625, 53.57293832648609],
                     [5.218505859375, 53.50111704294316],
                     [4.713134765624999, 53.20603255157844],
                     [4.5703125, 52.80940281068805],
                     [4.2626953125, 52.288322586002984],
                     [3.856201171875, 51.88327296443745],
                     [3.3508300781249996, 51.60437164681676],
                     [3.284912109375, 51.41291212935532],
                     [2.39501953125, 51.103521942404186],
                     [2.515869140625, 50.78510168548186],
                     [3.18603515625, 50.5064398321055],
                     [3.8452148437499996, 50.127621728300475],
                     [4.493408203125, 49.809631563563094],
                     [5.361328125, 49.475263243037986],
                     [6.35009765625, 49.36806633482156],
                     [6.602783203124999, 49.6462914122132],
                     [6.536865234375, 49.83798245308484],
                     [6.251220703125, 50.085344397538876],
                     [6.448974609375, 50.42251884281916],
                     [6.218261718749999, 50.75035931136963],
                     [6.13037109375, 51.034485632974125],
                     [6.2841796875, 51.32374658474385],
                     [6.218261718749999, 51.59754765771458],
                     [6.2841796875, 51.754240074033525],
                     [6.767578125, 51.896833883012484],
                     [7.086181640625, 52.17393169256849],
                     [7.0751953125, 52.482780222078226],
                     [6.844482421875, 52.482780222078226],
                     [6.83349609375, 52.5897007687178],
                     [7.0751953125, 52.6030475337285],
                     [7.218017578125001, 53.27178347923819]]
    study_area = ReadWriteShapes()
    study_area(Benelux_shape)

    geojson = study_area.shape

    # Try to do the same by creating a shapefile with QGIS, geojson online or a .kml file in google earth.
    # study_area.read_kml(kml_path)
    # study_area.read_geo_json(geojson_path)
    # study_area.read_shapefile(shapefile_path)

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
    polarisation = 'VV'

    from rippl.processing_templates.general_sentinel_1 import GeneralPipelines

    # Create the list of the 4 different stacks.
    track_no = 37
    stack_name = 'Benelux_track_37'
    tmp_directory = '/tmp/gertmulder'
    if not os.path.exists(tmp_directory):
        os.mkdir(tmp_directory)
    coreg_tmp_directory = '/tmp/gertmulder'
    if not os.path.exists(coreg_tmp_directory):
        os.mkdir(coreg_tmp_directory)
    ml_grid_tmp_directory = '/dev/shm/gertmulder'
    if not os.path.exists(ml_grid_tmp_directory):
        os.mkdir(ml_grid_tmp_directory)

    # For every track we have to select a master date. This is based on the search results earlier.
    # Choose the date with the lowest coverage to create an image with only the overlapping parts.
    # start_date = datetime.datetime(year=2016, month=3, day=3)
    # end_date = datetime.datetime(year=2016, month=4, day=18)

    master_date = datetime.datetime(year=2020, month=3, day=28)

    # Number of processes for parallel processing. Make sure that for every process at least 2GB of RAM is available
    # no_processes = 4

    s1_processing = GeneralPipelines(processes=no_processes)
    s1_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, master_date=master_date,
                                        cores=no_processes,
                                        track=track_no, stack_name=stack_name, polarisation=polarisation,
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
    dem_type = 'SRTM1'  # DEM type of data we download (SRTM1, SRTM3 and TanDEM-X are supported)

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
                                                      coreg_tmp_directory=coreg_tmp_directory,
                                                      tmp_directory=tmp_directory, baselines=False,
                                                      height_to_phase=True)
    shutil.rmtree(coreg_tmp_directory)
    os.mkdir(coreg_tmp_directory)
    shutil.rmtree(tmp_directory)
    os.mkdir(tmp_directory)

    """
    Now we can create calibrated amplitudes, interferograms and coherences.
    """

    # Load the images in blocks to temporary disk (or not if only coreg data is loaded to temp disk)
    temporal_baseline = 30
    min_timespan = temporal_baseline * 2
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
            for dx, dy in zip([50, 100, 200, 500, 1000, 2000], [50, 100, 200, 500, 1000, 2000]):
                # The actual creation of the calibrated amplitude images
                s1_processing.create_ml_coordinates(standard_type='oblique_mercator', dx=dx, dy=dy, buffer=0,
                                                    rounding=0)
                s1_processing.prepare_multilooking_grid(polarisation)
                s1_processing.create_calibrated_amplitude_multilooked(polarisation,
                                                                      coreg_tmp_directory=coreg_tmp_directory,
                                                                      tmp_directory=tmp_directory)
                s1_processing.create_output_tiffs_amplitude()

                s1_processing.create_ifg_network(temporal_baseline=temporal_baseline)
                s1_processing.create_interferogram_multilooked(polarisation, coreg_tmp_directory=ml_grid_tmp_directory,
                                                               tmp_directory=tmp_directory)
                s1_processing.create_coherence_multilooked(polarisation, coreg_tmp_directory=ml_grid_tmp_directory,
                                                           tmp_directory=tmp_directory)

                # Create output geotiffs
                s1_processing.create_output_tiffs_coherence_ifg()

                # Create lat/lon/incidence angle/DEM for multilooked grid.
                s1_processing.create_geometry_mulitlooked(baselines=True, height_to_phase=True)
                s1_processing.create_output_tiffs_geometry()

                # Do unwrapping
                if dx in [500, 1000, 2000]:
                    s1_processing.create_unwrapped_images(polarisation)
                    s1_processing.create_output_tiffs_unwrap()

                # The coreg temp directory will only contain the loaded input lines/pixels to do the multilooking. These
                # files will be called by every process so it can be usefull to load them in memory the whole time.
                # If not given, these files will be loaded in the regular tmp folder.
                if coreg_tmp_directory:
                    if os.path.exists(ml_grid_tmp_directory):
                        shutil.rmtree(ml_grid_tmp_directory)
                        os.mkdir(ml_grid_tmp_directory)

            for dx, dy in zip([50, 100, 200, 500, 1000, 2000], [50, 100, 200, 500, 1000, 2000]):
                # The actual creation of the calibrated amplitude images
                s1_processing.create_ml_coordinates(standard_type='UTM', dx=dx, dy=dy, buffer=0, rounding=0)
                s1_processing.prepare_multilooking_grid(polarisation)
                s1_processing.create_calibrated_amplitude_multilooked(polarisation,
                                                                      coreg_tmp_directory=coreg_tmp_directory,
                                                                      tmp_directory=tmp_directory)
                s1_processing.create_output_tiffs_amplitude()

                s1_processing.create_ifg_network(temporal_baseline=temporal_baseline)
                s1_processing.create_interferogram_multilooked(polarisation, coreg_tmp_directory=ml_grid_tmp_directory,
                                                               tmp_directory=tmp_directory)
                s1_processing.create_coherence_multilooked(polarisation, coreg_tmp_directory=ml_grid_tmp_directory,
                                                           tmp_directory=tmp_directory)

                # Create output geotiffs
                s1_processing.create_output_tiffs_coherence_ifg()

                # Create lat/lon/incidence angle/DEM for multilooked grid.
                s1_processing.create_geometry_mulitlooked(baselines=True, height_to_phase=True)
                s1_processing.create_output_tiffs_geometry()

                # Do unwrapping
                if dx in [500, 1000, 2000]:
                    s1_processing.create_unwrapped_images(polarisation)
                    s1_processing.create_output_tiffs_unwrap()

                # The coreg temp directory will only contain the loaded input lines/pixels to do the multilooking. These
                # files will be called by every process so it can be usefull to load them in memory the whole time.
                # If not given, these files will be loaded in the regular tmp folder.
                if coreg_tmp_directory:
                    if os.path.exists(ml_grid_tmp_directory):
                        shutil.rmtree(ml_grid_tmp_directory)
                        os.mkdir(ml_grid_tmp_directory)

            for dlat, dlon in zip([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02], [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]):
                # The actual creation of the calibrated amplitude images
                s1_processing.create_ml_coordinates(dlat=dlat, dlon=dlon, coor_type='geographic', buffer=0, rounding=0)
                s1_processing.prepare_multilooking_grid(polarisation)
                s1_processing.create_calibrated_amplitude_multilooked(polarisation,
                                                                      coreg_tmp_directory=coreg_tmp_directory,
                                                                      tmp_directory=tmp_directory)
                s1_processing.create_output_tiffs_amplitude()

                s1_processing.create_ifg_network(temporal_baseline=temporal_baseline)
                s1_processing.create_interferogram_multilooked(polarisation, coreg_tmp_directory=ml_grid_tmp_directory,
                                                               tmp_directory=tmp_directory)
                s1_processing.create_coherence_multilooked(polarisation, coreg_tmp_directory=ml_grid_tmp_directory,
                                                           tmp_directory=tmp_directory)

                # Create output geotiffs
                s1_processing.create_output_tiffs_coherence_ifg()

                # Create lat/lon/incidence angle/DEM for multilooked grid.
                s1_processing.create_geometry_mulitlooked(baselines=True, height_to_phase=True)
                s1_processing.create_output_tiffs_geometry()

                if dlat in [0.005, 0.01, 0.02]:
                    s1_processing.create_unwrapped_images(polarisation)
                    s1_processing.create_output_tiffs_unwrap()

                # The coreg temp directory will only contain the loaded input lines/pixels to do the multilooking. These
                # files will be called by every process so it can be usefull to load them in memory the whole time.
                # If not given, these files will be loaded in the regular tmp folder.
                if coreg_tmp_directory:
                    if os.path.exists(ml_grid_tmp_directory):
                        shutil.rmtree(ml_grid_tmp_directory)
                        os.mkdir(ml_grid_tmp_directory)

            if tmp_directory:
                if os.path.exists(tmp_directory):
                    shutil.rmtree(tmp_directory)
                    os.mkdir(tmp_directory)
