import datetime
import sys, getopt
import argparse

sys.path.extend(['/Users/gertmulder/software/rippl_main'])
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinel
from rippl.processing_templates.general_sentinel_1 import GeneralPipelines

if __name__ == '__main__':
    # First we read the input start and end date for the processing
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", help="Start date of processing yyyymmdd")
    parser.add_argument("-e", "--end_date", help="End date of processing yyyymmdd")

    args = parser.parse_args()
    start_date = datetime.datetime.strptime(args.start_date, '%Y%m%d')
    print('start date is ' + str(start_date.date()))
    end_date = datetime.datetime.strptime(args.end_date, '%Y%m%d')
    print('start date is ' + str(end_date.date()))

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
    study_area_shape = study_area.shape.buffer(0.2)

    # Track and data type of Sentinel data
    mode = 'IW'
    product_type = 'SLC'
    polarisation = ['VV', 'VH']

    # Create the list of the 4 different stacks.
    track_no = 37
    stack_name = 'Benelux_track_37'
    # For every track we have to select a master date. This is based on the search results earlier.
    # Choose the date with the lowest coverage to create an image with only the overlapping parts.
    master_date = datetime.datetime(year=2020, month=4, day=27)

    # Uncomment for testing
    # find_track = DownloadSentinel(start_date=start_date, end_date=end_date, shape=study_area.shape, sensor_mode=mode, polarisation=polarisation)
    # find_track.sentinel_search_ASF()
    # find_track.summarize_search_results(plot_cartopy=True, buffer=2)

    # Number of processes for parallel processing. Make sure that for every process at least 2GB of RAM is available
    no_processes = 1
    s1_processing = GeneralPipelines(processes=no_processes)
    s1_processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track_no,
                                               polarisation=polarisation, shapefile=study_area_shape, data=True, source='ASF')
