'''
This function connects to the sentinel database and creates a stack from there.
It can be used to create a whole new stack or update an old stack. This also includes the extension to a larger area
if needed.

The code works in the following way.
1. It reads the database (using sentinel_image_database.py) for a specific region, time span and polarisation
2. From there it reads or creates a reference image slice oversight list. This list contains all the slices.
3. Using this list it searches for new secondary slice files overlapping with the reference slices.
4. From the reference and slice lists a data_stack is created or extended. (Existing files/folders are not overwritten!)
'''

import datetime
import os
import zipfile
from random import shuffle
import logging

import numpy as np
from multiprocessing import get_context
from lxml import etree
from shapely.geometry import Polygon
import copy

from rippl.SAR_sensors.sentinel.sentinel_image_database import SentinelDatabase
from rippl.SAR_sensors.sentinel.sentinel_orbit_database import SentinelOrbitsDatabase
from rippl.SAR_sensors.sentinel.sentinel_read_data import write_sentinel_burst
from rippl.SAR_sensors.sentinel.sentinel_read_metadata import CreateSwathXmlRes
from rippl.SAR_sensors.sentinel.sentinel_image_download import DownloadSentinel
from rippl.SAR_sensors.sentinel.sentinel_orbit_download import DownloadSentinelOrbit
from rippl.SAR_sensors.sentinel.sentinel_burst_id import SentinelBurstId
from rippl.meta_data.stack import Stack
from rippl.user_settings import UserSettings
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

"""
# Test
import datetime
import numpy as np

import rippl
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.processing_templates.InSAR_processing import InSAR_Processing

Benelux_shape = [[7.218017578125001, 53.27178347923819],
                 [5.218505859375, 53.50111704294316],
                 [4.713134765624999, 53.20603255157844],
                 [3.3508300781249996, 51.60437164681676],
                 [3.8452148437499996, 50.127621728300475],
                 [4.493408203125, 49.809631563563094],
                 [6.35009765625, 49.36806633482156],
                 [6.83349609375, 52.5897007687178],
                 [7.218017578125001, 53.27178347923819]]
study_area = ReadWriteShapes()
study_area(Benelux_shape)
shape = study_area.shape.buffer(0.2)

# Track and data type of Sentinel data
mode = 'IW'
product_type = 'SLC'
polarisation = ['VV']

# Create the list of the 4 different stacks.
track_no = 37
stack_name = 'Benelux_NWP_track_37'
no_processes = 6

# For every track we have to select a primary date. This is based on the search results earlier.
# Choose the date with the lowest coverage to create an image with only the overlapping parts.
reference_date = datetime.datetime(year=2017, month=7, day=24)
start_date = datetime.datetime(year=2017, month=7, day=16)
end_date = datetime.datetime(year=2017, month=7, day=28)
processing = InSAR_Processing(processes=no_processes, stack_name=stack_name)
processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track_no, 
                                  shapefile=study_area.shape, data=True, source='ASF')
processing.create_sentinel_stack(start_date=start_date, end_date=end_date, reference_date=reference_date,
                                    cores=no_processes, track_no=track_no, polarisation=polarisation,
                                    shapefile=study_area.shape, mode=mode, product_type=product_type)

# Finally load the stack itself. If you want to skip the download step later, run this line before other steps!
processing.read_stack(start_date=start_date, end_date=end_date)
processing.create_coverage_shp_kml_geojson()

"""


class SentinelStack(SentinelDatabase, Stack):

    def __init__(self, data_stack_folder='', data_stack_name=''):

        self.settings = UserSettings()
        self.settings.load_settings()

        if not data_stack_folder:
            if not data_stack_name:
                raise NotADirectoryError('The data_stack name is empty, so directory cannot be created!')
            data_stack_folder = os.path.join(self.settings.settings['paths']['radar_data_stacks'],
                                             self.settings.settings['path_names']['SAR']['Sentinel-1'], data_stack_name)

        # Loads the class to read from the sentinel database
        SentinelDatabase.__init__(self)
        # Loads the class to work with a doris stack
        Stack.__init__(self, data_stack_folder)

        # Orbit and reference information
        self.orbits = []

        # Swath and slice information
        self.shape = Polygon()
        self.swaths = []
        self.slices = []
        self.reference_slices = []

        # Specific information individual slices
        self.slice_id = []
        self.slice_date = []
        self.slice_datetime = []
        self.slice_azimuth_seconds = []
        self.slice_range_seconds = []

        # Availability of the reference/secondary bursts in image
        self.burst_availability = []
        self.burst_availability_dates = []

        # init burst id
        self.burst_ids = []

    def download_sentinel_data(self, start_date='', end_date='', date='', dates='', time_window='', start_dates='', end_dates=''
                               , track='', polarisation='', shapefile='', radar_database_folder=None, orbit_folder=None,
                              data=True, orbit=True, source='ASF', n_processes=4):
        """
        Creation of data_stack of Sentinel-1 images including the orbits.

        :param start_date:
        :param end_date:
        :param track:
        :param polarisation:
        :param shapefile:
        :param radar_database_folder:
        :param orbit_folder:
        :return:
        """

        if not isinstance(shapefile, Polygon):
            raise TypeError('shapefile should be a shapely Polygon!')

        if data:
            if isinstance(polarisation, str):
                polarisation = [polarisation]

            # Download data and orbit
            for pol in polarisation:
                download_data = DownloadSentinel(start_date=start_date, end_date=end_date, end_dates=end_dates,
                                                 start_dates=start_dates, time_window=time_window, date=date, dates=dates,
                                                 shape=shapefile, track=track, polarisation=pol, n_processes=n_processes)
                if source == 'ASF':
                    download_data.sentinel_search_ASF()
                    download_data.sentinel_download_ASF(radar_database_folder)
                elif source == 'Copernicus':
                    download_data.sentinel_search_ESA()
                    download_data.sentinel_download_ESA(radar_database_folder)
                else:
                    logging.info('Source should be ESA or ASF')

        # Orbits
        if data and orbit:
            # Get all relevant overpasses
            if not orbit_folder:
                orbit_folder = self.settings.settings['paths']['orbit_database']

            precise_folder = os.path.join(orbit_folder, self.settings.settings['path_names']['SAR']['Sentinel-1'], 'precise')
            download_orbit = DownloadSentinelOrbit(precise_folder=precise_folder)
            download_orbit.download_orbits(overpasses=download_data.dates)

    def create_sentinel_stack(self, database_folder="", shapefile=None, track_no=None, orbit_folder=None, start_date="",
        end_date="", date="", dates="", time_window="", reference_date="", end_dates="", start_dates="", mode="IW",
        product_type="SLC", polarisation="VV", cores=4, remove_partial_secondaries=True):

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        if not database_folder:
            database_folder = os.path.join(
                self.settings.settings['paths']['radar_database'], self.settings.settings['path_names']['SAR']['Sentinel-1']
            )
        if not isinstance(shapefile, Polygon):
            raise TypeError('shapefile should be a shapely Polygon!')
        else:
            self.shape = shapefile
        if not track_no:
            raise ValueError("Track_no is missing!")
        if not orbit_folder:
            orbit_folder = os.path.join(
                self.settings.settings['paths']['orbit_database'], self.settings.settings['path_names']['SAR']['Sentinel-1']
            )

        # Select data products
        for pol in polarisation:
            logging.info('Start creating stack for polaristation ' + pol)
            SentinelDatabase.__call__(self, database_folder=database_folder, shapefile=shapefile, track_no=track_no,
                start_date=start_date, start_dates=start_dates, end_date=end_date, end_dates=end_dates, date=date,
                dates=dates, time_window=time_window, mode=mode, product_type=product_type, polarisation=pol)
            logging.info("Selected images:")

            if len(self.selected_images) == 0:
                logging.info("No SAR images found! Aborting")
                return

            for info in self.selected_images:
                logging.info(info)
            logging.info("")

            # Read the orbit database
            precise = os.path.join(orbit_folder, "precise")
            restituted = os.path.join(orbit_folder, "restituted")
            if not os.path.exists(precise) or not os.path.exists(restituted):
                logging.info(
                    "The orbit folder and or precise/restituted sub-folder do not exist. Will use the database folder for orbits"
                )
                self.orbits = SentinelOrbitsDatabase(database_folder, database_folder)
            else:
                self.orbits = SentinelOrbitsDatabase(precise, restituted)

            # Load list of slices from these data products.
            self.load_swath_metadata(pol)
            if reference_date:
                self.reference_datetime = reference_date
            elif len(self.reference_slice_date) > 0:
                self.reference_datetime = self.reference_slice_date[0]
            else:
                logging.info("Provide a reference date to create a data_stack")
                return

            existing_reference = self.read_reference_slice_list()
            self.assign_burst_ids(reference_date=reference_date, track=track_no)
            self.harmonize_burst_ids_different_dates(remove_partial_secondaries=remove_partial_secondaries)

            # Then adjust the range and azimuth timing to create uniform values for different dates.
            self.adjust_pixel_line_selected_bursts()

            # Finally write data to data files
            self.write_data_stack(cores)
            self.write_reference_slice_list()

    def load_swath_metadata(self, polarisation):
        """

        :param polarisation:
        :return:
        """
        logging.info('Reading swath .xml files for new images:')
        loaded_images = []

        for im in self.selected_images.keys():
            image = self.selected_images[im]

            if im.split('.')[0] not in loaded_images:
                loaded_images.append(im.split('.')[0])
            else:
                logging.info('Skipping duplicate image ' + im)
                continue

            for swath in image['swath_xml']:
                if os.path.basename(swath)[12:14] == polarisation.lower():

                    if image['path'].endswith('zip'):
                        archive = zipfile.ZipFile(image['path'], 'r')
                        xml_path = swath.replace('\\', '/')
                        swath_data = etree.parse(archive.open(xml_path))
                        xml_path = os.path.join(image['path'], swath)
                    else:
                        xml_path = os.path.join(image['path'], swath)
                        swath_data = etree.parse(xml_path)

                    logging.info('Read meta_data ' + os.path.basename(xml_path))
                    try:
                        xml_meta = CreateSwathXmlRes(xml_path, swath_data)
                        xml_meta.read_xml()
                        xml_meta.burst_swath_coverage()
                        self.swaths.append(xml_meta)
                    except Exception as e:
                        logging.warning('Failed to read meta_data ' + os.path.basename(xml_path) + '. ' + str(e))
        logging.info('')

    def assign_burst_ids(self, reference_date, track, load_reference_slices=True, load_secondary_slices=True,
                          mode='IW'):

        # Load burst IDs
        self.burst_ids = SentinelBurstId(mode=mode)
        burst_ids, burst_shapes = self.burst_ids.select_bursts(track=track, aoi=self.shape, min_overlap_area=0)

        if not (isinstance(reference_date, datetime.datetime) or isinstance(reference_date, datetime.date)):
            reference_date = datetime.datetime.strptime(reference_date, '%Y%m%d')
        elif isinstance(reference_date, datetime.datetime):
            reference_date = reference_date.date()

        # Take a time buffer of two days around the reference date (We have a 6-day buffer with Sentinel-1)
        reference_start = datetime.datetime.combine(reference_date - datetime.timedelta(days=2), datetime.time()).date()
        reference_end = datetime.datetime.combine(reference_date + datetime.timedelta(days=2), datetime.time()).date()
        stack_dates = [reference_date]

        # Iterate over swath list
        for swath in self.swaths:
            # Select if it overlaps with area of interest
            if self.shape.intersects(swath.swath_coverage):
                logging.info('Reading burst information from ' + os.path.basename(swath.swath_xml))
                try:
                    slices = swath(self.orbits)
                except Exception as e:
                    logging.warning('Failed reading burst information from ' + os.path.basename(swath.swath_xml) +
                          'with error ' + e)
                    continue

                for slice, slice_coverage, slice_time in zip(
                        slices, swath.burst_coverage, swath.burst_xml_dat['azimuthTimeStart']):

                    # Get the date of the slice, to check whether it is the reference or not.
                    readfile = slice.readfiles['original']
                    az_datetime_str = readfile.json_dict['Orig_first_pixel_azimuth_time (UTC)']
                    az_datetime = datetime.datetime.strptime(az_datetime_str, '%Y-%m-%dT%H:%M:%S.%f')
                    az_date = az_datetime.date()

                    # Some additional checking in case we have images that cross the date line
                    if az_date not in stack_dates:
                        day_diffs = np.abs(np.array(stack_dates) - az_date)
                        same_overpass, = np.where(day_diffs < datetime.timedelta(days=2))
                        if len(same_overpass) > 0:
                            az_date = stack_dates[same_overpass[0]]
                        else:
                            stack_dates.append(az_date)

                    # To make sure the number of seconds is always positive we subtract one day.
                    az_time_diff = (az_datetime - datetime.datetime.combine(az_date - datetime.timedelta(days=1), datetime.time()))
                    az_seconds = az_time_diff.seconds + az_time_diff.microseconds / 1000000
                    az_date_str = az_date.strftime('%Y-%m-%d')
                    az_time_str = az_datetime.time().strftime('%H:%M:%S.%f')

                    if reference_start < az_date < reference_end:
                        # Find the burst ID of this burst
                        burst_id, burst_shape = self.burst_ids.select_bursts(track=track, aoi=slice_coverage, min_overlap_area=65)
                        burst_id = burst_id[0]

                        if burst_id in burst_ids and 'slice_' + burst_id.replace(' ', '_') not in self.reference_slice_id:
                            # Check if this time already exists
                            self.reference_slice_id.append('slice_' + burst_id.replace(' ', '_'))
                            self.reference_slice_date.append(az_date_str)
                            self.reference_slice_datetime.append(az_datetime_str)
                            self.reference_slice_azimuth_seconds.append(az_seconds)
                            self.reference_slice_range_seconds.append(readfile.json_dict['Orig_range_time_to_first_pixel (2way) (ms)'] * 1e-3)
                            self.reference_slices.append(slice)

                    elif load_secondary_slices:
                        # Check if this time already exists
                        burst_id, burst_shape = self.burst_ids.select_bursts(track=track, aoi=slice_coverage, min_overlap_area=65)
                        burst_id = burst_id[0]

                        if burst_id in burst_ids and az_datetime_str not in self.slice_datetime:
                            self.slice_id.append('slice_' + burst_id.replace(' ', '_'))
                            self.slice_date.append(az_date_str)
                            self.slice_datetime.append(az_datetime_str)
                            self.slice_azimuth_seconds.append(az_seconds)
                            self.slice_range_seconds.append(readfile.json_dict['Orig_range_time_to_first_pixel (2way) (ms)'] * 1e-3)
                            self.slices.append(slice)

    def harmonize_burst_ids_different_dates(self, remove_partial_secondaries=False):
        """

        :param remove_partial_secondaries:
        :return:
        """

        # Remove all bursts that are not covered by the reference image
        del_list_reference_coverage = []
        for n, id in enumerate(self.slice_id):
            if id not in self.reference_slice_id:
                del_list_reference_coverage.append(n)

        for id in reversed(del_list_reference_coverage):
            del self.slice_id[id]
            del self.slice_datetime[id]
            del self.slice_date[id]
            del self.slice_azimuth_seconds[id]
            del self.slice_range_seconds[id]
            del self.slices[id]

        # Remove all bursts that are part of secondary images that do not have the same coverage as the reference image.
        del_list_secondary_coverage = []

        reference_num_bursts = len(self.reference_slices)
        dates, secondaries_num_bursts = np.unique(self.slice_date, return_counts=True)
        for date, secondary_num_burst in zip(dates, secondaries_num_bursts):
            logging.info('For date ' + date + ' ' + str(secondary_num_burst) + ' bursts out of ' + str(reference_num_bursts)
                         + ' are available.')
            if reference_num_bursts > secondary_num_burst:
                del_list_secondary_coverage.extend(list(np.where(np.array(self.slice_date) == date)[0]))

        if remove_partial_secondaries:
            del_list_secondary_coverage = np.sort(del_list_secondary_coverage)
            # Remove the slices without reference equivalent.
            for id in reversed(del_list_secondary_coverage):
                del self.slice_id[id]
                del self.slice_datetime[id]
                del self.slice_date[id]
                del self.slice_azimuth_seconds[id]
                del self.slice_range_seconds[id]
                del self.slices[id]

    def write_data_stack(self, num_cores):
        """
        Write the stack to disk. Folders that are already there are not overwritten! Also, if a part of the dataset
        is already removed

        Create the crops in parallel

        :param num_cores:
        :return:
        """

        if not self.skip_reference:
            n = len(self.reference_slices)
            id_num = np.arange(n)
            id_num = [id for id, slice in zip(id_num, self.reference_slices) if slice != []]
            id = copy.deepcopy(id_num)
            shuffle(id)
            reference_dat = [[self.data_stack_folder, slice, id, date, [no, len(id_num)]]
                         for slice, id, date, no
                          in zip(np.array(self.reference_slices)[id], np.array(self.reference_slice_id)[id],
                                 np.array(self.reference_slice_date)[id], id_num)]
            if num_cores > 1:
                with get_context("spawn").Pool(processes=num_cores, maxtasksperchild=5) as pool:
                    # Process in chunks of 100
                    chunk_size = 100
                    for i in range(int(np.ceil(len(reference_dat) / chunk_size))):
                        last_dat = np.minimum((i + 1) * chunk_size, len(reference_dat))
                        logging.info('Initializing reference bursts ' + str(i*chunk_size) + ' to ' + str(last_dat) + ' from total of ' + str(len(reference_dat)))
                        res = pool.map(write_sentinel_burst, reference_dat[i*chunk_size:last_dat])
            else:
                for m_dat in reference_dat:
                    write_sentinel_burst(m_dat)

        n = len(self.slices)
        id_num = np.arange(n)
        id = np.arange(n)
        shuffle(id)
        secondary_dat = [[self.data_stack_folder, slice, id, date, [no, n]]
                     for slice, id, date, no
                     in zip(np.array(self.slices)[id], np.array(self.slice_id)[id], np.array(self.slice_date)[id], id_num)]
        if num_cores > 1:
            with get_context("spawn").Pool(processes=num_cores, maxtasksperchild=5) as pool:
                # Process in chunks of 25
                chunk_size = 100
                for i in range(int(np.ceil(len(secondary_dat) / chunk_size))):
                    last_dat = np.minimum((i + 1) * chunk_size, len(secondary_dat))
                    logging.info('Initializing secondary bursts ' + str(i * chunk_size) + ' to ' + str(last_dat) + ' from total of ' + str(len(secondary_dat)))
                    res = pool.map(write_sentinel_burst, secondary_dat[i * chunk_size:last_dat])
        else:
            for s_dat in secondary_dat:
                write_sentinel_burst(s_dat)

    def adjust_pixel_line_selected_bursts(self):
        """
        For all sets of reference and secondary bursts adjust the azimuth and range time to the lowest available in the total
        set of bursts for that date.

        :return:
        """

        # Final step is to adjust all azimuth and range values.

        # Find the lowest azimuth value
        if self.reference_slices[0]:
            # This is only needed when the reference slices are assigned for the first time
            reference_lowest_az_time = np.min(self.reference_slice_azimuth_seconds)
            reference_lowest_ra_time = np.min(self.reference_slice_range_seconds)

            for slice in self.reference_slices:
                self.adjust_pixel_line(slice, reference_lowest_az_time, reference_lowest_ra_time)
            self.skip_reference = False
        else:
            self.skip_reference = True

        # Now do the same thing for the secondary pixels.
        dates = list(set(self.slice_date))
        for date in dates:
            slice_ids = np.where(np.array(self.slice_date) == date)[0]
            lowest_az_time = np.min(np.array(self.slice_azimuth_seconds)[slice_ids])
            lowest_ra_time = np.min(np.array(self.slice_range_seconds)[slice_ids])

            for slice_id in slice_ids:
                self.adjust_pixel_line(self.slices[slice_id], lowest_az_time, lowest_ra_time)

    @staticmethod
    def adjust_pixel_line(slice, lowest_az_time, lowest_ra_time):
        """
        Adjust the first pixel and line values.

        :param slice:
        :param lowest_az_time:
        :param lowest_ra_time:
        :return:
        """

        # Replace readfile values
        slice.readfiles['original'].az_first_pix_time = lowest_az_time
        slice.readfiles['original'].ra_first_pix_time = lowest_ra_time

        # Replace coordinate values
        pol = slice.readfiles['original'].json_dict['Polarisation']
        crop_key = [key for key in slice.processes['crop'].keys() if pol in key][0]
        crop_coor = slice.processes['crop'][crop_key].coordinates  # type:CoordinateSystem

        slice.readfiles['original'].first_line = int(np.round((crop_coor.orig_az_time - lowest_az_time) / crop_coor.az_step))
        slice.readfiles['original'].first_pixel = int(np.round((crop_coor.orig_ra_time - lowest_ra_time) / crop_coor.ra_step))
        slice.readfiles['original'].center_line += int(np.round((crop_coor.orig_az_time - lowest_az_time) / crop_coor.az_step))
        slice.readfiles['original'].center_pixel += int(np.round((crop_coor.orig_ra_time - lowest_ra_time) / crop_coor.ra_step))

        crop_coor.orig_first_line = copy.copy(crop_coor.first_line)
        crop_coor.first_line += int(np.round((crop_coor.orig_az_time - lowest_az_time) / crop_coor.az_step))
        crop_coor.center_line += int(np.round((crop_coor.orig_az_time - lowest_az_time) / crop_coor.az_step))
        crop_coor.az_time = lowest_az_time
        crop_coor.orig_first_pixel = copy.copy(crop_coor.first_pixel)
        crop_coor.first_pixel += int(np.round((crop_coor.orig_ra_time - lowest_ra_time) / crop_coor.ra_step))
        crop_coor.center_pixel += int(np.round((crop_coor.orig_ra_time - lowest_ra_time) / crop_coor.ra_step))
        crop_coor.ra_time = lowest_ra_time
