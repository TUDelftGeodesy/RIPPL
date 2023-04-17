'''
This function connects to the sentinel database and creates a stack from there.
It can be used to create a whole new stack or update an old stack. This also includes the extension to a larger area
if needed.

The code works in the following way.
1. It reads the database (using sentinel_database.py) for a specific region, time span and polarisation
2. From there it reads or creates a master image slice oversight list. This list contains all the slices.
3. Using this list it searches for new slave slice files overlapping with the master slices.
4. From the master and slice lists a data_stack is created or extended. (Existing files/folders are not overwritten!)
'''

import datetime
import os
import zipfile
from random import shuffle

import numpy as np
from multiprocessing import get_context
from lxml import etree
from shapely.geometry import Polygon
from shapely import speedups
speedups.disable()
import copy

from rippl.orbit_geometry.orbit_interpolate import OrbitInterpolate
from rippl.SAR_sensors.sentinel.sentinel_database import SentinelDatabase
from rippl.SAR_sensors.sentinel.sentinel_precise_orbit import SentinelOrbitsDatabase
from rippl.SAR_sensors.sentinel.sentinel_read_data import write_sentinel_burst
from rippl.SAR_sensors.sentinel.sentinel_swath_metadata import CreateSwathXmlRes
from rippl.meta_data.stack import Stack
from rippl.user_settings import UserSettings
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class SentinelStack(SentinelDatabase, Stack):

    def __init__(self, data_stack_folder='', data_stack_name=''):

        self.settings = UserSettings()
        self.settings.load_settings()

        if not data_stack_folder:
            if not data_stack_name:
                raise NotADirectoryError('The data_stack name is empty, so directory cannot be created!')
            data_stack_folder = os.path.join(self.settings.radar_data_stacks, self.settings.sar_sensor_name['sentinel1'], data_stack_name)

        # Loads the class to read from the sentinel database
        SentinelDatabase.__init__(self)
        # Loads the class to work with a doris stack
        Stack.__init__(self, data_stack_folder)

        # Orbit and master information
        self.orbits = []

        # Swath and slice information
        self.swaths = []
        self.slices = []
        self.master_slices = []

        # Specific information individual slices
        self.slice_swath_no = []
        self.slice_lat = []
        self.slice_lon = []
        self.slice_date = []
        self.slice_az_time = []
        self.slice_time = []
        self.slice_x = []
        self.slice_y = []
        self.slice_z = []
        self.slice_seconds = []
        self.slice_range_seconds = []
        self.slice_names = []

        # Availability of the master/slave bursts in image
        self.burst_availability = []
        self.burst_availability_dates = []

    def read_from_database(
        self,
        database_folder="",
        shapefile=None,
        track_no=None,
        orbit_folder=None,
        start_date="",
        end_date="",
        date="",
        dates="",
        time_window="",
        master_date="",
        end_dates="",
        start_dates="",
        mode="IW",
        product_type="SLC",
        polarisation="VV",
        cores=4,
    ):

        if not database_folder:
            database_folder = os.path.join(
                self.settings.radar_database, self.settings.sar_sensor_name["sentinel1"]
            )
        if not isinstance(shapefile, Polygon):
            if not shapefile:
                raise FileExistsError("Shapefile entry is empty!")
            elif not os.path.exists(shapefile):
                shapefile = os.path.join(self.settings.GIS_database, shapefile)
            if not os.path.exists(shapefile):
                raise FileExistsError(
                    "Shapefile not found! Either give the full path or the filename in the GIS database folder."
                )
        if not track_no:
            raise ValueError("Track_no is missing!")
        if not orbit_folder:
            orbit_folder = os.path.join(
                self.settings.orbit_database, self.settings.sar_sensor_name["sentinel1"]
            )

        # Select data products
        SentinelDatabase.__call__(
            self,
            database_folder=database_folder,
            shapefile=shapefile,
            track_no=track_no,
            start_date=start_date,
            start_dates=start_dates,
            end_date=end_date,
            end_dates=end_dates,
            date=date,
            dates=dates,
            time_window=time_window,
            mode=mode,
            product_type=product_type,
            polarisation=polarisation,
        )
        print("Selected images:")

        if len(self.selected_images) == 0:
            print("No SAR images found! Aborting")
            return

        for info in self.selected_images:
            print(info)
        print("")

        # Read the orbit database
        precise = os.path.join(orbit_folder, "precise")
        restituted = os.path.join(orbit_folder, "restituted")
        if not os.path.exists(precise) or not os.path.exists(restituted):
            print(
                "The orbit folder and or precise/restituted sub-folder do not exist. Will use the database folder for orbits"
            )
            self.orbits = SentinelOrbitsDatabase(database_folder, database_folder)
        else:
            self.orbits = SentinelOrbitsDatabase(precise, restituted)

        # Load list of slices from these data products.
        self.extract_swath_xml_orbit(polarisation)
        if master_date:
            self.master_date = master_date
        elif len(self.master_slice_date) > 0:
            self.master_date = self.master_slice_date[0]
        else:
            print("Provide a master date to create a data_stack")
            return

        existing_master = self.read_master_slice_list()
        self.create_slice_list(self.master_date, existing_master)

        # Now assign ids to new master and slave slices
        self.assign_slice_id()

        # Then adjust the range and azimuth timing to create uniform values for different dates.
        self.adjust_pixel_line_selected_bursts()

        # Finally write data to data files
        self.update_stack(cores)
        self.write_master_slice_list()


    def extract_swath_xml_orbit(self, polarisation):

        print('Reading swath .xml files for new images:')
        loaded_images = []

        for im in self.selected_images.keys():
            image = self.selected_images[im]

            if im.split('.')[0] not in loaded_images:
                loaded_images.append(im.split('.')[0])
            else:
                print('Skipping duplicate image ' + im)
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

                    print('Read meta_data ' + os.path.basename(xml_path))
                    try:
                        xml_meta = CreateSwathXmlRes(xml_path, swath_data)
                        xml_meta.read_xml()
                        xml_meta.burst_swath_coverage()
                        self.swaths.append(xml_meta)
                    except:
                        print('Failed to read meta_data ' + os.path.basename(xml_path))
        print('')

    def create_slice_list(self, master_start, existing_master=False):
        # master date should be in yyyy-mm-dd format
        # This function creates a list of all slices within the data_stack and reads the needed meta_data.
        load_new_master_slices = not existing_master
        self.iterate_swaths(master_start, adjust_date=False, load_slave_slices=False, load_new_master_slices=load_new_master_slices)

        # Adjust timing for master if needed.
        for seconds in self.master_slice_seconds:
            if seconds < 3600 or seconds > 82800:
                self.master_date_boundary = True
        if self.master_date_boundary:
            self.master_slice_seconds = np.array(self.master_slice_seconds)
            self.master_slice_seconds[self.master_slice_seconds < 7200] += 86400
            self.master_slice_seconds = list(self.master_slice_seconds)

        if len(self.master_slice_seconds) == 0:
            raise FileNotFoundError('No master date images found. Check if you selected a valid master date! Aborting..')

        # Reload the slices and calc coordinates
        if self.master_date_boundary:
            self.iterate_swaths(master_start, adjust_date=True, load_new_master_slices=load_new_master_slices)
            self.adjust_date = True
        else:
            self.iterate_swaths(master_start, adjust_date=False, load_new_master_slices=load_new_master_slices)
            self.adjust_date = False

    def iterate_swaths(self, master_start, adjust_date=False, load_slave_slices=True, load_new_master_slices=True):

        time_lim = 0.1
        if not isinstance(master_start, datetime.datetime):
            master_start = datetime.datetime.strptime(master_start, '%Y%m%d') - datetime.timedelta(seconds=7200)
        master_end = master_start + datetime.timedelta(days=1, seconds=14400)

        # Iterate over swath list
        for swath in self.swaths:
            # Select if it overlaps with area of interest
            if self.shape.intersects(swath.swath_coverage):
                print('Reading burst information from ' + os.path.basename(swath.swath_xml))
                try:
                    slices = swath(self.orbits, adjust_date=adjust_date)

                except:
                    print('Failed reading burst information from ' + os.path.basename(swath.swath_xml))
                    continue
                interp_orbit = OrbitInterpolate(slices[0].meta.orbits[list(slices[0].meta.orbits.keys())[0]])
                interp_orbit.fit_orbit_spline(vel=False, acc=False)

                for slice, slice_coverage, slice_time, slice_coors in zip(
                        slices, swath.burst_coverage, swath.burst_xml_dat['azimuthTimeStart'], swath.burst_center_coors):

                    readfile = slice.readfiles['original']
                    az_time, az_date_str = readfile.time2seconds(readfile.json_dict['Orig_first_pixel_azimuth_time (UTC)'], adjust_date=adjust_date)
                    az_date = datetime.datetime.strptime(readfile.json_dict['Orig_first_pixel_azimuth_time (UTC)'],
                                                               '%Y-%m-%dT%H:%M:%S.%f')

                    if master_start < az_date < master_end:
                        if self.master_shape.intersects(slice_coverage):
                            # Check if this time already exists

                            if len(self.master_slice_seconds) > 0:
                                time_min = np.abs(az_time - np.asarray(self.master_slice_seconds))
                                min_id = np.argmin(time_min)

                                if time_min[min_id] < time_lim:
                                    print('Duplicate slice found. Old slice replaced with new one.')
                                    for list_dat in [self.master_slice_x, self.master_slice_y, self.master_slice_z,
                                                     self.master_slice_lat, self.master_slice_lon, self.master_slice_date,
                                                     self.master_slice_swath_no, self.master_slice_az_time, self.master_slice_seconds,
                                                     self.master_slice_time, self.master_slice_range_seconds, self.master_slices]:
                                        list_dat.pop(min_id)
                                else:
                                    if not load_new_master_slices:
                                        print('Skipping master slice because it does not exist in original dataset')
                                        continue

                            xyz = interp_orbit.evaluate_orbit_spline([az_time])[0]
                            self.master_slice_x.append(xyz[0, 0])
                            self.master_slice_y.append(xyz[1, 0])
                            self.master_slice_z.append(xyz[2, 0])
                            self.master_slice_lat.append(slice_coors[1])
                            self.master_slice_lon.append(slice_coors[0])
                            self.master_slice_date.append(az_date_str)
                            self.master_slice_swath_no.append(int(slice.readfiles['original'].json_dict['Swath']))
                            self.master_slice_az_time.append(slice_time)
                            self.master_slice_seconds.append(az_time)
                            self.master_slice_time.append(readfile.json_dict['Orig_first_pixel_azimuth_time (UTC)'])
                            self.master_slice_range_seconds.append(readfile.json_dict['Orig_range_time_to_first_pixel (2way) (ms)'] * 1e-3)

                            slice.readfiles['original'].adjust_date = adjust_date
                            self.master_slices.append(slice)

                    elif self.shape.intersects(slice_coverage) and load_slave_slices:
                        # Check if this time already exists
                        dat_id = np.where(self.slice_date == az_date_str)[0]

                        if len(dat_id) >= 1:

                            time_min = np.abs(az_time - np.asarray(self.slice_seconds)[dat_id])
                            min_id = np.argmin(time_min)

                            if time_min[min_id] < time_lim:
                                if not adjust_date:
                                    print('Duplicate slice found. Old slice replaced with new one.')
                                for list_dat in [self.slice_x, self.slice_y, self.slice_z, self.slice_lat, self.slice_lon,
                                             self.slice_date, self.slice_swath_no, self.slice_az_time, self.slice_seconds,
                                             self.slice_time, self.slice_range_seconds, self.slices]:
                                    list_dat.pop(min_id)

                        xyz = interp_orbit.evaluate_orbit_spline([az_time])[0]
                        self.slice_x.append(xyz[0, 0])
                        self.slice_y.append(xyz[1, 0])
                        self.slice_z.append(xyz[2, 0])
                        self.slice_lat.append(slice_coors[1])
                        self.slice_lon.append(slice_coors[0])
                        self.slice_date.append(az_date_str)
                        self.slice_swath_no.append(int(readfile.json_dict['Swath']))
                        self.slice_az_time.append(slice_time)
                        self.slice_seconds.append(az_time)
                        self.slice_time.append(readfile.json_dict['Orig_first_pixel_azimuth_time (UTC)'])
                        self.slice_range_seconds.append(readfile.json_dict['Orig_range_time_to_first_pixel (2way) (ms)'] * 1e-3)

                        slice.readfiles['original'].adjust_date = adjust_date
                        self.slices.append(slice)

    def adjust_pixel_line_selected_bursts(self):
        """
        For all sets of master and slave bursts adjust the azimuth and range time to the lowest available in the total
        set of bursts for that date.

        :return:
        """

        # Final step is to adjust all azimuth and range values.

        # Find lowest azimuth value
        if np.sum(np.array(self.master_slice_range_seconds) == 0) == 0:
            # This is only needed when the master slices are assigned for the first time
            master_lowest_az_time = np.min(self.master_slice_seconds)
            master_lowest_ra_time = np.min(self.master_slice_range_seconds)

            for slice in self.master_slices:
                self.adjust_pixel_line(slice, master_lowest_az_time, master_lowest_ra_time, adjust_date=self.adjust_date)
            self.skip_master = False
        else:
            self.skip_master = True

        # Now do the same thing for the slave pixels.
        dates = list(set(self.slice_date))
        for date in dates:
            slice_ids = np.where(np.array(self.slice_date) == date)[0]
            lowest_az_time = np.min(np.array(self.slice_seconds)[slice_ids])
            lowest_ra_time = np.min(np.array(self.slice_range_seconds)[slice_ids])

            for slice_id in slice_ids:
                self.adjust_pixel_line(self.slices[slice_id], lowest_az_time, lowest_ra_time, adjust_date=self.adjust_date)

    @staticmethod
    def adjust_pixel_line(slice, lowest_az_time, lowest_ra_time, adjust_date):
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

        if adjust_date:
            orig_date = datetime.datetime.strptime(slice.readfiles['original'].date, '%Y-%m-%d')
            slice.readfiles['original'].date = (orig_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

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

    def assign_slice_id(self, remove_partial_slaves=False):
        # Create new master ids if needed
        known_ids = len(self.master_slice_number)
        total_ids = len(self.master_slice_az_time)

        print('Assign found bursts to data_stack. These can be part of current data_stack already!')

        if known_ids != total_ids:
            times = self.master_slice_seconds[known_ids:]
            swaths = self.master_slice_swath_no[known_ids:]
            dates = self.master_slice_date[known_ids:]
            slice_step = 2.7596

            if known_ids == 0:
                id = np.argmin(np.array(self.master_slice_seconds))
                ref_time = self.master_slice_seconds[id]
                ref_swath = self.master_slice_swath_no[id]
                ref_num = 500
            else:
                ref_swath = self.master_slice_swath_no[0]
                ref_time = self.master_slice_seconds[0]
                ref_num = self.master_slice_number[0]

            for time, swath, date in zip(times, swaths, dates):
                swath_diff = (swath - ref_swath) / 3.0 * slice_step

                num = int(np.round(ref_num + ((time - swath_diff) - ref_time) / slice_step))
                self.master_slice_number.append(num)
                slice = 'slice_' + str(num) + '_swath_' + str(swath)
                self.master_slice_names.append(slice)
                print('Assigned master burst ' + slice + ' at ' + date)

        print('')
        # Now link the slave slices and give them an id
        master_nums = np.asarray(self.master_slice_number)
        master_xyz = np.concatenate((np.asarray(self.master_slice_x)[None, :, None],
                                     np.asarray(self.master_slice_y)[None, :, None],
                                     np.asarray(self.master_slice_z)[None, :, None]))
        slave_xyz = np.concatenate((np.asarray(self.slice_x)[None, None, :],
                                    np.asarray(self.slice_y)[None, None, :],
                                    np.asarray(self.slice_z)[None, None, :]))
        dist = np.sum((master_xyz - slave_xyz)**2, axis=0)
        min_nums = np.argmin(dist, axis=0)
        min_dists = np.sqrt(dist[min_nums, np.arange(dist.shape[1])])

        # Assign slice values
        del_list = []
        for min_dist, min_num, i, date in zip(min_dists, min_nums, np.arange(len(min_dists)), self.slice_date):

            if min_dist > 1000:
                del_list.append(i)
                continue
            self.slice_number.append(self.master_slice_number[min_num])
            name = 'slice_' + (str(self.master_slice_number[min_num]) + '_swath_' +
                               str(self.master_slice_swath_no[min_num]))
            self.slice_names.append(name)
            print('Assigned slave burst ' + name + ' at ' + date)

        # Remove the slices without master equivalent.
        for id in reversed(del_list):
            del self.slice_swath_no[id]
            del self.slice_lat[id]
            del self.slice_lon[id]
            del self.slice_date[id]
            del self.slice_az_time[id]
            del self.slice_time[id]
            del self.slice_x[id]
            del self.slice_y[id]
            del self.slice_z[id]
            del self.slices[id]
            del self.slice_seconds[id]
            del self.slice_range_seconds[id]

        # Calculate the burst availability
        s_dates = np.sort(np.unique(np.array(self.slice_date)))
        m_names = np.sort(np.array(self.master_slice_names))
        n_burst = len(self.master_slice_names)
        self.burst_availability = np.zeros((n_burst, len(s_dates))).astype(np.bool8)

        for s_name, s_date in zip(self.slice_names, self.slice_date):
            date_id = np.where((s_dates == s_date))
            slice_id = np.where((m_names == s_name))
            self.burst_availability[slice_id, date_id] = 1
            self.burst_availability_dates = s_dates

        # In case we want only full coverage remove the not fully covered images.
        if remove_partial_slaves:
            no_full_date = s_dates[np.sum(self.burst_availability, axis=0) != len(m_names)]

            for del_date in no_full_date:
                burst_ids = np.sort(np.where(self.slice_date == np.array(del_date))[0])[::-1]
                print('Removing slave bursts from date ' + del_date + ' because they are not fully covered. Use the ' +
                      ' option remove_partial_slaves=False to included these dates.')
                for id in burst_ids:
                    del self.slice_swath_no[id]
                    del self.slice_lat[id]
                    del self.slice_lon[id]
                    del self.slice_date[id]
                    del self.slice_az_time[id]
                    del self.slice_time[id]
                    del self.slice_x[id]
                    del self.slice_y[id]
                    del self.slice_z[id]
                    del self.slices[id]
                    del self.slice_seconds[id]
                    del self.slice_range_seconds[id]

    def write_master_slice_list(self):
        # az_time, yyyy-mm-ddThh:mm:ss.ssssss, swath x, slice i, x xxxx, y yyyy, z zzzz, lat ll.ll, lon ll.ll, pol pp

        l = open(os.path.join(self.data_stack_folder, 'master_slice_list'), 'w+')
        for az_time, swath_no, number, x, y, z, lat, lon in zip(
            self.master_slice_az_time, self.master_slice_swath_no, self.master_slice_number,
            self.master_slice_x, self.master_slice_y, self.master_slice_z,
            self.master_slice_lat, self.master_slice_lon):

            l.write('az_time ' + az_time +
                    ',\tswath ' + str(swath_no) +
                    ',\tslice ' + str(number) +
                    ',\tx ' + str(int(x)) +
                    ',\ty ' + str(int(y)) +
                    ',\tz ' + str(int(z)) +
                    ',\tlat ' + '{0:.2f}'.format(lat) +
                    ',\tlon ' + '{0:.2f}'.format(lon) +
                    ' \n')

        l.close()

    def update_stack(self, num_cores):
        # Write the stack to disk. Folders that are already there are not overwritten! Also, if a part of the dataset
        # is already removed

        # Create the crops in parallel

        if not self.skip_master:
            n = len(self.master_slices)
            id_num = np.arange(n)
            id_num = [id for id, slice in zip(id_num, self.master_slices) if slice != []]
            id = copy.deepcopy(id_num)
            shuffle(id)
            master_dat = [[self.data_stack_folder, slice, number, swath_no, date, [no, len(id_num)]]
                         for slice, number, swath_no, date, no
                          in zip(np.array(self.master_slices)[id], np.array(self.master_slice_number)[id],
                                 np.array(self.master_slice_swath_no)[id], np.array(self.master_slice_date)[id], id_num)]
            if num_cores > 1:
                with get_context("spawn").Pool(processes=num_cores, maxtasksperchild=5) as pool:
                    # Process in blocks of 100
                    block_size = 100
                    for i in range(int(np.ceil(len(master_dat) / block_size))):
                        last_dat = np.minimum((i + 1) * block_size, len(master_dat))
                        print('Initializing master bursts ' + str(i*block_size) + ' to ' + str(last_dat) + ' from total of ' + str(len(master_dat)))
                        res = pool.map(write_sentinel_burst, master_dat[i*block_size:last_dat])
            else:
                for m_dat in master_dat:
                    write_sentinel_burst(m_dat)

        n = len(self.slices)
        id_num = np.arange(n)
        id = np.arange(n)
        shuffle(id)
        slave_dat = [[self.data_stack_folder, slice, number, swath_no, date, [no, n]]
                     for slice, number, swath_no, date, no
                     in zip(np.array(self.slices)[id], np.array(self.slice_number)[id],
                            np.array(self.slice_swath_no)[id], np.array(self.slice_date)[id], id_num)]
        if num_cores > 1:
            with get_context("spawn").Pool(processes=num_cores, maxtasksperchild=5) as pool:
                # Process in blocks of 25
                block_size = 100
                for i in range(int(np.ceil(len(slave_dat) / block_size))):
                    last_dat = np.minimum((i + 1) * block_size, len(slave_dat))
                    print('Initializing slave bursts ' + str(i * block_size) + ' to ' + str(last_dat) + ' from total of ' + str(len(slave_dat)))
                    res = pool.map(write_sentinel_burst, slave_dat[i * block_size:last_dat])
        else:
            for s_dat in slave_dat:
                write_sentinel_burst(s_dat)
