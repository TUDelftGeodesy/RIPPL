'''
This function connects to the sentinel database and creates a stack from there.
It can be used to create a whole new stack or update an old stack. This also includes the extension to a larger area
if needed.

The code works in the following way.
1. It reads the database (using sentinel_database.py) for a specific region, time span and polarisation
2. From there it reads or creates a master image slice oversight list. This list contains all the slices.
3. Using this list it searches for new slave slice files overlapping with the master slices.
4. From the master and slice lists a datastack is created or extended. (Existing files/folders are not overwritten!)
'''

import datetime
import os
import zipfile

import numpy as np
from multiprocessing import Pool
from lxml import etree

from rippl.orbit_geometry.orbit_interpolate import OrbitInterpolate
from rippl.SAR_sensors.sentinel.sentinel_database import SentinelDatabase
from rippl.SAR_sensors.sentinel.sentinel_precise_orbit import SentinelOrbitsDatabase
from rippl.SAR_sensors.sentinel.sentinel_read_data import write_sentinel_burst
from rippl.SAR_sensors.sentinel.sentinel_swath_metadata import CreateSwathXmlRes
from rippl.meta_data.stack import Stack


class SentinelStack(SentinelDatabase, Stack):

    def __init__(self, datastack_folder):

        # Loads the class to read from the sentinel database
        SentinelDatabase.__init__(self)
        # Loads the class to work with a doris stack
        Stack.__init__(self, datastack_folder)

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
        self.slice_names = []

        # Availability of the master/slave bursts in image
        self.burst_availability = []

    def read_from_database(self, database_folder, shapefile, track_no, orbit_folder='', start_date='2010-01-01',
                 end_date='2030-01-01', master_date='', mode='IW', product_type='SLC', polarisation='VV', cores=4):

        # Select data products
        SentinelDatabase.__call__(self, database_folder, shapefile, track_no, start_date, end_date, mode, product_type, polarisation)
        print('Selected images:')

        if len(self.selected_images) == 0:
            print('No SAR images found! Aborting')
            return

        for info in self.selected_images:
            print(info)
        print('')

        # Read the orbit database
        precise = os.path.join(orbit_folder, 'precise')
        restituted = os.path.join(orbit_folder, 'restituted')
        if not os.path.exists(precise) or not os.path.exists(restituted):
            print('The orbit folder and or precise/restituted sub-folder do not exist. Will use the database folder for orbits')
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
            print('Provide a master date to create a datastack')
            return

        self.create_slice_list(self.master_date)
        self.read_master_slice_list()

        # Now assign ids to new master and slave slices
        self.assign_slice_id()

        # Finally write data to data files
        self.update_stack(cores)
        self.write_master_slice_list()

    def extract_swath_xml_orbit(self, polarisation):

        print('Reading swath .xml files for new images:')

        for im in self.selected_images.keys():
            image = self.selected_images[im]

            for swath in image['swath_xml']:
                if os.path.basename(swath)[12:14] == polarisation.lower():

                    if image['path'].endswith('zip'):
                        archive = zipfile.ZipFile(image['path'], 'r')
                        xml_path = swath
                        swath_data = etree.parse(archive.open(xml_path))
                        xml_path = os.path.join(image['path'], swath)
                    else:
                        xml_path = os.path.join(image['path'], swath)
                        swath_data = etree.parse(xml_path)

                    xml_meta = CreateSwathXmlRes(xml_path, swath_data)
                    print('Read meta_data ' + os.path.basename(xml_path))
                    xml_meta.read_xml()
                    xml_meta.burst_swath_coverage()

                    self.swaths.append(xml_meta)

        print('')

    def create_slice_list(self, master_date):
        # master date should be in yyyy-mm-dd format
        # This function creates a list of all slices within the datastack and reads the needed meta_data.

        time_lim = 0.1
        master_start = datetime.datetime.strptime(master_date, '%Y-%m-%d')
        master_end = master_start + datetime.timedelta(days=1)

        # Iterate over swath list
        for swath in self.swaths:
            # Select if it overlaps with area of interest
            if self.shape.intersects(swath.swath_coverage):
                slices = swath(self.orbits)
                interp_orbit = OrbitInterpolate(slices[0].meta.orbits[list(slices[0].meta.orbits.keys())[0]])
                interp_orbit.fit_orbit_spline(vel=False, acc=False)

                for slice, slice_coverage, slice_time, slice_coors in zip(
                        slices, swath.burst_coverage, swath.burst_xml_dat['azimuthTimeStart'], swath.burst_center_coors):

                    b_t = datetime.datetime.strptime(slice_time, '%Y-%m-%dT%H:%M:%S.%f')
                    b_time = float(b_t.hour * 3600 + b_t.minute * 60 + b_t.second) + float(b_t.microsecond) / 1000000
                    if interp_orbit.t[-1] > 86400:
                        b_time += 86400

                    if master_start < b_t < master_end:
                        if self.master_shape.intersects(slice_coverage):
                            # Check if this time already exists
                            if len(self.master_slice_seconds) > 0 and np.min(np.abs(
                                            b_time - np.asarray(self.master_slice_seconds))) < time_lim:
                                print('Duplicate slice found. New slice removed from rippl.stack')
                                continue

                            xyz = interp_orbit.evaluate_orbit_spline([b_time])[0]
                            self.master_slice_x.append(xyz[0, 0])
                            self.master_slice_y.append(xyz[1, 0])
                            self.master_slice_z.append(xyz[2, 0])
                            self.master_slice_lat.append(slice_coors[1])
                            self.master_slice_lon.append(slice_coors[0])
                            self.master_slice_date.append(slice_time[:10])
                            self.master_slice_swath_no.append(int(slice.readfiles['original'].json_dict['Swath']))
                            self.master_slice_az_time.append(slice_time)
                            self.master_slice_seconds.append(b_time)
                            self.master_slice_time.append(b_t)

                            self.master_slices.append(slice)

                    elif self.shape.intersects(slice_coverage):
                        # Check if this time already exists
                        dat_id = np.where(self.slice_date == slice_time[:10])[0]

                        if len(dat_id) >= 1:
                            if len(self.slice_seconds) > 0 and np.min(np.abs(
                                            b_time - np.asarray(self.slice_seconds)[dat_id])) < time_lim:
                                print('Duplicate slice found. New slice removed from rippl.stack')
                                continue

                        xyz = interp_orbit.evaluate_orbit_spline([b_time])[0]
                        self.slice_x.append(xyz[0, 0])
                        self.slice_y.append(xyz[1, 0])
                        self.slice_z.append(xyz[2, 0])
                        self.slice_lat.append(slice_coors[1])
                        self.slice_lon.append(slice_coors[0])
                        self.slice_date.append(slice_time[:10])
                        self.slice_swath_no.append(int(slice.readfiles['original'].json_dict['Swath']))
                        self.slice_az_time.append(slice_time)
                        self.slice_seconds.append(b_time)
                        self.slice_time.append(b_t)

                        self.slices.append(slice)

    def assign_slice_id(self):
        # Create new master ids if needed
        known_ids = len(self.master_slice_number)
        total_ids = len(self.master_slice_az_time)

        print('Assign found bursts to datastack. These can be part of current datastack already!')

        if known_ids == total_ids:
            print('No new master slices found. Only new slave slices will be assigned')
        else:
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

        # Calculate the burst availability
        s_dates = np.unique(np.array(self.slice_date))
        m_names = np.array(self.master_slice_names)
        self.burst_availability = np.zeros((len(self.master_slice_names), len(s_dates)))

        for s_name, s_date in zip(self.slice_names, self.slice_date):
            date_id = np.where((s_dates == s_date))
            slice_id = np.where((m_names == s_name))
            self.burst_availability[slice_id, date_id] = 1

    def write_master_slice_list(self):
        # az_time, yyyy-mm-ddThh:mm:ss.ssssss, swath x, slice i, x xxxx, y yyyy, z zzzz, lat ll.ll, lon ll.ll, pol pp

        l = open(os.path.join(self.datastack_folder, 'master_slice_list'), 'w+')
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

        if num_cores > 1:
            pool = Pool(num_cores)

        master_dat = [[self.datastack_folder, slice, number, swath_no, date]
                     for slice, number, swath_no, date
                      in zip(self.master_slices, self.master_slice_number, self.master_slice_swath_no,
                      self.master_slice_date)]
        if num_cores > 1:
            res = pool.map(write_sentinel_burst, master_dat)
        else:
            for m_dat in master_dat:
                write_sentinel_burst(m_dat)

        slave_dat = [[self.datastack_folder, slice, number, swath_no, date]
                     for slice, number, swath_no, date
                     in zip(self.slices, self.slice_number, self.slice_swath_no, self.slice_date)]
        if num_cores > 1:
            res = pool.map(write_sentinel_burst, slave_dat)
        else:
            for s_dat in slave_dat:
                write_sentinel_burst(s_dat)
