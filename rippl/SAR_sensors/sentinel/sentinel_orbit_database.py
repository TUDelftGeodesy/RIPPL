# This function analyses a database of sentinel-1 precise orbit file and loads files for individual dates.
# First we check which files are available in the orbit folder and then load different files on request.
# The earlier requested dates are saved to memory, it is not needed to load these files again.

import os
import logging
from datetime import datetime, timedelta
from lxml import etree
from rippl.meta_data.orbit import Orbit
import numpy as np
from rippl.user_settings import UserSettings


class SentinelOrbitsDatabase(object):

    def __init__(self, precise_folder='', restituted_folder='', track_no=None):
        # To initialize this function we create a list of all the precise and restituted orbit files.
        # These are categorized in S1A and S1B

        logging.info('Reading precise and restituted orbit database')

        self.precise_files = dict()
        self.precise_files['S1A'] = {'start_time': [], 'end_time': [], 'file_name': []}
        self.precise_files['S1B'] = {'start_time': [], 'end_time': [], 'file_name': []}
        self.restituted_files = dict()
        self.restituted_files['S1A'] = {'start_time': [], 'end_time': [], 'file_name': []}
        self.restituted_files['S1B'] = {'start_time': [], 'end_time': [], 'file_name': []}

        # Initialize for the files loaded in memory
        self.precise_data = dict()
        self.precise_data['S1A'] = dict()
        self.precise_data['S1B'] = dict()
        self.restituted_data = dict()
        self.restituted_data['S1A'] = dict()
        self.restituted_data['S1B'] = dict()

        # Initialize the orbit ID listing for specific orbit
        self.track = track_no
        self.burst_shapes = dict()

        self.settings = UserSettings()
        self.settings.load_settings()

        database_folder = self.settings.settings['paths']['orbit_database']
        s1_name = self.settings.settings['path_names']['SAR']['Sentinel-1']

        if not restituted_folder:
            restituted_folder = os.path.join(database_folder, s1_name, 'restituted')
        if not precise_folder:
            precise_folder = os.path.join(database_folder, s1_name, 'precise')
        self.precise_folder = precise_folder
        self.restituted_folder = restituted_folder

        self.index_orbit_files()

    def index_orbit_files(self):
        """
        Index available orbit files on disk

        :return:
        """

        precise = os.listdir(self.precise_folder)
        restituted = os.listdir(self.restituted_folder)

        for filename in precise:
            if filename.startswith('S1B_OPER_AUX_POEORB_OPOD'):
                satellite = 'S1B'
            elif filename.startswith('S1A_OPER_AUX_POEORB_OPOD'):
                satellite = 'S1A'
            else:
                continue

            start_time = datetime.strptime(filename[42:57], '%Y%m%dT%H%M%S')
            end_time = datetime.strptime(filename[58:73], '%Y%m%dT%H%M%S')

            self.precise_files[satellite]['file_name'].append(os.path.join(self.precise_folder, filename))
            self.precise_files[satellite]['start_time'].append(start_time)
            self.precise_files[satellite]['end_time'].append(end_time)

        for filename in restituted:
            if filename.startswith('S1B_OPER_AUX_RESORB_OPOD'):
                satellite = 'S1B'
            elif filename.startswith('S1A_OPER_AUX_RESORB_OPOD'):
                satellite = 'S1A'
            else:
                continue

            start_time = datetime.strptime(filename[42:57], '%Y%m%dT%H%M%S')
            end_time = datetime.strptime(filename[58:73], '%Y%m%dT%H%M%S')

            self.restituted_files[satellite]['file_name'].append(os.path.join(self.precise_folder, filename))
            self.restituted_files[satellite]['start_time'].append(start_time)
            self.restituted_files[satellite]['end_time'].append(end_time)

        for satellite in ['S1A', 'S1B']:
            self.precise_files[satellite]['file_name'] = np.array(self.precise_files[satellite]['file_name'])
            self.precise_files[satellite]['start_time'] = np.array(self.precise_files[satellite]['start_time'])
            self.precise_files[satellite]['end_time'] = np.array(self.precise_files[satellite]['end_time'])

            self.restituted_files[satellite]['file_name'] = np.array(self.restituted_files[satellite]['file_name'])
            self.restituted_files[satellite]['start_time'] = np.array(self.restituted_files[satellite]['start_time'])
            self.restituted_files[satellite]['end_time'] = np.array(self.restituted_files[satellite]['end_time'])

    def __call__(self, overpass_time, satellite='S1A', input_orbit_type='precise'):
        """
        Call

        """

        if input_orbit_type not in ['RES', 'restituted', 'POE', 'precise']:
            logging.info('input orbit type not recognized')
            return False

        if input_orbit_type in ['POE', 'precise']:
            input_orbit_type = 'precise'
            ids = np.atleast_1d(np.squeeze(np.argwhere((self.precise_files[satellite]['start_time'] < overpass_time) *
                                         (self.precise_files[satellite]['end_time'] > overpass_time))))

            if len(ids) > 0:
                diff_time = (self.precise_files[satellite]['end_time'][ids] - self.precise_files[satellite]['start_time'][ids])
                ref_time = self.precise_files[satellite]['start_time'][ids] + diff_time
                diff_ref_time = np.abs(overpass_time - ref_time)
                id = ids[np.argmin(diff_ref_time)]

                orb_file = self.precise_files[satellite]['file_name'][id]
            else:  # If not available switch to restituted
                input_orbit_type = 'RES'

        if input_orbit_type in ['RES', 'restituted']:
            input_orbit_type = 'restituted'
            ids = np.atleast_1d(np.squeeze(np.argwhere((self.restituted_files[satellite]['start_time'] < overpass_time) *
                                         (self.restituted_files[satellite]['end_time'] > overpass_time))))

            # Check for either this hour or the one before
            if len(ids) > 0:
                diff_time = (self.restituted_files[satellite]['end_time'][ids] - self.restituted_files[satellite]['start_time'][ids])
                ref_time = self.restituted_files[satellite]['start_time'][ids] + diff_time
                diff_ref_time = np.abs(overpass_time - ref_time)
                id = ids[np.argmin(diff_ref_time)]

                orb_file = self.restituted_files[satellite]['file_name'][id]
            else:
                logging.info('No precise or restituted orbit file available for ' + str(overpass_time))
                return False

        return orb_file

    def interpolate_orbit(self, overpass_time, input_orbit_type='POE', satellite='S1A'):
        """

        :param overpass_time:
        :param input_orbit_type:
        :param satellite:
        :return:
        """

        # Get the needed file
        orb_file = self(overpass_time, satellite, input_orbit_type)

        logging.info('Loading orbits for ' + str(overpass_time))
        if input_orbit_type == 'precise':
            if os.path.basename(orb_file) in self.precise_data[satellite].keys():
                orbit_dat = self.precise_data[satellite][os.path.basename(orb_file)]
            else:
                orbit_meta, orbit_dat = self.orbit_read(orb_file, overpass_time, input_orbit_type)
                self.precise_data[satellite][os.path.basename(orb_file)] = orbit_dat
        if input_orbit_type == 'restituted':
            if os.path.basename(orb_file) in self.restituted_data[satellite].keys():
                orbit_dat = self.restituted_data[satellite][os.path.basename(orb_file)]
            else:
                orbit_meta, orbit_dat = self.orbit_read(orb_file, overpass_time, input_orbit_type)
                self.restituted_data[satellite][os.path.basename(orb_file)] = orbit_dat

        return orbit_dat

    @staticmethod
    def orbit_read(input_orbit, input_time, orbit_type='precise'):

        in_tree = etree.parse(input_orbit)
        metadata = {'Mission' : './/Earth_Explorer_Header/Fixed_Header/Mission',
                    'Validity_Start': './/Earth_Explorer_Header/Fixed_Header/Validity_Period/Validity_Start',
                    'Validity_Stop': './/Earth_Explorer_Header/Fixed_Header/Validity_Period/Validity_Stop'}

        # temp variables and parameters
        orbit_meta = dict()
        for key in metadata:
            orbit_meta[key] = in_tree.find(metadata[key])

        orbit_dat = {'orbitTime': [], 'orbitX': [], 'orbitY': [], 'orbitZ': [], 'velX': [], 'velY': [], 'velZ': []}

        # Read all orbit points within 1000 seconds.
        max_time_diff = timedelta(seconds=1000)
        max_time = input_time + max_time_diff
        min_time = input_time - max_time_diff

        for times in in_tree.findall('.//Data_Block/List_of_OSVs/OSV'):
            time = datetime.strptime(times[1].text[4:], '%Y-%m-%dT%H:%M:%S.%f')

            if min_time < time < max_time:
                seconds = float(time.hour * 3600 + time.minute * 60 + time.second) + float(time.microsecond) / 1000000

                orbit_dat['orbitTime'].append(seconds)
                orbit_dat['orbitX'].append(float(times[4].text))
                orbit_dat['orbitY'].append(float(times[5].text))
                orbit_dat['orbitZ'].append(float(times[6].text))
                orbit_dat['velX'].append(float(times[7].text))
                orbit_dat['velY'].append(float(times[8].text))
                orbit_dat['velZ'].append(float(times[9].text))

        orbit_object = Orbit()
        orbit_object.create_orbit(np.array(orbit_dat['orbitTime']),
                                  np.array(orbit_dat['orbitX']),
                                  np.array(orbit_dat['orbitY']),
                                  np.array(orbit_dat['orbitZ']),
                                  np.array(orbit_dat['velX']),
                                  np.array(orbit_dat['velY']),
                                  np.array(orbit_dat['velZ']),
                                  satellite='Sentinel-' + input_orbit[1:3],
                                  date=times[1].text[4:14],
                                  type=orbit_type)

        return orbit_meta, orbit_dat
