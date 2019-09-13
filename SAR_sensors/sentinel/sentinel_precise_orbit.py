# This function analyses a database of sentinel-1 precise orbit file and loads files for individual dates.
# First we check which files are available in the orbit folder and then load different files on request.
# The earlier requested dates are saved to memory, it is not needed to load these files again.

import os
from datetime import datetime, timedelta
from lxml import etree
from rippl.meta_data.orbit import Orbit
import numpy as np


class SentinelOrbitsDatabase(object):

    def __init__(self, precise_orbit_folder, restituted_orbit_folder):
        # To initialize this function we create a list of all the precise and restituted orbit files.
        # These are categorized in 1A and 1B

        print('Reading precise and restituted orbit database')

        self.precise_files = dict()
        self.precise_files['1A'] = dict()
        self.precise_files['1B'] = dict()
        self.restituted_files = dict()
        self.restituted_files['1A'] = dict()
        self.restituted_files['1B'] = dict()

        # Initialize for the files loaded in memory
        self.precise_data = dict()
        self.precise_data['1A'] = dict()
        self.precise_data['1B'] = dict()
        self.restituted_data = dict()
        self.restituted_data['1A'] = dict()
        self.restituted_data['1B'] = dict()

        precise = os.listdir(precise_orbit_folder)
        restituted = os.listdir(restituted_orbit_folder)

        self.latest_precise = 20140101

        for filename in precise:
            if filename.startswith('S1B_OPER_AUX_POEORB_OPOD'):
                type = '1B'
            elif filename.startswith('S1A_OPER_AUX_POEORB_OPOD'):
                type = '1A'
            else:
                continue

            start_time = datetime.strptime(filename[42:57], '%Y%m%dT%H%M%S')
            end_time = datetime.strptime(filename[58:73], '%Y%m%dT%H%M%S')

            ref_time = (start_time + (end_time - start_time) / 2).strftime('%Y%m%d')
            if int(ref_time) > self.latest_precise:
                self.latest_precise = int(ref_time)

            self.precise_files[type][ref_time] = os.path.join(precise_orbit_folder, filename)

        for filename in restituted:
            if filename.startswith('S1B_OPER_AUX_RESORB_OPOD'):
                type = '1B'
            elif filename.startswith('S1A_OPER_AUX_RESORB_OPOD'):
                type = '1A'
            else:
                continue

            end_time = datetime.strptime(filename[58:73], '%Y%m%dT%H%M%S')
            ref_time = (end_time - timedelta(hours=1)).strftime('%Y%m%d%H')

            if self.latest_precise > int(ref_time):  # If there is already a precise orbit file available
                continue

            self.restituted_files[type][ref_time] = os.path.join(restituted_orbit_folder, filename)

        self.latest_precise = datetime.strptime(str(self.latest_precise), '%Y%m%d')

    def interpolate_orbit(self, overpass_time, input_orbit_type='POE', satellite='1A'):
        # overpass_time should be a datetime object

        if input_orbit_type not in ['RES', 'restituted', 'POE', 'precise']:
            print('input orbit type not recognized')
            return False

        if input_orbit_type in ['POE', 'precise']:
            type = 'precise'
            ref_time = overpass_time.strftime('%Y%m%d')

            if ref_time in self.precise_files[satellite].keys():
                orb_file = self.precise_files[satellite][ref_time]
            else:  # If not available switch to restituted
                input_orbit_type = 'RES'

        if input_orbit_type in ['RES', 'restituted']:
            type = 'restituted'
            ref_time = overpass_time.strftime('%Y%m%d%H')
            ref_time_before = (overpass_time - timedelta(hours=1)).strftime('%Y%m%d%H')

            # Check for either this hour hour or the one before
            if ref_time in self.precise_files[satellite].keys():
                orb_file = self.precise_files[satellite][ref_time]
            elif ref_time_before in self.precise_files[satellite].keys():
                orb_file = self.precise_files[satellite][ref_time_before]
            else:
                print('No precise or restituted orbit file available')
                return False

        # print('Loading orbits for ' + ref_time)
        if type == 'precise':
            if os.path.basename(orb_file) in self.precise_files[satellite].keys():
                orbit_dat = self.precise_files[satellite][os.path.basename(orb_file)]
            else:
                orbit_meta, orbit_dat = self.orbit_read(orb_file, overpass_time, type)
                self.precise_files[satellite][os.path.basename(orb_file)] = orbit_dat
        if type == 'restituted':
            if os.path.basename(orb_file) in self.restituted_files[satellite].keys():
                orbit_dat = self.restituted_files[satellite][os.path.basename(orb_file)]
            else:
                orbit_meta, orbit_dat = self.orbit_read(orb_file, overpass_time, type)
                self.restituted_files[satellite][os.path.basename(orb_file)] = orbit_dat

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
                orbit_dat['orbitTime'].append(float(time.hour * 3600 + time.minute * 60 + time.second) + float(time.microsecond) / 1000000)
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
