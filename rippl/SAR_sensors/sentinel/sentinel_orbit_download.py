import os
import datetime
from eof.download import download_eofs
import numpy as np
import logging

from rippl.SAR_sensors.sentinel.sentinel_orbit_database import SentinelOrbitsDatabase
from rippl.user_settings import UserSettings

"""
# Test
dates = [datetime.datetime(year=2020, month=1, day=1, hour=4, minute=5), datetime.datetime(year=2020, month=1, day=5, hour=18)]

download = DownloadSentinelOrbit(use_asf=False)
files = download.download_orbits(dates, mission='S1A')

dates = [datetime.datetime(year=2024, month=1, day=30), datetime.datetime(year=2024, month=1, day=29)]

download = DownloadSentinelOrbit(use_copernicus=False)
files = download.download_orbits(dates, mission='S1B')
"""

class DownloadSentinelOrbit(object):

    def __init__(self, precise_folder='', restituted_folder='',
                 processes=4, use_copernicus=True, use_asf=True):
        # This script downloads all orbits files from the precise orbits website, when pages is set to a very high number.
        # By default only the first page for the last two days (restituted) is checked.

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

        self.database = SentinelOrbitsDatabase(self.precise_folder, self.restituted_folder)
        self.use_copernicus = use_copernicus
        self.use_asf = use_asf
        self.processes = processes

    def download_orbits(self, overpasses, mission='both'):
        """
        Download orbits from copernicus database

        """

        if self.use_copernicus:
            username_copernicus = self.settings.settings['accounts']['Copernicus']['username']
            password_copernicus = self.settings.settings['accounts']['Copernicus']['password']
        if self.use_asf:
            username_asf = self.settings.settings['accounts']['EarthData']['username']
            password_asf = self.settings.settings['accounts']['EarthData']['password']

        # For every overpass time search for an orbit file 1 hour before and 1 hour after to get the best overlapping
        # orbit file later on.
        dates = []
        for overpass in overpasses:
            before = overpass - datetime.timedelta(hours=1)
            dates.append(before)
            after = overpass + datetime.timedelta(hours=1)
            dates.append(after)

        if mission == 'both':
            missions = ['S1A', 'S1B']
        elif mission in ['S1A', 'S1B']:
            missions = [mission]
        else:
            raise TypeError('Mission should either be S1A, S1B or both')

        download_dates = []
        for date in dates:
            for mission in missions:
                if not self.database(date, satellite=mission):
                    download_dates.append(date)
                    break

        if len(download_dates) > 0:
            logging.info('Start downloading needed orbit files.')

        download_dates = np.array(download_dates)
        now = datetime.datetime.now()
        restituted_dates = download_dates[now - download_dates < datetime.timedelta(days=20)]

        out_files = []
        for mission in missions:

            # For all dates within 20 days download the restituted
            if len(restituted_dates) >= 1:
                mission_list = [mission for n in range(len(restituted_dates))]
                if self.use_copernicus and self.use_asf:
                    files = download_eofs(restituted_dates, orbit_type='restituted', max_workers=self.processes, save_dir=self.restituted_folder,
                                  cdse_user=username_copernicus, cdse_password=password_copernicus,
                                  asf_user=username_asf, asf_password=password_asf, missions=mission_list)
                elif self.use_copernicus:
                    files = download_eofs(restituted_dates, orbit_type='restituted', max_workers=self.processes, save_dir=self.restituted_folder,
                                  cdse_user=username_copernicus, cdse_password=password_copernicus, missions=mission_list)
                elif self.use_asf:
                    files = download_eofs(restituted_dates, orbit_type='restituted', max_workers=self.processes, save_dir=self.restituted_folder,
                                  asf_user=username_asf, asf_password=password_asf, force_asf=True, missions=mission_list)

                out_files.extend(files)

            # For all dates try to download the precise orbits
            if len(download_dates) >= 1:
                mission_list = [mission for n in range(len(download_dates))]
                if self.use_copernicus and self.use_asf:
                    files = download_eofs(download_dates, orbit_type='precise', max_workers=self.processes, save_dir=self.precise_folder,
                                  cdse_user=username_copernicus, cdse_password=password_copernicus,
                                  asf_user=username_asf, asf_password=password_asf, missions=mission_list)
                elif self.use_copernicus:
                    files = download_eofs(download_dates, orbit_type='precise', max_workers=self.processes, save_dir=self.precise_folder,
                                  cdse_user=username_copernicus, cdse_password=password_copernicus, missions=mission_list)
                elif self.use_asf:
                    files = download_eofs(download_dates, orbit_type='precise', max_workers=self.processes, save_dir=self.precise_folder,
                                  asf_user=username_asf, asf_password=password_asf, force_asf=True, missions=mission_list)

                # Remove downloaded restituted files
                for filename in files:
                    if 'RESORB' in str(filename):
                        os.remove(str(filename))

                out_files.extend(files)

            for out_file in out_files:
                logging.info('Downloaded orbit file ' + str(out_file))

        return out_files
