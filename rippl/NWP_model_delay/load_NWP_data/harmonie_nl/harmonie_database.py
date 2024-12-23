# This class creates an oversight of the available information from the harmonie_nl database.
# When calling this class with a datetime object, it will return the file on the closest moment as well as the
# closest step before and after. Both will be within a certain time frame (automatically set to 6 hours)

import datetime
import os
import numpy as np
import logging

from rippl.user_settings import UserSettings
from rippl.NWP_model_delay.load_NWP_data.harmonie_nl.harmonie_download import HarmonieDownload

class HarmonieDatabase():

    def __init__(self, database_folder=''):
        # If no database folder load info from user settings
        if not database_folder:
            user_settings = UserSettings()
            database_folder = os.path.join(user_settings.settings['paths']['NWP_model_database'],
                                           user_settings.settings['path_names']['NWP']['Harmonie-Arome'])

        # Possible harmonie cycles
        harmonie_versions = ['h38', 'h40']

        # Init the database for different Harmonie versions.
        self.version_folders = [os.path.join(database_folder, version) for version in harmonie_versions]

        # Search for files
        self.harmonie_types = []
        self.harmonie_grid = []
        self.harmonie_files = []
        self.harmonie_analysis_times = []
        self.harmonie_forecast_times = []

        for folder, h_type in zip(self.version_folders, harmonie_versions):

            if not os.path.exists(folder):
                os.mkdir(folder)
            try:
                h_files = next(os.walk(folder))[2]
            except Exception as e:
                logging.warning('No Harmonie files found in ' + folder + '. ' + str(e))

            for h_file in h_files:
                if not h_file.endswith('_GB'):
                    logging.info('Skipping ' + h_file + ' not a valid Harmonie file.')

                try:
                    lead_time = datetime.timedelta(hours=int(h_file[22:25]), minutes=int(h_file[25:27]))
                    analysis_time = datetime.datetime(year=int(h_file[9:13]), month=int(h_file[13:15]), day=int(h_file[15:17]),
                                                hour=int(h_file[17:19]), minute=int(h_file[19:21]))
                    forecast_time = analysis_time + forecast_time
                    h_grid = h_file[5:8]

                    self.harmonie_types.append(h_type)
                    self.harmonie_grid.append(h_grid)
                    self.harmonie_files.append(os.path.join(folder, h_file))
                    self.harmonie_analysis_times.append(analysis_time)
                    self.harmonie_forecast_times.append(forecast_time)

                except Exception as e:
                    logging.warning('Failed to load ' + h_file + ' ' + str(e))

    def __call__(self, overpass_time, h_type='all', interval_hours=6):

        if h_type == 'all':
            type_list = range(len(self.harmonie_forecast_times))
        elif h_type in ['h38', 'h40']:
            type_list = np.where(self.harmonie_types == h_type)[0]
        else:
            return

        # Find closest datetime with overpass time
        output_date = HarmonieDownload.find_closest_dataset(overpass_time, interval_hours=interval_hours)

        if output_date in self.harmonie_forecast_times:
            n = np.where(self.harmonie_forecast_times == output_date)[0]

            if n in type_list:
                return self.harmonie_files[n]
            else:
                return False
        else:
            return False