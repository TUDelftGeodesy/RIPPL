import os
import datetime
from typing import List, Optional
import numpy as np

from rippl.user_settings import UserSettings
from rippl.NWP_model_delay.load_NWP_data.harmonie_nl.harmonie_api_request import OpenDataAPI

class HarmonieDownload():

    def __init__(self, overpass_times: List[datetime.datetime], dataset='all_layers', api_key=''):
        # Input paramters should be:
        # - start_date/end_date > yyyy-mm-dd
        # - vars > ['var1', var2' etc] (list)
        # - station_type one of 'daily_rainfall', 'daily_weather', 'hourly_weather'

        # To create project_functions strings
        self.download_path = ''

        # Dates and coverage of period
        if not isinstance(overpass_times, list):
            overpass_times = [overpass_times]
        for overpass_time in overpass_times:
            if not isinstance(overpass_time, datetime.datetime):
                raise TypeError('overpass_times should consist of datetime.datetime instances. Aborting.')

        self.overpass_times = overpass_times

        # Get API key
        if not api_key:
            user_settings = UserSettings()
            self.api_key = user_settings.settings['accounts']['KNMI']['api_key']
        else:
            self.api_key = api_key

        # Define the dataset names
        if dataset == 'all_layers':
            self.dataset_name = 'harmonie_arome_cy43_p5'
            self.dataset_version = '1.0'

        # Get available datasets
        self.knmi_downloader = OpenDataAPI(api_token=self.api_key)
        self.available_files = self.knmi_downloader.list_files(self.dataset_name, self.dataset_version, params={'maxKeys': 1000})

    def download_data_file(self, download_folder):
        # Download the KNMI data file
        if os.path.exists(download_folder):
            self.download_folder = download_folder
        else:
            print('Download folder does not exist')
            return

        # Download the relevant datafiles for the overpass times of interest.
        self.download_files = []
        for overpass_time in self.overpass_times:
            # First find the closest 6 hour interval time.
            path = 'wget -O ' + self.download_file
            self.download_files.append(self.download_file)
            if not os.path.exists(self.download_file) or os.stat(self.download_file).st_size == 0:
                try:
                    os.system(path)
                except:
                    if os.path.exists(self.download_file):
                        os.remove(self.download_file)

    @staticmethod
    def find_closest_dataset(overpass_time: datetime.datetime, interval_hours=1, interval_minutes=0, date_type='closest'):
        """
        Find the closest dataset time assuming an interval of a fixed number of hours.

        """

        minute_steps = interval_minutes + interval_hours * 60
        hour_opts = np.floor(np.arange(0, 24 * 60, minute_steps) / 60).astype(np.int32)
        minute_opts = np.remainder(np.arange(0, 24 * 60, minute_steps), 60)

        # Get the day before and after
        yesterday = (overpass_time - datetime.timedelta(days=1)).date()
        today = overpass_time.date()
        tomorrow = (overpass_time + datetime.timedelta(days=1)).date()

        time_opts = []
        for day in [yesterday, today, tomorrow]:
            for hour, minute in zip(hour_opts, minute_opts):
                time_opts.append(datetime.datetime.combine(day, datetime.time(hour=hour, minute=minute)))

        if date_type == 'before':
            seconds_diff = np.array(time_opts) - overpass_time
            seconds_diff[seconds_diff > datetime.timedelta(seconds=0)] = datetime.timedelta(9999999)
        elif date_type == 'after':
            seconds_diff = np.array(time_opts) - overpass_time
            seconds_diff[seconds_diff < datetime.timedelta(seconds=0)] = datetime.timedelta(9999999)
        elif date_type == 'closest':
            seconds_diff = np.abs(np.array(time_opts) - overpass_time)
            pass
        else:
            raise TypeError('Only the options before/after/closest are possible!')

        min_id = np.argmin(np.abs(seconds_diff))
        nwp_datetime = np.array(time_opts)[min_id]

        return nwp_datetime
