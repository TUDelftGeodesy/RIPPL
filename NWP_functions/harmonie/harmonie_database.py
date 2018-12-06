# This class creates an oversight of the available information from the harmonie database.
# When calling this class with a datetime object, it will return the file on the closest moment as well as the
# closest step before and after. Both will be within a certain time frame (automatically set to 6 hours)

import datetime
import os
import numpy as np


class HarmonieDatabase():

    def __init__(self, database_folder='/media/gert/Data/weather_models/harmonie_data'):

        # Init the database for different Harmonie versions.
        self.h37_folder = os.path.join(database_folder, 'h37')
        self.h38_folder = os.path.join(database_folder, 'h38')
        self.h40_folder = os.path.join(database_folder, 'h40')

        # Search for files
        self.h_types = []
        self.h_files = []
        self.h_dates = []
        self.analysis_times = []
        self.forecast_times = []

        for folder, h_type in zip([self.h37_folder, self.h38_folder, self.h40_folder], ['h37', 'h38', 'h40']):

            h_files = next(os.walk(folder))[2]
            for h_file in h_files:

                try:
                    fc_time = datetime.timedelta(hours=int(h_file[22:25]), minutes=int(h_file[25:27]))
                    an_time = datetime.datetime(year=int(h_file[9:13]), month=int(h_file[13:15]), day=int(h_file[15:17]),
                                                hour=int(h_file[17:19]), minute=int(h_file[19:21]))
                    time = an_time + fc_time

                    self.h_types.append(h_type)
                    self.h_files.append(os.path.join(folder, h_file))
                    self.h_dates.append(time)
                    self.analysis_times.append(an_time)
                    self.forecast_times.append(fc_time)

                except:
                    print('Failed to load ' + h_file)

    def __call__(self, time, time_frame='', h_type='all'):

        if not time_frame:
            time_frame = datetime.timedelta(hours=6)

        if h_type == 'all':
            type_list = range(len(self.h_dates))
        elif h_type in ['h37', 'h38', 'h40']:
            type_list = np.where(self.h_types == h_type)[0]
        else:
            return

        time_diff = np.array([(time - h_time).seconds + (time - h_time).days * 24 * 3600 for h_time in self.h_dates])
        diff_max = time_frame.seconds

        closest_id = np.argmin(np.abs(np.array(time_diff)))
        closest = self.h_files[type_list[closest_id]]
        closest_time = self.h_dates[type_list[closest_id]]
        closest_diff = time_diff[type_list[closest_id]]

        if np.sum(time_diff < 0) > 0:
            before = np.argmin(np.abs(np.array(time_diff)[time_diff < 0]))
            before_id = np.where(time_diff < 0)[0][before]
            before = self.h_files[type_list[before_id]]
            before_time = self.h_dates[type_list[before_id]]
            before_diff = time_diff[type_list[before_id]]
        else:
            before = ''
            before_diff = datetime.timedelta(days=9999)
            before_time = datetime.datetime(year=1900, month=1, day=1)

        if np.sum(time_diff >= 0) > 0:
            after = np.argmin(np.array(time_diff)[time_diff >= 0])
            after_id = np.where(time_diff  >= 0)[0][after]
            after = self.h_files[type_list[after_id]]
            after_time = self.h_dates[type_list[after_id]]
            after_diff = time_diff[type_list[after_id]]
        else:
            after = ''
            after_diff = datetime.timedelta(days=9999)
            after_time = datetime.datetime(year=2100, month=1, day=1)

        if np.abs(closest_diff) > diff_max:
            closest = ''
        if np.abs(before_diff) > diff_max:
            before = ''
        if np.abs(after_diff) > diff_max:
            after = ''

        return [closest, before, after], [closest_time, before_time, after_time]
