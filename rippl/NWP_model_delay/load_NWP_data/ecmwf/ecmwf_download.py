# This class both downloads and reads an CDS dataset as a preprocessing step
import datetime
import os
from typing import Optional, List
from joblib import Parallel, delayed
import numpy as np

from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_api_request import CDSRequest
from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_database import ECMWFDatabase
from rippl.user_settings import UserSettings

"""
# Test download of different types of ECMWF data:
# - ERA5 model levels
# - ERA5 pressure levels
# - CERRA model levels
# - CERRA pressure levels

lonlim = [-2, 12]
latlim = [45, 56]
overpass_time = datetime.datetime(year=2020, month=1, day=1, hour=5, minute=23, second=0)
downloads = dict()

for data_type in ['reanalysis-era5-pressure-levels', 'reanalysis-cerra-pressure-levels', 'reanalysis-cerra-model-levels', 'reanalysis-era5-complete']:    
    if 'era5' in data_type:               
        downloads[data_type] = CDSdownload(data_type=data_type, latlim=latlim, lonlim=lonlim, overpass_times=[overpass_time], processes=1)
    else:
        downloads[data_type] = CDSdownload(data_type=data_type, overpass_times=[overpass_time], processes=1)
    
    downloads[data_type].prepare_download()
    downloads[data_type].download(parallel=False)

"""

class CDSdownload:
    
    def __init__(self, overpass_times: List[datetime.datetime],
                 latlim: Optional[List[int]] = None,
                 lonlim: Optional[List[int]] = None,
                 ecmwf_data_folder='',
                 data_type='reanalysis-era5-pressure-levels',
                 processes=4):
        # In the init function we mainly check the different dates, data folders and extend of dataset.
        self.data_type = data_type

        # Find the times steps for different CDS datasets
        if data_type == 'reanalysis-cerra-model-levels':
            self.an_step = 3
            self.fc_step = None
            self.grid_size = None
            self.model_levels = 106
            self.pressure_levels = None
            self.dataset_class = None
            self.type_folder = 'cerra_model_levels'
        elif data_type == 'reanalysis-cerra-pressure-levels':
            self.an_step = 3
            self.fc_step = 1
            self.grid_size = None
            self.model_levels = None
            self.pressure_levels = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 750, 700, 600, 500, 400, 300, 250,
                                    200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]
            self.dataset_class = None
            self.type_folder = 'cerra_pressure_levels'
        elif data_type == 'reanalysis-era5-pressure-levels':
            self.an_step = 1
            self.fc_step = None
            self.grid_size = 0.25
            self.model_levels = None
            self.pressure_levels = [1000, 975, 950, 925, 900, 875, 850, 825 ,800, 775, 750, 700, 650, 600, 550, 500,
                                    450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5,
                                    3, 2, 1]
            self.dataset_class = None
            self.type_folder = 'era5_pressure_levels'
        elif data_type == 'reanalysis-era5-complete':
            self.an_step = 1
            self.fc_step = None
            self.grid_size = 0.25
            self.model_levels = 137
            self.pressure_levels = None
            self.dataset_class = 'ea'
            self.type_folder = 'era5_model_levels'
        elif data_type == 'oper':
            self.an_step = 6
            self.fc_step = None
            self.grid_size = 0.25
            self.pressure_levels = None
            self.model_levels = 137
            self.dataset_class = 'od'
            self.type_folder = 'operational_hres'
        else:
            raise TypeError('CDS dataset type not recognized! Types should be one of oper, reanalysis-era5-complete and '
                            'reanalysis-era5-pressure-levels')
            return

        # Init folder
        if not ecmwf_data_folder:
            settings = UserSettings()
            ecmwf_data_folder = os.path.join(settings.settings['paths']['NWP_model_database'],
                                             settings.settings['path_names']['NWP']['ECMWF'])
        self.ecmwf_data_type_folder = os.path.join(ecmwf_data_folder, self.type_folder)

        if not os.path.exists(ecmwf_data_folder):
            os.mkdir(ecmwf_data_folder)
        if not os.path.exists(self.ecmwf_data_type_folder):
            os.mkdir(self.ecmwf_data_type_folder)

        self.filenames = []
        self.n_processes = processes
        
        # Date should be in datetime format.
        self.overpass_times = overpass_times
        self.database = ECMWFDatabase()
        self.time = []

        # area and grid strings
        if self.grid_size:
            self.grid_str = str(self.grid_size) + '/' + str(self.grid_size)
        else:
            self.grid_str = None

        if latlim and lonlim and 'cerra' not in self.data_type:
            # Boundaries
            self.lat = np.sort(latlim)
            self.lon = np.sort(lonlim)
            self.area_str_mars = str(self.lat[1]) + '/' + str(self.lon[0]) + '/' + str(self.lat[0]) + '/' + str(self.lon[1])
            self.area_list = [str(self.lat[1]), str(self.lon[0]), str(self.lat[0]), str(self.lon[1])]
            self.file_area_str = (('n' + str(self.lat[1]).zfill(2) if self.lat[1] >= 0 else 's' + str(np.abs(self.lat[1])).zfill(2)) +
                                  ('e' + str(self.lon[0]).zfill(3) if self.lon[0] >= 0 else 'w' + str(np.abs(self.lon[0])).zfill(3)) +
                                  '_' +
                                  ('n' + str(self.lat[0]).zfill(2) if self.lat[0] >= 0 else 's' + str(np.abs(self.lat[0])).zfill(2)) +
                                  ('e' + str(self.lon[1]).zfill(3) if self.lon[1] >= 0 else 'w' + str(np.abs(self.lon[1])).zfill(3)))
        else:
            self.file_area_str = 'full_resolution'
            self.area_str_mars = '90/-180/-90/180'
            self.area_list = ['90', '-180', '-90', '180']
            self.lat = np.array([-90, 90])
            self.lon = np.array([-180, 180])

        # Initialize variable of final product
        self.requests = dict()
        self.ecmwf_data = dict()

    def check_overpass_available(self, overpass_time):
        """

        :return:
        """

        if self.fc_step:
            data_time = CDSdownload.find_closest_dataset(overpass_time, interval_hours=self.fc_step)
            interval = self.fc_step
        else:
            data_time = CDSdownload.find_closest_dataset(overpass_time, interval_hours=self.an_step)
            interval = self.an_step

        filename = self.database(data_time, ecmwf_type=self.type_folder, interval_hours=interval)

        return filename

    def prepare_download(self):
        # This function downloads CDS data for the area within the bounding box
        file_base = os.path.join(self.ecmwf_data_type_folder, 'ecmwf_' + self.type_folder + '_')

        # Find the closest hour to the overpass time (They should all be the same)
        for overpass_time in self.overpass_times:

            before = CDSdownload.find_closest_dataset(overpass_time, interval_hours=self.an_step, date_type='before')
            after = CDSdownload.find_closest_dataset(overpass_time, interval_hours=self.an_step, date_type='after')
            if self.fc_step:
                before_fc = CDSdownload.find_closest_dataset(overpass_time, interval_hours=self.fc_step, date_type='before')
                after_fc = CDSdownload.find_closest_dataset(overpass_time, interval_hours=self.fc_step, date_type='after')
                after = before # Use the same analysis time for both
                interval = self.fc_step
            else:
                before_fc = before
                after_fc = after
                interval = self.an_step

            for an_time, fc_time in zip([before, after], [before_fc, after_fc]):

                # Always request datasets per month (This is the advised method to request ECMWF model data)
                month = an_time.strftime('%Y%m')

                # Check whether file already exists in database
                if self.database(fc_time, ecmwf_type=self.type_folder, interval_hours=interval)[0]:
                    continue

                if month not in self.requests.keys():
                    self.requests[month] = dict()
                    self.requests[month]['year'] = an_time.strftime('%Y')
                    self.requests[month]['month'] = an_time.strftime('%m')
                    self.requests[month]['days'] = []
                    self.requests[month]['hours'] = []
                    if self.fc_step:
                        self.requests[month]['forecast_hours'] = []
                    self.requests[month]['atmosphere'] = []
                    self.requests[month]['download_atmosphere'] = file_base + self.file_area_str + '_' + month + '_atmosphere.grb'
                    self.requests[month]['surface'] = []
                    self.requests[month]['download_surface'] = file_base + self.file_area_str + '_' + month + '_surface.grb'

                # Output files. Check for copy to be sure
                time_str = an_time.strftime('%Y%m%dT%H') + '+' + str(int((fc_time - an_time).seconds // 3600)).zfill(2)
                if file_base + time_str + '_atmosphere.grb' not in self.requests[month]['atmosphere']:
                    day = an_time.strftime('%d')
                    if day not in self.requests[month]['days']:
                        self.requests[month]['days'].append(day)

                    hour = an_time.strftime('%H:%M')
                    if hour not in self.requests[month]['hours']:
                        self.requests[month]['hours'].append(hour)

                    if 'forecast_hours' in self.requests[month].keys():
                        forecast_hour = (fc_time - an_time).seconds // 3600
                        if not str(forecast_hour) in self.requests[month]['forecast_hours']:
                            self.requests[month]['forecast_hours'].append(str(forecast_hour))

                    self.requests[month]['atmosphere'].append(file_base + self.file_area_str + '_' + time_str + '_atmosphere.grb')
                    self.requests[month]['surface'].append(file_base + self.file_area_str + '_' + time_str + '_surface.grb')

        # Create input lists for mars requests
        for key in self.requests.keys():
            if self.model_levels:
                level_list = [str(level + 1) + '/' for level in range(self.model_levels)]
                self.requests[key]['level_list'] = ''.join(level_list)[:-1]
            if self.pressure_levels:
                level_list = [str(level) + '/' for level in self.pressure_levels]
                self.requests[key]['pressure_list'] = ''.join(level_list)[:-1]
            if self.model_levels:
                self.requests[key]['levels'] = [str(level + 1) for level in range(self.model_levels)]
            if self.pressure_levels:
                self.requests[key]['pressures'] = [str(level) for level in self.pressure_levels]

            hour_list = [hour + '/' for hour in self.requests[key]['hours']]
            self.requests[key]['hour_list'] = ''.join(hour_list)[:-1]

            month_base = an_time.strftime('%Y-%m-')
            day_list = [month_base + day + '/' for day in self.requests[key]['days']]
            self.requests[key]['day_list'] = ''.join(day_list)[:-1]

    def download(self, parallel=True):

        # Surface files
        mars_request = CDSRequest(self.data_type, self.dataset_class, grid=self.grid_str,
                                  area_mars=self.area_str_mars, area_list=self.area_list, dataset='surface')
        if parallel:
            Parallel(n_jobs=self.n_processes)(delayed(mars_request)(request) for request in self.requests.values())
        else:
            for request in self.requests.values():
                mars_request(request)

        # Atmosphere files
        mars_request = CDSRequest(self.data_type, self.dataset_class, grid=self.grid_str,
                                  area_mars=self.area_str_mars, area_list=self.area_list, dataset='atmosphere')
        if parallel:
            Parallel(n_jobs=self.n_processes)(delayed(mars_request)(request) for request in self.requests.values())
        else:
            for request in self.requests.values():
                mars_request(request)

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
