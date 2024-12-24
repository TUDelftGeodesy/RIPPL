# This class creates an oversight of the available information from the ecmwf database.
# When calling this class with a datetime object, it will return the file on the closest moment as well as the
# closest step before and after.

import datetime
import os
import numpy as np
import logging
import xarray as xr
from shapely.geometry import Polygon

from rippl.user_settings import UserSettings
from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_download import CDSdownload

class ECMWFDatabase():

    def __init__(self, database_folder=''):
        # If no database folder load info from user settings
        if not database_folder:
            user_settings = UserSettings()
            database_folder = os.path.join(user_settings.settings['paths']['NWP_model_database'],
                                           user_settings.settings['path_names']['NWP']['ECMWF'])

        if not os.path.exists(database_folder):
            os.mkdir(database_folder)

        # Possible ecmwf data types
        ecmwf_versions = ['era5_pressure_levels', 'era5_model_levels', 'cerra_pressure_levels', 'cerra_model_levels', 'operational_hres']

        # Init the database for different Harmonie versions.
        self.version_folders = [os.path.join(database_folder, version) for version in ecmwf_versions]

        # Search for files
        self.ecmwf_types = []
        self.ecmwf_files = []
        self.ecmwf_analysis_time = []
        self.ecmwf_forecast_time = []           # Same as analysis in case analysis time is used
        self.ecmwf_areas = []
        self.ecmwf_data_type = []

        for folder, ecmwf_type in zip(self.version_folders, ecmwf_versions):
            if not os.path.exists(folder):
                os.mkdir(folder)

            try:
                ecmwf_files = next(os.walk(folder))[2]
            except Exception as e:
                logging.warning('No ECMWF files found in ' + folder + '. ' + str(e))

            for ecmwf_file in ecmwf_files:
                if not ecmwf_file.endswith('.nc'):
                    logging.info('Skipping ' + ecmwf_file + ' not a valid ECMWF file.')
                    continue

                try:
                    analysis_time, forecast_time, date_type, shape = self.get_ecmwf_datetime_area(ecmwf_file)

                    self.ecmwf_types.append(ecmwf_type)
                    self.ecmwf_files.append(os.path.join(folder, ecmwf_file))
                    self.ecmwf_forecast_time.append(forecast_time)
                    self.ecmwf_analysis_time.append(analysis_time)
                    self.ecmwf_areas.append(shape)
                    self.ecmwf_data_type.append(ecmwf_file.split('_')[-1][:-3])

                except Exception as e:
                    logging.warning('Failed to load ' + ecmwf_file + ' ' + str(e))

        self.ecmwf_types = np.array(self.ecmwf_types)
        self.ecmwf_forecast_time = np.array(self.ecmwf_forecast_time)
        self.ecmwf_data_type = np.array(self.ecmwf_data_type)
        self.ecmwf_files = np.array(self.ecmwf_files)

    def __call__(self, overpass_time, ecmwf_type='all', interval_hours=1, area=None, buffer=1):

        if ecmwf_type == 'all':
            type_list = np.ones(self.ecmwf_types.shape).astype(np.bool_)
        elif ecmwf_type in ['cerra_pressure_levels', 'cerra_model_levels', 'era5_pressure_levels', 'era5_model_levels', 'operational_hres']:
            type_list = np.array([ecmwf_type == ecmwf_type_val for ecmwf_type_val in self.ecmwf_types])
        else:
            return

        if isinstance(area, Polygon):
            area = area.buffer(buffer)
            contains = [ecmwf_area.contains(area) for ecmwf_area in self.ecmwf_areas]
        else:
            contains = np.ones(self.ecmwf_types.shape).astype(np.bool_)

        # Find closest datetime with overpass time
        output_date = CDSdownload.find_closest_dataset(overpass_time, interval_hours=interval_hours, date_type='closest')

        if output_date in self.ecmwf_forecast_time:
            atmosphere = self.ecmwf_files[(self.ecmwf_forecast_time == output_date) *
                                          (self.ecmwf_data_type == 'atmosphere') *
                                          type_list * contains]
            surface = self.ecmwf_files[(self.ecmwf_forecast_time == output_date) *
                                       (self.ecmwf_data_type == 'surface') *
                                       type_list * contains]

            if len(atmosphere) > 0 and len(surface) > 0:
                return [atmosphere[0], surface[0]]
            else:
                return [None, None]
        else:
            return [None, None]

    @staticmethod
    def get_ecmwf_datetime_area(file):
        """
        Get the datetime from file

        """

        filename = os.path.basename(file)
        date_str = filename.split('_')[-2]
        if '+' in date_str:
            an_str = date_str[:-3]
            fc_str = date_str[-2:]
        else:
            an_str = date_str
            fc_str = ''

        # Get the area
        area_str1 = filename.split('_')[-4]
        area_str2 = filename.split('_')[-3]

        if area_str1 + '_' + area_str2 == 'full_resolution':
            shape = Polygon([[-90, -180], [90, -180], [90, 180], [-90, 180], [-90, -180]])
        else:
            lats = []
            lons = []
            for n, area_str in enumerate([area_str1, area_str2]):
                lats.append(int(area_str[1:3]) if area_str[0] == 'n' else -int(area_str[1:3]))
                lons.append(int(area_str[4:7]) if area_str[3] == 'e' else -int(area_str[4:7]))
            shape = Polygon([[lats[0], lons[0]], [lats[0], lons[1]], [lats[1], lons[1]], [lats[1], lons[0]], [lats[0], lons[0]]])

        # Get the date
        if len(an_str) == 6:
            analysis_time = datetime.datetime.strptime(an_str, '%Y%m')
            date_type = 'month'
        elif len(an_str) == 8:
            analysis_time = datetime.datetime.strptime(an_str, '%Y%m%d')
            date_type = 'day'
        elif len(an_str) == 11:
            analysis_time = datetime.datetime.strptime(an_str, '%Y%m%dT%H')
            date_type = 'hour'

        if fc_str:
            forecast_time = analysis_time + datetime.timedelta(hours=int(fc_str))
        else:
            forecast_time = analysis_time

        return analysis_time, forecast_time, date_type, shape

    @staticmethod
    def split_grib_file(file, input_step='month'):
        """
        Split .grib file in files per hour

        """

        dat_folder = os.path.dirname(file)
        an_date, fc_date, dat_type, shape = ECMWFDatabase.get_ecmwf_datetime_area(file)
        if input_step is not dat_type:
            raise TypeError('Input type of file ' + dat_type + ' does not match with input step ' + input_step +
                            '. Aborting.')

        # Load as xarray variable
        in_data = xr.open_dataset(file, engine='cfgrib', indexpath='', backend_kwargs={'read_keys': ['pv']})
        date_times = np.atleast_1d(in_data.time.values)
        date_steps = np.atleast_1d(in_data.step.values)

        for time in date_times:
            for step in date_steps:
                date_str = (time.astype('datetime64[h]').item().strftime('%Y%m%dT%H') + '+'
                            + str(step.astype('timedelta64[h]').item().seconds // 3600).zfill(2))

                splits = os.path.basename(file).split('_')
                splits[-2] = date_str
                new_file_name = os.path.join(dat_folder, '_'.join(splits)[:-4] + '.nc')

                if len(date_times) > 1 and len(date_steps) > 1:
                    new_data = in_data.sel(time=time, step=step)
                elif len(date_times) > 1:
                    new_data = in_data.sel(time=time)
                elif len(date_steps) > 1:
                    new_data = in_data.sel(step=step)
                else:
                    new_data = in_data

                # Save data as netcdf file
                new_data.to_netcdf(new_file_name)
