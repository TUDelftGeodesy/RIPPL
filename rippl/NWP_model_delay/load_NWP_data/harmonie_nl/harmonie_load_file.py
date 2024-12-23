# Load the Harmonie data from the data files as input for ray tracing.
import numpy as np
import cfgrib
import logging
import os
import pyproj

from rippl.NWP_model_delay.load_NWP_data.harmonie_nl.harmonie_database import HarmonieDatabase
from rippl.NWP_model_delay.ray_tracing_NWP_data.model_to_delay import ModelToDelay

class HarmonieData(object):

    def __init__(self):
        # Initialize the model data
        self.model_data = dict()
        self.filenames = []

        # Load the level information
        self.times = []

        # Load the database
        self.database = HarmonieDatabase()

    def load_harmonie(self, overpass, filename=''):
        # This function resamples the Harmonie grid to different 2d cross-section for different lines.
        # Input is:
        # - dat_file    > the Harmonie .grib file
        # - lat / lon   > the far range points on a line (we assume a straight line in lat/lon)
        # - heading     > heading of ray from point to satellite (degrees from north [0-360])
        # - max_dist    > determines how long we follow this direction (4-5 degrees should be fine
        #

        # What is the exact time we are looking for. (round to 6 hours)
        if not filename:
            filename, date = self.database(overpass)

        time_str = date.strftime('%Y%m%dT%H%M')
        self.times.append(time_str)

        # Read in the Harmonie data for the selected area.
        # Load data from grib file
        # For now we assume that we need the whole dataset
        dat_nums = ['33', '34', '11', '51', '76', '58']
        dat_types = ['Wind_u', 'Wind_v', 'Temperature', 'Specific humidity', 'Specific cloud liquid water content',
                     'Specific cloud ice water content']
        dat_names = ['U component of wind', 'V component of wind', 'Temperature', 'Specific humidity', 'Cloud water',
                     'CIWC Cloud ice kg m**-2']

        # Load .grib file model levels
        # logging.info('Calculate delays from Harmonie data for time ' + time
        if not filename in self.filenames:
            logging.info('Loading data file ' + filename)
            self.filenames.append(filename)

            cycle = int(os.path.basename(filename)[2:4])

            self.model_data[time_str] = dict()

            for dat_type, dat_num, dat_name in zip(dat_types, dat_nums, dat_names):
                try:
                    if cycle == 40:
                         var_data = cfgrib.open_dataset(filename, backend_kwargs={
                            'filter_by_keys':  {'typeOfLevel': 'hybrid', 'parameterName': str(dat_num)}})
                    else:
                        if dat_name == 'CIWC Cloud ice kg m**-2':
                            var_data = cfgrib.open_dataset(filename, backend_kwargs={
                                'filter_by_keys': {'typeOfLevel': 'hybrid', 'parameterName': dat_name}})
                        else:
                            var_data = cfgrib.open_dataset(filename, backend_kwargs={
                                'filter_by_keys': {'typeOfLevel': 'hybrid', 'name': dat_name}})
                except:
                    logging.warning('Variable ' + dat_type + ' is missing from dataset.')

                var_name = list(var_data.data_vars.keys())[0]
                self.model_data[time_str][dat_type] = var_data[var_name]

            # Calculate pressure and heights for model levels.
            if cycle == 40:
                geo_data = cfgrib.open_dataset(filename, backend_kwargs={
                    'filter_by_keys':  {'typeOfLevel': 'heightAboveGround', 'parameterName': '6'}})
            else:
                geo_data = cfgrib.open_dataset(filename, backend_kwargs={
                    'filter_by_keys':  {'typeOfLevel': 'heightAboveGround', 'name': 'Geopotential'}})
            geo_varname = list(geo_data.data_vars.keys())[0]
            geo = geo_data[geo_varname]

            # Latitude/Longitude values
            coor_names = list(geo.indexes.keys())
            self.model_data[time_str]['projection'] = pyproj.CRS.from_epsg(4326)
            self.model_data[time_str]['latitude'] = (
                    geo['latitude'].values[:, None] * np.ones(len(geo['longitude']))[None, :])
            self.model_data[time_str]['longitude'] = (
                    geo['longitude'].values[None, :] * np.ones(len(geo['latitude']))[:, None])

            # Surface height values
            self.model_data[time_str]['surface_height'] = ModelToDelay.geopotential_height_to_real_height(
                self.model_data[time_str]['latitude'], geo.values / 9.80665)

            if cycle == 40:
                pressure_data = cfgrib.open_dataset(filename, backend_kwargs={'read_keys': ['pv'],
                    'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'parameterName': '1'}})
            elif type == 'Num':
                pressure_data = cfgrib.open_dataset(filename, backend_kwargs={'read_keys': ['pv'],
                    'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'name': 'Pressure'}})
            pressure_varname = list(pressure_data.data_vars.keys())[0]
            pressure = pressure_data[pressure_varname]
            
            pressure_ground = pressure.values
            levels = len(pressure.attrs['GRIB_pv']) // 2 - 1
            a = np.array(pressure.attrs['GRIB_pv'][:levels + 1])
            b = np.array(pressure.attrs['GRIB_pv'][levels + 1:])
            self.model_data[time_str]['levels'] = levels

            # Now load a and b values to calculate pressure levels and heights.
            self.model_data[time_str]['pressures'] = pressure_ground * b[:, None, None] + a[:, None, None]

    def remove_harmonie(self, filename):

        self.model_data.pop(filename)
