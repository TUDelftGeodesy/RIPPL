# Load the ECMWF data from the data files as input for ray tracing.
import datetime

import numpy as np

from rippl.NWP_model_delay.ray_tracing.model_reference import ModelReference


class ECMWFData(object):

    def __init__(self, n_levels):

        # Initialize the model data
        self.model_data = dict()
        self.surface_model_files = []
        self.surface_model_data = dict()
        self.atmosphere_model_files = []
        self.atmosphere_model_data = dict()

        # Load the level information
        self.levels = n_levels
        self.times = []
        self.a, self.b = ModelReference.get_a_b_coef(self.levels)

    def load_ecmwf(self, date, filename):
        # This function resamples the ECMWF grid to different 2d cross-section for different lines.
        # Input is:
        # - dat_file    > the ECMWF .grib file
        # - lat / lon   > the far range points on a line (we assume a straight line in lat/lon)
        # - heading     > heading of ray from point to satellite (degrees from north [0-360])
        # - max_dist    > determines how long we follow this direction (4-5 degrees should be fine

        # Load pygrib
        try:
            import pygrib
        except:
            ImportError('Pygrib package not installed!')

        # What is the exact time we are looking for? (round to 6 hours)
        minute = date.minute
        if minute > 30:
            used_date = date + datetime.timedelta(hours=1)
        else:
            used_date = date
        hour = used_date.hour
        data_date = used_date.strftime('%Y%m%d')
        time = date.strftime('%Y%m%dT%H%M')
        self.times.append(time)

        # Read in the ECMWF data for the selected area.
        # Load data from grib file
        # For now we assume that we need the whole dataset TODO select only part of ECMWF data

        # Load .grib file model levels
        # print('Calculate delays from ECMWF data for time ' + time)

        if not filename + '_atmosphere.grb' in self.atmosphere_model_files:
            # print('Loading data file ' + filename + '_atmosphere.grb')
            self.atmosphere_model_data[filename] = pygrib.index(filename + '_atmosphere.grb', 'name', 'level', 'hour', 'dataDate')
            self.atmosphere_model_files.append(filename + '_atmosphere.grb')
        if not filename + '_surface.grb' in self.surface_model_files:
            # print('Loading data file ' + filename + '_surface.grb')
            self.surface_model_files.append(filename + '_surface.grb')
            self.surface_model_data[filename] = pygrib.index(filename + '_surface.grb', 'name', 'hour', 'dataDate')

        dat_types = ['Specific humidity', 'Temperature', 'Specific cloud liquid water content',
                     'Specific cloud ice water content']

        var = self.atmosphere_model_data[filename](name='Temperature', level=1, hour=hour,
                                                            dataDate=data_date)[0]
        latitudes = np.unique(var['latitudes'])
        longitudes = np.unique(var['longitudes'])
        self.model_data['latitudes'] = latitudes
        self.model_data['longitudes'] = longitudes
        dat_shape = (self.levels, len(latitudes), len(longitudes))

        self.model_data[time] = dict()

        for dat_type in dat_types:

            self.model_data[time][dat_type] = np.zeros(shape=dat_shape)

            for level in range(1, self.levels + 1):
                var = self.atmosphere_model_data[filename](name=dat_type, level=level, hour=hour,
                                                                    dataDate=data_date)[0]
                self.model_data[time][dat_type][level - 1, :, :] = np.flipud(var.values)

        # Calculate pressure and heights for model levels.
        # We have different files for surface files (surface pressure is different)
        geo = self.surface_model_data[filename].select(name='Geopotential', hour=hour,
                                                                dataDate=data_date)[0]
        self.model_data['geo_h'] = np.flipud(geo.values) / 9.80665

        try:
            log_p = self.surface_model_data[filename].select(name='Logarithm of surface pressure', hour=hour,
                                                                  dataDate=data_date)[0]
            geo_p = np.flipud(np.exp(log_p.values))
        except:
            log_p = self.surface_model_data[filename].select(name='Surface pressure', hour=hour,
                                                                  dataDate=data_date)[0]
            geo_p = np.flipud(log_p.values)

        # Now load a and b values to calculate pressure levels and heights.
        self.model_data[time]['pressures'] = geo_p * self.b[:, None, None] + self.a[:, None, None]

    def remove_ecmwf(self, time):

        self.model_data.pop(time)
