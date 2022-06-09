# Load the ECMWF data from the data files as input for ray tracing.
import numpy as np

from rippl.NWP_model_delay.ray_tracing.model_reference import ModelReference


class HarmonieData(object):

    def __init__(self):
        # Load pygrib

        # Initialize the model data
        self.model_data = dict()
        self.grib_files = []
        self.grib_data = dict()

        # Load the level information
        self.levels = 65
        self.times = []

    def load_harmonie(self, date, filename):
        # This function resamples the ECMWF grid to different 2d cross-section for different lines.
        # Input is:
        # - dat_file    > the ECMWF .grib file
        # - lat / lon   > the far range points on a line (we assume a straight line in lat/lon)
        # - heading     > heading of ray from point to satellite (degrees from north [0-360])
        # - max_dist    > determines how long we follow this direction (4-5 degrees should be fine
        #

        # What is the exact time we are looking for. (round to 6 hours)
        time = date.strftime('%Y%m%dT%H%M')
        self.times.append(time)

        # Read in the ECMWF data for the selected area.
        # Load data from grib file
        # For now we assume that we need the whole dataset TODO select only part of ECMWF data

        # Load .grib file model levels
        # print('Calculate delays from Harmonie data for time ' + time)

        try:
            import pygrib
        except:
            ImportError('pygrib package not installed!')

        if not filename in self.grib_files:
            # print('Loading data file ' + filename + '_atmosphere.grb')

            self.grib_data[filename] = pygrib.index(filename, 'name', 'level')
            type = 'Name'

            try:
                var = self.grib_data[filename](name='Temperature', level=1)[0]
            except:
                self.grib_data[filename] = pygrib.index(filename, 'parameterName', 'level')
                var = self.grib_data[filename](parameterName='51', level=1)[0]
                type = 'Num'

            self.grib_files.append(filename)
        else:
            type = 'Name'

            try:
                var = self.grib_data[filename](name='Temperature', level=1)[0]
            except:
                var = self.grib_data[filename](parameterName='51', level=1)[0]
                type = 'Num'

        dat_nums = ['33', '34', '11', '51', '76', '58']
        dat_types = ['Wind_u', 'Wind_v', 'Temperature', 'Specific humidity', 'Specific cloud liquid water content',
                     'Specific cloud ice water content']
        dat_names = ['Wind_u', 'Wind_v', 'Temperature', 'Specific humidity', 'Cloud water',
                     'CIWC Cloud ice kg m**-2']

        latitudes = np.unique(var['latitudes'])
        longitudes = np.unique(var['longitudes'])
        self.model_data['latitudes'] = latitudes
        self.model_data['longitudes'] = longitudes
        dat_shape = (self.levels, len(latitudes), len(longitudes))

        self.model_data[time] = dict()

        for dat_type, dat_num, dat_name in zip(dat_types, dat_nums, dat_names):

            self.model_data[time][dat_type] = np.zeros(shape=dat_shape)

            for level in range(1, self.levels + 1):
                try:
                    if type == 'Name':
                        var = self.grib_data[filename](name=dat_name, level=level)[0]
                    elif type == 'Num':
                        var = self.grib_data[filename](parameterName=dat_num, level=level)[0]

                    self.model_data[time][dat_type][level - 1, :, :] = np.transpose(var.values)
                except:
                    self.model_data[time][dat_type][level - 1, :, :] = np.zeros((dat_shape[1], dat_shape[2]))

        # Calculate pressure and heights for model levels.
        if type == 'Name':
            geo = self.grib_data[filename](name='Geopotential', level=0)[0]
        elif type == 'Num':
            geo = self.grib_data[filename](parameterName='6', level=0)[0]
        self.model_data['geo_h'] = np.transpose(geo.values / 9.80665)

        if type == 'Name':
            log_p = self.grib_data[filename](name='Pressure', level=0)[0]
        elif type == 'Num':
            log_p = self.grib_data[filename](parameterName='1', level=0)[0]
        geo_p = np.transpose(log_p.values)
        self.a = log_p.pv[:66]
        self.b = log_p.pv[66:]

        # Now load a and b values to calculate pressure levels and heights.
        self.model_data[time]['pressures'] = geo_p * self.b[:, None, None] + self.a[:, None, None]

    def remove_harmonie(self, filename):

        self.model_data.pop(filename)
