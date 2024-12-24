# Load the CDS data from the data files as input for ray tracing.
import datetime

import numpy as np
import xarray as xr
import logging
import pyproj

from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_database import ECMWFDatabase
from rippl.NWP_model_delay.ray_tracing_NWP_data.model_to_delay import ModelToDelay

"""
# Test download of different types of ECMWF data:
# - ERA5 model levels
# - ERA5 pressure levels
# - CERRA model levels
# - CERRA pressure levels

for data_type in ['era5_pressure_levels', 'cerra_pressure_levels', 
                  'cerra_model_levels', 'era5_model_levels']

lonlim = [-2, 12]
latlim = [45, 56]
overpass_time = datetime.datetime(year=2020, month=1, day=1, hour=5, minute=23, second=0)

loaded_data = dict()

if 'era5' in data_type:               
    loaded_data[data_type] = CDSdownload(data_type=data_type, latlim=latlim, lonlim=lonlim, overpass_times=[overpass_time], processes=1)
else:
    loaded_data[data_type] = CDSdownload(data_type=data_type, overpass_times=[overpass_time], processes=1)

"""

class CDSData(object):

    def __init__(self, model_type='era5_pressure_levels'):

        # Initialize the database
        self.database = ECMWFDatabase()
        model_types = ['era5_pressure_levels', 'era5_model_levels', 'cerra_pressure_levels', 'cerra_model_levels']
        if model_type in model_types:
            self.model_type = model_type
        else:
            raise TypeError(model_type + ' not in ' + model_types)

        # Initialize the model data
        self.model_data = dict()
        self.surface_model_files = []
        self.surface_model_data = dict()
        self.atmosphere_model_files = []
        self.atmosphere_model_data = dict()

        # Level and time information
        self.times = []

    def load_ecmwf(self, time='', atmosphere_filename='', surface_filename=''):
        # This function loads the ECMWF data based on either the overpass time of the satellite or on the names of the
        # surface and atmosphere files themselves.

        # What is the exact time we are looking for?
        if isinstance(time, datetime.datetime) and (not atmosphere_filename or not surface_filename):
            atmosphere_filename, surface_filename = self.database(time, ecmwf_type=self.model_type)
            if not atmosphere_filename or not surface_filename:
                raise FileNotFoundError('NWP data for ' + time.strftime('%Y%m%dT%H:M') + ' not found. Aborting.')

        an_time, fc_time, date_type, shape = ECMWFDatabase.get_ecmwf_datetime_area(atmosphere_filename)
        self.times.append(fc_time)
        time_str = fc_time.strftime('%Y%m%dT%H%M')
        self.model_data[time_str] = {}

        # Index for dataset
        time_index = np.datetime64(an_time)
        step_index = np.datetime64(fc_time) - np.datetime64(an_time)
        # Read in the CDS data for the selected area.
        # Load data from grib file
        # Load .grib file model levels
        # logging.info('Calculate delays from CDS data for time ' + time)
        if not atmosphere_filename in self.atmosphere_model_files:
            logging.info('Loading data file ' + atmosphere_filename)
            atmosphere = xr.open_dataset(atmosphere_filename)
            self.atmosphere_model_data[time_str] = atmosphere
            self.atmosphere_model_files.append(atmosphere_filename)
        if not surface_filename in self.surface_model_files:
            logging.info('Loading data file ' + surface_filename)
            self.surface_model_files.append(surface_filename)
            surface = xr.open_dataset(surface_filename)
            self.surface_model_data[time_str] = surface

        # Set the projection string:
        if 'cerra' in atmosphere_filename:
            proj_str = '+proj=lcc +lat_0=50 +lat_1=50 +lat_2=50 +lon_0=8 +R=6371229'
            self.model_data[time_str]['projection'] = pyproj.CRS.from_proj4(proj_str)
            self.model_data[time_str]['x'] = (np.arange(-534, 535) * 5500)[None, :] * np.ones([1069, 1])
            self.model_data[time_str]['y'] = (np.arange(-534, 535) * 5500)[:, None] * np.ones([1, 1069])
            self.model_data[time_str]['latitude'] = atmosphere['latitude'].values
            self.model_data[time_str]['longitude'] = atmosphere['longitude'].values
            self.model_data[time_str]['longitude'][self.model_data[time_str]['longitude'] > 180] -= 360
        if 'era5' in atmosphere_filename:
            self.model_data[time_str]['x'] = []
            self.model_data[time_str]['y'] = []
            self.model_data[time_str]['projection'] = pyproj.CRS.from_epsg(4326)
            self.model_data[time_str]['latitude'] = (
                    atmosphere['latitude'].values[:, None] * np.ones(len(atmosphere['longitude']))[None, :])
            self.model_data[time_str]['longitude'] = (
                    atmosphere['longitude'].values[None, :] * np.ones(len(atmosphere['latitude']))[:, None])

        # Load the input variables
        dat_types = ['Specific humidity', 'Relative humidity', 'Temperature', 'Specific cloud liquid water content',
                     'Specific cloud ice water content', 'U component of wind', 'V component of wind']
        dat_codes = ['q', 'r', 't', 'cswc', 'ciwc', 'u', 'v']

        var_names = list(atmosphere.variables.keys())
        for dat_code, dat_type in zip(dat_codes, dat_types):
            if dat_code in var_names:
                self.model_data[time_str][dat_type] = np.squeeze(atmosphere[dat_code])

        # Calculate pressure and heights for model levels.
        # We have different files for surface files (surface pressure is different)
        if 'z' in surface.keys():
            geo_height = surface['z'] / 9.80665
            self.model_data[time_str]['surface_height'] = ModelToDelay.geopotential_height_to_real_height(
                self.model_data[time_str]['latitude'], geo_height)
        elif 'h' in surface.keys():        # Otherwise use orography.
            self.model_data[time_str]['surface_height'] = surface['h'].values()

        if 'isobaricInhPa' in atmosphere.keys():
            # Make a full cube for the derived pressures
            pressures_layers = np.array(list(np.squeeze(atmosphere['isobaricInhPa'].values[:, None, None]))) * 100
            pressures = np.zeros(len(pressures_layers) + 1)
            pressures[0] = pressures_layers[0] + (pressures_layers[0] - pressures_layers[1]) / 2
            pressures[1:-1] = (pressures_layers[:-1] + pressures_layers[1:]) / 2 # Get values top and bottom layers
            self.model_data[time_str]['pressures'] = (pressures[:, None, None] *
                                                  np.ones(self.model_data[time_str]['latitude'].shape)[None, :, :])

            # Load and add the surface pressure in the lowest layer
            mean_sea_level_pressure = surface['msl'][:, :]
            self.model_data[time_str]['pressures'][0, :, :] = mean_sea_level_pressure
            self.model_data[time_str]['surface_height'] = np.zeros(mean_sea_level_pressure.shape)

            # Adjust all other layers that are lower than the pressure at sea level
            diff_values = self.model_data[time_str]['pressures'] - mean_sea_level_pressure.values[None, :, :]
            diff_values[diff_values < 0] = 0
            self.model_data[time_str]['pressures'] = self.model_data[time_str]['pressures'] - diff_values

        else:
            # Estimate the pressures based of the geopotential at the surface.
            if 'z' in surface.keys():
                geo_p = surface['z']
            elif 'h' in surface.keys():
                height = surface['h'].values()
                geo_height = ModelToDelay.real_height_to_geopotential_height(self.model_data[time_str]['latitude'], height)
                geo_p = geo_height * 9.80665

            # Now load a and b values to calculate pressure levels and heights.
            coefficients = np.array(atmosphere.t.attrs['GRIB_pv'])
            levels = int(len(coefficients) / 2 - 1)
            a = np.flip(coefficients[:levels + 1])
            b = np.flip(coefficients[levels + 1:])
            self.model_data[time_str]['levels'] = levels
            self.model_data[time_str]['pressures'] = geo_p * b[:, None, None] + a[:, None, None]

            # Finally flip all layers upside down to start from the ground layer. (Not sure whether the model levels
            # are always called hybrid levels)
            for dat_type in dat_types:
                self.model_data[time_str][dat_type] = self.model_data[time_str][dat_type].isel(hybrid=slice(None, None, -1))

        # Convert relative to specific humidity
        if 'r' in var_names and 'q' not in var_names:
            mid_pressures = (self.model_data[time_str]['pressures'][:-1, :, :] +
                             self.model_data[time_str]['pressures'][1:, :, :]) / 2

            self.model_data[time_str]['Specific humidity'] = ModelToDelay.relative_to_specific_humidity(
                self.model_data[time_str]['Temperature'], self.model_data[time_str]['Relative humidity'],
                mid_pressures
            )

        self.model_data[time_str]['levels'] = self.model_data[time_str]['pressures'].shape[0] - 1

    def remove_ecmwf(self, time):

        self.model_data.pop(time)
