import logging
import os

from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_database import ECMWFDatabase

class CDSRequest(object):

    def __init__(self, data_type, dataset_class, grid, area_mars, area_list, dataset):

        try:
            import cdsapi
        except:
            raise ImportError('Cannot import cdsapi or ecmwfapi')

        self.grid = grid
        self.area_mars = area_mars
        self.area_list = area_list
        self.dataset_class = dataset_class
        self.data_type = data_type
        self.dataset = dataset
        self.server = cdsapi.Client()

    def __call__(self, request):

        logging.info('Start download ECMWF data. Monitor your requests on https://cds.climate.copernicus.eu/cdsapp#!/yourrequests')
        if self.dataset == 'surface':
            target = request['download_surface']
            logging.info('Downloading CDS surface data to ' + target)

            if self.data_type in ['reanalysis-era5-pressure-levels', 'reanalysis-era5-complete']:

                self.server.retrieve('reanalysis-era5-single-levels', {
                     'product_type': 'reanalysis',
                     'format': 'grib',
                     'grid': self.grid,
                     'year': request['year'],
                     'month': request['month'],
                     'day': request['days'],
                     'time': request['hours'],
                     'area': self.area_list,
                     'variable': [
                         'boundary_layer_height', 'geopotential', 'mean_sea_level_pressure',
                         'surface_pressure', 'total_column_water_vapour',
                     ]},
                    target)

            elif self.data_type in ['reanalysis-cerra-pressure-levels', 'reanalysis-cerra-model-levels']:
                req = {
                     'data_type': 'reanalysis',
                     'level_type': 'surface_or_atmosphere',
                     'format': 'grib',
                     'year': request['year'],
                     'month': request['month'],
                     'day': request['days'],
                     'time': request['hours'],
                     'variable': [
                         '2m_relative_humidity', '2m_temperature', 'mean_sea_level_pressure',
                         'surface_pressure', 'total_column_integrated_water_vapour'
                     ]}

                if 'forecast_hours' in request.keys():
                    req['leadtime_hour'] = request['forecast_hours']
                    req['product_type'] = 'forecast'
                else:
                    req['product_type'] = 'analysis'
                    req['variable'].append('orography')

                self.server.retrieve('reanalysis-cerra-single-levels', request=req, target=target)

            elif self.data_type == 'oper':
                # Only possible if you have access via ecmwfapi.
                # https://pypi.org/project/ecmwf-api-client/

                self.service.retrieve({
                    "class": self.dataset_class,
                    "date": request['day_list'],
                    'grid': self.grid,
                    "expver": "1",
                    'area': self.area_mars,
                    "levelist": "1",
                    "levtype": "ml",
                    "param": "129/134/137/151",
                    "step": "0",
                    "stream": "oper",
                    "time": request['hour_list'],
                    "type": "an"},
                    target)

        elif self.dataset == 'atmosphere':
            target = request['download_atmosphere']
            logging.info('Downloading CDS surface data to ' + target)

            if self.data_type == 'reanalysis-era5-pressure-levels':

                self.server.retrieve('reanalysis-era5-pressure-levels', {
                     'product_type': 'reanalysis',
                     'format': 'grib',
                     'area': self.area_list,
                     'grid': self.grid,
                     'pressure_level': request['pressures'],
                     'year': request['year'],
                     'month': request['month'],
                     'day': request['days'],
                     'time': request['hours'],
                     'variable': [
                         'geopotential', 'specific_cloud_ice_water_content',
                         'specific_humidity',
                         'specific_snow_water_content', 'temperature', 'u_component_of_wind',
                         'v_component_of_wind',
                     ]},
                    target)

            elif self.data_type == 'reanalysis-era5-complete':

                self.server.retrieve(self.data_type, {
                    "date": request['day_list'],
                    'grid': self.grid,
                    'area': self.area_mars,
                    "levelist": request['level_list'],
                    "levtype": "ml",
                    "param": "130/131/132/133/246/247",
                    "stream": "oper",
                    "time": request['hour_list'],
                    "type": "an",
                }, target)

            elif self.data_type == 'reanalysis-cerra-model-levels':

                self.server.retrieve('reanalysis-cerra-model-levels', {
                     'data_type': 'reanalysis',
                     'format': 'grib',
                     "model_level": request['levels'],
                     'year': request['year'],
                     'month': request['month'],
                     'day': request['days'],
                     'time': request['hours'],
                     'variable': [
                         'specific_humidity', 'temperature', 'u_component_of_wind',
                         'v_component_of_wind',
                     ]},
                    target)

            elif self.data_type == 'reanalysis-cerra-pressure-levels':
                req = {
                    'data_type': 'reanalysis',
                    'format': 'grib',
                    'pressure_level': request['pressures'],
                    'year': request['year'],
                    'month': request['month'],
                    'day': request['days'],
                    'time': request['hours'],
                    'variable': [
                        'geopotential', 'relative_humidity' , 'temperature', 'u_component_of_wind',
                        'v_component_of_wind',
                    ]}

                if 'forecast_hours' in request.keys():
                    req['leadtime_hour'] = request['forecast_hours']
                    req['product_type'] = 'forecast'
                    req['variable'].extend(['specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content'])
                else:
                    req['product_type'] = 'analysis'

                self.server.retrieve('reanalysis-cerra-pressure-levels', request=req, target=target)

        # Finally unpack for specific times and remove downloaded file.
        ECMWFDatabase.split_grib_file(target, input_step='month')

        # Finally remove the target file after splitting
        os.remove(target)
