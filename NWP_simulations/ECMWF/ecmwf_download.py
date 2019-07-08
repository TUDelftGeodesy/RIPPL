# This class both downloads and reads an ECMWF dataset as a preprocessing step
import calendar
import datetime
import os
import numpy as np
from rippl.NWP_simulations.ECMWF.ecmwf_type import ECMWFType
from rippl.NWP_simulations.ECMWF.ecmwf_parallel_download import parallel_download
import pygrib
import multiprocessing
import copy


class ECMWFdownload(ECMWFType):
    
    def __init__(self, latlim, lonlim, ecmwf_data_folder, data_type='era_interim', processes=8):
        # In the init function we mainly check the different dates, data folders and extend of dataset.

        ECMWFType.__init__(self, data_type)

        # Init folder
        self.ecmwf_data_folder = os.path.join(ecmwf_data_folder, data_type)
        if not os.path.exists(self.ecmwf_data_folder):
            os.mkdir(self.ecmwf_data_folder)
        self.egm96 = []

        self.filenames = []
        self.latlim = latlim
        self.lonlim = lonlim
        self.bbox = np.asarray([latlim, lonlim])
        self.n_processes = processes
        
        # Date should be in datetime format.
        self.dates = []
        self.time = []

        # Initialize variable of final product
        self.requests = dict()
        self.ecmwf_data = dict()

    def prepare_download(self, dates):
        # Check the dates

        self.dates = dates
        self.time = [date.strftime('%Y%m%d%H') for date in self.dates]

        # Check if the requested dates are valid:
        for date in self.dates:
            if date.minute != 0 or date.second != 0 or date.microsecond != 0:
                print('Data is only available on hourly level')
                return
            if self.data_type == 'interim' and np.remainder(date.hour, 6) != 0:
                print('era interim date is only available for every 6 hours')
                return

        # This function downloads ECMWF data for the area within the bounding box
    
        # create bounding box string assume [[lat_min, lat_max], [lon_min, lon_max]]
        if self.bbox[0, 1] < 0:
            ul_lat = 's' + str(abs(int(self.bbox[0, 1]))).zfill(2)
        else:
            ul_lat = 'n' + str(abs(int(self.bbox[0, 1]))).zfill(2)
        if self.bbox[0, 0] < 0:
            lr_lat = 's' + str(abs(int(self.bbox[0, 0]))).zfill(2)
        else:
            lr_lat = 'n' + str(abs(int(self.bbox[0, 0]))).zfill(2)
        if self.bbox[1, 0] < 0:
            ul_lon = 'w' + str(abs(int(self.bbox[1, 0]))).zfill(3)
        else:
            ul_lon = 'e' + str(abs(int(self.bbox[1, 0]))).zfill(3)
        if self.bbox[1, 1] < 0:
            lr_lon = 'w' + str(abs(int(self.bbox[1, 1]))).zfill(3)
        else:
            lr_lon = 'e' + str(abs(int(self.bbox[1, 1]))).zfill(3)
    
        path_str = ul_lat + ul_lon + '_' + lr_lat + lr_lon
        type2str = {'interim': '_erai_', 'era5': '_era5_', 'oper': '_oper_'}

        # Define output filenames
        self.filenames = []
        
        if self.data_type in ['interim', 'era5']:
            dates = [date.strftime('%Y%m') for date in self.dates]
        elif self.data_type in ['oper']:
            dates = [date.strftime('%Y%m%d') for date in self.dates]

        for date in dates:
            if date not in self.requests.keys():
                self.requests[date] = dict()

                if self.data_type in ['interim', 'era5']:
                    d = datetime.datetime.strptime(date, '%Y%m')
                    days = calendar.monthrange(d.year, d.month)
                    request_date = (d.strftime('%Y-%m-') + str(1).zfill(2) + '/to/' +
                                    d.strftime('%Y-%m-') + str(days[1]).zfill(2))
                elif self.data_type in ['oper']:
                    d = datetime.datetime.strptime(date, '%Y%m%d')
                    request_date = d.strftime('%Y-%m-%d')
                else:
                    return

                self.requests[date]['atmosphere'] = os.path.join(self.ecmwf_data_folder, 'ecmwf' + type2str[self.data_type] + date + '_' + path_str + '_atmosphere.grb')
                self.requests[date]['surface'] = os.path.join(self.ecmwf_data_folder, 'ecmwf' + type2str[self.data_type] + date + '_' + path_str + '_surface.grb')
                self.requests[date]['request'] = request_date

        dates = [date.strftime('%Y%m%d') for date in self.dates]
        for date in dates:
            self.filenames.append(os.path.join(self.ecmwf_data_folder, 'ecmwf' + type2str[self.data_type] +
                                                   date + '_' + path_str))

    def download(self):

        bb_str = str(int(self.bbox[0, 1])) + '/' + str(int(self.bbox[1, 0])) + '/' + str(
            int(self.bbox[0, 0])) + '/' + str(int(self.bbox[1, 1]))

        # create levels and time string
        level_list = ''
        for l in range(1, self.levels+1):
            level_list += str(l) + '/'
        level_list = level_list[:-1]

        t_list = ''
        for t in range(0, 24, int(self.t_step.seconds // 3600)):
            t_list += str(t).zfill(2) + ':00:00' + '/'

        t_list = t_list[:-1]
        grid = str(self.grid_size) + '/' + str(self.grid_size)

        surface_files = []
        surface_dates = []
        atmosphere_files = []
        atmosphere_dates = []

        for key in self.requests.keys():
            # set all project_functions parameters
            if not os.path.exists(self.requests[key]['surface']):
                surface_files.append(self.requests[key]['surface'])
                surface_dates.append(self.requests[key]['request'])
            if not os.path.exists(self.requests[key]['atmosphere']):
                atmosphere_files.append(self.requests[key]['atmosphere'])
                atmosphere_dates.append(self.requests[key]['request'])

        if len(surface_files) > 0:
            print('Surface files to be downloaded')
            for surface_file in surface_files:
                print(surface_file)

        # Surface files
        input = dict()
        input['data_type'] = self.data_type
        input['dataset_class'] = self.dataset_class
        input['t_list'] = t_list
        input['bb_str'] = bb_str
        input['grid'] = grid
        input['level_list'] = level_list
        input['dataset'] = 'surface'

        inputs = []
        for date, target in zip(surface_dates, surface_files):
            input['date'] = date
            input['target'] = target
            inputs.append(copy.copy(input))

        self.pool = multiprocessing.Pool(self.n_processes, maxtasksperchild=1)
        self.pool.map(parallel_download, inputs)
        self.pool.close()

        if len(atmosphere_files) > 0:
            print('Atmosphere files to be downloaded')
            for atmosphere_file in atmosphere_files:
                print(atmosphere_file)

        input['dataset'] = 'atmosphere'
        inputs = []
        for date, target in zip(atmosphere_dates, atmosphere_files):
            input['date'] = date
            input['target'] = target
            inputs.append(copy.copy(input))

        # Atmosphere files
        self.pool = multiprocessing.Pool(self.n_processes, maxtasksperchild=1)
        self.pool.map(parallel_download, inputs)
        self.pool.close()
        self.pool = []

        # Finally split the monthly values to daily values
        self.split_monthly_to_daily(surface_files)
        self.split_monthly_to_daily(atmosphere_files)

    def split_monthly_to_daily(self, file_paths):

        """
        Test data....
        dir = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/weather_models/ecmwf_data/era5/'
        file_paths = os.listdir(dir)
        file_paths = [os.path.join(dir, f) for f in file_paths]
        file_path = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/weather_models/ecmwf_data/era5/ecmwf_era5_201804_n56w002_n45e012_atmosphere.grb'
        """

        for file_path in file_paths:
            folder = os.path.dirname(file_path)
            grib_file = os.path.basename(file_path)

            if grib_file[6:10] in ['era5', 'erai'] and grib_file[17] == '_':

                date = datetime.datetime.strptime(grib_file[11:17], '%Y%m')
                day = datetime.timedelta(days=1)
                month = date.month

                daily_dates = []
                daily_files = []

                while date.month == month:
                    grb_file = os.path.join(folder, grib_file[:11] + date.strftime('%Y%m%d') + grib_file[17:])

                    if not os.path.exists(grb_file):
                        daily_dates.append(date)
                        daily_files.append(grb_file)
                    date += day

                if len(daily_files) > 0:
                    grib_data = pygrib.index(file_path, 'month', 'day', 'year')

                    for date, file in zip(daily_dates, daily_files):
                        msgs = grib_data.select(year=date.year, month=date.month, day=date.day)

                        grbout = open(file, 'wb')
                        for msg in msgs:
                            grbout.write(msg.tostring())
                        grbout.close()

                    grib_data.close()

