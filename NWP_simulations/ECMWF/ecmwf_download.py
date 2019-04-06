# This class both downloads and reads an ECMWF dataset as a preprocessing step
import calendar
import datetime
import os
import numpy as np
from ecmwfapi import ECMWFDataServer, ECMWFService
from rippl.NWP_simulations.ECMWF.ecmwf_type import ECMWFType
from joblib import Parallel, delayed
from rippl.NWP_simulations.ECMWF.ecmwf_mars_request import MarsRequest


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

        # Surface files
        mars_request = MarsRequest(self.data_type, self.dataset_class, t_list, bb_str, grid, level_list, 'surface')
        Parallel(n_jobs=self.n_processes)(delayed(mars_request)(date, target) for
                                          date, target in
                                          zip(surface_dates, surface_files))

        # Atmosphere files
        mars_request = MarsRequest(self.data_type, self.dataset_class, t_list, bb_str, grid, level_list, 'atmosphere')
        Parallel(n_jobs=self.n_processes)(delayed(mars_request)(date, target) for
                                          date, target in
                                          zip(atmosphere_dates, atmosphere_files))
