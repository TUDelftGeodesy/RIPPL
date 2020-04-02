from datetime import timedelta
import os
import locale


class ECMWFType(object):

    def __init__(self, data_type='interim', data_archive=''):
        # Set locale
        locale.setlocale(locale.LC_ALL, 'en_US.utf8')

        self.data_type = data_type

        # Find the times steps for different ECMWF datasets
        if data_type == 'interim':
            self.t_step = timedelta(hours=6)
            self.data_t_step = 'month'
            self.grid_size = 0.25
            self.levels = 60
            self.dataset_class = 'ei'
        elif data_type == 'era5':
            self.t_step = timedelta(hours=1)
            self.data_t_step = 'month'
            self.grid_size = 0.25
            self.levels = 137
            self.dataset_class = 'ea'
        elif data_type == 'oper':
            self.t_step = timedelta(hours=6)
            self.data_t_step = 'day'
            self.grid_size = 0.10
            self.levels = 137
            self.dataset_class = 'od'
        else:
            print('ECMWF dataset type not recognized!')
            return

        self.data_folder = os.path.join(data_archive, data_type)
