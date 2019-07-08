import os
from ecmwfapi import ECMWFService, ECMWFDataServer
from cdsapi import Client
from time import sleep
import random

class MarsRequest(object):

    def __init__(self, data_type, dataset_class, t_list, bb_str, grid, level_list, dataset):

        self.t_list = t_list
        self.bb_str = bb_str
        self.grid = grid
        self.dataset_class = dataset_class
        self.data_type = data_type
        self.level_list = level_list
        self.dataset = dataset

        if self.data_type == 'interim':
            self.server = ECMWFDataServer()
        elif self.data_type == 'era5':
            self.server = Client()
        else:
            self.service = ECMWFService("mars")

    def __call__(self, date, target):

        if os.path.exists(target):
            print('ECMWF surface data is already downloaded at ' + target)
            return

        sleep(random.randint(1, 30))

        while not os.path.exists(target):

            print('Downloading ECMWF file' + target)

            if self.dataset == 'surface':
                if self.data_type == 'interim':

                    self.server.retrieve({
                        "class": self.dataset_class,
                        "dataset": self.data_type,
                        "date": date,
                        "expver": "1",
                        "grid": self.grid,
                        "levtype": "sfc",
                        "param": "129.128/134.128",
                        "step": "0",
                        "stream": "oper",
                        "time": self.t_list,
                        "type": "an",
                        "area": self.bb_str,
                        "target": target,
                    })

                elif self.data_type == 'era5':

                    # Translate to cds date info
                    year = date[:4]
                    month = date[5:7]
                    last_day = int(date[22:])
                    days = [str(d).zfill(2) for d in range(1, last_day)]
                    time = [str(h).zfill(2) + ':00' for h in range(24)]
                    bb_str = self.bb_str.split('/')
                    grid_str = self.grid.split('/')

                    self.server.retrieve('reanalysis-era5-single-levels',
                        {
                            "product_type": 'reanalysis',
                            "variable": ['orography', 'surface_pressure'],
                            "year": year,
                            "month": month,
                            "day": days,
                            "grid": grid_str,
                            "time": time,
                            "area": bb_str,
                            "format": "grib"
                        },
                        target)

                elif self.data_type == 'oper':

                    self.service.execute({
                        "class": self.dataset_class,
                        # "dataset": self.data_type,
                        "date": date,
                        "expver": "1",
                        "grid": self.grid,
                        "levtype": "sfc",
                        "param": "129.128/134.128",
                        "step": "0",
                        "stream": "oper",
                        "time": self.t_list,
                        "type": "an",
                        "area": self.bb_str},
                        target)

            elif self.dataset == 'atmosphere':
                if self.data_type == 'interim':

                    self.server.retrieve({
                        "class": self.dataset_class,
                        "dataset": self.data_type,
                        "date": date,
                        "expver": "1",
                        "grid": self.grid,
                        "levelist": self.level_list,
                        "levtype": "ml",
                        "param": "130.128/133.128/246.128/247.128",
                        "step": "0",
                        "stream": "oper",
                        "time": self.t_list,
                        "type": "an",
                        "area": self.bb_str,
                        "target": target
                    })

                elif self.data_type == 'era5':

                    self.server.retrieve('reanalysis-era5-complete',
                        {
                            "class": self.dataset_class,
                            "date": date,
                            "expver": "1",
                            "grid": self.grid,
                            "levelist": self.level_list,
                            "levtype": "ml",
                            "param": "130.128/133.128/246.128/247.128",
                            "step": "0",
                            "stream": "oper",
                            "time": self.t_list,
                            "type": "an",
                            "area": self.bb_str
                        },
                        target)

                elif self.data_type == 'oper':

                    self.service.execute({
                        "class": self.dataset_class,
                        # "dataset": self.data_type,
                        "date": date,
                        "expver": "1",
                        "grid": self.grid,
                        "levelist": self.level_list,
                        "levtype": "ml",
                        "param": "130.128/133.128/246.128/247.128",
                        "step": "0",
                        "stream": "oper",
                        "time": self.t_list,
                        "type": "an",
                        "area": self.bb_str},
                        target)

            print('Failed to download, retrying in 100 seconds')
            sleep(100)
