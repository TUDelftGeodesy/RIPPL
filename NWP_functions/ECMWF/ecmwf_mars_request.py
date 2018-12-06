import os
from ecmwfapi import ECMWFService, ECMWFDataServer


class MarsRequest(object):

    def __init__(self, data_type, dataset_class, t_list, bb_str, grid, level_list, dataset):

        self.t_list = t_list
        self.bb_str = bb_str
        self.grid = grid
        self.dataset_class = dataset_class
        self.data_type = data_type
        self.level_list = level_list
        self.dataset = dataset

        if self.data_type in ['interim', 'era5']:
            self.server = ECMWFDataServer()
        else:
            self.service = ECMWFService("mars")

    def __call__(self, date, target):

        if not os.path.exists(target):
            print('Downloading ECMWF surface data to ' + target)

            if self.dataset == 'surface':
                if self.data_type in ['interim', 'era5']:

                    self.server.retrieve({
                        "class": self.dataset_class,
                        "dataset": self.data_type,
                        "date": date,
                        "expver": "1",
                        "grid": self.grid,
                        "levelist": "1",
                        "levtype": "ml",
                        "param": "129.128/152.128",
                        "step": "0",
                        "stream": "oper",
                        "time": self.t_list,
                        "type": "an",
                        "area": self.bb_str,
                        "target": target,
                    })

                elif self.data_type in ['oper']:

                    self.service.execute({
                        "class": self.dataset_class,
                        # "dataset": self.data_type,
                        "date": date,
                        "expver": "1",
                        "grid": self.grid,
                        "levelist": "1",
                        "levtype": "ml",
                        "param": "129.128/152.128",
                        "step": "0",
                        "stream": "oper",
                        "time": self.t_list,
                        "type": "an",
                        "area": self.bb_str},
                        target)

            elif self.dataset == 'atmosphere':
                if self.data_type in ['interim', 'era5']:

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
                        "target": target,
                    })

                elif self.data_type in ['oper']:

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
        else:
            print('ECMWF surface data is already downloaded at ' + target)
