import numpy as np
import datetime


class InterferogramNetwork(object):

    def __init__(self, image_dates, master_date=[], image_baselines=[], network_type='temp_baseline', temporal_baseline=60, temporal_no=3, spatial_baseline=2000):

        self.type = network_type
        self.temp_baseline = temporal_baseline
        self.spat_baseline = spatial_baseline
        self.temp_no = temporal_no          # Number of images before and after image in time used to form a network of ifg.

        date_int = np.sort([int(key) for key in image_dates])
        if not master_date:
            master_int = date_int[0]
        else:
            master_int = int(master_date)
        self.master_date = datetime.datetime.strptime(str(master_int), '%Y%m%d')
        self.dates = np.array([datetime.datetime.strptime(str(date), '%Y%m%d') for date in date_int])
        self.spat_baselines = image_baselines

        self.ifg_pairs = []

        if self.type == 'temp_baseline':
            self.temporal_baseline()
        elif self.type == 'daisy_chain':
            self.daisy_chain()
        elif self.type == 'single_master':
            self.single_master()

    def temporal_baseline(self):

        days = np.array([diff.days for diff in self.dates - np.min(self.dates)])

        # Define network based on network type
        for n in np.arange(len(days)):
            ids = np.where((days - days[n] > 0) * (days - days[n] <= self.temp_baseline))[0]
            for id in ids:
                self.ifg_pairs.append([n, id])

    def daisy_chain(self):

        n_im = len(self.dates)
        for i in np.arange(n_im):
            for n in np.arange(0, self.temp_no + 1):
                if n + i < n_im and i != n:
                    self.ifg_pairs.append([i, n])

    def single_master(self):

        master_n = np.where(self.dates == self.master_date)[0][0]

        for n in np.arange(len(self.dates)):
            if n != master_n:
                self.ifg_pairs.append([master_n, n])
