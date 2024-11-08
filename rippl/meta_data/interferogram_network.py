import numpy as np
import datetime


class InterferogramNetwork(object):

    def __init__(self, image_dates, perpendicular_baselines=[], single_reference_date='',
                 max_temporal_baseline=0, daisy_chain_width=0, max_perpendicular_baseline=0):

        self.single_reference = single_reference_date
        self.max_temporal_baseline = max_temporal_baseline
        self.max_perpendicular_baseline = max_perpendicular_baseline
        self.temp_no = daisy_chain_width      # Number of images before and after image in time used to form a network of ifg.

        date_int = np.sort([int(key) for key in image_dates])
        if not single_reference_date:
            primary_int = date_int[0]
        else:
            primary_int = int(single_reference_date)
        self.primary_date = datetime.datetime.strptime(str(primary_int), '%Y%m%d')
        self.dates = np.array([datetime.datetime.strptime(str(date), '%Y%m%d') for date in date_int])
        self.perpendicular_baselines = perpendicular_baselines

        self.ifg_pairs = set()

        if single_reference_date:
            ifg_pairs = self.single_primary()
            if len(self.ifg_pairs) == 0:
                self.ifg_pairs = ifg_pairs
            else:
                self.ifg_pairs.intersection(ifg_pairs)
        if daisy_chain_width:
            ifg_pairs = self.daisy_chain()
            if len(self.ifg_pairs) == 0:
                self.ifg_pairs = ifg_pairs
            else:
                self.ifg_pairs.intersection(ifg_pairs)
        if max_perpendicular_baseline:
            ifg_pairs = self.find_max_perpendicular_baseline()
            if len(self.ifg_pairs) == 0:
                self.ifg_pairs = ifg_pairs
            else:
                self.ifg_pairs.intersection(ifg_pairs)
        if max_temporal_baseline:
            ifg_pairs = self.find_max_temporal_baseline()
            if len(self.ifg_pairs) == 0:
                self.ifg_pairs = ifg_pairs
            else:
                self.ifg_pairs.intersection(ifg_pairs)

        self.ifg_pairs = list(self.ifg_pairs)

    def find_max_temporal_baseline(self):

        days = np.array([diff.days for diff in self.dates - np.min(self.dates)])
        ifg_pairs = []

        # Define network based on network type
        for n in np.arange(len(days)):
            ids = np.where((days - days[n] > 0) * (days - days[n] <= self.max_temporal_baseline))[0]
            for id in ids:
                ifg_pairs.append([n, id])

        return set(tuple(ifg) for ifg in ifg_pairs)

    def daisy_chain(self):

        ifg_pairs = []

        n_im = len(self.dates)
        for i in np.arange(n_im):
            for n in np.arange(0, self.temp_no + 1):
                if n + i < n_im and i != n:
                    ifg_pairs.append([i, n])

        return set(tuple(ifg) for ifg in ifg_pairs)

    def single_primary(self):

        primary_n = np.where(self.dates == self.primary_date)[0][0]
        ifg_pairs = []

        for n in np.arange(len(self.dates)):
            if n != primary_n:
                ifg_pairs.append([primary_n, n])

        return set(tuple(ifg) for ifg in ifg_pairs)

    def find_max_perpendicular_baseline(self, max_perpendicular_baseline):
        """
        Define a network of interferograms based on a maximum perpendicular baseline
        """

        baselines = np.array(self.perpendicular_baselines)
        ifg_pairs = []

        if not len(baselines) == len(self.dates):
            raise ValueError('Number of baselines and dates should be the same!')

        # Define network based on network type
        for n in np.arange(baselines):
            ids = np.where((baselines - baselines[n] > 0) * (baselines - baselines[n] <= self.max_perpendicular_baseline))[0]
            for id in ids:
                ifg_pairs.append([n, id])

        return set(tuple(ifg) for ifg in ifg_pairs)
