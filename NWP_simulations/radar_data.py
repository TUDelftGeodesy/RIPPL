import datetime
import numpy as np
import locale
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates

class RadarData(OrbitCoordinates):

    """
    :type t_step = float
    """

    def __init__(self, time_interp, t_step, interval):
        # Set locale
        locale.setlocale(locale.LC_ALL, 'en_US.utf8')

        # Init the rounding time step and meta_data
        self.t_step = t_step
        self.time_interp = time_interp
        
        # Initialize date specific information
        self.meta = []
        self.dates = []
        self.date_times = []
        self.model_times = dict()
        self.times = dict()
        self.weights = dict()

        # Intervals for calculation (interval) and output (fine_interval)
        self.interval = interval
        self.lines = []
        self.pixels = []
        self.fine_grid = []

    def match_overpass_weather_model(self, date):
        # This method updates the overpass information. This will be done for the initialization, but can easily be
        # used to generate data for the slave images in a stack.

        key_date = date.strftime('%Y%m%d')
        self.dates.append(key_date)
        date_day = datetime.datetime(year=date.year, month=date.month, day=date.day)
        step_time = (((date - date_day).seconds // self.t_step.seconds) * self.t_step.seconds)
        step_date = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=int(step_time//3600), minute=int((step_time-(step_time//3600 * 3600))//60))
        time_diffs = [date - step_date, (step_date + self.t_step) - date]

        if self.time_interp not in ['linear', 'nearest']:
            print('Interpolation method between time step should either be linear or nearest')
        elif self.time_interp == 'linear':
            self.model_times[key_date] = [step_date, step_date + self.t_step]
            self.times[key_date] = [step_date.strftime('%Y%m%d%H%M'), (step_date + self.t_step).strftime('%Y%m%d%H%M')]
            self.weights[key_date] = [(time_diffs[1].seconds + time_diffs[1].microseconds / 1000000.0) / self.t_step.seconds,
                            (time_diffs[0].seconds + time_diffs[0].microseconds / 1000000.0) / self.t_step.step.seconds]
        elif self.time_interp == 'nearest':
            id = np.argmin(time_diffs)
            self.model_times[key_date] = [step_date + (id * self.t_step)]
            self.times[key_date] = [(step_date + (id * self.t_step)).strftime('%Y%m%d%H%M')]
            self.weights[key_date] = [1.0]

        for t in self.model_times[key_date]:
            if t not in self.date_times:
                self.date_times.append(t)
