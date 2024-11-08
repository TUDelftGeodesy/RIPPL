import numpy as np
from scipy.interpolate import RectBivariateSpline
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ModelInterpolateDelays(object):

    def __init__(self, in_coor, out_coor, heights, interp_type='linear', t_step=1, height_interval=10, split=True):

        self.split_signal = split
        if self.split_signal:
            self.run_data = ['total', 'hydrostatic', 'wet', 'liquid']
        else:
            self.run_data = ['total']

        # Initialize database
        self.splines = dict()
        if not isinstance(in_coor, CoordinateSystem) or not isinstance(out_coor, CoordinateSystem):
            raise TypeError('In_coor and out_coor of interpolate delays should be CoordinateSystem objects.')
        self.in_coor = in_coor
        self.out_coor = out_coor
        if self.out_coor.grid_type == 'geographic':
            self.out_lines, self.out_pixels = self.out_coor.create_latlon_grid()
        elif self.out_coor.grid_type == 'projection':
            self.out_pixels, self.out_lines = self.out_coor.create_xy_grid()
        self.out_heights = heights

        if self.in_coor.grid_type == 'geographic':
            lines, pixels = self.in_coor.create_latlon_grid()
        elif self.in_coor.grid_type == 'projection':
            pixels, lines = self.in_coor.create_xy_grid()
        self.in_lines = lines[:, 0]
        self.in_pixels = pixels[0, :]

        # The delays at certain time steps
        self.interp_time = []
        self.interp_delays = {'total': {}, 'wet': {}, 'liquid': {}, 'hydrostatic': {}}

        # Interpolation grid
        self.out_line_array = []
        self.out_pixel_array = []
        self.out_height_grid = []

        # Further details
        self.interp_type = interp_type
        self.dh = int(height_interval)
        self.dh_bins = dict()
        self.dh_line_coor = dict()
        self.dh_pixel_coor = dict()
        self.t_step = t_step

        self.h_min = int(np.floor(np.min(self.out_heights) / float(self.dh))) -1
        self.h_max = int(np.ceil(np.max(self.out_heights) / float(self.dh))) +1
        self.h_steps = np.arange(self.h_min, self.h_max) * self.dh

    def add_delays(self, splines):
        # Here we add delays from model timesteps.
        self.splines = splines

    def interpolate_points(self):
        # This function interpolates from low coverage points calculated in find_point_delays to high coverage.
        # Interpolation is done by a 2d linear interpolation after selection based on heights.

        # logging.info('Start interpolation from coarse to specific points in radar grid')
        times = list(self.splines['total'].keys())

        # Finially calculate the splines of the corresponding rays.
        # Save for all types or just the total delays
        # Create output grid
        for t in times:
            self.interp_time.append(t)
            for run_type in self.run_data:
                self.interp_delays[run_type][t] = np.zeros(self.out_heights.shape)

        for i in range(len(self.h_steps)):
            h_bin = (self.out_heights > self.h_steps[i]) * (self.out_heights <= self.h_steps[i] + self.dh)

            for t in times:
                for run_type in self.run_data:
                    # Evaluate splines for this area.
                    grid_delay = self.splines[run_type][t].evaluate_splines([self.h_steps[i]]).reshape((len(self.in_lines), len(self.in_pixels)))

                    # Resample to new grid
                    resampling_grid = RectBivariateSpline(self.in_lines, self.in_pixels, grid_delay, kx=1, ky=1)
                    delays = resampling_grid.ev(self.out_lines[h_bin], self.out_pixels[h_bin])
                    self.interp_delays[run_type][t][h_bin] = delays
