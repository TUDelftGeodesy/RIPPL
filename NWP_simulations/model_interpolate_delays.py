import numpy as np
from rippl.orbit_resample_functions.resample import Resample


class ModelInterpolateDelays(object):

    def __init__(self, lines, pixels, interp_type='linear', t_step=1, height_interval=10, split=True):

        self.split_signal = split
        if self.split_signal:
            self.run_data = ['total', 'wet', 'liquid', 'hydrostatic']
        else:
            self.run_data = ['total']

        # Initialize database
        self.splines = dict()
        self.lines = lines
        self.pixels = pixels

        # The delays at certain time steps
        self.interp_time = []
        self.interp_delays = dict()
        self.interp_delays['total'] = dict()
        self.interp_delays['wet'] = dict()
        self.interp_delays['hydrostatic'] = dict()
        self.interp_delays['liquid'] = dict()

        # Interpolation points
        self.out_pixels = []
        self.out_lines = []
        self.out_heights = []

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

    def add_interp_points(self, line, pixel, height):
        # Interpolation points
        self.out_pixels = pixel
        self.out_lines = line
        self.out_heights = height

        if not len(line) == len(pixel) == len(height):
            print('Input line, pixel and height should have the same size')
            return

        h_min = int(np.floor(np.min(self.out_heights) / float(self.dh)))
        h_max = int(np.ceil(np.max(self.out_heights) / float(self.dh)))
        h_steps = np.arange(h_min, h_max) * self.dh
        d_pix = self.pixels[1] - self.pixels[0]
        d_lin = self.lines[1] - self.lines[0]

        self.h_steps = []
        # Divide in dh bins:
        # print('Divide heights of points in bins')
        for h_step in h_steps:
            id = str(h_step)

            points = np.where((self.out_heights > h_step) * (self.out_heights <= (h_step + self.dh)))[0].astype(np.int32)
            if len(points) > 0:
                self.dh_bins[id] = points
                # print('Bin between ' + id + ' and ' + str(h_step + self.dh) + ' has ' + str(len(self.dh_bins[id])) + ' pixels')
                self.dh_line_coor[id] = (self.out_lines[self.dh_bins[id]] - self.lines[0]).astype(np.float32) / d_lin
                self.dh_pixel_coor[id] = (self.out_pixels[self.dh_bins[id]] - self.pixels[0]).astype(np.float32)  / d_pix
                self.h_steps.append(h_step)

        self.h_steps = np.array(self.h_steps)

    def add_interp_grid(self, lines, pixels, heights):
        # Load the interpolation grid, for a full grid. lines and pixels should be equidistant.
        # TODO finish grid interpolation
        self.out_line_array = lines
        self.out_pixel_array = pixels
        self.out_height_grid = heights

        # Calc the new coordinates in the new grid.


        # Calc the contribution of the 4 surrounding pixels for every


    def add_delays(self, splines):
        # Here we add delays from model timesteps.

        self.splines = splines

    def interpolate_points(self):
        # This function interpolates from low coverage points calculated in find_point_delays to high coverage.
        # Interpolation is done by a 2d linear interpolation after selection based on heights.

        # print('Start interpolation from coarse to specific points in radar grid')
        times = self.splines['total'].keys()
        size = (len(self.lines), len(self.pixels))

        # Finially calculate the splines of the corresponding rays.
        # Save for all types or just the total delays
        for t in times:
            self.interp_time.append(t)

            for run_type in self.run_data:

                # Create ouput grid
                self.interp_delays[run_type][t] = np.zeros(len(self.out_heights))

                # print('Interpolate for points of interest for ' + run_type + ' delay for time ' + t)
                # Then divide all points over different height bins. (dh of 5 or 10 meter should be good enough when compared to
                # SRTM accuracy.

                # Now do the actual interpolation for different height bins seperately.
                for i in self.h_steps:
                    # print('Evaluating height between ' + str(i) + ' and ' + str(i + self.dh))

                    # Find ids and coordinates of points within certain height bin
                    coor_l = self.dh_line_coor[str(i)]
                    coor_p = self.dh_pixel_coor[str(i)]

                    # Evaluate splines for this area.
                    grid_delay = self.splines[run_type][t].evaluate_splines(np.asarray([i])).reshape(size)

                    # Resample to new grid
                    delays = Resample.resample_grid(grid_delay, coor_l, coor_p, w_type='linear', table_size=[50, 50])
                    self.interp_delays[run_type][t][self.dh_bins[str(i)]] = delays
