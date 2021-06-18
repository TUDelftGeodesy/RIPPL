# This is the main script to calculate delays
import numpy as np

from rippl.NWP_model_delay.ray_tracing.massive_spline import MassiveSpline
from scipy.interpolate import RectBivariateSpline

class ModelRayTracing(object):

    def __init__(self, split_signal=True, height_step=10):
        # Some basic parameters for the model data
        self.delay_data = dict()
        self.t = []
        self.lev = []
        self.dist_steps = ''
        self.step_size = ''
        self.pix_coor = []

        # Initialize utilized variables
        self.R = 6371000.0
        self.degree2rad = np.pi / 180

        # Next images are only created when split_signal is True
        self.split_signal = split_signal
        if self.split_signal:
            self.run_data = ['total', 'wet', 'liquid', 'hydrostatic']
        else:
            self.run_data = ['total']

        # Initialize slice variables (first we make slice over different lines in the image)
        self.slice_lat = []
        self.slice_lon = []
        self.slice_heights = dict()

        self.slice_delays = dict()
        self.slice_delays['total'] = dict()
        self.slice_delays['wet'] = dict()
        self.slice_delays['hydrostatic'] = dict()
        self.slice_delays['liquid'] = dict()
        
        # Initialize ray tracing variables (ray tracing is done for individual pixels in a slice)
        self.ray_height = dict()

        self.ray_delays = dict()
        self.ray_delays['total'] = dict()
        self.ray_delays['wet'] = dict()
        self.ray_delays['hydrostatic'] = dict()
        self.ray_delays['liquid'] = dict()

        self.spline_delays = dict()
        self.spline_delays['total'] = dict()
        self.spline_delays['hydrostatic'] = dict()
        self.spline_delays['wet'] = dict()
        self.spline_delays['liquid'] = dict()

        # Initialize the geometry loaded from radar data
        self.lines = np.array([])
        self.pixels = np.array([])
        self.heights = np.array([])
        self.azimuth_angle = np.array([])
        self.elevation_angle = np.array([])
        self.lat = np.array([])
        self.lon = np.array([])

        # Initialize the results for the final output
        self.heights = np.array([])
        self.out_delay_points = dict()
        self.out_delay_points['total'] = []
        self.out_delay_points['hydrostatic'] = []
        self.out_delay_points['wet'] = []
        self.out_delay_points['liquid'] = []

    def load_geometry(self, azimuth_angle, elevation_angle, lats, lons, heights):
        # Load geometry of radar image
        self.azimuth_angle = azimuth_angle * self.degree2rad
        self.elevation_angle = elevation_angle * self.degree2rad
        self.lat = lats * self.degree2rad
        self.lon = lons * self.degree2rad
        self.heights = heights

    def load_delay(self, delay_data):
        # Load delays from ECMWF data
        self.delay_data = delay_data

    def calc_cross_sections(self):
        # Calc end points of heading lines
        # This can be done assuming a constant 2d grid or using ellipsoid equations.
        # The step size is chosen in kilometers.

        # Some basic parameters for the model data
        self.t = [t for t in self.delay_data.keys() if t not in ['latitudes', 'longitudes']]
        self.lev = self.delay_data[self.t[0]]['total_delay'].shape[0]

        # Convert to pixel and line values
        model_lat = self.delay_data['latitudes'] * self.degree2rad
        model_lon = self.delay_data['longitudes'] * self.degree2rad
        d_lat = model_lat[1] - model_lat[0]
        d_lon = model_lon[1] - model_lon[0]
        no = np.size(self.lat)

        # Calc steps assuming swath with of 250 km and 150 km atmosphere thickness
        self.step_size = d_lat / 1.5 / self.degree2rad * 111100       # split every cell in 4 steps (can be changed if needed...)
        atmo_thickness = 100000
        self.dist_steps = int(np.ceil((atmo_thickness) / self.step_size))

        # For calculations of cross sections we only use the first rows of all coordinates. Other points are part
        # of the generated slices.
        slice_heading = np.ravel(self.azimuth_angle)

        # Calculate the new coordinate using heading and distance
        distance_arc = np.arange(0, self.dist_steps) * self.step_size / self.R

        # Create lat/lon slices
        self.slice_lat = np.zeros([self.lat.size, self.dist_steps])
        self.slice_lon = np.zeros([self.lat.size, self.dist_steps])
        self.slice_lat[:, :] = np.arcsin(np.sin(np.ravel(self.lat))[:, None] * np.cos(distance_arc)[None, :] +
                                    np.cos(np.ravel(self.lat))[:, None] * np.sin(distance_arc)[None, :] * np.cos(slice_heading)[:, None])
        self.slice_lon[:, :] = np.ravel(self.lon)[:, None] + np.arctan2(
            np.sin(slice_heading)[:, None] * np.sin(distance_arc)[None, :] * np.cos(np.ravel(self.lat))[:, None],
            np.cos(distance_arc)[None, :] - np.sin(np.ravel(self.lat))[:, None] * np.sin(self.slice_lat))

        mat_size = (self.lev, no, self.slice_lat.shape[1])
        h_size = (self.lev + 1, no, self.slice_lat.shape[1])

        for t in self.t:
            self.slice_heights[t] = np.zeros(h_size).astype(np.float32)
            self.slice_delays['total'][t] = np.zeros(mat_size).astype(np.float32)

            if self.split_signal:
                self.slice_delays['wet'][t] = np.zeros(mat_size).astype(np.float32)
                self.slice_delays['hydrostatic'][t] = np.zeros(mat_size).astype(np.float32)
                self.slice_delays['liquid'][t] = np.zeros(mat_size).astype(np.float32)

            # Interpolate the values at these points for different delay types and levels.
            for l in range(self.lev):
                interp_delays = RectBivariateSpline(model_lat, model_lon, self.delay_data[t]['total_delay'][l, :, :], kx=1, ky=1)
                self.slice_delays['total'][t][l, :, :] = interp_delays.ev(self.slice_lat, self.slice_lon)

                if self.split_signal:
                    interp_delays = RectBivariateSpline(model_lat, model_lon, self.delay_data[t]['wet'][l, :, :], kx=1, ky=1)
                    self.slice_delays['wet'][t][l, :, :] = interp_delays.ev(self.slice_lat, self.slice_lon)
                    interp_delays = RectBivariateSpline(model_lat, model_lon, self.delay_data[t]['hydrostatic'][l, :, :], kx=1, ky=1)
                    self.slice_delays['hydrostatic'][t][l, :, :] = interp_delays.ev(self.slice_lat, self.slice_lon)
                    interp_delays = RectBivariateSpline(model_lat, model_lon, self.delay_data[t]['liquid'][l, :, :], kx=1, ky=1)
                    self.slice_delays['liquid'][t][l, :, :] = interp_delays.ev(self.slice_lat, self.slice_lon)

            # Also find the heights at these points.
            for l in range(self.lev + 1):
                interp_delays = RectBivariateSpline(model_lat, model_lon, self.delay_data[t]['heights'][l, :, :], kx=1, ky=1)
                self.slice_heights[t][l, :, :] = interp_delays.ev(self.slice_lat, self.slice_lon)

    def find_point_delays(self, spline_num=10):
        # This function calculate the point delays based on:
        # - model line slices (see interpolate ECMWF in resample.py)
        # - elevation angle (we assume input going from left to right in the slices)
        # - lat/lon to find position in slice
        # - line to link with slice (maybe we will work with 2d grids later on)
        # The function will return:
        # - The delay at specified height and up and down as defined in diff_h. (intervals of 10m defined by dh)

        # print('Calculate total delay along ray using radar geometry and weather model delay values')
        # print('')

        for t in self.t:
            # Convert heights and distances to resolution of pixels on the ground
            iter_coor = np.zeros(self.lat.size)
            heights = self.slice_heights[t] / self.step_size
            coor_rem = np.ravel(iter_coor - np.floor(iter_coor))
            ground_h = heights[-1, :, 0]

            # Check where they cross the pressure level boundaries and grid cell boundaries
            # This is done by iterating starting from the lowest level.
            # 1. Check if the ray crosses the top boundary of the cell, within the cell boundary.
            #   1a If true, calculate the new coordinate for the next layer
            #   1b If false calculate the coordinate where it leaves this cell
            #   1c Calculate the total delay in this step and add it to the delay for this layer
            # 2. Iterate till all rays reached the top of this layer.

            # Define the line from the rays in the slice coordinates
            rc = np.ravel(np.tan(self.elevation_angle))
            slope = np.ravel(1.0 / np.sin(self.elevation_angle))
            iter_coor = np.ravel(iter_coor)
            a_ray = np.ravel(rc)
            b_ray = np.ravel(ground_h) - iter_coor * a_ray

            h_size = (self.lev + 1, np.size(self.lat))
            d_size = (self.lev, np.size(self.lat))
            self.ray_height[t] = np.zeros(h_size)
            self.ray_height[t][-1, :] = ground_h
            self.ray_delays['total'][t] = np.zeros(d_size)
            if self.split_signal:
                self.ray_delays['hydrostatic'][t] = np.zeros(d_size)
                self.ray_delays['wet'][t] = np.zeros(d_size)
                self.ray_delays['liquid'][t] = np.zeros(d_size)

            for l in np.arange(self.lev - 1, -1, -1):
                id = np.arange(self.lat.size)

                no = 0
                no_max = 3

                while len(id) > 0 and no < no_max:
                    no += 1

                    h0 = heights[l, id, np.floor(iter_coor[id]).astype(np.int32)]
                    h1 = heights[l, id, np.floor(iter_coor[id]).astype(np.int32) + 1]

                    h00 = heights[l + 1, id, np.floor(iter_coor[id]).astype(np.int32)]
                    h11 = heights[l + 1, id, np.floor(iter_coor[id]).astype(np.int32) + 1]
                    coor = (iter_coor[id]).astype(np.int32)

                    # Define the line of layer and check crossing
                    a_l = h1 - h0
                    b_l = h0 - coor * a_l

                    # Calculate the crossing of the lines.
                    # 1. Where do we cross the top of this cell?
                    # 2. Is this point within the current cell? If not take the border and mark as unfinished.
                    cross_x = (b_l - b_ray[id]) / (a_ray[id] - a_l)

                    # Check which cells are not yet at the cell top
                    outside = np.floor(cross_x - coor) != 0
                    cross_x[outside] = np.floor(cross_x[outside])
                    cross_y = a_l * cross_x + b_l

                    # Calculate the delay for this part
                    start = iter_coor[id] - coor
                    end = cross_x - coor

                    # Add evaluated parts of this cell.
                    self.ray_height[t][l, id] = cross_y

                    # Now calculate actual delay for different types
                    for run_type in self.run_data:
                        d0 = self.slice_delays[run_type][t][l, id, np.floor(iter_coor[id]).astype(np.int32)]
                        d1 = self.slice_delays[run_type][t][l, id, np.floor(iter_coor[id]).astype(np.int32) + 1]
                        delay = self.calc_delay_layer(d0, d1, h0 - h00, h1 - h11, start, end, a_ray[id])
                        self.ray_delays[run_type][t][l, id] += delay

                    # Prepare for new iteration
                    iter_coor[id] = cross_x
                    id = id[outside]

                # In case we have some erroneous ray tracing values
                h_average = np.mean(self.ray_height[t][l, :][~np.isnan(self.ray_delays[run_type][t][l, :])])
                self.ray_height[t][l, :][np.isnan(self.ray_delays[run_type][t][l, :])] = h_average

                for run_type in self.run_data:
                    self.ray_delays[run_type][t][l, :][np.isnan(self.ray_delays[run_type][t][l, :])] = 0.001

            # Finally calculate the splines of the corresponding rays.
            # print('Calculate splines of slant delays vs height for time ' + t)

            for run_type in self.run_data:
                # Calculate the cumulative sum
                self.ray_delays[run_type][t] *= np.expand_dims(slope, axis=0)
                total_delay = np.cumsum(self.ray_delays[run_type][t], axis=0)

                # interpolate over the last [-diff_h, diff_h] interval to get the variation
                # The function should be exponential, but is generally well fitted using a cubic spline. We only
                # interpolate over the last n layers, as we expect all topography within these n layers.
                # If extrapolation is needed because the point is lower than the model topography we will use an
                # extrapolation of the lowest layer.

                height = np.fliplr(np.transpose(self.ray_height[t][-spline_num:, :]) * self.step_size)
                delays = np.fliplr(np.transpose(total_delay[-spline_num:, :]))

                self.spline_delays[run_type][t] = MassiveSpline(height, delays, 'linear')

    def remove_delay(self, t):

        self.spline_delays['total'].pop(t)
        self.slice_heights.pop(t)
        self.slice_delays['total'].pop(t)
        self.ray_height.pop(t)
        self.ray_delays['total'].pop(t)

        if self.split_signal:
            self.slice_delays['wet'].pop(t)
            self.slice_delays['hydrostatic'].pop(t)
            self.slice_delays['liquid'].pop(t)

            self.spline_delays['wet'].pop(t)
            self.spline_delays['hydrostatic'].pop(t)
            self.spline_delays['liquid'].pop(t)

            self.ray_delays['wet'].pop(t)
            self.ray_delays['hydrostatic'].pop(t)
            self.ray_delays['liquid'].pop(t)

    @staticmethod
    def calc_delay_layer(d0, d1, h0, h1, start, end, s, formula='simplified'):
        # This function calculates the total delay for a ray that crosses a model layer.
        # d1 and d2 are the delay values for the left and right side of the cell.
        # h1 and h2 are the thickness of the layer at the left and right side of the cell
        # start and end are the coordinates where the line starts and ends traversing the cell.
        # slope is the slope of the line
        # Width of every cell is expected to be one.

        # Calculations can be done using a simplified or complex formula. Generally, the simplified one will suffice.

        if formula == 'simplified':
            # This method assumes the height of the layer constant over the whole interval.
            # integral of slope * x * (x * (d2-d1) + d1) / h

            x_mean = ((start + end) / 2)
            h_mean = h0 + (h1 - h0) * x_mean
            d_mean = d0 + (d1 - d0) * x_mean
            d = d_mean * (end - start) * s / h_mean

        else:
            # integrates over a varying h
            # TODO Implement a more complex but accurate formula. Maybe not needed as the error due to exponential delay over height is larger.
            print('Not yet implemented')
            return

        return d

