# This is the main script to calculate delays
import copy

import numpy as np
import pyproj

from rippl.NWP_model_delay.ray_tracing_NWP_data.massive_spline import MassiveSpline
from rippl.NWP_model_delay.timing_NWP_data.change_timing import AdjustTiming
from scipy.interpolate import RectBivariateSpline

class ModelRayTracing(object):

    def __init__(self, split_signal=True, height_step=10):
        # Some basic parameters for the model data

        self.times = []

        # Next images are only created when split_signal is True
        self.split_signal = split_signal
        if self.split_signal:
            self.run_data = ['total_delay', 'wet_delay', 'liquid_delay', 'hydrostatic_delay']
            run_dict = {'total_delay': {}, 'wet_delay': {}, 'liquid_delay': {}, 'hydrostatic_delay': {}}
        else:
            self.run_data = ['total_delay']
            run_dict = {'total_delay': {}}

        # Init delay data
        self.delay_data = copy.deepcopy(run_dict)

        # Initialize slice variables (first we make slice over different lines in the image)
        self.slice_heights = dict()
        self.slice_step_size = dict()
        self.slice_delays = copy.deepcopy(run_dict)
        
        # Initialize ray tracing variables (ray tracing is done for individual pixels in a slice)
        self.ray_height = dict()
        self.ray_delays = copy.deepcopy(run_dict)
        self.spline_delays = copy.deepcopy(run_dict)

        # Initialize the geometry loaded from radar data
        self.heights = np.array([])
        self.azimuth_angle = np.array([])
        self.elevation_angle = np.array([])
        self.lat = np.array([])
        self.lon = np.array([])

        # Initialize the results for the final output
        self.out_delay_points = copy.deepcopy(run_dict)

    def load_geometry(self, azimuth_angle, elevation_angle, lats, lons, heights):
        # Load geometry of radar image
        self.azimuth_angle = np.deg2rad(azimuth_angle)
        self.elevation_angle = np.deg2rad(elevation_angle)
        self.lat = np.deg2rad(lats)
        self.lon = np.deg2rad(lons)
        self.heights = heights

    def load_delay(self, delay_data):
        # Load delays from ERA5 data
        self.delay_data = delay_data

    def calc_cross_sections(self):
        # Calc end points of heading lines
        # This can be done assuming a constant 2d grid or using ellipsoid equations.
        # The step size is chosen in kilometers.

        # Some basic parameters for the model data
        self.times = list(self.delay_data.keys())

        for time in self.times:
            no_levels = self.delay_data[time]['total_delay'].shape[0]

            # Convert to pixel and line values
            lat = self.delay_data[time]['latitude']
            lon = self.delay_data[time]['longitude']
            if 'step_x' in self.delay_data[time].keys() and 'step_y' in self.delay_data[time].keys():
                step_x = self.delay_data[time]['step_x']
                step_y = self.delay_data[time]['step_y']
            else:
                az_x, az_y, step_x, step_y = AdjustTiming.calc_steps_and_heading(time, lat, lon)
            no = np.size(self.lat)

            # Calc steps assuming swath with of 250 km and 150 km atmosphere thickness
            self.slice_step_size[time] = np.minimum(np.min(step_x), np.min(step_y)) / 2 # split every cell in 2 steps (can be changed if needed...)
            atmo_thickness = 150000
            dist_steps = int(np.ceil((atmo_thickness) / np.abs(self.slice_step_size[time])))

            # For calculations of cross-sections we only use the first rows of all coordinates. Other points are part
            # of the generated slices.
            slice_heading = np.ravel(self.azimuth_angle)

            # Calculate the new coordinate using heading and distance
            R = 6371000.0
            distance_arc = np.arange(0, dist_steps) * self.slice_step_size[time] / R

            # Create lat/lon slices
            slice_lat = np.zeros([self.lat.size, dist_steps])
            slice_lon = np.zeros([self.lat.size, dist_steps])
            slice_lat[:, :] = np.arcsin(np.sin(np.ravel(self.lat))[:, None] * np.cos(distance_arc)[None, :] +
                                        np.cos(np.ravel(self.lat))[:, None] * np.sin(distance_arc)[None, :] * np.cos(slice_heading)[:, None])
            slice_lon[:, :] = np.ravel(self.lon)[:, None] + np.arctan2(
                np.sin(slice_heading)[:, None] * np.sin(distance_arc)[None, :] * np.cos(np.ravel(self.lat))[:, None],
                np.cos(distance_arc)[None, :] - np.sin(np.ravel(self.lat))[:, None] * np.sin(slice_lat))

            # If the coordinate system of the model is projected, the calculated points should be transformed to the
            # projected coordinate system first. Otherwise, lat/lon coordinates can be used.
            if self.delay_data[time]['projection'].is_projected:
                input_crs = pyproj.CRS(4326)    # Geographic coordinates WGS84
                transformer = pyproj.Transformer.from_crs(input_crs, self.delay_data[time]['projection'])

                self.slice_x, self.slice_y = transformer.transform(slice_lat, slice_lon, radians=True)
                model_y = self.delay_data[time]['y'][:, 0]
                model_x = self.delay_data[time]['x'][0, :]
            elif self.delay_data[time]['projection'].is_geographic:
                self.slice_x = np.rad2deg(slice_lon)
                self.slice_y = np.rad2deg(slice_lat)

                model_y = self.delay_data[time]['latitude'][:, 0]
                model_x = self.delay_data[time]['longitude'][0, :]
            else:
                raise ValueError('No projection defined')

            mat_size = (no_levels, no, slice_lat.shape[1])
            h_size = (no_levels + 1, no, slice_lat.shape[1])
            self.slice_heights[time] = np.zeros(h_size).astype(np.float32)
            self.slice_delays['total_delay'][time] = np.zeros(mat_size).astype(np.float32)

            if self.split_signal:
                self.slice_delays['wet_delay'][time] = np.zeros(mat_size).astype(np.float32)
                self.slice_delays['hydrostatic_delay'][time] = np.zeros(mat_size).astype(np.float32)
                self.slice_delays['liquid_delay'][time] = np.zeros(mat_size).astype(np.float32)

            # Define the limits of the area for the bivariate spline. Extend not larger than needed.
            min_y = np.min(self.slice_y)
            max_y = np.max(self.slice_y)
            min_x = np.min(self.slice_x)
            max_x = np.max(self.slice_x)

            if model_y[-1] < model_y[0]:
                flip_data = True
                y0 = np.max(np.argwhere(model_y > max_y))
                y1 = np.min(np.argwhere(model_y < min_y)) + 1
            else:
                flip_data = False
                y0 = np.max(np.argwhere(model_y < min_y))
                y1 = np.min(np.argwhere(model_y > max_y)) + 1
            x0 = np.max(np.argwhere(model_x < min_x))
            x1 = np.min(np.argwhere(model_x > max_x)) + 1

            # Interpolate the values at these points for different delay types and levels.
            for l in range(no_levels):
                if self.split_signal:
                    data_types = ['total_delay', 'wet_delay', 'hydrostatic_delay', 'liquid_delay']
                else:
                    data_types = ['total_delay']

                for data_type in data_types:
                    if flip_data:
                        interp_delays = RectBivariateSpline(np.flip(model_y[y0:y1]), model_x[x0:x1],
                                                            np.flipud(self.delay_data[time][data_type][l, y0:y1, x0:x1]), kx=1, ky=1)
                        self.slice_delays[data_type][time][l, :, :] = interp_delays.ev(self.slice_y, self.slice_x)
                    else:
                        interp_delays = RectBivariateSpline(model_y[y0:y1], model_x[x0:x1],
                                                            self.delay_data[time][data_type][l, y0:y1, x0:x1], kx=1, ky=1)
                        self.slice_delays[data_type][time][l, :, :] = interp_delays.ev(self.slice_y, self.slice_x)

            # Also find the heights at these points.
            for l in range(no_levels + 1):
                if flip_data:
                    interp_delays = RectBivariateSpline(np.flip(model_y[y0:y1]), model_x[x0:x1],
                                                        np.flipud(self.delay_data[time]['heights'][l, y0:y1, x0:x1]), kx=1, ky=1)
                    self.slice_heights[time][l, :, :] = interp_delays.ev(self.slice_y, self.slice_x)
                else:
                    interp_delays = RectBivariateSpline(model_y[y0:y1], model_x[x0:x1],
                                                        self.delay_data[time]['heights'][l, y0:y1, x0:x1], kx=1, ky=1)
                    self.slice_heights[time][l, :, :] = interp_delays.ev(self.slice_y, self.slice_x)

    def find_point_delays(self, spline_num=10, spline_type='linear'):
        # This function calculate the point delays based on:
        # - model line slices (see interpolate ERA5 in resample.py)
        # - elevation angle (we assume input going from left to right in the slices)
        # - lat/lon to find position in slice
        # - line to link with slice (maybe we will work with 2d grids later on)
        # The function will return:
        # - The delay at specified height and up and down as defined in diff_h. (intervals of 10m defined by dh)

        # logging.info('Calculate total delay along ray using radar geometry and weather model delay values')
        # logging.info('')

        for time in self.times:
            no_levels = self.delay_data[time]['total_delay'].shape[0]
            # Convert heights and distances to resolution of pixels on the ground
            iter_coor = np.zeros(self.lat.size)
            heights = self.slice_heights[time] / np.abs(self.slice_step_size[time])
            coor_rem = np.ravel(iter_coor - np.floor(iter_coor))
            ground_h = heights[0, :, 0]

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

            h_size = (no_levels + 1, np.size(self.lat))
            d_size = (no_levels, np.size(self.lat))
            self.ray_height[time] = np.zeros(h_size)
            self.ray_height[time][0, :] = ground_h
            self.ray_delays['total_delay'][time] = np.zeros(d_size)
            if self.split_signal:
                self.ray_delays['hydrostatic_delay'][time] = np.zeros(d_size)
                self.ray_delays['wet_delay'][time] = np.zeros(d_size)
                self.ray_delays['liquid_delay'][time] = np.zeros(d_size)

            for l in np.arange(no_levels):
                id = np.arange(self.lat.size)

                no = 0
                no_max = 3

                while len(id) > 0 and no < no_max:
                    no += 1

                    h0 = heights[l + 1, id, np.floor(iter_coor[id]).astype(np.int32)]
                    h1 = heights[l + 1, id, np.floor(iter_coor[id]).astype(np.int32) + 1]

                    h00 = heights[l, id, np.floor(iter_coor[id]).astype(np.int32)]
                    h11 = heights[l, id, np.floor(iter_coor[id]).astype(np.int32) + 1]
                    coor = (iter_coor[id]).astype(np.int32)

                    # Define the line of layer and check crossing
                    a_l = h1 - h0
                    b_l = h0 - coor * a_l

                    # Calculate the crossing of the lines.
                    # 1. Where do we cross the top of this cell?
                    # 2. Is this point within the current cell? If not take the border and mark as unfinished.
                    cross_x = (b_l - b_ray[id]) / (a_ray[id] - a_l)
                    cross_x[cross_x < 0] = 0

                    # Check which cells are not yet at the cell top
                    outside = np.floor(cross_x - coor) != 0
                    cross_x[outside] = np.floor(cross_x[outside])
                    cross_y = a_l * cross_x + b_l

                    # Calculate the delay for this part
                    start = iter_coor[id] - coor
                    end = cross_x - coor

                    # Add evaluated parts of this cell.
                    self.ray_height[time][l + 1, id] = cross_y

                    # Now calculate actual delay for different types
                    for run_type in self.run_data:
                        d0 = self.slice_delays[run_type][time][l, id, np.floor(iter_coor[id]).astype(np.int32)]
                        d1 = self.slice_delays[run_type][time][l, id, np.floor(iter_coor[id]).astype(np.int32) + 1]
                        delay = self.calc_delay_layer(d0, d1, h0 - h00, h1 - h11, start, end, a_ray[id])
                        self.ray_delays[run_type][time][l, id] += delay

                    # Prepare for new iteration
                    iter_coor[id] = cross_x
                    id = id[outside]

                # In case we have some erroneous ray tracing values
                h_average = np.mean(self.ray_height[time][l, :][~np.isnan(self.ray_delays[run_type][time][l, :])])
                self.ray_height[time][l, :][np.isnan(self.ray_delays[run_type][time][l, :])] = h_average

                for run_type in self.run_data:
                    self.ray_delays[run_type][time][l, :][np.isnan(self.ray_delays[run_type][time][l, :])] = 0.001

            # Finally calculate the splines of the corresponding rays.
            # logging.info('Calculate splines of slant delays vs height for time ' + t)
            no_zero_height = np.sum((np.diff(self.slice_heights[time][:, :, 0], axis=0) <= 1), axis=0)

            for run_type in self.run_data:
                # Calculate the cumulative sum
                self.ray_delays[run_type][time] *= np.expand_dims(slope, axis=0)
                total_delay = np.flip(np.cumsum(np.flip(self.ray_delays[run_type][time], axis=0), axis=0), axis=0)

                # interpolate over the last [-diff_h, diff_h] interval to get the variation
                # The function should be exponential, but is generally well-fitted using a cubic spline. We only
                # interpolate over the last n layers, as we expect all topography within these n layers.
                # If extrapolation is needed because the point is lower than the model topography we will use an
                # extrapolation of the lowest layer.
                height = np.zeros((len(no_zero_height), spline_num))
                delays = np.zeros((len(no_zero_height), spline_num))

                for val in np.unique(no_zero_height):
                    ids = np.where(no_zero_height == val)
                    height[ids, :] = np.transpose(np.squeeze(self.ray_height[time][val:spline_num + val, ids])) * np.abs(self.slice_step_size[time])
                    delays[ids, :] = np.transpose(np.squeeze(total_delay[val:spline_num + val, ids]))

                self.spline_delays[run_type][time] = MassiveSpline(height, delays, spline_type)

    def remove_delay(self, t):

        self.slice_heights.pop(t)
        self.ray_height.pop(t)
        self.slice_step_size.pop(t)

        for run_data in self.run_data:
            self.spline_delays[run_data].pop(t)
            self.slice_delays[run_data].pop(t)
            self.ray_delays[run_data].pop(t)

    @staticmethod
    def calc_delay_layer(d0, d1, h0, h1, start, end, s):
        # This function calculates the total delay for a ray that crosses a model layer.
        # d1 and d2 are the delay values for the left and right side of the cell.
        # h1 and h2 are the thickness of the layer at the left and right side of the cell
        # start and end are the coordinates where the line starts and ends traversing the cell.
        # slope is the slope of the line
        # Width of every cell is expected to be one.

        x_mean = ((start + end) / 2)
        h_mean = h0 + (h1 - h0) * x_mean
        d_mean = d0 + (d1 - d0) * x_mean
        d = d_mean * (end - start) * s / h_mean

        return d
