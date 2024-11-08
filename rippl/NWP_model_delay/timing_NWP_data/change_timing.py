# This class both downloads and reads an ERA5 dataset as a preprocessing step
import numpy as np
import datetime
from cartopy.geodesic import Geodesic

from scipy.interpolate.interpolate import RectBivariateSpline


class AdjustTiming(object):
    """
    :type model_data = dict

    """

    def __init__(self, time_step=5):

        self.model_data = dict()
        self.times = list()
        self.time_step = time_step      # Calculation time step in minutes

    def load_model_delay(self, model_data):

        self.model_data = model_data
        self.times = [t for t in self.model_data.keys()]

    def change_timing(self, time_diff=[]):

        for time in self.times:
            # Start by creating the interpolation values for the individual model realisations
            v_interp = dict()
            u_interp = dict()
            t_interp = dict()
            e_interp = dict()
            w_interp = dict()
            i_interp = dict()

            # Calculate the steps in meters.
            lat = self.model_data[time]['latitude']
            lon = self.model_data[time]['longitude']
            az_x, az_y, step_x, step_y = self.calc_steps_and_heading(time, lat, lon)

            model_vars = list(self.model_data[time].keys())
            for interp_var, model_var in zip([u_interp, v_interp, t_interp, e_interp, w_interp, i_interp],
                                             ['U component of wind', 'V component of wind', 'Temperature', 'Specific humidity',
                                              'Specific cloud liquid water content',
                                              'Specific cloud ice water content']):

                if model_var in model_vars:
                    data = self.model_data[time][model_var]
                    no_layers = data.shape[0]
                    u_size = data.shape[1]
                    v_size = data.shape[2]

                    for i in range(no_layers):
                        interp_var[str(i)] = RectBivariateSpline(range(u_size), range(v_size), data[i, :, :], kx=1, ky=1)

                # Split in times below and above 0.
            time_diff = np.array(time_diff)
            times_before = time_diff[time_diff < 0]
            times_after = time_diff[time_diff > 0]

            for times in [times_before, times_after]:
                if len(times) == 0:
                    continue

                # Check if steps are a multiple of the time step
                if not np.max(np.remainder(times, self.time_step)) == 0:
                    raise ValueError('All selected times should be a multiple of the time step!')

                # First get the total time difference and define the number of steps.
                n_steps = int(np.floor(np.max(np.abs(times)) / self.time_step))
                if np.max(times) < 0:
                    time_step_model = -self.time_step
                else:
                    time_step_model = self.time_step

                # Then calculate the u/v shifts
                new_x_grid, new_y_grid = np.meshgrid(range(v_size), range(u_size))
                new_x_grid = np.tile(new_x_grid.astype('float32')[None, :, :], [no_layers, 1, 1])
                new_y_grid = np.tile(new_y_grid.astype('float32')[None, :, :], [no_layers, 1, 1])

                u_shift = self.model_data[time]['U component of wind'] * time_step_model * 60
                v_shift = self.model_data[time]['V component of wind'] * time_step_model * 60

                step_time = 0
                for time_step in list(np.ones(n_steps) * time_step_model):
                    for n in range(no_layers):
                        u_shift[n, :, :] = u_interp[str(n)].ev(new_x_grid[n, :, :], new_y_grid[n, :, :]) * time_step_model * 60
                        v_shift[n, :, :] = v_interp[str(n)].ev(new_x_grid[n, :, :], new_y_grid[n, :, :]) * time_step_model * 60

                    # To convert u, v speeds we need to take into account that the x, y grid is rotated with respect
                    # to the north/east direction due to the projection.
                    x_shift = np.sin(az_x) * u_shift / step_x + np.sin(az_y) * v_shift / step_y
                    y_shift = np.cos(az_x) * u_shift / step_x + np.cos(az_y) * v_shift / step_y
                    new_x_grid += x_shift
                    new_y_grid += y_shift
                    step_time += time_step_model

                    if step_time in times:
                        # Finally add the outcomes to the new grid.
                        time_str = (datetime.datetime.strptime(time, '%Y%m%dT%H%M') + datetime.timedelta(minutes=step_time)).strftime('%Y%m%dT%H%M')
                        if time_str != time:
                            self.model_data[time_str] = dict()
                        self.model_data[time_str]['pressures'] = self.model_data[time]['pressures']
                        self.model_data[time_str]['Specific humidity'] = np.zeros(data.shape)
                        self.model_data[time_str]['Temperature'] = np.zeros(data.shape)

                        # These values are not available in all models
                        if 'Specific cloud liquid water content' in model_vars:
                            self.model_data[time_str]['Specific cloud liquid water content'] = np.zeros(data.shape)
                        if 'Specific cloud ice water content' in model_vars:
                            self.model_data[time_str]['Specific cloud ice water content'] = np.zeros(data.shape)

                        for n in range(no_layers):
                            self.model_data[time_str]['Specific humidity'][n, :, :] = \
                                e_interp[str(n)].ev(new_x_grid[n, :, :], new_y_grid[n, :, :])
                            self.model_data[time_str]['Temperature'][n, :, :] = \
                                t_interp[str(n)].ev(new_x_grid[n, :, :], new_y_grid[n, :, :])

                            # These values are not available in all models
                            if 'Specific cloud liquid water content' in model_vars:
                                self.model_data[time_str]['Specific cloud liquid water content'][n, :, :] = \
                                    w_interp[str(n)].ev(new_x_grid[n, :, :], new_y_grid[n, :, :])
                            if 'Specific cloud ice water content' in model_vars:
                                self.model_data[time_str]['Specific cloud ice water content'][n, :, :] = \
                                    i_interp[str(n)].ev(new_x_grid[n, :, :], new_y_grid[n, :, :])

    @staticmethod
    def calc_steps_and_heading(time, lat, lon):
        """
        Calulcate the steps and heading of a dataset based on latitudes/longitudes and x/y coordinates

        :return:
        """

        coor_shape = lat.shape
        dlat_x = np.zeros(coor_shape)
        dlon_x = np.zeros(coor_shape)
        for coor_dat, in_dat in zip([dlat_x, dlon_x], [lat, lon]):
            coor_dat[:, 1:] = np.diff(in_dat, axis=0)
            coor_dat[:, 0] = coor_dat[:, 1]

        dlat_y = np.zeros(coor_shape)
        dlon_y = np.zeros(coor_shape)
        for coor_dat, in_dat in zip([dlat_y, dlon_y], [lat, lon]):
            coor_dat[1:, :] = np.diff(in_dat, axis=0)
            coor_dat[0, :] = coor_dat[1, :]

        shp = [lat.size, 1]
        [step_x, az_x, az_x2] = Geodesic.inverse(np.concatenate((lat.reshape(shp), lon.reshape(shp)), axis=1),
                                                 np.concatenate(((lat + dlat_x).reshape(shp), (lon + dlon_x).reshape(shp)), axis=1))
        [step_y, az_y, az_y2] = Geodesic.inverse(np.concatenate((lat.reshape(shp), lon.reshape(shp)), axis=1),
                                                 np.concatenate(((lat + dlat_y).reshape(shp), (lon + dlon_y).reshape(shp)), axis=1))

        return az_x, az_y, step_x, step_y

    def remove_model_data(self, time):

        self.model_data.pop(time)
