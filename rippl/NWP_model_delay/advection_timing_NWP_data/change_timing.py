# This class both downloads and reads an ECMWF dataset as a preprocessing step
import numpy as np
import datetime

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
        self.times = [t for t in self.model_data.keys() if len(t) == 13]
        self.model_lat = self.model_data['latitudes']
        self.model_lon = self.model_data['longitudes']

    def change_timing(self, time_diff=[]):

        for time in self.times:
            # Start by creating the interpolation values for the individual model realisations
            v_interp = dict()
            u_interp = dict()
            t_interp = dict()
            e_interp = dict()
            w_interp = dict()
            i_interp = dict()

            # steps in meters
            R = 6371
            lat_step = np.deg2rad(np.diff(self.model_lat)[0]) * 6371000
            lon_step = np.deg2rad(np.diff(self.model_lon)[0]) * 6371000 * np.cos(np.deg2rad(np.mean(self.model_lat)))

            for interp_var, model_var in zip([u_interp, v_interp, t_interp, e_interp, w_interp, i_interp],
                                             ['Wind_u', 'Wind_v', 'Temperature', 'Specific humidity',
                                              'Specific cloud liquid water content',
                                              'Specific cloud ice water content']):

                data = self.model_data[time][model_var]
                no_layers = data.shape[0]
                u_size = data.shape[1]
                v_size = data.shape[2]

                for i in range(no_layers):
                    interp_var[str(i)] = RectBivariateSpline(range(u_size), range(v_size), data[i, :, :], kx=1, ky=1)

            for t_diff in time_diff:
                # First get the total time difference and define the number of steps.
                n_steps = int(np.floor(np.abs(t_diff) / self.time_step))
                final_step = np.abs(t_diff) - (n_steps * self.time_step)
                if t_diff < 0:
                    time_step_model = -self.time_step
                    final_step = -final_step
                else:
                    time_step_model = self.time_step

                # Then calculate the u/v shifts
                new_u_grid, new_v_grid = np.meshgrid(range(v_size), range(u_size))
                new_u_grid = np.tile(new_u_grid.astype('float32')[None, :, :], [no_layers, 1, 1])
                new_v_grid = np.tile(new_v_grid.astype('float32')[None, :, :], [no_layers, 1, 1])
                u_shift = self.model_data[time]['Wind_u'] / lon_step * time_step_model * 60
                v_shift = self.model_data[time]['Wind_v'] / lat_step * time_step_model * 60

                for step in list(np.ones(n_steps) * time_step_model) + [final_step]:
                    for n in range(no_layers):
                        u_shift[n, :, :] = u_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :]) * step / lon_step * 60
                        v_shift[n, :, :] = v_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :]) * step / lat_step * 60

                    new_u_grid += u_shift
                    new_v_grid += v_shift

                # Finally add the outcomes to the new grid.
                time_str = (datetime.datetime.strptime(time, '%Y%m%dT%H%M') + datetime.timedelta(minutes=t_diff)).strftime('%Y%m%dT%H%M')
                if time_str != time:
                    self.model_data[time_str] = dict()
                self.model_data[time_str]['pressures'] = self.model_data[time]['pressures']
                self.model_data[time_str]['Specific humidity'] = np.zeros(data.shape)
                self.model_data[time_str]['Temperature'] = np.zeros(data.shape)
                self.model_data[time_str]['Specific cloud liquid water content'] = np.zeros(data.shape)
                self.model_data[time_str]['Specific cloud ice water content'] = np.zeros(data.shape)

                for n in range(no_layers):
                    self.model_data[time_str]['Specific humidity'][n, :, :] = \
                        e_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :])
                    self.model_data[time_str]['Temperature'][n, :, :] = \
                        t_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :])
                    self.model_data[time_str]['Specific cloud liquid water content'][n, :, :] = \
                        w_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :])
                    self.model_data[time_str]['Specific cloud ice water content'][n, :, :] = \
                        i_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :])

    def remove_model_data(self, time):

        self.model_data.pop(time)
