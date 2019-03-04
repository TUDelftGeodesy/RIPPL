# This class both downloads and reads an ECMWF dataset as a preprocessing step
import numpy as np
from collections import defaultdict

from rippl.NWP_functions.model_reference import ModelReference


class ModelToDelay(object):

    """
    :type model_data = dict

    """

    def __init__(self, levels, geoid_file):
        # In the init function we mainly check the different dates, data folders and extend of dataset.

        # Initialize variable of final product
        self.levels = levels
        self.model_data = dict()
        self.delay_data = defaultdict()
        self.geoid_file = geoid_file
        self.geoid = ''

    def load_model_delay(self, model_data):

        self.model_data = model_data
        self.times = [t for t in self.model_data.keys() if len(t) == 13]

    def model_to_delay(self, method='pressures'):
        # This function resamples the ECMWF grid to different 2d cross-section for different lines.
        # Input is:
        # - dat_file    > the ECMWF .grib file
        # - lat / lon   > the far range points on a line (we assume a straight line in lat/lon)
        # - heading     > heading of ray from point to satellite (degrees from north [0-360])
        # - max_dist    > determines how long we follow this direction (4-5 degrees should be fine

        # Load geoid information for these coordinates if needed
        if len(self.geoid) == 0:
            lats = self.model_data['latitudes']
            lons = self.model_data['longitudes']
            self.geoid = ModelReference.get_geoid(self.geoid_file, lats, lons) / 100

        if 'latitudes' not in self.delay_data.keys() or 'longitudes' not in self.delay_data.keys():
            self.delay_data['latitudes'] = self.model_data['latitudes']
            self.delay_data['longitudes'] = self.model_data['longitudes']

        # Calculate geometric heights based on latitudes and graviation constant
        gm = 9.80655
        delta = 0.6077
        eps = 0.622

        Rd = 287.0586
        k1 = 7.76e-07
        k2b = 23.30e-07
        k3 = 3.75e-03
        klw = 1.45

        a_wgs = 6378137.000
        e_wgs = 0.081819  # excentricity WGS84
        f_wgs = 0.003352811  # flattening
        g_equ = 9.7803253359  # Gravity constant at equator
        ks_som = 0.001931853  # constant formula Somigliana
        sin_lat = np.sin(self.model_data['latitudes'] * np.pi / 180) ** 2

        for time in self.times:

            # Preassign the heights and delays
            tot_size = (self.levels, self.model_data['latitudes'].size, self.model_data['longitudes'].size)
            if time not in self.delay_data.keys():
                self.delay_data[time] = dict()
            self.delay_data[time]['heights'] = np.zeros(shape=self.model_data[time]['pressures'].shape)
            self.delay_data[time]['wet_delay'] = np.zeros(shape=tot_size)
            self.delay_data[time]['liquid_delay'] = np.zeros(shape=tot_size)
            self.delay_data[time]['hydrostatic_delay'] = np.zeros(shape=tot_size)
            self.delay_data[time]['wet_delay_2'] = np.zeros(shape=tot_size)
            self.delay_data[time]['hydrostatic_delay_2'] = np.zeros(shape=tot_size)
            self.delay_data[time]['total_delay'] = np.zeros(shape=tot_size)
            self.delay_data[time]['total_delay_2'] = np.zeros(shape=tot_size)

            # Calculate radius based on
            r_lat = a_wgs * (1 - f_wgs * sin_lat)
            # Calculate gravity based on the formula of Somigliana https://en.wikipedia.org/wiki/Normal_gravity_formula#cite_note-1
            g_lat = g_equ * (1 + ks_som * sin_lat) / np.sqrt(1 - e_wgs ** 2 * sin_lat)

            # Calculate geometric height and gravity at surface.
            h1 = r_lat[:, None] * self.model_data['geo_h'] / ((g_lat / gm * r_lat)[:, None] - self.model_data['geo_h'])
            self.model_data[time]['surface_height'] = h1
            self.delay_data[time]['heights'][self.levels, :, :] = h1
            g1 = g_lat[:, None] * (r_lat[:, None] / (r_lat[:, None] + h1)) ** 2
            num = len(self.model_data['latitudes']) * len(self.model_data['latitudes'])

            # Calculate the heigths and delays for the different layers. We start at the surface, so in reverse order.
            for l in np.arange(self.levels - 1, -1, -1):

                T = self.model_data[time]['Temperature'][l, :, :]
                q0 = self.model_data[time]['Specific humidity'][l, :, :]
                p0 = self.model_data[time]['pressures'][l + 1, :, :]
                p1 = self.model_data[time]['pressures'][l, :, :]

                # For the top layer we do not go to pressure 0
                if l == 0:
                    p1 = p1 + 1

                # First guess
                h0 = h1
                g0 = g1
                dh = 100.0
                dh_0 = 0.0

                # Now iterate to find the geometric heights of the different levels.
                while np.sum(np.abs(dh - dh_0) < 0.1) < num:
                    dh_0 = dh
                    g1 = g_lat[:, None] * (r_lat[:, None] / (r_lat[:, None] + h1)) ** 2
                    g2 = (g0 + g1) / 2.0
                    dh = - Rd * T * (1.0 + delta * q0) / g2 * np.log(p1 / p0)
                    h1 = h0 + dh

                if l == 0:
                    p1 = p1 - 1

                self.delay_data[time]['heights'][l, :, :] = h1
                frac_ew = self.model_data[time]['Specific humidity'][l, :, :] / \
                          (eps + (1.0 - eps) * self.model_data[time]['Specific humidity'][l, :, :])
                mean_p = (p0 + p1) / 2
                plw = (self.model_data[time]['Specific cloud liquid water content'][l, :, :] +
                       self.model_data[time]['Specific cloud ice water content'][l, :, :]) * (p0 - p1) / g2 / 1000

                # Now we can follow two approaches:
                # 1. Calculate refractivity at mid-level and multiply with height
                # 2. Calculate total refractivity from number of particles.
                # We think the second approach is more accurate but the first will be given to here.

                # Liquid water content is calculated same way for both methods.

                self.delay_data[time]['liquid_delay'][l, :, :] = klw * plw

                # approach 1 (Not used, because causes error in highest layer)
                if method == 'heights':
                    self.delay_data[time]['wet_delay'][l, :, :] = dh * mean_p * frac_ew * (k2b / T + k3 / T ** 2)
                    self.delay_data[time]['hydrostatic_delay'][l, :, :] = dh * mean_p * k1 / T
                # approach 2
                elif method == 'pressures':
                    self.delay_data[time]['wet_delay'][l, :, :] = Rd / g2 * (p0 - p1) * frac_ew * (k2b + k3 / T)
                    self.delay_data[time]['hydrostatic_delay'][l, :, :] = Rd / g2 * (p0 - p1) * (1 - frac_ew) * k1

                self.delay_data[time]['total_delay'][l, :, :] = (self.delay_data[time]['hydrostatic_delay'][l, :, :] +
                                                              self.delay_data[time]['wet_delay'][l, :, :] +
                                                              self.delay_data[time]['liquid_delay'][l, :, :])

            # This step is to correct the ECMWF data for the geoid, as we work with data with respect to the ellipse
            # in further scripts. You can disable this by setting the geoid flag to false.

            self.delay_data[time]['heights'] += self.geoid[None, :, :]

    def remove_delay(self, time):

        self.delay_data.pop(time)
