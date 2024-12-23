# This class both downloads and reads an ERA5 dataset as a preprocessing step
import numpy as np
from collections import defaultdict
import os

from rippl.user_settings import UserSettings
from rippl.external_dems.geoid import GeoidInterp


class ModelToDelay(object):

    """
    :type model_data = dict

    """

    def __init__(self):
        # In the init function we mainly check the different dates, data folders and extend of dataset.

        # Initialize variable of final product
        self.model_data = dict()
        self.delay_data = dict()

    def load_model_delay(self, model_data):

        self.model_data = model_data
        self.times = list(self.model_data.keys())

    def model_to_delay(self, method='pressures'):
        # This function resamples the ERA5 grid to different 2d cross-section for different lines.
        # Input is:
        # - dat_file    > the ERA5 .grib file
        # - lat / lon   > the far range points on a line (we assume a straight line in lat/lon)
        # - heading     > heading of ray from point to satellite (degrees from north [0-360])
        # - max_dist    > determines how long we follow this direction (4-5 degrees should be fine

        # Calculate geometric heights based on latitudes and graviation constant
        Rv = 461.5              # specific gas constant water vapor
        Rd = 287.0586           # specific gas constant air
        eps = Rd / Rv           # ratio of molar weights of dry and moist air
        delta = 1 / eps - 1     # conversion for virtual temperature
        k1 = 77.6
        k2 = 64.8
        k2b = 23.3
        k3 = 377600
        klw = 1.45

        for time in self.times:
            # Load geoid information for these coordinates if needed
            lats = self.model_data[time]['latitude']
            lons = self.model_data[time]['longitude']
            settings = UserSettings()
            egm_96_file = os.path.join(settings.settings['paths']['DEM_database'], 'geoid', 'egm96.dat')
            egm_96 = GeoidInterp.create_geoid(egm_96_file=egm_96_file, lat=lats, lon=lons)
            self.delay_data[time] = dict()
            self.delay_data[time]['geoid'] = np.transpose(egm_96)

            # Load projection information
            self.delay_data[time]['projection'] = self.model_data[time]['projection']
            self.delay_data[time]['x'] = self.model_data[time]['x']
            self.delay_data[time]['y'] = self.model_data[time]['y']
            if 'step_x' in self.model_data[time].keys() and 'step_y' in self.model_data[time].keys():
                self.delay_data[time]['step_x'] = self.model_data[time]['step_x']
                self.delay_data[time]['step_y'] = self.model_data[time]['step_y']
            self.delay_data[time]['latitude'] = self.model_data[time]['latitude']
            self.delay_data[time]['longitude'] = self.model_data[time]['longitude']

            # Preassign the heights and delays
            levels = self.model_data[time]['levels']
            tot_size = (levels, self.model_data[time]['latitude'].shape[0], self.model_data[time]['longitude'].shape[1])
            self.delay_data[time]['heights'] = np.zeros(shape=self.model_data[time]['pressures'].shape)
            self.delay_data[time]['wet_delay'] = np.zeros(shape=tot_size)
            self.delay_data[time]['liquid_delay'] = np.zeros(shape=tot_size)
            self.delay_data[time]['hydrostatic_delay'] = np.zeros(shape=tot_size)
            self.delay_data[time]['total_delay'] = np.zeros(shape=tot_size)
            self.delay_data[time]['heights'][0, :, :] = self.model_data[time]['surface_height']

            r_lat, g_lat = self.radius_graviational_const_from_latitude(self.model_data[time]['latitude'][:, 0])
            g1 = g_lat[:, None] * (r_lat[:, None] / (r_lat[:, None] + self.model_data[time]['surface_height'])) ** 2
            num = self.model_data[time]['latitude'].size

            h1 = self.model_data[time]['surface_height']

            # Calculate the heigths and delays for the different layers. We start at the surface, so in reverse order.
            for l in np.arange(levels):

                T = self.model_data[time]['Temperature'][l, :, :]
                q0 = self.model_data[time]['Specific humidity'][l, :, :]
                p0 = self.model_data[time]['pressures'][l, :, :]
                p1 = self.model_data[time]['pressures'][l + 1, :, :]

                # For the top layer we do not go to pressure 0
                if l == 0:
                    p1 = p1 + 1

                # First guess
                h0 = h1
                g0 = g1
                dh = 100.0
                dh_0 = 0.0

                if l == levels - 1:
                    p1 = p1 + 1

                # Now iterate to find the geometric heights of the different levels.
                while np.sum(np.abs(dh - dh_0) < 0.1) < num:
                    dh_0 = dh
                    g1 = g_lat[:, None] * (r_lat[:, None] / (r_lat[:, None] + h1)) ** 2 # New top level gravity estimation
                    g2 = (g0 + g1) / 2.0                        # Mean gravity over model level
                    Tv = T * (1.0 + eps * q0)                   # Virtual temperature
                    dh = - Rd * Tv / g2 * np.log(p1 / p0)       # Depth of model level layer
                    h1 = h0 + dh

                self.delay_data[time]['heights'][l + 1, :, :] = h1

                # Now we can follow two approaches:
                # 1. Calculate refractivity at mid-level and multiply with height
                # 2. Calculate total refractivity from number of particles.
                # We think the second approach is more accurate but the first will be given to here.
                q = self.model_data[time]['Specific humidity'][l, :, :]
                mean_p = (p0 + p1) / 2  # Mean pressure (equivalent to Rd * rho)
                rho = mean_p / (Rd * T)

                # Liquid water content is calculated same way for both methods.
                # If we use the CERRA data, this is not always available, so it will be skipped and kept at zero
                # in that case. The contribution to the total delay is minimal and insignificant in most cases.
                # This does therefore not create any problems in the final calculation as other factors like model
                # resolution and/or uncertainty in model variables have a much larger effect.
                vars = list(self.model_data[time].keys())
                if 'Specific cloud liquid water content' in vars and 'Specific cloud ice water content' in vars:
                    plw = (self.model_data[time]['Specific cloud liquid water content'][l, :, :] +
                           self.model_data[time]['Specific cloud ice water content'][l, :, :]) * (p0 - p1) / g2 / 1000
                    self.delay_data[time]['liquid_delay'][l, :, :] = klw * plw / 10**8

                # Everything according to Meteorological applications of a surface network of Global Positioning System receivers, S. De Haan 2008, page 22
                # approach 1 based on heights
                if method == 'heights':
                    self.delay_data[time]['wet_delay'][l, :, :] = dh * rho * q * Rd / eps * (-k1 * eps + k2 + k3 / T)
                    self.delay_data[time]['hydrostatic_delay'][l, :, :] = dh * k1 * rho * Rd
                # approach 2 based on pressures
                elif method == 'pressures':
                    self.delay_data[time]['wet_delay'][l, :, :] = (Rd / g2 / eps * (p0 - p1) * q *
                                                                   (-k1 * eps + k2 + k3 / T) / 10**8)
                    self.delay_data[time]['hydrostatic_delay'][l, :, :] = Rd / g2 * (p0 - p1) * k1 / 10**8

                # Add the three different types of delays together.
                self.delay_data[time]['total_delay'][l, :, :] = (self.delay_data[time]['hydrostatic_delay'][l, :, :] +
                                                              self.delay_data[time]['wet_delay'][l, :, :] +
                                                              self.delay_data[time]['liquid_delay'][l, :, :])

            # This step is to correct the ERA5 data for the geoid, as we work with data with respect to the ellipsoid
            # in further scripts. You can disable this by setting the geoid flag to false.
            self.delay_data[time]['heights'] += self.delay_data[time]['geoid'][None, :, :]

    @staticmethod
    def radius_graviational_const_from_latitude(latitudes):
        """
        Get the earth radius and graviational constant at geoid
        reference: https://en.wikipedia.org/wiki/Normal_gravity_formula#cite_note-1
        """

        a_wgs = 6378137.000
        e_wgs = 0.081819  # excentricity WGS84
        f_wgs = 0.003352811  # flattening
        g_equ = 9.7803253359  # Gravity constant at equator
        ks_som = 0.001931853  # constant formula Somigliana
        sin_lat = np.sin(latitudes * np.pi / 180) ** 2

        # Calculate radius based on
        r_lat = a_wgs * (1 - f_wgs * sin_lat)
        # Calculate gravity based on the formula of Somigliana https://en.wikipedia.org/wiki/Normal_gravity_formula#cite_note-1
        g_lat = g_equ * (1 + ks_som * sin_lat) / np.sqrt(1 - e_wgs ** 2 * sin_lat)

        return r_lat, g_lat

    @staticmethod
    def geopotential_height_to_real_height(latitudes, geo_height):
        """
        Calculate height based on latitude and geopotential height
        reference: https://en.wikipedia.org/wiki/Normal_gravity_formula#cite_note-1

        """

        g_equ = 9.7803253359  # Gravity constant at equator
        r_lat, g_lat = ModelToDelay.radius_graviational_const_from_latitude(latitudes)

        # Calculate geometric height and gravity at surface.
        height = r_lat * geo_height / ((g_lat / g_equ * r_lat) - geo_height)

        return height

    @staticmethod
    def real_height_to_geopotential_height(latitudes, height):
        """
        Calculate height based on latitude and geopotential height
        reference: https://en.wikipedia.org/wiki/Normal_gravity_formula#cite_note-1

        """

        g_equ = 9.7803253359  # Gravity constant at equator
        r_lat, g_lat = ModelToDelay.radius_graviational_const_from_latitude(latitudes)

        # Calculate geometric height and gravity at surface.
        geo_height = height * (g_lat / g_equ * r_lat) / (r_lat + height)

        return geo_height

    @staticmethod
    def water_vapor_pressure(T, RH):
        Tc = T - 273.15
        e_sat = 0.61121 * np.exp((18.678 - (Tc / 234.5)) * (Tc / (257.14 + Tc))) * 1000
        e = e_sat * RH / 100

        return e

    @staticmethod
    def relative_to_specific_humidity(T, RH, p):
        """
        Calculate specific humidity

        """

        Rv = 461.5  # specific gas constant water vapor
        Rd = 287.0586  # specific gas constant air

        e = ModelToDelay.water_vapor_pressure(T, RH)
        w = e * Rd / (Rv * (p - e))
        q = w / (w + 1)

        return q

    def remove_delay(self, time):

        self.delay_data.pop(time)
