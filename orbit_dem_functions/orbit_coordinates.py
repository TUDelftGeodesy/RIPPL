# Class to calculate the geometry of a radar image based on the orbits.
import datetime
import os
import numpy as np
from rippl.orbit_dem_functions.orbit_interpolate import OrbitInterpolate
from rippl.image_data import ImageData


class OrbitCoordinates(OrbitInterpolate, ImageData):

    def __init__(self, meta='', dem_grid=''):
        # This function needs several input files. This works for now, but ideally this information should all be
        # contained in the .res file.

        # Load data from res file
        if isinstance(meta, str):
            if len(meta) != 0:
                ImageData.__init__(self, meta, 'single')
        elif isinstance(meta, ImageData):
            self.__dict__ = meta.__dict__.copy()

        # interpolate the orbit (from InterpolateOrbit)
        OrbitInterpolate.__init__(self, meta)
        self.fit_orbit_spline(vel=True, acc=True)

        if 'readfiles' in self.processes:
            meta_dat = 'readfiles'
            meta_crop = 'crop'
        elif 'coreg_readfiles' in self.processes:
            meta_dat = 'coreg_readfiles'
            meta_crop = 'coreg_crop'

        # Get main variables from the .res file
        self.az_time = self.processes[meta_dat]['First_pixel_azimuth_time (UTC)']
        az_seconds = (datetime.datetime.strptime(self.az_time, '%Y-%m-%dT%H:%M:%S.%f') -
                           datetime.datetime.strptime(self.az_time[:10], '%Y-%m-%d'))
        self.az_seconds = az_seconds.seconds + az_seconds.microseconds / 1000000.0
        self.ra_time = float(self.processes[meta_dat]['Range_time_to_first_pixel (2way) (ms)']) / 1000
        self.az_step = 1 / float(self.processes[meta_dat]['Pulse_Repetition_Frequency (computed, Hz)'])
        self.ra_step = 1 / float(self.processes[meta_dat]['Range_sampling_rate (computed, MHz)']) / 1000000

        self.degree2rad = np.asarray([np.pi / 180])
        self.rad2degree = np.asarray([180 / np.pi])
        self.center_phi = float(self.processes[meta_dat]['Scene_centre_latitude']) * self.degree2rad
        self.center_lambda = float(self.processes[meta_dat]['Scene_centre_longitude']) * self.degree2rad

        # Read pixel coordinate information
        self.first_line = int(self.processes[meta_crop]['crop_first_line'])
        self.first_pixel = int(self.processes[meta_crop]['crop_first_pixel'])
        lines = int(self.processes[meta_crop]['crop_lines'])
        pixels = int(self.processes[meta_crop]['crop_pixels'])
        self.size = (lines, pixels)

        # Define the main variables for this function
        self.az_times = np.asarray([])          # azimuth times of different lines
        self.ra_times = np.asarray([])          # range times of different pixels
        self.lines = np.asarray([])             # line numbers
        self.pixels = np.asarray([])            # pixel numbers
        self.regular = True
        self.xyz_orbit = np.asarray([])         # the orbit coordinates at azimuth times
        self.vel_orbit = np.asarray([])         # the orbit velocities at azimuth times
        self.acc_orbit = np.asarray([])         # the orbit acceleration at azimuth times
        self.lat = np.asarray([])               # latitude of points on the ground (WGS84)
        self.lon = np.asarray([])               # longitude of points on the ground (WGS84)
        self.height = np.asarray([])            # heights of points on the ground (WGS84)
        self.lat_orbit = np.asarray([])         # latitude of azimuth points along orbit (WGS84)
        self.lon_orbit = np.asarray([])         # longitude of azimuth points along orbit (WGS84)
        self.height_orbit = np.asarray([])      # height of azimuth points along orbit (WGS84)
        self.x = np.asarray([])                 # cartesian coordinates on the ground
        self.y = np.asarray([])                 # cartesian coordinates on the ground
        self.z = np.asarray([])                 # cartesian coordinates on the ground
        self.geoid = np.asarray([])             # geoid correction for points on the ground (EGM96)

        # Variables that can be used for ray tracing
        self.off_nadir_angle = np.asarray([])   # off nadir angle for all pixels
        self.heading = np.asarray([])           # heading of satellite
        self.azimuth_angle = np.asarray([])     # azimuth angle from points on the ground
        self.elevation_angle = np.asarray([])   # elevation angle from points on the ground to the satellite

        # constants
        self.sol = 299792458                        # speed of light [m/s]
        self.ellipsoid = [6378137.0, 6356752.3141]  # lenght of axis of ellips [m]
        self.maxiter = 10                           # maximum number of iterations radar to xyz calculations
        self.maxiter_coor = 1                       # maximum number of iterations lat/lon calculations
        self.criterpos = 0.00000001                 # iteration accuracy [m]

        # Load dem
        self.dem_grid = dem_grid
        if dem_grid:
            if os.path.exists(dem_grid):
                self.dem_data = np.memmap(dem_grid, dtype=np.float32, shape=self.size, mode='r')
            else:
                print('DEM file cannot be loaded. Path does not exist')
                self.dem_data = []
        else:
            # print('No DEM file loaded. Make sure you run the create_radar_dem function if you want to calculate' +
            #      ' xyz coordinates of specific points')
            self.dem_data = []

    # The next two processing_steps convert from and to pixel coordinates and range/azimuth times.
    def lp_time(self, lines='', pixels='', regular=True):
        # This function calculates the azimuth and range timing (two way), based on a resfile.
        # If there are no specific lines or pixels specified, we will assume you want to work with the first pixel.

        if len(lines) == 0:
            lines = self.lines
        else:
            self.lines = lines
        if len(pixels) == 0:
            pixels = self.pixels
        else:
            self.pixels = pixels

        self.regular = regular

        if not regular:
            if len(lines) != len(pixels) or len(pixels) == 0 or len(lines) == 0:
                print('For irregular lines/pixels arrays both arrays should be equal length and not empty')

            all_lines = np.arange(np.min(lines), np.max(lines) + 1)
            all_pixels = np.arange(np.min(pixels), np.max(pixels) + 1)

            self.az_times = self.az_seconds + self.az_step * (all_lines - 1)
            self.ra_times = self.ra_time + self.ra_step * (all_pixels - 1)
            self.xyz_orbit, self.vel_orbit, self.acc_orbit = self.evaluate_orbit_spline(self.az_times, vel=True, acc=True)

            # Reduce the line number to create right az/ra ids
            self.lines = lines - np.min(lines)
            self.pixels = pixels - np.min(pixels)

        else:
            self.az_times = self.az_seconds + self.az_step * (lines - 1)
            self.ra_times = self.ra_time + self.ra_step * (pixels - 1)

        # Evaluate the orbit based on the given az_times
        self.xyz_orbit, self.vel_orbit, self.acc_orbit = self.evaluate_orbit_spline(self.az_times, vel=True, acc=True)

    def time_lp(self, az_times=None, ra_times=None):
        # This function calculates the azimuth and range timing (two way), based on a resfile.
        # If there are no specific lines or pixels specified, we will assume you want to work with the first pixel.

        if az_times:
            self.az_times = np.asarray(az_times)
        if ra_times:
            self.ra_times = np.asarray(ra_times)

        self.lines = ((self.az_times - self.az_seconds) / self.az_step) + 1
        self.pixels = ((self.ra_times - self.ra_time) / self.ra_step) + 1

        # Evaluate the orbit based on the given az_times
        self.xyz_orbit, self.vel_orbit, self.acc_orbit = self.evaluate_orbit_spline(self.az_times, vel=True, acc=True)

    def load_height(self):
        # This function loads the heights from the dem file, based on the line and pixel values

        if len(self.dem_data) == 0:
            print('There is no DEM file loaded')
            return
        if len(self.pixels) == 0 or len(self.lines) == 0:
            print('Variables lines and pixels should be created first. Using either the lp_time or time_lp method')

        self.height = self.dem_data[(self.lines - self.first_line)[:, None], self.pixels - self.first_pixel]

    # The next two function are used to calculate xyz coordinates on the ground.
    # To do so we also need the heights of the points on the ground.
    def lph2xyz(self):
        # LPH2XYZ   Convert radar line/pixel coordinates into Cartesian coordinates.
        #   XYZ=LPH2XYZ(LINE,PIXEL,HEIGHT,IMAGE,ORBFIT) converts the radar coordinates
        #   LINE and PIXEL into Cartesian coordinates XYZ. HEIGHT contains the height
        #   of the pixel above the ellipsoid. IMAGE contains the image metadata
        #   and ORBFIT the orbit fit, proceduced respectively by the METADATA and
        #   ORBITFIT processing_steps.
        #
        #   [XYZ,SATVEC]=LPH2XYZ(...) also outputs the corresponding satellite
        #   position and velocity, with SATVEC a matrix with in it's columns the
        #   position X(m), Y(m), Z(m) and velocity X_V(m/s), Y_V(m/s), Z_V(m/s),
        #   with rows corresponding to rows of XYZ.
        #
        #   [...]=LPH2XYZ(...,'VERBOSE',0,'MAXITER',10,'CRITERPOS',1e-6,'ELLIPSOID',
        #   [6378137.0 6356752.3141]) includes optional input arguments VERBOSE,
        #   MAXITER, CRITERPOS and ELLIPSOID  to overide the default verbosity level,
        #   interpolator exit criteria and ellipsoid. The defaults are the WGS-84
        #   ELLIPSOID=[6378137.0 6356752.3141], MAXITER=10, CRITERPOS=1e-6 and
        #   VERBOSE=0.
        #
        #   Example:
        #       [image, orbit] = metadata('master.res');
        #       orbfit = orbitfit(orbit);
        #       xyz = lph2xyz(line,pixel,0,image,orbfit);
        #
        #   See also METADATA, ORBITFIT, ORBITVAL, XYZ2LP and XYZ2T.
        #
        #   (c) Petar Marinkovic, Hans van der Marel, Delft University of Technology, 2007-2014.

        #   Created:    20 June 2007 by Petar Marinkovic
        #   Modified:   13 March 2014 by Hans van der Marel
        #                - added description and input argument checking
        #                - use orbit fitting procedure
        #                - added output of satellite position and velocity
        #                - original renamed to LPH2XYZ_PM
        #                6 April 2014 by Hans van der Marel
        #                - improved handling of optional parameters
        #                5 July 2017 by Gert Mulder
        #                - converted to python code
        #                - created read of .res files
        #                - vectorized to optimize for speed

        if len(self.lines) == 0:
            print('First define for which pixels or which azimuth/range times you want to compute the xyz coordinates')
            return
        if len(self.height) == 0:
            print('First find the heights of the invidual pixels. This can be done using the create DEM function')

        ell_a = self.ellipsoid[0]
        ell_b = self.ellipsoid[1]
        ell_e2 = 1 - ell_b ** 2 / ell_a ** 2

        # Some preparations to get the start conditions
        height = np.ravel(self.height)

        h = np.mean(height)
        ell_a_2 = (ell_a + height)**2    # Preparation for distance on ellips with certain height
        ell_b_2 = (ell_b + height)**2    # Preparation for distance on ellips with certain height
        Ncenter = ell_a / np.sqrt(1 - ell_e2 * (np.sin(self.center_phi) ** 2))
        scenecenterx = (Ncenter + h) * np.cos(self.center_phi) * np.cos(self.center_lambda)
        scenecentery = (Ncenter + h) * np.cos(self.center_phi) * np.sin(self.center_lambda)
        scenecenterz = (Ncenter + h - ell_e2 * Ncenter) * np.sin(self.center_phi)

        # These arrays are only in the azimuth direction
        possatx = self.xyz_orbit[0, :]
        possaty = self.xyz_orbit[1, :]
        possatz = self.xyz_orbit[2, :]
        velsatx = self.vel_orbit[0, :]
        velsaty = self.vel_orbit[1, :]
        velsatz = self.vel_orbit[2, :]

        # First guess
        if self.regular:
            num = len(self.lines) * len(self.pixels)
            shp = (len(self.lines), len(self.pixels))
        else:
            num = len(self.lines)
            shp = len(self.lines)

        height = []
        posonellx = np.ones(num) * scenecenterx
        posonelly = np.ones(num) * scenecentery
        posonellz = np.ones(num) * scenecenterz

        # 1D id, 2D row and column ids (used to extract information
        az = np.arange(len(self.lines)).astype(np.int32)[:, None]

        if self.regular:
            az_id = np.ravel(az * np.ones((1, len(self.pixels)))).astype(np.int32)
            range_dist = np.ravel((self.sol * self.ra_times[None, :] / 2) ** 2 * np.ones((len(self.lines), 1)))
        else:
            az_id = self.lines
            range_dist = np.ravel((self.sol * self.ra_times[self.pixels] / 2)**2)

        # Next parameter defines which points still needs another iteration to solve. If the precisions are met,
        # this point will be removed from the dataset.
        solve_ids = np.arange(num).astype(np.int32)

        for iterate in range(self.maxiter):

            # Distance of orbit points with start point
            dsat_Px = np.take(posonellx, solve_ids) - np.take(possatx, az_id)
            dsat_Py = np.take(posonelly, solve_ids) - np.take(possaty, az_id)
            dsat_Pz = np.take(posonellz, solve_ids) - np.take(possatz, az_id)

            # Equations 1. range line perpendicular to orbit
            #           2. range time times speed of light same as distance orbit to point
            #           3. point on ellipsoid
            equations = np.zeros(shape=(3, len(solve_ids)))

            equations[0, :] = -(np.take(velsatx, az_id) * dsat_Px + np.take(velsaty, az_id) *
                                dsat_Py + np.take(velsatz, az_id) * dsat_Pz)
            equations[1, :] = -(dsat_Px**2 + dsat_Pz**2 + dsat_Py**2 - range_dist)  # Add average atmospheric delay?
            equations[2, :] = -((np.take(posonellx, solve_ids)**2 + np.take(posonelly, solve_ids)**2) /
                                ell_a_2 + (np.take(posonellz, solve_ids)**2 / ell_b_2) - 1)

            # derivatives of 3 components for linearization
            derivatives = np.zeros(shape=(3, 3, len(solve_ids)))

            derivatives[0, 0, :] = np.take(velsatx, az_id)
            derivatives[1, 0, :] = np.take(velsaty, az_id)
            derivatives[2, 0, :] = np.take(velsatz, az_id)
            derivatives[0, 1, :] = 2 * dsat_Px
            derivatives[1, 1, :] = 2 * dsat_Py
            derivatives[2, 1, :] = 2 * dsat_Pz
            derivatives[0, 2, :] = (2 * np.take(posonellx, solve_ids)) / ell_a_2
            derivatives[1, 2, :] = (2 * np.take(posonelly, solve_ids)) / ell_a_2
            derivatives[2, 2, :] = (2 * np.take(posonellz, solve_ids)) / ell_b_2
            dsat_Px = []
            dsat_Py = []
            dsat_Pz = []

            # Solve system of equations
            solpos = np.linalg.solve(derivatives.swapaxes(0, 2), equations.swapaxes(0, 1)).swapaxes(0, 1)

            # Update solution
            posonellx[solve_ids] += solpos[0, :]
            posonelly[solve_ids] += solpos[1, :]
            posonellz[solve_ids] += solpos[2, :]
            derivatives = []

            # Check which ids are close enough
            not_finished = np.ravel(np.argwhere(((np.abs(solpos[0, :]) < self.criterpos) *
                                             (np.abs(solpos[1, :]) < self.criterpos) *
                                             (np.abs(solpos[2, :]) < self.criterpos)) == False))

            # If all points are found we can stop the iteration
            if len(not_finished) == 0:
                # print('All points located within ' + str(iterate + 1) + ' iterations.')
                break

            # prepare for next iteration by removing values from these variables
            solve_ids = np.take(solve_ids, not_finished)
            az_id = np.take(az_id, not_finished)
            range_dist = np.take(range_dist, not_finished)
            ell_a_2 = np.take(ell_a_2, not_finished)
            ell_b_2 = np.take(ell_b_2, not_finished)

            # If some point are not found within the iteration time, give a warning
            if iterate == self.maxiter - 1:
                print(str(len(solve_ids)) + 'did not converge within ' + str(
                    self.maxiter) + ' iterations. Maybe use more iterations or less stringent criteria?')

        self.x = np.reshape(posonellx, shp)
        posonellx = []
        self.y = np.reshape(posonelly, shp)
        posonelly = []
        self.z = np.reshape(posonellz, shp)
        posonellz = []

    # This function is mainly used to find the coordinates of known points on the ground in the radar grid
    def xyz2lp(self, x, y, z, az_times=''):
        # XYZ2T   Convert Cartesian coordinates into radar azimuth and range time.
        #   [T_AZI,T_RAN]=XYZ2T(XYZ,resfile) converts the Cartesian coordinates
        #   XYZ into radar azimuth time T_AZI and range time T_RAN. IMAGE contains
        #   image metadata and orbits.
        #
        #   [...]=XYZ2T(...,t0, maxiter, criterpos) includes
        #   optional input arguments VERBOSE, MAXITER and CRITERPOS to override
        #   the default verbosity level and interpolator exit criteria. The defaults
        #   are MAXITER=10, CRITERPOS=1e-10.
        #   With the t0 value you can add a first guess for the azimuth time.
        #   Especially for large datasets it is usefull to start with a small subset of points
        #   and run the function again after a first interpolation for all other points.
        #
        #   See also ORBITFIT, ORBITVAL, XYZ2LP and LPH2XYZ.
        #
        #   (c) Hans van der Marel, Delft University of Technology, 2014.
        #
        #   Created:    20 June 2007 by Petar Marinkovic
        #   Modified:   13 March 2014 by Hans van der Marel
        #                - added description and input argument checking
        #                - completely reworked orbit fitting procedure
        #                - added output of satellite position and velocity
        #                - original renamed to XYZ2T_PM
        #                6 April 2014 by Hans van der Marel
        #                - improved handling of optional parameters
        #                6 July 2017 by Gert Mulder
        #                - rewrite in python
        #                - optimize fit of orbit using cubic spline
        #                - vectorize to improve speed

        n = np.size(x)

        if not x.shape == y.shape == z.shape:
            print('x,y,z coordinate matrices should have same dimensions')

        old_shape = x.shape
        xyz = np.concatenate((np.ravel(x)[None, :], np.ravel(y)[None, :], np.ravel(z)[None, :]), axis=0)
        x = []
        y = []
        z = []

        # Make a first guess of the azimuth times:
        if len(az_times) == 0:
            az_times = self.az_seconds * np.ones((1, xyz.shape[1]))
        solve_ids = np.arange(n)

        # Now start the iteration to find the optimal point
        for iterate in range(self.maxiter):

            # Prepare next iteration
            if iterate != 0:
                sort = np.argsort(az_times[0, solve_ids])
                solve_ids = solve_ids[sort]
                sort = []
                az_time = az_times[0, solve_ids]
            else:
                az_time = np.ravel(az_times)

            possat = self.evaluate_orbit_spline(az_time, pos=True, acc=False, vel=False, sorted=True)[0]
            delta = xyz[:, solve_ids] - possat

            accsat = self.evaluate_orbit_spline(az_time, pos=False, acc=True, vel=False, sorted=True)[2]
            s1 = np.sum(delta * accsat, axis=0)
            accsat = []
            velsat = self.evaluate_orbit_spline(az_time, pos=False, acc=False, vel=True, sorted=True)[1]
            s0 = np.sum(delta * velsat, axis=0)
            delta = []
            s2 = np.sum(velsat * velsat, axis=0)
            velsat = []

            t_diff = -s0 / (s1 - s2)
            az_times[0, solve_ids] += t_diff

            # remove approximation which are close enough
            finished = np.ravel(np.abs(t_diff) > self.criterpos)

            # Now remove info from points which are finished
            solve_ids = solve_ids[finished]

            if np.sum(finished) == 0:
                # print('Finished within ' + str(iterate) + ' iterations.')
                break

            if iterate == self.maxiter - 1:
                print(str(len(solve_ids)) + 'did not converge within ' + str(
                    self.maxiter) + ' iterations. Maybe use more iterations or less stringent criteria?')

        # Calculate range times
        dist_diff = self.evaluate_orbit_spline(np.ravel(az_times))[0] - xyz
        range_dist = np.sqrt(np.sum(dist_diff**2, axis=0))
        dist_diff = []
        ra_times = range_dist / self.sol * 2

        lines = (((az_times - self.az_seconds) / self.az_step) + 1.0).reshape(old_shape)
        pixels = (((ra_times - self.ra_time) / self.ra_step) + 1.0).reshape(old_shape)

        return lines, pixels

    # Next two processing_steps convert back and forth from cartesian to lat/lon
    # This function is used to find known lat/lon/h points to convert to xyz, which can be used to find the radar
    # coordinates later on.
    @staticmethod
    def ell2xyz(lat, lon, height, ell_axis=''):
        # convert geodetic coordinate phi,lambda,height to geocentered cartesian
        # coordinate
        #
        # output:
        # x,y,z              ---  geocentered cartesian coordinate
        #
        # input:
        # ell_axis      ---  semimajor amd semiminor axis of ellipsoid
        # lat,lon,height  ---  geodetic coordinate,latitude,longitude(in degree),height
        # ********************************************

        if len(ell_axis) == 2:
            ell_a = ell_axis[0]
            ell_b = ell_axis[1]
        else:
            # Default values based on WGS84
            ell_a = 6378137.0
            ell_b = 6356752.3141

        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        # Calculate squared first eccentricity
        ell_e2 = (ell_a ** 2 - ell_b ** 2) / ell_a ** 2

        N = ell_a / np.sqrt(1 - ell_e2 * (np.sin(lat)) ** 2)
        Nph = N + height

        x = Nph * np.cos(lat) * np.cos(lon)
        y = Nph * np.cos(lat) * np.sin(lon)
        z = (Nph - ell_e2 * N) * np.sin(lat)

        return x, y, z

    # This method is used to convert the calculated xyz points to find the lat, lon, h on the ellipsoid.
    def xyz2ell(self, method=1, h_diff=False, pixel=True, orbit=False):
        # convert geodetic coordinate phi,lambda,height to geocentered cartesian
        # coordinate
        #
        # output:
        # lat,lon,height  ---  geodetic coordinate,latitude,longitude(in degree),height
        #
        # input:
        # ell_axis      ---  semimajor amd semiminor axis of ellipsoid
        # x,y,z              ---  geocentered cartesian coordinate

        # Calculations after Fukushima (2006)
        # Transformation from Cartesian to geodetic coordinates accelerated by Halleys method
        # Second method from http://www.navipedia.net/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
        # ********************************************

        if len(self.x) == 0:
            print('First calculate the cartesian xyz coordinates before converting to lat/lon coordinates')
            return
        if not pixel and not orbit:
            print('Neither orbit coordinates nor pixel coordinates are selected')
            return

        a = self.ellipsoid[0]
        b = self.ellipsoid[1]

        p = dict()
        z = dict()

        types = []
        if orbit:
            z['orbit'] = self.xyz_orbit[2, :]
            p['orbit'] = np.sqrt(self.xyz_orbit[0, :]**2 + self.xyz_orbit[1, :]**2)
            types.append('orbit')
            self.lon_orbit = np.arctan(self.xyz_orbit[1, :] / self.xyz_orbit[1, :])
        if pixel:
            z['pixel'] = np.ravel(self.z)
            p['pixel'] = np.sqrt(np.ravel(self.x)**2 + np.ravel(self.y)**2)
            types.append('pixel')
            self.lon = np.arctan(self.y / self.x).astype(np.float32).reshape(self.x.shape) / np.pi * 180

        for t in types:
            if method == 1:
                # Calculate squared first eccentricity
                e = np.sqrt(1 - b ** 2 / a ** 2)
                E = e**2
                P = 1 / a * p[t]
                ec = np.sqrt(1 - e ** 2)
                Z = 1 / a * ec * np.abs(z[t])

                T = np.abs(z[t]) / (ec * p[t])

                # Now do the first iteration. Just one iteration is generally close enough.
                # Maybe check for extreme heights 10000 km+
                for i in range(self.maxiter_coor):
                    Ts = np.sqrt(1 + T**2)
                    g0 = P * T - Z - E * T / Ts
                    g1 = P - E / (Ts**3)
                    g2 = - 3 * E * T / (Ts**5)

                    T -= g0 / (g1 - g2 * g0 / (2 * g1))

                lat = np.sign(z[t]) * np.arctan(T / ec)
                if h_diff or t == 'orbit':
                    height = (ec * p[t] + np.abs(z[t]) * T - b * np.sqrt(1 + T ** 2)) / np.sqrt(ec**2 + T**2)

            else:
                # ESA procedure
                e = np.sqrt(1 - b**2 / a**2)

                lat = np.arctan(z[t] / ((1 - e ** 2) * p[t]))

                for i in range(self.maxiter_coor):
                    N = a / (np.sqrt(1 - e**2 * np.sin(lat)**2))
                    height = p[t] / np.cos(lat) - N
                    lat = np.arctan(z[t] / ((1 - e ** 2 * (N / (N + height))) * p[t]))

            if t == 'orbit':
                self.lat_orbit = lat.astype(np.float32)
                self.height_orbit = height
            elif t == 'pixel':
                self.lat = np.reshape(lat, self.lon.shape).astype(np.float32).reshape(self.x.shape) / np.pi * 180

        if h_diff:
            return np.reshape(height, self.height.shape) - self.height

    # Next two processing_steps are used to find the heading and inclination angle of the satellite at certain pixel, range
    # combinations, as well as the azimuth and elevation angle on the ground.
    def xyz2orbit_heading_off_nadir(self):
        # Calculates the heading and off-nadir angle from the satellite

        if len(self.x) == 0:
            print('First calculate the cartesian xyz coordinates before you run this function')
            return
        elif len(self.lat_orbit) == 0:
            print('To calculate the heading we also need the latitude. Compute the lat/lon first.')
            return

        # orbit vector
        if self.regular:
            x_diff = self.xyz_orbit[0, :][:, None] - self.x
            y_diff = self.xyz_orbit[1, :][:, None] - self.y
            z_diff = self.xyz_orbit[2, :][:, None] - self.z
        else:
            x_diff = self.xyz_orbit[0, self.lines] - self.x
            y_diff = self.xyz_orbit[1, self.lines] - self.y
            z_diff = self.xyz_orbit[2, self.lines] - self.z

        ray = np.stack((x_diff, y_diff, z_diff), axis=0)
        ray = ray / np.sqrt(np.sum(ray**2, axis=0))
        x_diff = []
        y_diff = []
        z_diff = []

        # Now find the correction for the ellipsoid
        # Calc tangent plane
        x_tan = 2 * self.xyz_orbit[0, :] / (self.ellipsoid[0] + self.height_orbit)**2
        y_tan = 2 * self.xyz_orbit[1, :] / (self.ellipsoid[0] + self.height_orbit)**2
        z_tan = 2 * self.xyz_orbit[2, :] / (self.ellipsoid[1] + self.height_orbit)**2

        # Calc normalized vector
        N = np.transpose(np.vstack((x_tan, y_tan, z_tan)))
        N = N / np.sqrt(np.sum(N**2, axis=1))[:, None]
        x_tan = []
        y_tan = []
        z_tan = []

        # Calc off nadir angle
        if self.regular:
            self.off_nadir_angle = np.arccos(np.einsum('jik,ij->ik', ray, N)).astype(np.float32) / np.pi * 180
        else:
            self.off_nadir_angle = np.arccos(np.einsum('ji,ij->i', ray, N[self.lines, :])).astype(np.float32) / np.pi * 180

        # Calc vector on tangent plant to satellite and to north
        north_plane = np.stack((-self.xyz_orbit[0, :], -self.xyz_orbit[1, :], np.zeros((N.shape[0]))), axis=0)
        north_vector = OrbitCoordinates.project_on_plane(np.transpose(N), north_plane, regular=False)

        # Next part is only used for the heading, which is the same for every line.
        v_plane = OrbitCoordinates.project_on_plane(np.transpose(N), self.vel_orbit, regular=False)

        # Based on https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
        dot = np.einsum('ki,ki->i', v_plane, north_vector)
        det = np.einsum('ki,ki->i', np.transpose(N), np.cross(v_plane, north_vector, axis=0))
        v_plane = []
        north_vector = []

        if self.regular:
            self.heading = np.arctan2(det, dot).astype(np.float32) / np.pi * 180
        else:
            self.heading = (np.arctan2(det, dot).astype(np.float32) / np.pi * 180)[self.lines]

    def xyz2scatterer_azimuth_elevation(self):
        # Calculates the azimuth and elevation angle of a point on the ground based on the point on the
        # ground (xyz) and the point in orbit

        if len(self.x) == 0:
            print('First calculate the cartesian xyz coordinates before you run this function')
            return

        # point to orbit vector
        if self.regular:
            x_diff = self.xyz_orbit[0, :][:, None] - self.x
            y_diff = self.xyz_orbit[1, :][:, None] - self.y
            z_diff = self.xyz_orbit[2, :][:, None] - self.z
        else:
            x_diff = self.xyz_orbit[0, self.lines] - self.x
            y_diff = self.xyz_orbit[1, self.lines] - self.y
            z_diff = self.xyz_orbit[2, self.lines] - self.z

        # Calc normalized vector ground to satellite
        diff = np.stack((x_diff, y_diff, z_diff), axis=0)
        ray = diff / np.sqrt(np.sum(diff**2, axis=0))
        diff = []
        x_diff = []
        y_diff = []
        z_diff = []

        # Calc tangent plane
        x_tan = 2 * self.x / (self.ellipsoid[0] + self.height)**2
        y_tan = 2 * self.y / (self.ellipsoid[0] + self.height)**2
        z_tan = 2 * self.z / (self.ellipsoid[1] + self.height)**2

        # Calc normalized vector normal to surface ellipsoid
        N = np.stack((x_tan, y_tan, z_tan), axis=0)
        N = N / np.sqrt(np.sum(N**2, axis=0))
        x_tan = []
        y_tan = []
        z_tan = []

        # Calc elevation angle
        if self.regular:
            self.elevation_angle = np.arccos(np.einsum('ijk,ijk->jk', ray, -N)).astype(np.float32) / np.pi * 180 - 90
        else:
            self.elevation_angle = np.arccos(np.einsum('ij,ij->j', ray, -N)).astype(np.float32) / np.pi * 180 - 90

        # Calc vector on tangent plant to satellite and to north
        if self.regular:
            north_plane = np.stack((-self.x, -self.y, np.zeros((self.x.shape[0], self.x.shape[1]))), axis=0)
        else:
            north_plane = np.stack((-self.x, -self.y, np.zeros((self.x.shape[0]))), axis=0)

        north_vector = OrbitCoordinates.project_on_plane(N, north_plane, regular=self.regular)
        ray_vector = OrbitCoordinates.project_on_plane(N, ray, regular=self.regular)

        # Based on https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
        if self.regular:
            dot = np.einsum('kij,kij->ij', ray_vector, north_vector)
            det = np.einsum('kij,kij->ij', N, np.cross(ray_vector, north_vector, axis=0))
        else:
            dot = np.einsum('ki,ki->i', ray_vector, north_vector)
            det = np.einsum('ki,ki->i', N, np.cross(ray_vector, north_vector, axis=0))
        ray_plane = []
        north_vector = []
        self.azimuth_angle = np.arctan2(det, dot).astype(np.float32) / np.pi * 180

    @staticmethod
    def project_on_plane(N, vector, regular=False):

        N = N / np.sqrt(np.sum(N ** 2, axis=0))

        if regular:
            out_vector = vector - np.einsum('kij,kij->ij', vector, N) * N
        else:
            out_vector = vector - np.einsum('ki,ki->i', vector, N) * N

        return out_vector