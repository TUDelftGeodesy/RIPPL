# This function extracts the orbit from a .res file and uses it to calculate a polynomial for the orbit.
# This polynomial is calculate based on 100 seconds of the orbit and can have max 5 degrees.

import numpy as np
from scipy.interpolate import CubicSpline
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.orbit import Orbit


class OrbitInterpolate(ImageProcessingData):

    """
    :type self.orbit_fit = np.ndarray
    """

    def __init__(self, orbit):

        # Load data from res file
        if not isinstance(orbit, Orbit):
            print('Input for input interpolate should be an orbit object')

        # Initialize the orbit values
        self.t = np.array(orbit.t)
        
        self.x = np.array(orbit.x)
        self.y = np.array(orbit.y)
        self.z = np.array(orbit.z)
        
        self.v_x = np.array(orbit.v_x)
        self.v_y = np.array(orbit.v_y)
        self.v_z = np.array(orbit.v_z)
        
        self.a_x = np.array(orbit.a_x)
        self.a_y = np.array(orbit.a_y)
        self.a_z = np.array(orbit.a_z)

        # Initialize the needed variables
        self.orbit_spline = np.array([])
        self.orbit_fit = np.array([])

    def fit_orbit(self, degree=3, vel=True, acc=True):
        # ORBITFIT   Fit low degree polynomial to short segment of satellite orbit.
        #   ORBFIT=ORBITFIT(ORBIT,DEGREE) fits a polynomial of degree DEGREE to
        #   the orbit state vector ORBIT. ORBIT is matrix with 4 columns,
        #   with respectively t(s), X(m), Y(m), Z(m), and optionally X_V(m/s),
        #   Y_V(m/s), and Z_V(m/s). ORBIT is an output structure which contains
        #   the coefficients of the polynomial fits. The default value for
        #   DEGREE is 3.
        #
        #   ORBFIT=ORBITFIT(ORBIT) fits a degree 3 polynomial to the orbit.
        #
        #   See also evaluate_orbit, lph2xyz, lp2t, t2lp, xyz2lp
        #
        #   (c) Hans van der Marel, Delft University of Technology, 2014.

        #   Created:    14 March 2014 by Hans van der Marel
        #   Modified:   5 Juli 2017 rewritten to python by Gert Mulder

        # Now find the polynomials for x,y,z and their derivatives.
        x_poly = np.polyfit(self.t, self.x, degree)
        y_poly = np.polyfit(self.t, self.y, degree)
        z_poly = np.polyfit(self.t, self.z, degree)
        fit_len = 3

        if vel or acc:
            if len(self.v_x) > 0 and len(self.v_y) > 0 and len(self.v_z) > 0:
                x_der = np.polyfit(self.t, self.v_x, degree - 1)
                y_der = np.polyfit(self.t, self.v_y, degree - 1)
                z_der = np.polyfit(self.t, self.v_z, degree - 1)
            else:
                x_der = np.polyder(x_poly)
                y_der = np.polyder(y_poly)
                z_der = np.polyder(z_poly)
            fit_len = 6
            if acc:
                if len(self.a_x) > 0 and len(self.a_y) > 0 and len(self.a_z) > 0:
                    x_acc = np.polyfit(self.t, self.a_x, degree - 2)
                    y_acc = np.polyfit(self.t, self.a_y, degree - 2)
                    z_acc = np.polyfit(self.t, self.a_z, degree - 2)
                else:
                    x_acc = np.polyder(x_der)
                    y_acc = np.polyder(y_der)
                    z_acc = np.polyder(z_der)
                fit_len = 9

        self.orbit_fit = np.zeros(shape=(fit_len, degree + 1))
        self.orbit_fit[0, :] = x_poly
        self.orbit_fit[1, :] = y_poly
        self.orbit_fit[2, :] = z_poly
        if vel or acc:
            self.orbit_fit[3, 1:] = x_der
            self.orbit_fit[4, 1:] = y_der
            self.orbit_fit[5, 1:] = z_der
            if acc:
                self.orbit_fit[6, 2:] = x_acc
                self.orbit_fit[7, 2:] = y_acc
                self.orbit_fit[8, 2:] = z_acc
        self.orbit_fit = np.fliplr(self.orbit_fit)

    def evaluate_orbit(self, az_times):
        # Input argument checking and default values
        # ORBITVAL   Compute satellite state vector from orbit fit.
        #   SATVEC=ORBITVAL(TAZI,ORBFIT) computes the satellite state vector SATVEC
        #   at time TAZI from the orbit fit ORBFIT. ORBFIT must have been computed
        #   with the function ORBITFIT. TAZI is a scalar or vector with the time
        #   in seconds (in the day). SATVEC is a matrix with in it's columns
        #   the position X(m), Y(m), Z(m) and velocity X_V(m/s), Y_V(m/s), Z_V(m/s),
        #   with rows corresponding to time TAZI.
        #

        #   Created:    14 March 2014 by Hans van der Marel
        #   Modified:   5 July 2017 translated to python by Gert Mulder

        if len(self.orbit_fit) == 0:
            print('First an orbit fit should be created before you can evaluate it')
            return

        orbit_xyz = np.zeros(shape=(len(az_times), 6))

        deg = self.orbit_fit.shape[1]
        par = self.orbit_fit.shape[0]

        # Evaluate the polynomials for the given azimuth times.
        for p in range(par):
            for d in range(deg):
                if d == 0:
                    orbit_xyz[:, p] += self.orbit_fit[p, d]
                else:
                    orbit_xyz[:, p] += self.orbit_fit[p, d] * az_times**d

        return orbit_xyz

    def fit_orbit_spline(self, vel=True, acc=True):
        # ORBITFIT   Fit low degree polynomial to short segment of satellite orbit.
        #   ORBFIT=ORBITFIT(ORBIT,DEGREE) fits a cubic spline to
        #   the orbit state vector ORBIT. ORBIT is matrix with 4 columns,
        #   with respectively t(s), X(m), Y(m), Z(m),
        #   See also evaluate_orbit, lph2xyz, lp2t, t2lp, xyz2lp
        #
        #   (c) Gert Mulder, Delft University of Technology, 2017.

        #   Created:   5 July 2017 by Gert Mulder

        # Now find the polynomials for x,y,z and their derivatives.
        x_poly = CubicSpline(self.t, self.x)
        y_poly = CubicSpline(self.t, self.y)
        z_poly = CubicSpline(self.t, self.z)
        fit_len = 3

        if vel or acc:
            if len(self.v_x) > 0 and len(self.v_y) > 0 and len(self.v_z) > 0:
                x_der = CubicSpline(self.t, self.v_x)
                y_der = CubicSpline(self.t, self.v_y)
                z_der = CubicSpline(self.t, self.v_z)
            else:
                x_der = x_poly.derivative()
                y_der = y_poly.derivative()
                z_der = z_poly.derivative()
            fit_len = 6
            if acc:
                if len(self.a_x) > 0 and len(self.a_y) > 0 and len(self.a_z) > 0:
                    x_acc = CubicSpline(self.t, self.a_x)
                    y_acc = CubicSpline(self.t, self.a_y)
                    z_acc = CubicSpline(self.t, self.a_z)
                else:
                    x_acc = x_der.derivative()
                    y_acc = y_der.derivative()
                    z_acc = z_der.derivative()
                fit_len = 9

        self.orbit_spline = np.zeros(shape=(fit_len, 4, len(self.t) - 1))
        self.orbit_spline[0, :, :] = x_poly.c
        self.orbit_spline[1, :, :] = y_poly.c
        self.orbit_spline[2, :, :] = z_poly.c
        if vel or acc:
            c_size = x_der.c.shape[0]
            self.orbit_spline[3, 4-c_size:, :] = x_der.c
            self.orbit_spline[4, 4-c_size:, :] = y_der.c
            self.orbit_spline[5, 4-c_size:, :] = z_der.c
            if acc:
                c_size = x_acc.c.shape[0]
                self.orbit_spline[6, 4-c_size:, :] = x_acc.c
                self.orbit_spline[7, 4-c_size:, :] = y_acc.c
                self.orbit_spline[8, 4-c_size:, :] = z_acc.c
        self.orbit_spline = np.fliplr(self.orbit_spline)

    def evaluate_orbit_spline(self, az_times, pos=True, vel=False, acc=False, sorted=False):
        # Input argument checking and default values
        # ORBITVAL   Compute satellite state vector from orbit fit.
        #   SATVEC=ORBITVAL(TAZI,ORBFIT) computes the satellite state vector SATVEC
        #   at time TAZI from the orbit fit based on a cubic spline
        #
        #   Created:    5 July 2017 by Gert Mulder

        if len(self.orbit_spline) == 0 or len(self.t) == 0:
            print('First the orbit spline should be calculated')
            return

        deg = self.orbit_spline.shape[1]

        if isinstance(az_times, list):
            az_times = np.array(az_times)

        if not sorted and len(az_times) > 1:
            sort_ids = np.argsort(az_times)
            az_times = az_times[sort_ids]

        # We assume evenly spaced intervals here. This is normally the case for orbit vectors.
        tot_interval = self.t[-1] - self.t[0]
        step_size = tot_interval / (len(self.t) - 1)
        interval_id = np.int16(np.floor((az_times - self.t[0]) / step_size))

        interval_id[interval_id >= len(self.t)] = len(self.t) - 2
        interval_id[interval_id < 0] = 0

        eq_times = az_times - self.t[interval_id]
        n = len(az_times)

        # Because there will be many pixels within the same slots, we can seperate the array in parts based on the
        # interval id.
        insert_ids = np.searchsorted(interval_id, np.arange(interval_id[0], interval_id[-1]), side='right')
        starts = np.concatenate(([0], insert_ids))
        ends = np.concatenate((insert_ids, [len(interval_id)]))
        ids = np.arange(interval_id[0], interval_id[-1] + 1)

        # Initialize result dat
        if pos:
            position = np.zeros(shape=(3, n))
        else:
            position = []
        if vel:
            velocity = np.zeros(shape=(3, n))
        else:
            velocity = []
        if acc:
            acceleration = np.zeros(shape=(3, n))
        else:
            acceleration = []

        for id, start, end in zip(ids, starts, ends):
            times = eq_times[start:end]

            if pos:
                # Evaluate the polynomials for the given azimuth times.
                for p in range(3):
                    for d in range(deg):
                        if d == 0:
                            position[p, start:end] += self.orbit_spline[p, d, id]
                        else:
                            position[p, start:end] += self.orbit_spline[p, d, id] * times**d

            if vel:
                # Evaluate the polynomials for the given azimuth times.
                for p in range(3, 6):
                    for d in range(deg):
                        if d == 0:
                            velocity[p-3, start:end] += self.orbit_spline[p, d, id]
                        else:
                            velocity[p-3, start:end] += self.orbit_spline[p, d, id] * times ** d

            if acc:
                # Evaluate the polynomials for the given azimuth times.
                for p in range(6, 9):
                    for d in range(deg):
                        if d == 0:
                            acceleration[p-6, start:end] += self.orbit_spline[p, d, id]
                        else:
                            acceleration[p-6, start:end] += self.orbit_spline[p, d, id] * times ** d

        if not sorted and len(az_times) > 1:
            if pos:
                position = position[:, sort_ids]
            if vel:
                velocity = velocity[:, sort_ids]
            if acc:
                acceleration = acceleration[:, sort_ids]

        return position, velocity, acceleration
