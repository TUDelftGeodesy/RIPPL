# This class can interpolate a massive number of splines.
import numpy as np
import logging


class MassiveSpline(object):
    """Cubic spline data interpolator.
    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable [1]_. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.
    Parameters
    ----------
    x : array_like, shape (n,)
        2-d array containing values of the independent variable [n_functions, n_points].
        Values must be real, finite and in strictly increasing order.
    y : array_like [
        2-d array containing values of the dependent variable [n_functions, n_points].
    type : string
        Describes which kind of spline you want to use


    This function is adapted from the scipy.interpolate.CubicSpline function:

    Copyright (c) 2001, 2002 Enthought, Inc.
    All rights reserved.

    Copyright (c) 2003-2017 SciPy Developers.
    All rights reserved.

    Adaptions made by Gert Mulder, TU Delft, 2019

    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.

    This processing_steps is created to fastly produce many cubic spline interpolations for short time-series.


    """

    def __init__(self, x, y, type='linear'):

        if not type in ['linear', 'quadratic', 'cubic']:
            logging.info('Only linear, quadratic and cubic interpolation are implemented')
            return

        if len(x.shape) == 1:
            x = x[:, None]
        if len(y.shape) == 1:
            y = y[:, None]

        if x.shape[1] < 4:
            raise ValueError("`x` must contain at least 4 elements.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("The length of `y` along `axis`={0} doesn't "
                             "match the length of `x`".format(0))

        if not np.all(np.isfinite(x)):
            raise ValueError("`x` must contain only finite values.")
        if not np.all(np.isfinite(y)):
            raise ValueError("`y` must contain only finite values.")
        if x.shape[1] > 20:
            raise ValueError(
                "Size of the generated matrix is over 20, this will generate huge matrixes. This function"
                "is designed to work with many spline solutions of short lengths. example 10 points, but"
                "1000 different solutions.")

        self.x = x
        self.y = y

        self.points = x.shape[1]
        self.n = x.shape[0]
        self.dx = np.diff(x, axis=1)

        # Initialize the coefficients
        self.c = np.zeros((4, self.n, self.points - 1))

        if np.any(self.dx <= 0):
            raise ValueError("`x` must be strictly increasing sequence.")

        self.slope = np.diff(y, axis=1) / self.dx

        if type == 'linear':
            self.linear()
        elif type == 'quadratic':
            self.quadratic()
        elif type == 'cubic':
            self.cubic()

    def linear(self):
        # Convert the slope and
        self.c[2, :, :] = self.slope
        self.c[3, :, :] = self.y[:, :-1]

    def quadratic(self):
        # Get a series of matrixes to solve the quadratic equations

        logging.info('To be done')

    def cubic(self, bc='', bv=''):
        # Find derivative values at each x[i] by solving a tridiagonal
        # system.
        A = np.zeros((self.n, self.points, self.points))  # This are the banded matrices.
        b = np.empty((self.n, self.points), dtype=self.y.dtype)

        # Filling the system for i=1..n-2
        #                         (x[i-1] - x[i]) * s[i-1] +\
        # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
        #                         (x[i] - x[i-1]) * s[i+1] =\
        #       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
        #           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))

        A[:, range(1, self.points - 1), range(1, self.points - 1)] = 2 * (self.dx[:, -1:] + self.dx[:, 1:])  # The diagonal
        A[:, range(1, self.points - 1), range(2, self.points)] = self.dx[:, :-1]  # The upper diagonal
        A[:, range(1, self.points - 1), range(0, self.points - 2)] = self.dx[:, 1:]  # The lower diagonal

        b[:, 1:-1] = 3 * (self.dx[:, 1:] * self.slope[:, :-1] + self.dx[:, :-1] * self.slope[:, 1:])

        if len(bc) == 0:
            bc = [None, None]
        if len(bv) == 0:
            bv = [0, 0]

        # Define the boundary conditions.
        if not bc[0]:
            A[:, 0, 0] = self.dx[:, 1]
            A[:, 0, 1] = self.x[:, 2] - self.x[:, 0]
            d = self.x[:, 2] - self.x[:, 0]
            b[:, 0] = ((self.dx[:, 0] + 2 * d) * self.dx[:, 1] * self.slope[:, 0] +
                       self.dx[:, 0] ** 2 * self.slope[:, 1]) / d
        elif bc[0] == '1st':
            A[:, 0, 0] = 1
            A[:, 0, 1] = 0
            b[:, 0] = bv[0]
        elif bc[0] == '2nd':
            A[:, 0, 0] = 2 * self.dx[:, 0]
            A[:, 0, 1] = self.dx[:, 0]
            b[:, 0] = -0.5 * bv[0] * self.dx[:, 0]**2 + 3 * (self.y[:, 1] - self.y[:, 0])

        if not bc[1]:
            A[:, -1, -1] = self.dx[:, -2]
            A[:, -1, -2] = self.x[:, -1] - self.x[:, -3]
            d = self.x[:, -1] - self.x[:, -3]
            b[:, -1] = ((self.dx[:, -1] ** 2 * self.slope[:, -2] +
                         (2 * d + self.dx[:, -1]) * self.dx[:, -2] * self.slope[:, -1]) / d)
        elif bc[1] == '1st':
            A[:, -1, -1] = 1
            A[:, -1, -2] = 0
            b[:, -1] = bv[1]
        elif bc[1] == '2nd':
            A[:, -1, -1] = 2 * self.dx[:, -1]
            A[:, -1, -2] = self.dx[:, -1]
            b[:, -1] = -0.5 * bv[1] * self.dx[:, -1] ** 2 + 3 * (self.y[:, -1] - self.y[:, -2])

        s = np.linalg.solve(A, b)

        # Compute coefficients in PPoly form.
        t = (s[:, :-1] + s[:, 1:] - 2 * self.slope) / self.dx

        self.c[0, :, :] = t / self.dx
        self.c[1, :, :] = (self.slope - s[:, :-1]) / self.dx - t
        self.c[2, :, :] = s[:, :-1]
        self.c[3, :, :] = self.y[:, :-1]


    def evaluate_splines(self, new_x):
        # This function evaluate the calculated splines.
        # The input of this function is one or more x values on which all processing_steps are evaluated. If the values are
        # outside this interval we extrapolate from the last spline.

        n = self.x.shape[0]
        output_size = (n, len(new_x))
        id = np.zeros(output_size).astype(np.int32)
        dx = np.zeros(output_size)
        val_out = np.zeros(output_size)

        for i, nx in zip(range(len(new_x)), new_x):
            # First define the interval for which we have to extract the coordinates.
            check = (self.x[:, :-1] <= nx).astype(np.int8) + (self.x[:, 1:] > nx).astype(np.int8)
            inside = np.sum(check == 2, axis=1).astype(np.bool_)

            id[inside, i] = np.argmax(check[inside, :], axis=1)
            id[(inside == False) * (nx >= self.x[:, -1]), i] = self.x.shape[1] - 2

            # Scale the values to the size of intervals
            dx[:, i] = (nx - self.x[range(n), id[:, i]])
            # / (self.x[range(n), id[:, i] + 1] - self.x[range(n), id[:, i]])

            # Evaluate using the cubic spline
            val_out[:, i] = self.c[3, range(n), id[:, i]] + self.c[2, range(n), id[:, i]] * dx[:, i] + \
                            self.c[1, range(n), id[:, i]] * dx[:, i] ** 2 + self.c[0, range(n), id[:, i]] * dx[:, i]**3

        return val_out
