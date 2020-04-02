# This function does the resampling of a radar grid based on different kernels.
import numpy as np
from scipy.interpolate import RectBivariateSpline


class Regural2irregular(object):

    def __init__(self, w_type='4p_cubic', table_size=None):
        # Both the resampling and interp table scripts are staticmethod which makes them usable outside this class.
        # However, if you want to do the processing using the same interpolation kernel you can use this class to keep
        # those stable.
        self.window = self.create_interp_table(w_type, table_size)
        self.table_size = [self.window.shape[0] - 1, self.window.shape[1] - 1]
        self.w_type = w_type

    def __call__(self, orig_grid, new_grid_lines, new_grid_pixels, step_size=100000000):
        # Do the actual resampling.
        return self.resample_grid(orig_grid, new_grid_lines, new_grid_pixels, self.w_type, step_size, self.table_size, self.window)

    @staticmethod
    def resample_grid(orig_grid, new_grid_lines, new_grid_pixels, w_type='', step_size=100000000, table_size='', window=''):
        # Here the actual resampling is done. If the output is a grid, the variables new_grid_lines and new_grid_pixels
        # should be the same size as the output grid.

        # If the table size was not defined, it defaults to [1000, 100]
        if len(table_size) == 0:
            table_size = [1000, 100]
        # If you use the same resampling window many times, you can also pass it with the function
        if len(window) == 0:
            window = Regural2irregular.create_interp_table(w_type, table_size=table_size)
        w_steps = [1.0 / table_size[0], 1.0 / table_size[1]]
        window_size = [window.shape[2], window.shape[3]]

        out_dim = len(new_grid_lines.shape)

        # Because the resampling can take up a lot of memory, we split the operation in a number of steps defined by
        # the size of the dataset. The size of the steps is defined by the number of pixels defined in step_size
        if out_dim == 2:
            line_steps = np.ceil(float(step_size) / new_grid_lines.shape[1])
            steps = np.ceil(new_grid_lines.shape[0] / float(line_steps))
        else:       # Assume that we have a 1D array
            steps = np.ceil(new_grid_lines.size / float(step_size))
            line_steps = step_size

        # Create the output grid.
        out_grid = np.zeros(new_grid_lines.shape).astype(orig_grid.dtype)

        # Now use the location of the new pixels to extract the weights and the values from the input grid.
        # After adding up and multiplying the weights and values we have the out going grid.
        for step in range(int(steps)):
            line_0 = int(line_steps * step)
            line_1 = np.minimum(line_steps * (step + 1), new_grid_lines.shape[0]).astype(np.int32)

            if out_dim == 2:
                grid_lines = np.ravel(new_grid_lines[line_0:line_1, :])
                grid_pixels = np.ravel(new_grid_pixels[line_0:line_1, :])
            else:
                grid_lines = new_grid_lines[line_0:line_1]
                grid_pixels = new_grid_pixels[line_0:line_1]

            line_id = np.floor(grid_lines).astype('int32')
            pixel_id = np.floor(grid_pixels).astype('int32')
            l_window_id = table_size[0] - np.round((grid_lines - line_id) // w_steps[0]).astype(np.int32)
            p_window_id = table_size[1] - np.round((grid_pixels - pixel_id) // w_steps[1]).astype(np.int32)

            # Check wether the pixels are far enough from the border of the image. Otherwise interpolation kernel cannot
            # Be applied. Missing pixels will be filled with zeros.
            half_w_line = window_size[0] // 2
            half_w_pixel = window_size[1] // 2
            valid_vals = (((line_id - half_w_line + 1) >= 0) * ((line_id + half_w_line) < orig_grid.shape[0]) *
                          ((pixel_id - half_w_pixel + 1) >= 0) * ((pixel_id + half_w_pixel) < orig_grid.shape[1]))

            # Pre assign the final values of this step.
            out_vals = np.zeros(len(l_window_id)).astype(orig_grid.dtype)

            # Calculate individually for different pixels in the image window. Saves a lot of memory space...
            for i in np.arange(window_size[0]):
                for j in np.arange(window_size[1]):
                    # First we assign the weighting windows to the pixels.
                    weights = window[l_window_id, p_window_id, i, j]

                    # Find the original grid values and add to the out values, applying the weights.
                    i_im = i - half_w_line + 1
                    j_im = j - half_w_pixel + 1
                    out_vals[valid_vals] += orig_grid[line_id[valid_vals] + i_im, pixel_id[valid_vals] + j_im] * weights[valid_vals]

            # Finally assign the grid to the outgoing grid
            if out_dim == 2:
                out_grid[line_0:line_1, :] = out_vals.reshape((line_1 - line_0, new_grid_lines.shape[1]), order='C')
            else:
                out_grid[line_0:line_1] = out_vals

        # Return resampling result
        return out_grid

    @staticmethod
    def create_interp_table(w_type, custom_kernel=None, kernel_size=None, table_size=None):
        # This function creates a lookup table for radar image interpolation.
        # Possible types are nearest neighbour, linear, cubic convolution kernel, knab window, raised cosine kernel or
        # a truncated sinc.
        # If we use custom the input kernel should be a RectBivariateSpline (scipy)

        if not table_size:
            table_size = [1000, 100]

        if w_type == 'custom':
            if kernel_size == None:
                kernel_size = [5, 5]
            if not isinstance(custom_kernel, RectBivariateSpline):
                raise TypeError('The custom kernel should be a scipy.interpolate.RectBivariateSpline!')

        az_coor = np.arange(table_size[0] + 1).astype('float32') / table_size[0]
        ra_coor = np.arange(table_size[1] + 1).astype('float32') / table_size[1]

        if w_type == 'nearest_neighbour':
            d_az = np.vstack((1 - np.round(az_coor), np.round(az_coor)))
            d_ra = np.vstack((1 - np.round(ra_coor), np.round(ra_coor)))
        elif w_type == 'linear':
            d_az = np.vstack((az_coor, 1 - az_coor))
            d_ra = np.vstack((az_coor, 1 - az_coor))
        elif w_type == '4p_cubic':
            a = -1.0
            az_coor_2 = az_coor + 1
            ra_coor_2 = ra_coor + 1

            d_az_r = np.vstack(((a+2) * az_coor**3 - (a+3) * az_coor**2 + 1,
                                a * az_coor_2**3 - (5*a) * az_coor_2**2 + 8*a * az_coor_2 - 4*a))
            d_az = np.vstack((np.fliplr(np.flipud(d_az_r)), d_az_r))
            d_ra_r = np.vstack(((a + 2) * ra_coor ** 3 - (a + 3) * ra_coor ** 2 + 1,
                                a * ra_coor_2 ** 3 - (5 * a) * ra_coor_2 ** 2 + 8 * a * ra_coor_2 - 4 * a))
            d_ra = np.vstack((np.fliplr(np.flipud(d_ra_r)), d_ra_r))

        elif w_type == '6p_cubic':
            a = -0.5
            b = 0.5
            az_coor_2 = az_coor + 1
            ra_coor_2 = ra_coor + 1
            az_coor_3 = az_coor + 2
            ra_coor_3 = ra_coor + 2

            d_az_r = np.vstack(((a-b+2) * az_coor**3 - (a-b+3) * az_coor**2 + 1,
                                a * az_coor_2**3 - (5*a-b) * az_coor_2**2 + (8*a - 3*b) * az_coor_2 - (4*a - 2*b),
                                b * az_coor_3**3 - 8*b * az_coor_3**2 + 21*b * az_coor_3 - 18*b))
            d_az = np.vstack((np.fliplr(np.flipud(d_az_r)), d_az_r))
            d_ra_r = np.vstack(((a - b + 2) * ra_coor ** 3 - (a - b + 3) * ra_coor ** 2 + 1,
                                a * ra_coor_2 ** 3 - (5 * a - b) * ra_coor_2 ** 2 + (8 * a - 3 * b) * ra_coor_2 - (
                                4 * a - 2 * b),
                                b * ra_coor_3 ** 3 - 8 * b * ra_coor_3 ** 2 + 21 * b * ra_coor_3 - 18 * b))
            d_ra = np.vstack((np.fliplr(np.flipud(d_ra_r)), d_ra_r))
        elif w_type == 'custom':
            az_values = np.arange(np.ceil(-kernel_size[0] / 2), np.ceil(kernel_size[0] / 2))[:, None] * np.ones((1, kernel_size[1]))
            ra_values = np.arange(np.ceil(-kernel_size[1] / 2), np.ceil(kernel_size[1] / 2))[None, :] * np.ones((kernel_size[0], 1))

            az_wind_vals = np.ravel(az_values[:, :, None, None] + az_coor[None, None, :, None] + np.ones(ra_coor.shape)[None, None, None, :])
            ra_wind_vals = np.ravel(ra_values[:, :, None, None] + np.ones(az_coor.shape)[None, None, :, None] + ra_coor[None, None, None, :])

            window = custom_kernel(az_wind_vals, ra_wind_vals).reshape([kernel_size[0], kernel_size[1], len(az_coor), len(ra_coor)])
        else:
            print('Use nearest_neighbour, linear, 4p_cubic or 6p_cubic as kernel.')
            return

        # TODO Add the 6, 8 and 16 point truncated sinc + raised cosine interpolation

        # Calculate the 2d window
        if w_type != 'custom':
            window = np.einsum('ij,kl->jlik', d_az, d_ra)

        return window
