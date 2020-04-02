"""

This class holds a number of methods to select the input window in case of a resampling script.

1. resampling regular > irregular, rectangle selection based on new resampling coordinates
2. resampling regular > irregular, bool grid (True/False) selection based on new resampling coordinates
3. resampling irregular > regular, rectangle selection based on the irregular coordinates of the input data
4. resampling irregular > regular, bool grid (True/False) selection based on the irregular coordinates of the input data

Not implemented:
5. regular multilooking, based on coordinate settings of both input and output
6. irregular multilooking, not available at the moment. Possibly this will be added to the conversion grid function
        some moment in the future. (Sorting arrays to large for memory is not really trivial...)

"""

import numpy as np
from skimage.morphology.binary import binary_dilation


class SelectInputWindow():

    @staticmethod
    def input_irregular_rectangle(input_lines, input_pixels, s_lin=0, s_pix=0, shape=[0, 0], buf=3):
        # Calculate the needed rectangle area for the input, given the needed output line and pixels.

        line_min_y = np.min(input_lines, 1)
        line_max_y = np.max(input_lines, 1)
        line_min_x = np.min(input_lines, 0)
        line_max_x = np.max(input_lines, 0)

        lines_inside_y = (line_max_y > (s_lin - buf)) * (line_min_y < (s_lin + shape[0] + buf))
        lines_inside_x = (line_max_x > (s_lin - buf)) * (line_min_x < (s_lin + shape[0] + buf))

        pixel_min_y = np.min(input_pixels, 1)
        pixel_max_y = np.max(input_pixels, 1)
        pixel_min_x = np.min(input_pixels, 0)
        pixel_max_x = np.max(input_pixels, 0)

        pixels_inside_y = (pixel_max_y > (s_pix - buf)) * (pixel_min_y < (s_pix + shape[1] + buf))
        pixels_inside_x = (pixel_max_x > (s_pix - buf)) * (pixel_min_x < (s_pix + shape[1] + buf))

        line_ids = np.argwhere(lines_inside_y * pixels_inside_y)
        pixel_ids = np.argwhere(lines_inside_x * pixels_inside_x)

        in_s_lin = line_ids[0][0]
        in_s_pix = pixel_ids[0][0]

        in_shape = (line_ids[-1][0] - in_s_lin, pixel_ids[-1][0] - in_s_pix)

        return in_s_lin, in_s_pix, in_shape

    @staticmethod
    def input_irregular_bool_grid(input_lines, input_pixels, s_lin=0, s_pix=0, shape=[0, 0], buf=3):
        # In addition to the selected rectangle shaped area we also check which pixels within this region are needed
        # given a certain buffer.

        # First get the needed rectangle.
        in_s_lin, in_s_pix, in_shape = SelectInputWindow.input_irregular_rectangle(input_lines, input_pixels, s_lin, s_pix, shape, buf)

        # Then check all the pixels within that box that are needed given the buffer.
        lines = input_lines[in_s_lin:in_s_lin + in_shape[0], in_s_pix:in_s_pix + in_shape[1]]
        pixels = input_pixels[in_s_lin:in_s_lin + in_shape[0], in_s_pix:in_s_pix + in_shape[1]]
        bool_grid = (lines >= s_lin - buf) * (lines <= s_lin + shape[0] + buf) * \
                    (pixels >= s_pix - buf) * (pixels <= s_pix + shape[1] + buf)
        del lines, pixels

        # To be sure that we do not miss needed adjacent pixel we let the data grow with an additional pixel.
        bool_grid = SelectInputWindow.grow_selection(bool_grid, buf=4)

        return bool_grid, in_s_lin, in_s_pix, in_shape

    @staticmethod
    def output_irregular_rectangle(output_lines, output_pixels, max_shape=[], buf=3):

        in_s_lin = int(np.floor(np.maximum(np.min(output_lines) - buf, 0)))
        in_s_pix = int(np.floor(np.maximum(np.min(output_pixels) - buf, 0)))
        if len(max_shape) == 0:
            in_e_lin = int(np.ceil(np.max(output_lines) + buf))
            in_e_pix = int(np.ceil(np.max(output_pixels) + buf))
        else:
            in_e_lin = int(np.ceil(np.minimum(np.max(output_lines) + buf, max_shape[0])))
            in_e_pix = int(np.ceil(np.minimum(np.max(output_pixels) + buf, max_shape[1])))

        in_shape = [in_e_lin - in_s_lin, in_e_pix - in_s_pix]

        return in_s_lin, in_s_pix, in_shape

    @staticmethod
    def output_irregular_bool_grid(output_lines, output_pixels, max_shape=[], buf=3):
        # A rectangular selection is sometimes much larger than the actual area needed because the area of interest is
        # rotated.

        in_s_lin, in_s_pix, in_shape = SelectInputWindow(output_lines, output_pixels, max_shape, buf)

        lines = np.arange(in_s_lin, in_shape[0])
        pixels = np.arange(in_s_pix, in_shape[1])

        # This method check for every input row what the max and min line and pixel values are.
        row_max_lin = np.int(np.max(output_lines, 1))
        row_min_lin = np.int(np.min(output_lines, 1))
        row_max_pix = np.int(np.max(output_pixels, 1))
        row_min_pix = np.int(np.min(output_pixels, 1))

        # the line values are now used to check which rows cross certain columns of the image
        in_between_matrix = (row_max_lin[None, :] - lines[:, None] > 0) * (row_min_lin[None, :] - lines[:, None] < 0)
        in_between_matrix[in_between_matrix == 0] = np.nan
        # Knowing the which columns to detect we can identify the maximum and minimum line value for every column.
        max_pixel_values = row_max_pix[:, None] * np.ones((len(lines), len(row_max_lin))) * in_between_matrix
        row_max = np.nanmax(max_pixel_values, 1)
        min_pixel_values = row_min_pix[:, None] * np.ones((len(lines), len(row_max_lin))) * in_between_matrix
        row_min = np.nanmin(min_pixel_values)

        del max_pixel_values, min_pixel_values, in_between_matrix

        # Finally construct the image coverage.
        row_len = len(pixels)
        bool_grid = np.r_([[start + row_len * row_id, end + row_len * row_id, 1]
                            for start, end, row_id in zip(row_min, row_max, range(len(lines)))])
        bool_grid = np.reshape(bool_grid, (len(lines), len(pixels)))

        return bool_grid, in_s_lin, in_s_pix, in_shape

    @staticmethod
    def grow_selection(input_grid, buf=3):

        filter = np.ones((buf * 2 + 1, buf * 2 + 1))
        output_grid = binary_dilation(input_grid, filter)

        return output_grid
