"""
This code is used to create one test case for this step.

Important to note:
- This will be applied on burst level only. Therefore it can give discontinuities on the burst borders
- In first instance this is only a test script, do not try to include it in your processing chain yet!



"""

import numpy as np



# First load a stack of images for the test


# Select two images with a glacier in it


# Now divide the whole images in blocks of certain size


# If needed oversample these blocks before applying the correlation windows.


# Apply complex and amplitude correlation for these blocks to get a first estimate of the blocks.


# Calculate a pixel based shift for the first round


# Iterate this process over smaller blocks.

class CoregWindows():

    @staticmethod
    def select_window_locations(input_dat, shape, window_no=200, priority_mask=None):
        """
        Function to select the locations of the windows in an image. Inputs are the shapes of the windows, the number
        of windows and a priority mask. In first instance the mask will just select which areas are used and which are
        not. Maybe this will changed in a density mask later on.

        :param input_dat:
        :param shape:
        :param window_no:
        :param priority_mask:
        :return:
        """



    @staticmethod
    def select_window(input_dat, loc, shape, init_shift=[0, 0], oversample=1, resample_type='4p_cubic'):
        """
        Select the window from the full image. Defined location is the upper left corner. If oversample is used the
        data is oversampled including the next pixel to the lower right (So every window becomes exactly
        shape*oversample size)

        :param np.ndarray input_dat: Input data to extract the window from. Numpy or numpy memmap array
        :param tuple loc: Location of the window in the image
        :param tuple shape: Size of the window
        :param tuple init_shift: Initial shift if this is refinement step
        :param int oversample: Number of times to oversample the image. This can give a more accurate doing the shift.
        :return:
        """

    @staticmethod
    def remove_phase_ramp(coordinates, input_window_data, lines, pixels):
        """
        Remove the phase ramp from the selected windows. This is needed for resampling and image matching.

        :param CoordinateSystem coordinates: Coordinate system of full image
        :param np.ndarray lines: Lines of window
        :param np.ndarray pixels: Pixels of window
        :return: output_window data
        """

        # Use the available functions to remove phase ramp.

        return output_window_data

    @staticmethod
    def find_shift_fft(input_1, input_2, amplitude_only=False):
        """
        Search for the shift with the best correlation using the two images.

        :param np.ndarray input_1: Input from first image
        :param np.ndarray input_2: Input from second image
        :param bool amplitude_only: Is the image matching done using amplitude only or the full complex signal.
                                    Default is the use of complex data.
        :return:
        """




        return shift, max_correlation


    @staticmethod
    def zero_padding(input_dat, over_size=1):
        """
        Create zero padding for fft calculation of radar grid.

        :param np.ndarray input_dat: Input data that needs to be zero padded
        :param int over_size: How many times do we over size the zero padded area. Common is two times to get higher
                                resolution, but not needed in general.
        :return: output_dat
        """

        shape = input_dat.shape
        new_shape = np.array([2**(np.ceil(np.log2(shape[0]))), 2**(np.ceil(np.log2(shape[1])))]) * 2**np.int(over_size)

        output_dat = np.zeros(new_shape)
        output_dat[:shape[0], :shape[1]] = input_dat

        return output_dat
