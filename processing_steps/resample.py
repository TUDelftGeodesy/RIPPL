# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other

import numpy as np
from doris_processing.image_data import ImageData
from collections import defaultdict, OrderedDict
import logging
import os


class Resample(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, master_meta='', slave_meta='', s_lin=0, s_pix=0, lines=0, buf=5, warning=False, input_dat=''):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(slave_meta, str):
            if len(slave_meta) != 0:
                self.slave = ImageData(slave_meta, 'single')
        elif isinstance(slave_meta, ImageData):
            self.slave = slave_meta
        if isinstance(master_meta, str):
            if len(master_meta) != 0:
                self.master = ImageData(master_meta, 'single')
        elif isinstance(master_meta, ImageData):
            self.master = master_meta

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.shape = self.slave.data_sizes['combined_coreg']['New_line']
        if lines != 0:
            self.shape = [np.minimum(lines, self.shape[0] - s_lin), self.shape[1] - s_pix]

        self.s_lin = s_lin
        self.s_pix = s_pix
        self.e_lin = self.s_lin + self.shape[0]
        self.e_pix = self.s_pix + self.shape[1]

        self.new_line = self.slave.image_load_data_memory('combined_coreg', self.s_lin, self.s_pix, self.shape, 'New_line')
        self.new_pixel = self.slave.image_load_data_memory('combined_coreg', self.s_lin, self.s_pix, self.shape, 'New_pixel')

        if len(self.new_line) == 0 or len(self.new_pixel) == 0:
            print('Missing new lines or pixels for resample of ' + self.slave.folder + '. Aborting..')
            return

        # Look for the region from the image we have to load.
        orig_s_lin = self.slave.data_limits['crop']['Data'][0] - 1
        self.s_lin_source = int(np.floor(np.maximum(np.min(self.new_line) - buf - orig_s_lin, 0)))
        orig_s_pix = self.slave.data_limits['crop']['Data'][1] - 1
        self.s_pix_source = int(np.floor(np.maximum(np.min(self.new_pixel) - buf - orig_s_pix, 0)))
        orig_lines = self.slave.data_sizes['crop']['Data'][0]
        self.e_lin_source = int(np.ceil(np.minimum(np.max(self.new_line) + buf - orig_s_lin, orig_lines)))
        orig_pixels = self.slave.data_sizes['crop']['Data'][1]
        self.e_pix_source = int(np.ceil(np.minimum(np.max(self.new_pixel) + buf - orig_s_pix, orig_pixels)))

        self.shape_source = [self.e_lin_source - self.s_lin_source, self.e_pix_source - self.s_pix_source]

        # Find the coordinate of the first based on which the new_line and new_pixel are calculated.
        self.first_line = self.s_lin_source + orig_s_lin
        self.first_pixel = self.s_pix_source + orig_s_pix

        # Select the required area. Possibly it already covers the required area, then we do not need to load new data.
        # However, this will only be the case if we have loaded a larger tile already.
        if input_dat in ['deramp', 'crop']:
            input_step = input_dat
        elif self.slave.process_control['deramp'] == '1':
            if warning:
                print('We use the deramped image data for resampling.')
            input_step = 'deramp'
        else:
            if warning:
                print('No deramping information found. Use the original SLC data as input')
            input_step = 'crop'

        # Load needed data from crop
        self.crop = self.slave.image_load_data_memory(input_step, self.s_lin_source, self.s_pix_source, self.shape_source, 'Data')
        self.resampled = []

    def __call__(self, w_type='4p_cubic', table_size='', window=''):

        if len(self.new_line) == 0 or len(self.new_pixel) == 0 or len(self.crop) == 0:
            print('Missing data for resample of ' + self.slave.folder + '. Aborting..')
            return False

        # Here the actual resampling is done. If the output is a grid, the variables new_line and new_pixel
        # should be the same size as the output grid.

        try:

            # If you use the same resampling window many times, you can also pass it with the function.
            if len(table_size) != 2:
                table_size = [1000, 100]
            if len(window) == 0:
                window, w_steps = self.create_interp_table(w_type, table_size=table_size)
            window_size = [window.shape[2], window.shape[3]]
            w_steps = [1.0 / (window.shape[0] - 1), 1.0 / (window.shape[1] - 1)]

            # Now use the location of the new pixels to extract the weights and the values from the input grid.
            # After adding up and multiplying the weights and values we have the out going grid.
            line_id = np.floor(self.new_line).astype('int32')
            pixel_id = np.floor(self.new_pixel).astype('int32')
            l_window_id = table_size[0] - np.round((self.new_line - line_id) / w_steps[0]).astype(np.int32)
            p_window_id = table_size[1] - np.round((self.new_pixel - pixel_id) / w_steps[1]).astype(np.int32)
            line_id -= self.first_line
            pixel_id -= self.first_pixel

            # Check wether the pixels are far enough from the border of the image. Otherwise interpolation kernel cannot
            # Be applied. Missing pixels will be filled with zeros.
            half_w_line = window_size[0] / 2
            half_w_pixel = window_size[1] / 2
            valid_vals = (((line_id - half_w_line + 1) >= 0) * ((line_id + half_w_line) < self.crop.shape[0]) *
                          ((pixel_id - half_w_pixel + 1) >= 0) * ((pixel_id + half_w_pixel) < self.crop.shape[1]))

            # Pre assign the final values of this step.
            self.resampled = np.zeros(l_window_id.shape).astype(self.crop.dtype)

            # Calculate individually for different pixels in the image window. Saves a lot of memory space...
            for i in np.arange(window_size[0]):
                for j in np.arange(window_size[1]):
                    # First we assign the weighting windows to the pixels.
                    weights = window[l_window_id, p_window_id, i, j]

                    # Find the original grid values and add to the out values, applying the weights.
                    i_im = i - half_w_line + 1
                    j_im = j - half_w_pixel + 1
                    self.resampled[valid_vals] += self.crop[line_id[valid_vals] + i_im,
                                                            pixel_id[valid_vals] + j_im] * weights[valid_vals]

            self.add_meta_data(self.master, self.slave)
            self.slave.image_new_data_memory(self.resampled, 'resample', self.s_lin, self.s_pix, 'Data')

            return True

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed resampling for ' + self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed resampling for ' + self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_output_files(meta, to_disk=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = ['Data']

        for s in to_disk:
            meta.image_create_disk('resample', s)

    @staticmethod
    def add_meta_data(master, slave):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        meta_info = OrderedDict()

        meta_info['Data_output_file'] = 'resampled.raw'
        meta_info['Data_output_format'] = 'complex_int'

        meta_info['Data_lines'] = master.processes['crop']['Data_lines']
        meta_info['Data_pixels'] = master.processes['crop']['Data_pixels']
        meta_info['Data_first_line'] = master.processes['crop']['Data_first_line']
        meta_info['Data_first_pixel'] = master.processes['crop']['Data_first_pixel']

        slave.image_add_processing_step('resample', meta_info)

    @staticmethod
    def processing_info():
        # Information on this processing step
        input_dat = defaultdict()
        input_dat['slave']['crop'] = ['Data']
        input_dat['slave']['combined_coreg'] = ['New_line', 'New_pixel']

        output_dat = dict()
        output_dat['slave']['resample'] = ['Data']

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_interp_table(w_type, table_size=None):
        # This function creates a lookup table for radar image interpolation.
        # Possible types are nearest neighbour, linear, cubic convolution kernel, knab window, raised cosine kernel or
        # a truncated sinc.
    
        if not table_size:
            table_size = [1000, 100]
    
        az_coor = np.arange(table_size[0] + 1).astype('float32') / table_size[0]
        ra_coor = np.arange(table_size[1] + 1).astype('float32') / table_size[1]
    
        if w_type == 'nearest_neighbour':
            d_az = np.vstack((1 - np.round(az_coor), np.round(az_coor)))
            d_ra = np.vstack((1 - np.round(ra_coor), np.round(ra_coor)))
        elif w_type == 'linear':
            d_az = np.vstack((az_coor, 1 - az_coor))
            d_ra = np.vstack((ra_coor, 1 - ra_coor))
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
        else:
            print('Use nearest_neighbour, linear, 4p_cubic or 6p_cubic as kernel.')
            return
    
        # TODO Add the 6, 8 and 16 point truncated sinc + raised cosine interpolation
    
        # Calculate the 2d window
        window = np.einsum('ij,kl->jlik', d_az, d_ra)
    
        return window, [1.0 / table_size[0], 1.0 / table_size[1]]
