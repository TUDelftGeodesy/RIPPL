# The following class creates an interferogram from a master and slave image.

from doris_processing.image_data import ImageData
from collections import OrderedDict, defaultdict
import numpy as np
import logging
import os


class Multilook(object):
    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, step, meta, ra=20, az=5, data_type='Data', meta_type='interferogram', s_lin=0, s_pix=0, offset='', shape=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, meta_type)
        elif isinstance(meta, ImageData):
            self.meta = meta

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        if len(shape) != 2:
            self.shape = self.meta.data_sizes[step][data_type]
        else:
            self.shape = shape

        if len(offset) != 2:
            offset = [0, 0]
        self.offset = offset
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.s_calc_lin = (s_lin + self.offset[0]) % az
        self.s_calc_pix = (s_pix + self.offset[1]) % ra
        self.e_lin = self.s_lin + (self.shape[0] - (((s_lin + self.offset[0]) % az) / az) * az)
        self.e_pix = self.s_pix + (self.shape[1] - (((s_pix + self.offset[1]) % ra) / ra) * ra)
        self.new_s_lin = (self.s_calc_lin + self.s_lin) / az
        self.new_s_pix = (self.s_calc_pix + self.s_pix) / ra
        self.ra = ra
        self.az = az
        self.step = step
        self.data_type = data_type

        # Load data
        self.data = self.meta.image_load_data_memory(step, self.s_pix, self.s_lin, self.shape, data_type)
        self.multilooked = []

    def __call__(self):
        if len(self.data) == 0:
            print('Missing input data for multilooking for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Calculate the multilooked image.

            ir = np.arange(0, (self.e_lin - self.s_calc_lin), self.az)
            jr = np.arange(0, (self.e_pix - self.s_calc_pix), self.ra)
            n = 1. / (self.ra * self.az)
            self.multilooked = n * np.add.reduceat(np.add.reduceat(self.data, ir), jr, axis=1)

            self.meta.image_new_data_memory(self.multilooked, self.step, self.new_s_lin, self.new_s_pix, self.data_type)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed multilooking for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed multilooking for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    def create_output_files(self, to_disk=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        self.meta.image_create_disk(self.step, self.data_type)

    def add_meta_data(self):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        meta_info = OrderedDict()

        meta_info[dat + 'output_file'] = dat + '.raw'
        meta_info[dat + 'output_format'] = t

        meta_info[dat + 'lines'] = str(self.orig_s_lin + self.orig_lines)
        meta_info[dat + 'pixels'] = str(self.orig_s_pix + self.orig_pixels)
        meta_info[dat + 'first_line'] = str(self.orig_s_lin + 1)
        meta_info[dat + 'first_pixel'] = str(self.orig_s_pix + 1)
        meta_info[dat + 'range_interval'] = str(self.interval[1])
        meta_info[dat + 'azimuth_interval'] = str(self.interval[0])
        meta_info[dat + 'range_offset'] = str(self.buffer[1])
        meta_info[dat + 'azimuth_offset'] = str(self.buffer[0])

        self.meta.image_add_processing_step(step, meta_info)

    @staticmethod
    def processing_info():
        # Information on this processing step
        input_dat = defaultdict()
        input_dat['master']['correct_earth_dem_phase'] = ['Data']
        input_dat['slave']['correct_earth_dem_phase'] = ['Data']

        output_dat = dict()
        output_dat['ifgs']['interferogram'] = ['Data']

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 3

        return input_dat, output_dat, mem_use
