# Function to make an estimate the topographic and/or atmospheric error.
# This will be based on baselines, incidence angles, the derived DEM and statistical analysis of the ifg.

import numpy as np
from collections import OrderedDict, defaultdict
from image_data import ImageData
import os
import logging


class EarthTopoPhase(object):
    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, master_meta, slave_meta, s_lin=0, s_pix=0, lines=0, input_step=''):
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
        self.shape = self.slave.data_sizes['resample']['Data']
        if lines != 0:
            self.shape = [np.minimum(lines, self.shape[0] - s_lin), self.shape[1] - s_pix]

        self.s_lin = s_lin
        self.s_pix = s_pix
        self.e_lin = self.s_lin + self.shape[0]
        self.e_pix = self.s_pix + self.shape[1]

        if input_step not in ['reramp', 'resample']:
            if self.slave.process_control['reramp'] == '1':
                print('We use the reramped image data for correction.')
                input_step = 'reramp'
            else:
                print('No reramping information found. Use the original resampled data as input')
                input_step = 'resample'

        # Load data
        self.new_pixel = self.slave.image_load_data_memory('combined_coreg', self.s_lin, self.s_pix, self.shape, 'New_pixel')
        self.resampled = self.slave.image_load_data_memory(input_step, self.s_lin, self.s_pix, self.shape, 'Data')

        self.first_pixel = self.slave.data_limits['resample']['Data'][1] + self.s_pix
        self.last_pixel = self.slave.data_limits['resample']['Data'][1] + self.e_pix

        self.resampled_corrected = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.resampled) == 0 or len(self.new_pixel) == 0:
            print('Missing input data for processing earth_topo_phase for ' + self.slave.folder + '. Aborting..')
            return False

        try:
            # Get the range shift
            ra_shift = self.new_pixel - np.arange(self.first_pixel, self.last_pixel)[None, :]

            # Then calculate the ramp
            az_time = 1 / float(self.slave.processes['readfiles']['Range_sampling_rate (computed, MHz)']) / 1000000
            c = 299792458
            wave_length = float(self.slave.processes['readfiles']['Radar_wavelength (m)'])
            conversion_factor = c * az_time / wave_length
            phase_shift = np.remainder(ra_shift * conversion_factor, 1) * np.pi * 2
            del ra_shift

            # Finally correct the data
            self.resampled_corrected = (self.resampled * np.exp(1j * phase_shift)).astype('complex64')
            self.add_meta_data(self.master, self.slave)
            self.slave.image_new_data_memory(self.resampled_corrected, 'earth_topo_phase', self.s_lin, self.s_pix, 'Data')

            return True

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing earth_topo_phase for ' +
                              self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing earth_topo_phase for ' +
                  self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_output_files(meta, to_disk=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = ['Data']

        for s in to_disk:
            meta.image_create_disk('earth_topo_phase', s)

    def save_to_disk(self, to_disk=''):

        if not to_disk:
            to_disk = ['Data']

        for s in to_disk:
            self.slave.image_memory_to_disk('earth_topo_phase', s)

    @staticmethod
    def add_meta_data(master, slave):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        meta_info = OrderedDict()

        dat = 'Data_'
        meta_info[dat + 'output_file'] = 'resampled_corrected.raw'
        meta_info[dat + 'output_format'] = 'complex_int'
        meta_info[dat + 'lines'] = master.processes['crop']['Data_lines']
        meta_info[dat + 'pixels'] = master.processes['crop']['Data_pixels']
        meta_info[dat + 'first_line'] = master.processes['crop']['Data_first_line']
        meta_info[dat + 'first_pixel'] = master.processes['crop']['Data_first_pixel']
        slave.image_add_processing_step('earth_topo_phase', meta_info)

    @staticmethod
    def processing_info():
        # Information on this processing step
        input_dat = defaultdict()
        input_dat['slave']['resample'] = ['Data']
        input_dat['slave']['combined_coreg'] = ['New_pixel']

        output_dat = dict()
        output_dat['slave']['earth_topo_phase'] = ['Data']

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use
