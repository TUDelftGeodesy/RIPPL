# Function to make an estimate the topographic and/or atmospheric error.
# This will be based on baselines, incidence angles, the derived DEM and statistical analysis of the ifg.

import numpy as np
from collections import OrderedDict, defaultdict
from rippl.image_data import ImageData
import os
import logging
from rippl.coordinate_system import CoordinateSystem


class EarthTopoPhase(object):
    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0, input_step=''):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.shape = coordinates.shape
        if lines != 0:
            self.shape = [np.minimum(lines, self.shape[0] - s_lin), self.shape[1] - s_pix]

        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates

        if input_step not in ['reramp', 'resample']:
            if self.meta.process_control['reramp'] == '1':
                # print('We use the reramp image data for correction.')
                input_step = 'reramp'
            else:
                # print('No reramping information found. Use the original resample data as input')
                input_step = 'resample'

        # Load data
        self.new_pixel = self.meta.image_load_data_memory('geometrical_coreg', self.s_lin, self.s_pix, self.shape, 'new_pixel' + coordinates.sample)
        self.resample = self.meta.image_load_data_memory(input_step, self.s_lin, self.s_pix, self.shape, input_step + coordinates.sample)

        self.resample_corrected = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.resample) == 0 or len(self.new_pixel) == 0:
            print('Missing input data for processing earth_topo_phase for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Then calculate the ramp
            az_time = 1 / float(self.meta.processes['readfiles']['Range_sampling_rate (computed, MHz)']) / 1000000
            c = 299792458
            wave_length = float(self.meta.processes['readfiles']['Radar_wavelength (m)'])
            conversion_factor = c * az_time / wave_length
            phase_shift = np.remainder(self.new_pixel * conversion_factor, 1) * np.pi * 2

            # Finally correct the data
            self.resample_corrected = (self.resample * np.exp(1j * phase_shift)).astype('complex64')
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.resample_corrected, 'earth_topo_phase', self.s_lin, self.s_pix)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing earth_topo_phase for ' +
                              self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing earth_topo_phase for ' +
                  self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'earth_topo_phase' in meta.processes.keys():
            meta_info = meta.processes['earth_topo_phase']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['earth_topo_phase'], ['complex_int'], meta_info)
        meta.image_add_processing_step('earth_topo_phase', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='slave', reramp=True):

        recursive_dict = lambda: defaultdict(recursive_dict)

        in_coordinates = CoordinateSystem()
        in_coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])

        input_dat = recursive_dict()
        input_dat[meta_type]['geometrical_coreg']['new_pixel' + in_coordinates.sample]['file'] = 'new_pixel' + in_coordinates.sample + '.raw'
        input_dat[meta_type]['geometrical_coreg']['new_pixel' + in_coordinates.sample]['coordinates'] = in_coordinates
        input_dat[meta_type]['geometrical_coreg']['new_pixel' + in_coordinates.sample]['slice'] = True

        # Input file should always be a full resolution grid.
        if reramp:
            input_dat[meta_type]['reramp']['reramp' + in_coordinates.sample]['file'] = 'reramp' + in_coordinates.sample + '.raw'
            input_dat[meta_type]['reramp']['reramp' + in_coordinates.sample]['coordinates'] = in_coordinates
            input_dat[meta_type]['reramp']['reramp' + in_coordinates.sample]['slice'] = True
        else:
            input_dat[meta_type]['resample']['resample' + in_coordinates.sample]['file'] = 'resample' + in_coordinates.sample + '.raw'
            input_dat[meta_type]['resample']['resample' + in_coordinates.sample]['coordinates'] = in_coordinates
            input_dat[meta_type]['resample']['resample' + in_coordinates.sample]['slice'] = True

        output_dat = recursive_dict()
        output_dat[meta_type]['earth_topo_phase']['earth_topo_phase' + coordinates.sample]['file'] = 'earth_topo_phase' + coordinates.sample + '.raw'
        output_dat[meta_type]['earth_topo_phase']['earth_topo_phase' + coordinates.sample]['coordinates'] = coordinates
        output_dat[meta_type]['earth_topo_phase']['earth_topo_phase' + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('earth_topo_phase', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('earth_topo_phase', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('earth_topo_phase', file_type, coordinates)
