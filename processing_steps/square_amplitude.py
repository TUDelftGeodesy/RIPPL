# The following class creates an interferogram from a master and slave image.

from rippl.image_data import ImageData
from collections import OrderedDict, defaultdict
import numpy as np
from rippl.coordinate_system import CoordinateSystem
import logging
import os


class SquareAmplitude(object):

    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0, step='earth_topo_phase', file_type=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        if isinstance(coordinates, CoordinateSystem):
            self.coordinates = coordinates

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.shape = coordinates.shape
        if lines != 0:
            self.shape = [np.minimum(lines, self.shape[0] - s_lin), self.shape[1] - s_pix]

        self.s_lin = s_lin
        self.s_pix = s_pix

        # Load data
        if file_type == '':
            self.file_type = step
        else:
            self.file_type = file_type
        self.step = step
        self.slc_dat = self.meta.image_load_data_memory(self.step, self.s_lin, self.s_pix, self.shape, self.file_type + coordinates.sample, warn=False)

        # Initialize output
        self.squared = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.slc_dat) == 0:
            print('Missing input data to square SLC data for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Square image
            self.squared = np.abs(self.slc_dat)**2

            # If needed do the multilooking step
            self.add_meta_data(self.meta, self.coordinates, self.step, self.file_type)
            self.meta.image_new_data_memory(self.squared, 'square_amplitude', self.s_lin, self.s_pix, 'square_amplitude' + self.coordinates.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            if not os.path.exists(self.meta.folder):
                os.makedirs(self.meta.folder)
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception(
                'Failed processing squared amplitude for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing squared amplitude for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates, step='earth_topo_phase', file_type='earth_topo_phase'):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'square_amplitude' in meta.processes.keys():
            meta_info = meta.processes['square_amplitude']
        else:
            meta_info = OrderedDict()

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')
            return

        type_str = 'square_amplitude'
        meta_info[type_str + '_input_step'] = step
        meta_info[type_str + '_input_file_type'] = file_type
        meta_info = coordinates.create_meta_data([type_str], ['real4'], meta_info)

        meta.image_add_processing_step('square_amplitude', meta_info)

    @staticmethod
    def processing_info(coordinates, step='earth_topo_phase', file_type='earth_topo_phase', meta_type='slave'):

        in_coordinates = CoordinateSystem()
        in_coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Data input file from a random step / file type
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat[meta_type][step][file_type + in_coordinates.sample]['file'] = file_type + in_coordinates.sample + '.raw'
        input_dat[meta_type][step][file_type + in_coordinates.sample]['coordinates'] = in_coordinates
        input_dat[meta_type][step][file_type + in_coordinates.sample]['slice'] = True
        input_dat[meta_type][step][file_type + in_coordinates.sample]['coor_change'] = 'multilook'

        # line and pixel output files.
        output_dat = recursive_dict()
        output_dat[meta_type]['square_amplitude']['square_amplitude' + coordinates.sample]['file'] = 'square_amplitude' + coordinates.sample + '.raw'
        output_dat[meta_type]['square_amplitude']['square_amplitude' + coordinates.sample]['coordinates'] = coordinates
        output_dat[meta_type]['square_amplitude']['square_amplitude' + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram.
        mem_use = 2

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('square_amplitude', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('square_amplitude', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('square_amplitude', file_type, coordinates)

