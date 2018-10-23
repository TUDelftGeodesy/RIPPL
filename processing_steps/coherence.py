# The following class creates an interferogram from a master and slave image.

from image_data import ImageData
from collections import OrderedDict, defaultdict
import numpy as np
from processing_steps.interfero import Interfero
from coordinate_system import CoordinateSystem
import logging
import os


class Coherence(object):

    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, master_meta, ifg_meta, coordinates, s_lin=0, s_pix=0, lines=0):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, ImageData) and isinstance(master_meta, ImageData):
            self.slave = meta
            self.master = master_meta
        else:
            return

        if isinstance(ifg_meta, ImageData):
            self.ifg = ifg_meta
        else:
            Interfero.create_meta_data(master_meta, meta)

        if isinstance(coordinates, CoordinateSystem):
            self.coordinates = [coordinates]
        else:
            self.coordinates = coordinates

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.shape = self.coordinates.shape
        if lines != 0:
            self.shape = [np.minimum(lines, self.shape[0] - s_lin), self.shape[1] - s_pix]

        # Load input data (squared amplitud of slave/master) and the interferogram.
        self.master_squared = self.master.image_load_data_memory('square_amplitude', self.s_lin, self.s_pix, self.shape,
                                                                 'square_amplitude' + coordinates.sample, warn=False)
        self.slave_squared = self.slave.image_load_data_memory('square_amplitude', self.s_lin, self.s_pix, self.shape,
                                                               'square_amplitude' + coordinates.sample, warn=False)
        self.ifg_dat = self.slave.image_load_data_memory('interferogram', self.s_lin, self.s_pix, self.shape,
                                                         'interferogram' + coordinates.sample, warn=False)

        self.coherence = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.master_squared) == 0 or len(self.slave_squared) == 0 or len(self.ifg_dat) == 0:
            print('Missing input data for processing coherence for ' + self.ifg.folder + '. Aborting..')
            return False

        try:
            # Calculate new coherence from squared amplitudes and interferogram
            self.coherence = self.ifg_dat / np.sqrt(self.slave_squared * self.master_squared)

            # If needed do the multilooking step
            self.add_meta_data(self.ifg, self.coordinates)
            self.ifg.image_new_data_memory(self.coherence, 'coherence', self.s_lin, self.s_pix, 'coherence' + self.coordinates.sample)

            return True

        except Exception:
            log_file = os.path.join(self.ifg.folder, 'error.log')
            if not os.path.exists(self.ifg.folder):
                os.makedirs(self.ifg.folder)
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception(
                'Failed processing coherence for ' + self.ifg.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing coherence for ' + self.ifg.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(ifg_meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'coherence' in ifg_meta.processes.keys():
            meta_info = ifg_meta.processes['coherence']
        else:
            meta_info = OrderedDict()

        for coor in coordinates:
            if not isinstance(coor, CoordinateSystem):
                print('coordinates should be an CoordinateSystem object')
                return

            meta_info = coor.create_meta_data(['coherence' + coor.sample], ['complex_float'])

        ifg_meta.image_add_processing_step('coherence', meta_info)

    @staticmethod
    def processing_info(coordinates):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()
        coor = CoordinateSystem()
        for res_type in ['slave', 'master']:
            input_dat[res_type]['square_amplitude']['square_amplitude']['file'] = ['square_amplitude' + coordinates.sample + '.raw']
            input_dat[res_type]['square_amplitude']['square_amplitude']['coordinates'] = coor
            input_dat[res_type]['square_amplitude']['square_amplitude']['slice'] = coor.slice

        input_dat['ifg']['interferogram']['interferogram']['file'] = ['interferogram' + coordinates.sample + '.raw']
        input_dat['ifg']['interferogram']['interferogram']['coordinates'] = coor
        input_dat['ifg']['interferogram']['interferogram']['slice'] = coor.slice

        # line and pixel output files.
        output_dat = defaultdict()
        for coor in coordinates:
            output_dat['ifg']['coherence']['coherence']['file'] = ['coherence' + coor.sample + '.raw']
            output_dat['ifg']['coherence']['coherence']['coordinates'] = coor
            output_dat['ifg']['coherence']['coherence']['slice'] = coor.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('coherence', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_create_disk('coherence', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('coherence', file_type, coordinates)

