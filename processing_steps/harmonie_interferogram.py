# The following class creates an interferogram from a master and slave image.

from rippl.image_data import ImageData
from rippl.coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import os
import logging
import numpy as np


class HarmonieInterferogram(object):

    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, master_meta, coordinates, ifg_meta, s_lin=0, s_pix=0, lines=0,
                 step='harmonie_aps', file_type='harmonie_aps'):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, ImageData) and isinstance(master_meta, ImageData):
            self.slave = meta
            self.master = master_meta
        else:
            return

        if isinstance(ifg_meta, ImageData):
            self.ifg = ifg_meta

        if isinstance(coordinates, CoordinateSystem):
            self.coordinates = coordinates

        shape = coordinates.shape
        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        self.shape = [l, shape[1] - s_pix]
        self.s_lin = s_lin
        self.s_pix = s_pix

        # Input data
        self.step = step
        if file_type == '':
            self.file_type = step
        else:
            self.file_type = file_type

        # Currently not possible to perform this step in slices because it includes multilooking. Maybe this will be
        # able later on. (Convert to different grids and slicing can cause problems at the sides of the slices.)
        self.master_dat = self.master.image_load_data_memory(self.step,  self.s_lin, self.s_pix, self.shape, self.file_type + self.coordinates.sample)
        self.slave_dat = self.slave.image_load_data_memory(self.step, self.s_lin, self.s_pix, self.shape, self.file_type + self.coordinates.sample, warn=False)

        self.interferogram = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.slave_dat) == 0 or len(self.master_dat) == 0:
            print('Missing input data for creating interferogram for ' + self.ifg.folder + '. Aborting..')
            return False

        # This function is a special case as it allows the creation of the interferogram together with multilooking
        # of this interferogram in one step. It is therefore a kind of nested function.
        # This is done like this because it prevents the creation of intermediate interferograms which can take a large
        # part of memory or disk space.
        # For the coherence this is not needed, therefore a different approach is used there.

        try:
            self.interferogram = self.master_dat - self.slave_dat

            # Save meta data and results
            self.add_meta_data(self.ifg, self.coordinates, self.step, self.file_type)
            self.ifg.image_new_data_memory(self.interferogram, 'harmonie_interferogram', 0, 0, file_type='harmonie_interferogram' + self.coordinates.sample)

            return True

        except Exception:
            log_file = os.path.join(self.ifg.folder, 'error.log')
            if not os.path.exists(self.ifg.folder):
                os.makedirs(self.ifg.folder)
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating interferogram for ' +
                              self.ifg.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating interferogram for ' +
                  self.ifg.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(ifg_meta, coordinates, step='', type=''):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'harmonie_interferogram' in ifg_meta.processes.keys():
            meta_info = ifg_meta.processes['harmonie_interferogram']
        else:
            meta_info = OrderedDict()

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')
            return

        meta_info = coordinates.create_meta_data(['harmonie_interferogram'], ['real4'], meta_info)
        meta_info['harmonie_interferogram' + coordinates.sample + '_input_step'] = step
        meta_info['harmonie_interferogram' + coordinates.sample + '_input_type'] = type
        ifg_meta.image_add_processing_step('harmonie_interferogram', meta_info)

    @staticmethod
    def processing_info(coordinates, step='harmonie_aps', file_type='harmonie_aps'):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        file_type = file_type

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat['slave'][step][file_type + coordinates.sample]['file'] = file_type + coordinates.sample + '.raw'
        input_dat['slave'][step][file_type + coordinates.sample]['coordinates'] = coordinates
        input_dat['slave'][step][file_type + coordinates.sample]['slice'] = coordinates.slice

        input_dat['master'][step][file_type + coordinates.sample]['file'] = file_type + coordinates.sample + '.raw'
        input_dat['master'][step][file_type + coordinates.sample]['coordinates'] = coordinates
        input_dat['master'][step][file_type + coordinates.sample]['slice'] = coordinates.slice

        # line and pixel output files.
        output_dat = recursive_dict()
        output_dat['ifg']['harmonie_interferogram']['harmonie_interferogram' + coordinates.sample]['file'] = 'harmonie_interferogram' + coordinates.sample + '.raw'
        output_dat['ifg']['harmonie_interferogram']['harmonie_interferogram' + coordinates.sample]['coordinates'] = coordinates
        output_dat['ifg']['harmonie_interferogram']['harmonie_interferogram' + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 2

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('harmonie_interferogram', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('harmonie_interferogram', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('harmonie_interferogram', file_type, coordinates)
