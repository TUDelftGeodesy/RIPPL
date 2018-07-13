# The following class does the coregistration and resampling of a slave image to the geometry of a master image
# The coregistration includes the following main steps:
#   - Calculate the geometry of the master image and if needed the dem of the master based on the master
#   - Calculate the line and pixel coordinates of the master image in the slave line and pixel coordinates
#   - (optional) Create a mask for the areas used to check shift using cross-correlation
#   - (optional) Calculate the window shifts for large or small windows to verify the geometrical shift
#   - (optional) Adapt calculated geometrical shift using windows.
#   - Resample the slave grid to master coordinates
#   - (optional) Calculate the baselines between the two images

from doris_processing.image_data import ImageData
from doris_processing.orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from collections import OrderedDict, defaultdict
import copy
import numpy as np
import os
import logging


class GeometricalCoreg(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, master_meta, slave_meta, s_lin=0, s_pix=0, lines=0):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

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
        self.shape = self.master.data_sizes['crop']['Data']
        if lines != 0:
            self.shape = [np.minimum(lines, self.shape[0] - s_lin), self.shape[1] - s_pix]

        self.s_lin = s_lin
        self.s_pix = s_pix
        self.e_lin = self.s_lin + self.shape[0]
        self.e_pix = self.s_pix + self.shape[1]

        self.new_line = []
        self.new_pixel = []

        self.orbits = OrbitCoordinates(self.slave)

        # Load data
        self.X = self.master.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, 'X')
        self.Y = self.master.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, 'Y')
        self.Z = self.master.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, 'Z')

    def __call__(self):
        # Calculate the new line and pixel coordinates based orbits / geometry
        if len(self.X) == 0 or len(self.Y) == 0 or len(self.Z) == 0:
            print('Missing input data for geometrical coregistration for ' + self.slave.folder + '. Aborting..')
            return False

        try:
            self.new_line, self.new_pixel = self.orbits.xyz2lp(self.X, self.Y, self.Z)

            self.add_meta_data(self.master, self.slave)
            self.slave.image_new_data_memory(self.new_line, 'combined_coreg', self.s_lin, self.s_pix, 'New_line')
            self.slave.image_new_data_memory(self.new_pixel, 'combined_coreg', self.s_lin, self.s_pix, 'New_pixel')

            return True

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed geometrical coregistration for ' +
                              self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed geometrical coregistration for ' +
                  self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_output_files(self, to_disk=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = ['new_line', 'new_pixel']

        for s in to_disk:
            self.slave.image_create_disk('reramp', s)

    @staticmethod
    def add_meta_data(master, slave):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        meta_info = OrderedDict()

        meta_info['Master reference date'] = master.processes['readfiles']['First_pixel_azimuth_time (UTC)'][:10]

        for dat, dat_type in zip(['New_line_', 'New_pixel_'], ['real8', 'real8']):
            meta_info[dat + 'output_file'] = dat[:-1] + '.raw'
            meta_info[dat + 'output_format'] = dat_type
            meta_info[dat + 'lines'] = master.processes['crop']['Data_lines']
            meta_info[dat + 'pixels'] = master.processes['crop']['Data_pixels']
            meta_info[dat + 'first_line'] = master.processes['crop']['Data_first_line']
            meta_info[dat + 'first_pixel'] = master.processes['crop']['Data_first_pixel']

        slave.image_add_processing_step('geometrical_coreg', meta_info)
        slave.image_add_processing_step('combined_coreg', meta_info)

        slave.image_add_processing_step('coreg_readfiles', copy.deepcopy(master.processes['readfiles']))
        slave.image_add_processing_step('coreg_orbits', copy.deepcopy(master.processes['orbits']))
        slave.image_add_processing_step('coreg_crop', copy.deepcopy(master.processes['crop']))

    @staticmethod
    def processing_info():
        # Information on this processing step
        input_dat = defaultdict()
        input_dat['master']['geocode'] = ['X', 'Y', 'Z']

        output_dat = dict()
        output_dat['slave']['combined_coreg'] = ['New_line', 'New_pixel']

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use
