# The following class does the coregistration and resampling of a slave image to the geometry of a master image
# The coregistration includes the following main steps:
#   - Calculate the geometry of the master image and if needed the dem of the master based on the master
#   - Calculate the line and pixel coordinates of the master image in the slave line and pixel coordinates
#   - (optional) Create a mask for the areas used to check shift using cross-correlation
#   - (optional) Calculate the window shifts for large or small windows to verify the geometrical shift
#   - (optional) Adapt calculated geometrical shift using windows.
#   - Resample the slave grid to master coordinates
#   - (optional) Calculate the baselines between the two images

from image_data import ImageData
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
from find_coordinates import FindCoordinates
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

    def __init__(self, cmaster_meta, slave_meta, coordinates, s_lin=0, s_pix=0, lines=0):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(slave_meta, ImageData) and isinstance(cmaster_meta, ImageData):
            self.slave = slave_meta
            self.cmaster = cmaster_meta
        else:
            return

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates
        self.shape, self.lines, self.pixels, fl, fp, self.sample, self.multilook, self.oversample, self.offset = \
            GeometricalCoreg.find_coordinates(self.cmaster, s_lin, s_pix, lines, coordinates)

        # Load data
        self.X = self.cmaster.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, 'X')
        self.Y = self.cmaster.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, 'Y')
        self.Z = self.cmaster.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, 'Z')

        # Initialize output
        self.new_line = []
        self.new_pixel = []

        self.orbits = OrbitCoordinates(self.slave)

    def __call__(self):
        # Calculate the new line and pixel coordinates based orbits / geometry
        if len(self.X) == 0 or len(self.Y) == 0 or len(self.Z) == 0:
            print('Missing input data for geometrical coregistration for ' + self.slave.folder + '. Aborting..')
            return False

        try:
            self.new_line, self.new_pixel = self.orbits.xyz2lp(self.X, self.Y, self.Z)

            self.add_meta_data(self.cmaster, self.slave, self.coordinates)
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
    def find_coordinates(cmaster, s_lin, s_pix, lines, coordinates):

        if isinstance(coordinates, CoordinateSystem):
            if not coordinates.grid_type == 'radar_coordinates':
                print('Other grid types than radar coordinates not supported yet.')
                return
        else:
            print('coordinates should be an CoordinateSystem object')

        shape = cmaster.image_get_data_size('crop', 'crop')

        first_line = cmaster.data_offset['crop']['crop'][0]
        first_pixel = cmaster.data_offset['crop']['crop'][1]
        sample, multilook, oversample, offset, [lines, pixels] = \
            FindCoordinates.interval_lines(shape, s_lin, s_pix, lines, multilook, oversample, offset)

        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        shape = [l, shape[1] - s_pix]

        lines = lines + first_line
        pixels = pixels + first_pixel

        return shape, lines, pixels, first_line, first_pixel, sample, multilook, oversample, offset

    @staticmethod
    def add_meta_data(cmaster, slave, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'geometrical_coreg' in slave.processes.keys():
            meta_info = slave.processes['geometrical_coreg']
        else:
            meta_info = OrderedDict()

        meta_info['Master_reference_date'] = cmaster.processes['readfiles']['First_pixel_azimuth_time (UTC)'][:10]
        meta_info = coordinates.create_meta_data(['New_line', 'New_pixel'], ['real8', 'real8'], meta_info)

        slave.image_add_processing_step('geometrical_coreg', meta_info)
        slave.image_add_processing_step('combined_coreg', meta_info)

        # Add the information from the master file to the slave .res files.
        slave.image_add_processing_step('coreg_readfiles', copy.deepcopy(cmaster.processes['readfiles']))
        slave.image_add_processing_step('coreg_orbits', copy.deepcopy(cmaster.processes['orbits']))
        slave.image_add_processing_step('coreg_crop', copy.deepcopy(cmaster.processes['crop']))

    @staticmethod
    def processing_info(coordinates):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()
        for t in ['X', 'Y', 'Z']:
            input_dat['cmaster']['geocode'][t]['file'] = [t + coordinates.sample + '.raw']
            input_dat['cmaster']['geocode'][t]['coordinates'] = coordinates
            input_dat['cmaster']['geocode'][t]['slice'] = coordinates.slice

        # line and pixel output files.
        output_dat = defaultdict()
        for step in ['geometrical_coreg', 'combined_coreg']:
            for t in ['New_line', 'New_pixel']:
                output_dat['slave'][step][t]['files'] = [t + coordinates.sample + '.raw']
                output_dat['slave'][step][t]['multilook'] = coordinates
                output_dat['slave'][step][t]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, output_file_steps=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not output_file_steps:
            meta_info = meta.processes['geocode']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_create_disk('geocode', s)

    @staticmethod
    def save_to_disk(meta, output_file_steps=''):

        if not output_file_steps:
            meta_info = meta.processes['geocode']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_memory_to_disk('geocode', s)