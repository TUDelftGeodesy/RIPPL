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
from processing_steps.radar_dem import RadarDem
import copy
import os
import logging


class GeometricalCoreg(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, cmaster_meta, meta, coordinates, s_lin=0, s_pix=0, lines=0):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, ImageData) and isinstance(cmaster_meta, ImageData):
            self.slave = meta
            self.cmaster = cmaster_meta
        else:
            return

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates
        self.shape, self.lines, self.pixels = RadarDem.find_coordinates(self.cmaster, s_lin, s_pix, lines, coordinates)

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

            self.add_meta_data(self.slave, self.cmaster, self.coordinates)
            self.slave.image_new_data_memory(self.new_line, 'geometrical_coreg', self.s_lin, self.s_pix, 'new_line')
            self.slave.image_new_data_memory(self.new_pixel, 'geometrical_coreg', self.s_lin, self.s_pix, 'new_pixel')

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
    def add_meta_data(meta, cmaster_meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'geometrical_coreg' in meta.processes.keys():
            meta_info = meta.processes['geometrical_coreg']
        else:
            meta_info = OrderedDict()

        meta_info['Master_reference_date'] = cmaster_meta.processes['readfiles']['First_pixel_azimuth_time (UTC)'][:10]
        meta_info = coordinates.create_meta_data(['new_line', 'new_pixel'], ['real8', 'real8'], meta_info)

        meta.image_add_processing_step('geometrical_coreg', meta_info)
        meta.image_add_processing_step('combined_coreg', meta_info)

        # Add the information from the master file to the slave .res files.
        meta.image_add_processing_step('coreg_readfiles', copy.deepcopy(cmaster_meta.processes['readfiles']))
        meta.image_add_processing_step('coreg_orbits', copy.deepcopy(cmaster_meta.processes['orbits']))
        meta.image_add_processing_step('coreg_crop', copy.deepcopy(cmaster_meta.processes['crop']))

    @staticmethod
    def processing_info(coordinates, meta_type=''):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        for t in ['X', 'Y', 'Z']:
            input_dat['cmaster']['geocode'][t + coordinates.sample]['file'] = t + coordinates.sample + '.raw'
            input_dat['cmaster']['geocode'][t + coordinates.sample]['coordinates'] = coordinates
            input_dat['cmaster']['geocode'][t + coordinates.sample]['slice'] = True

        # line and pixel output files.
        output_dat = recursive_dict()
        for step in ['geometrical_coreg', 'combined_coreg']:
            for t in ['new_line', 'new_pixel']:
                output_dat['slave'][step][t + coordinates.sample]['files'] = t + coordinates.sample + '.raw'
                output_dat['slave'][step][t + coordinates.sample]['multilook'] = coordinates
                output_dat['slave'][step][t + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('geometrical_coreg', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('geometrical_coreg', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('geometrical_coreg', file_type, coordinates)
