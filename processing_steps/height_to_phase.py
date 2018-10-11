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
from find_coordinates import FindCoordinates
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import numpy as np
import os
import logging


class HeightToPhase(object):
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
        self.shape, lines, self.pixels, fl, fp, self.sample, self.multilook, self.oversample, self.offset = \
            HeightToPhase.find_coordinates(self.cmaster, s_lin, s_pix, lines, coordinates)

        # Information on conversion from range to distances.
        sol = 299792458  # speed of light [m/s]
        self.ra2m = 1 / float(self.cmaster.processes['readfiles']['Range_sampling_rate (computed, MHz)']) / 1000000 * sol
        self.dist_first_pix = float(self.cmaster.processes['readfiles']['Range_time_to_first_pixel (2way) (ms)']) \
                              / 1000 / 2 * sol
        self.wavelength = 1.0 / float(self.cmaster.processes['readfiles']['RADAR_FREQUENCY (HZ)']) * sol

        # Load data
        self.baseline = self.cmaster.image_load_data_memory('baseline', self.s_lin, self.s_pix, self.shape,
                                                            'Perpendicular_baseline')
        self.incidence = (90 - self.cmaster.image_load_data_memory('azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                                       self.shape, 'Elevation_angle')) / 180 * np.pi

        # Initialize output
        self.h2ph = []

    def __call__(self):
        # Calculate the new line and pixel coordinates based orbits / geometry
        if len(self.baseline) == 0 or len(self.incidence) == 0:
            print('Missing input data for geometrical coregistration for ' + self.slave.folder + '. Aborting..')
            return False

        try:
            # First get the master and slave positions.
            R = self.ra2m * (self.pixels[None, :] - 1) + self.dist_first_pix
            self.h2ph = self.baseline / (self.wavelength * R * np.sin(self.incidence)) * 4 * np.pi

            # Save meta data
            self.add_meta_data(self.slave, self.coordinates)
            self.slave.image_new_data_memory(self.h2ph, 'height_to_phase', self.s_lin, self.s_pix,
                                             file_type='height_to_phase' + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed height to phase calculation for ' +
                              self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed height to phase calculation for ' +
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
            FindCoordinates.interval_lines(shape, s_lin, s_pix, lines, coordinates)

        shape = [len(lines), len(pixels)]
        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        shape = [l, shape[1] - s_pix]

        return shape, lines, pixels, first_line, first_pixel, sample, multilook, oversample, offset

    @staticmethod
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'height_to_phase' in meta.processes.keys():
            meta_info = meta.processes['height_to_phase']
        else:
            meta_info = OrderedDict()

        coordinates.create_meta_data(['height_to_phase'], ['real4'], meta_info)

        meta.image_add_processing_step('height_to_phase', meta_info)

    @staticmethod
    def processing_info(coordinates):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()
        input_dat['slave']['baseline']['Perpendicular_baseline']['file'] = ['Perpendicular_baseline' + coordinates.sample + '.raw']
        input_dat['slave']['baseline']['Perpendicular_baseline']['multilook'] = coordinates
        input_dat['slave']['baseline']['Perpendicular_baseline']['slice'] = coordinates.slice

        input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle']['file'] = ['Elevation_angle' + coordinates.sample + '.raw']
        input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle']['multilook'] = coordinates
        input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle']['slice'] = coordinates.slice

        # line and pixel output files.
        output_dat = defaultdict()
        output_dat['slave']['height_to_phase']['height_to_phase']['files'] = ['height_to_phase' + coordinates.sample + '.raw']
        output_dat['slave']['height_to_phase']['height_to_phase']['files'] = coordinates
        output_dat['slave']['height_to_phase']['height_to_phase']['files'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, output_file_steps=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not output_file_steps:
            meta_info = meta.processes['height_to_phase']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_create_disk('height_to_phase', s)

    @staticmethod
    def save_to_disk(meta, output_file_steps=''):

        if not output_file_steps:
            meta_info = meta.processes['height_to_phase']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_memory_to_disk('height_to_phase', s)
