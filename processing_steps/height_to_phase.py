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
from radar_dem import RadarDem
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

    def __init__(self, meta, cmaster_meta, coordinates, s_lin=0, s_pix=0, lines=0):
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
        self.sample = self.coordinates.sample
        self.shape, lines, self.pixels = RadarDem.find_coordinates(self.cmaster, s_lin, s_pix, lines, coordinates)

        # Information on conversion from range to distances.
        sol = 299792458  # speed of light [m/s]
        self.ra2m = 1 / float(self.cmaster.processes['readfiles']['Range_sampling_rate (computed, MHz)']) / 1000000 * sol
        self.dist_first_pix = float(self.cmaster.processes['readfiles']['Range_time_to_first_pixel (2way) (ms)']) \
                              / 1000 / 2 * sol
        self.wavelength = 1.0 / float(self.cmaster.processes['readfiles']['RADAR_FREQUENCY (HZ)']) * sol

        # Load data
        self.baseline = self.slave.image_load_data_memory('baseline', self.s_lin, self.s_pix, self.shape,
                                                            'perpendicular_baseline')
        self.incidence = (90 - self.cmaster.image_load_data_memory('azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                                       self.shape, 'elevation_angle')) / 180 * np.pi

        # Initialize output
        self.h2ph = []

    def __call__(self):
        # Calculate the new line and pixel coordinates based orbits / geometry
        if len(self.baseline) == 0 or len(self.incidence) == 0:
            print('Missing input data for geometrical coregistration for ' + self.slave.folder + '. Aborting..')
            return False

        try:
            no0 = (self.baseline != 0) * (self.incidence != 0)
            lines, pixels = np.where(no0)
            del lines

            self.h2ph = np.zeros(self.baseline.shape).astype(np.float32)

            if np.sum(no0) > 0:
                # First get the master and slave positions.
                R = self.ra2m * (self.pixels[pixels] - 1) + self.dist_first_pix
                self.h2ph[no0] = self.baseline[no0] / (R * np.sin(self.incidence[no0]))

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
    def processing_info(coordinates, meta_type=''):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat['slave']['baseline']['perpendicular_baseline']['file'] = ['perpendicular_baseline' + coordinates.sample + '.raw']
        input_dat['slave']['baseline']['perpendicular_baseline']['coordinates'] = coordinates
        input_dat['slave']['baseline']['perpendicular_baseline']['slice'] = True

        input_dat['cmaster']['azimuth_elevation_angle']['elevation_angle']['file'] = ['elevation_angle' + coordinates.sample + '.raw']
        input_dat['cmaster']['azimuth_elevation_angle']['elevation_angle']['coordinates'] = coordinates
        input_dat['cmaster']['azimuth_elevation_angle']['elevation_angle']['slice'] = coordinates.slice

        # line and pixel output files.
        output_dat = recursive_dict()
        output_dat['slave']['height_to_phase']['height_to_phase']['files'] = ['height_to_phase' + coordinates.sample + '.raw']
        output_dat['slave']['height_to_phase']['height_to_phase']['files'] = coordinates
        output_dat['slave']['height_to_phase']['height_to_phase']['files'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('height_to_phase', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('height_to_phase', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('height_to_phase', file_type, coordinates)
