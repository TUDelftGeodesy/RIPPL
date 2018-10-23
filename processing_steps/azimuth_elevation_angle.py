# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from image_data import ImageData
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from collections import OrderedDict, defaultdict
from find_coordinates import FindCoordinates
from coordinate_system import CoordinateSystem
import numpy as np
import logging
import os


class AzimuthElevationAngle(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0, orbit=False, scatterer=True):
        # There are three options for processing:

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # Load coordinates
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates
        self.shape, self.lines, self.pixels, self.sample, self.multilook, self.oversample, self.offset = \
            AzimuthElevationAngle.find_coordinates(meta, s_lin, s_pix, lines, coordinates)

        # Load data
        x_key = 'X' + self.sample
        y_key = 'Y' + self.sample
        z_key = 'Z' + self.sample
        dem_key = 'radar_DEM' + self.sample
        self.x = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, x_key)
        self.y = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, y_key)
        self.z = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, z_key)
        self.height = self.meta.image_load_data_memory('radar_DEM', self.s_lin, self.s_pix, self.shape, dem_key)

        self.orbits = OrbitCoordinates(self.meta)
        self.orbits.x = self.x
        self.orbits.y = self.y
        self.orbits.z = self.z
        self.orbits.height = self.height

        self.orbits.lp_time(lines=self.lines, pixels=self.pixels)

        # And the output data
        self.orbit = orbit
        self.scatterer = scatterer
        if self.scatterer:
            self.az_angle = []
            self.elev_angle = []
        if self.orbit:
            self.heading = []
            self.off_nadir_angle = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.x) == 0 or len(self.y) == 0 or len(self.z) == 0 or len(self.height) == 0:
            print('Missing input data for processing azimuth_elevation_angle for ' + self.meta.folder + '. Aborting..')
            return False

        # Here the actual geocoding is done.
        try:
            self.add_meta_data(self.meta, self.multilook, self.oversample, self.coordinates)

            # Then calculate the x,y,z coordinates
            if self.scatterer:
                self.orbits.xyz2scatterer_azimuth_elevation()
                self.az_angle = self.orbits.azimuth_angle
                self.elev_angle = self.orbits.elevation_angle

                # Data can be saved using the create output files and add meta data function.
                self.meta.image_new_data_memory(self.az_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='Azimuth_angle' + self.sample)
                self.meta.image_new_data_memory(self.elev_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='Elevation_angle' + self.sample)
            if self.orbit:
                self.orbits.xyz2ell(orbit=True)
                self.orbits.xyz2orbit_heading_off_nadir()
                self.heading = self.orbits.heading[:, None]
                self.off_nadir_angle = self.orbits.off_nadir_angle

                # Data can be saved using the create output files and add meta data function.
                self.meta.image_new_data_memory(self.heading, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='Heading' + self.sample)
                self.meta.image_new_data_memory(self.off_nadir_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='Off_nadir_angle' + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing azimuth_elevation_angle for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing azimuth_elevation_angle for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def find_coordinates(meta, s_lin, s_pix, lines, coordinates):

        if isinstance(coordinates, CoordinateSystem):
            if not coordinates.grid_type == 'radar_coordinates':
                print('Other grid types than radar coordinates not supported yet.')
                return
        else:
            print('coordinates should be an CoordinateSystem object')

        shape = meta.image_get_data_size('crop', 'crop')
        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        shape = [l, shape[1] - s_pix]

        first_line = meta.data_offset['crop']['crop'][0]
        first_pixel = meta.data_offset['crop']['crop'][1]
        sample, multilook, oversample, offset, [lines, pixels] = \
            FindCoordinates.interval_lines(shape, s_lin, s_pix, lines, coordinates.multilook, coordinates.oversample, coordinates.offset)

        lines = lines + first_line
        pixels = pixels + first_pixel

        return shape, lines, pixels, sample, multilook, oversample, offset

    @staticmethod
    def add_meta_data(meta, coordinates, scatterer=True, orbit=True):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'azimuth_elevation_angle' in meta.processes.keys():
            meta_info = meta.processes['azimuth_elevation_angle']
        else:
            meta_info = OrderedDict()

        if scatterer:
            meta_info = coordinates.create_meta_data(['Azimuth_angle', 'Elevation_angle'], ['real4', 'real4'], meta_info)
        if orbit:
            meta_info = coordinates.create_meta_data(['Heading', 'Off_nadir_angle'], ['real4', 'real4'], meta_info)

        meta.image_add_processing_step('azimuth_elevation_angle', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='cmaster', scatterer=True, orbit=False):

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()
        for t in ['X', 'Y', 'Z']:
            input_dat[meta_type]['geocode'][t]['file'] = [t + coordinates.sample + '.raw']
            input_dat[meta_type]['geocode'][t]['coordinates'] = coordinates
            input_dat[meta_type]['geocode'][t]['slice'] = coordinates.slice

        file_type = []
        if scatterer:
            file_type.extend(['Azimuth_angle', 'Elevation_angle'])
        if orbit:
            file_type.extend(['Heading', 'Off_nadir_angle'])

        # 2 or 4 output files.
        output_dat = defaultdict()
        for t in file_type:
            output_dat[meta_type]['azimuth_elevation_angle'][t]['file'] = [t + coordinates.sample + '.raw']
            output_dat[meta_type]['azimuth_elevation_angle'][t]['multilook'] = coordinates
            output_dat[meta_type]['azimuth_elevation_angle'][t]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('azimuth_elevation_angle', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_create_disk('azimuth_elevation_angle', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('azimuth_elevation_angle', file_type, coordinates)
