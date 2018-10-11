# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from image_data import ImageData
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from find_coordinates import FindCoordinates
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import numpy as np
import os
import logging


class Geocode(object):

    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0):
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
            Geocode.find_coordinates(meta, s_lin, s_pix, lines, coordinates)

        dat_key = 'radar_DEM' + self.sample
        self.height = self.meta.image_load_data_memory('radar_DEM', self.s_lin, self.s_pix, self.shape, dat_key)
        self.orbits = OrbitCoordinates(self.meta)
        self.orbits.height = self.height

        self.orbits.lp_time(lines=self.lines, pixels=self.pixels)

        # And the output data
        self.lat = []
        self.lon = []
        self.x_coordinate = []
        self.y_coordinate = []
        self.z_coordinate = []

    def __call__(self):
        # Here the actual geocoding is done.
        # Check if needed data is loaded
        if len(self.height) == 0:
            print('Missing input data for geocoding for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Then calculate the x,y,z coordinates
            self.orbits.lph2xyz()
            self.x_coordinate = self.orbits.x
            self.y_coordinate = self.orbits.y
            self.z_coordinate = self.orbits.z

            # Finally calculate the lat/lon coordinates
            self.orbits.xyz2ell()
            self.lat = self.orbits.lat
            self.lon = self.orbits.lon

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.lat, 'geocode', self.s_lin, self.s_pix, file_type='Lat' + self.sample)
            self.meta.image_new_data_memory(self.lon, 'geocode', self.s_lin, self.s_pix, file_type='Lon' + self.sample)
            self.meta.image_new_data_memory(self.x_coordinate, 'geocode', self.s_lin, self.s_pix, file_type='X' + self.sample)
            self.meta.image_new_data_memory(self.y_coordinate, 'geocode', self.s_lin, self.s_pix, file_type='Y' + self.sample)
            self.meta.image_new_data_memory(self.z_coordinate, 'geocode', self.s_lin, self.s_pix, file_type='Z' + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed geocoding for ' +
                              self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed geocoding for ' +
                  self.meta.folder + '. Check ' + log_file + ' for details.')

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
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'geocode' in meta.processes.keys():
            meta_info = meta.processes['geocode']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['Lat', 'Lon', 'X', 'Y', 'Z'],
                                                 ['real4', 'real4', 'real8', 'real8', 'real8'], meta_info)

        meta.image_add_processing_step('geocode', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='cmaster'):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed Dem, Dem_line and Dem_pixel
        input_dat = defaultdict()
        input_dat[meta_type]['radar_DEM']['radar_DEM']['file'] = ['Data_' + coordinates.sample + '.raw']
        input_dat[meta_type]['radar_DEM']['radar_DEM']['coordinates'] = coordinates
        input_dat[meta_type]['radar_DEM']['radar_DEM']['slice'] = coordinates.slice

        # One output file created radar dem
        output_dat = defaultdict()
        for t in ['lat', 'lon', 'X', 'Y', 'Z']:
            output_dat[meta_type]['geocode'][t]['files'] = [t + coordinates.sample + '.raw']
            output_dat[meta_type]['geocode'][t]['coordinates'] = coordinates
            output_dat[meta_type]['geocode'][t]['slice'] = coordinates.slice

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