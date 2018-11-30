# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from image_data import ImageData
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from processing_steps.radar_dem import RadarDem
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
        self.shape, self.lines, self.pixels = RadarDem.find_coordinates(meta, s_lin, s_pix, lines, coordinates)
        self.sample = coordinates.sample

        dat_key = 'radar_DEM' + self.sample
        self.height = self.meta.image_load_data_memory('radar_DEM', self.s_lin, self.s_pix, self.shape, dat_key)
        self.orbits = OrbitCoordinates(self.meta)

        self.no0 = (self.height != 0)
        self.orbits.height = self.height[self.no0]

        # And the output data
        self.lat = np.zeros(self.height.shape).astype(np.float32)
        self.lon = np.zeros(self.height.shape).astype(np.float32)
        self.x_coordinate = np.zeros(self.height.shape).astype(np.float64)
        self.y_coordinate = np.zeros(self.height.shape).astype(np.float64)
        self.z_coordinate = np.zeros(self.height.shape).astype(np.float64)

    def __call__(self):
        # Here the actual geocoding is done.
        # Check if needed data is loaded
        if len(self.height) == 0:
            print('Missing input data for geocoding for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            if np.sum(self.no0) > 0:
                l_id, p_id = np.where(self.no0)
                self.orbits.lp_time(lines=self.lines[l_id], pixels=self.pixels[p_id], regular=False)

                # Then calculate the x,y,z coordinates
                self.orbits.lph2xyz()
                self.x_coordinate[self.no0] = self.orbits.x
                self.y_coordinate[self.no0] = self.orbits.y
                self.z_coordinate[self.no0] = self.orbits.z

                # Finally calculate the lat/lon coordinates
                self.orbits.xyz2ell()
                self.lat[self.no0] = self.orbits.lat
                self.lon[self.no0] = self.orbits.lon

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.lat, 'geocode', self.s_lin, self.s_pix, file_type='lat' + self.sample)
            self.meta.image_new_data_memory(self.lon, 'geocode', self.s_lin, self.s_pix, file_type='lon' + self.sample)
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
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'geocode' in meta.processes.keys():
            meta_info = meta.processes['geocode']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['lat', 'lon', 'X', 'Y', 'Z'],
                                                 ['real4', 'real4', 'real8', 'real8', 'real8'], meta_info)

        meta.image_add_processing_step('geocode', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='cmaster'):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed Dem, Dem_line and Dem_pixel
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat[meta_type]['radar_DEM']['radar_DEM' + coordinates.sample]['file'] = 'radar_DEM' + coordinates.sample + '.raw'
        input_dat[meta_type]['radar_DEM']['radar_DEM' + coordinates.sample]['coordinates'] = coordinates
        input_dat[meta_type]['radar_DEM']['radar_DEM' + coordinates.sample]['slice'] = coordinates.slice

        # One output file created radar dem
        output_dat = recursive_dict()
        for t in ['lat', 'lon', 'X', 'Y', 'Z']:
            output_dat[meta_type]['geocode'][t + coordinates.sample]['files'] = t + coordinates.sample + '.raw'
            output_dat[meta_type]['geocode'][t + coordinates.sample]['coordinates'] = coordinates
            output_dat[meta_type]['geocode'][t + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('geocode', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('geocode', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('geocode', file_type, coordinates)
