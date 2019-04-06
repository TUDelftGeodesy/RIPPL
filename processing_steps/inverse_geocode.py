"""
This script is mainly used to fit a DEM to the line and pixel coordinates of a specific orbit.
To do so we follow the next 3 steps:
1. Based on image metadata we load the DEM.
2. For irregular lat/lon grids, we load lat/lon files. / For regular lat/lon grids we calculate it from metadata
3. If needed, we correct the height grid for the geoid based on EGM96
4. Calculate the xyz cartesian coordinates for every pixel from latitude, longitude and height using the WGS84 ellipsoid
5. Finally the xyz coordinates are used to find line and pixel coordinates of the grid.

This steps follows the create SRTM DEM function or load the external DEM function.

The main function is called inverse geocoding, because the performed steps are also done during geocoding but in the
opposite order, where the line and pixel coordinates are converted to lat/lon/height coordinates via cartesian coordi-
nates.

The results of this step is a grid of line and pixel coordinates

"""

from rippl.orbit_resample_functions.orbit_coordinates import OrbitCoordinates
from rippl.coordinate_system import CoordinateSystem
from rippl.image_data import ImageData
from collections import OrderedDict, defaultdict
import numpy as np
import logging
import os


class InverseGeocode(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0):
        # There are three options for processing:

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        dem_key = 'DEM' + coordinates.sample
        shape = np.array(self.meta.data_sizes['import_DEM'][dem_key])
        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        self.shape = [l, shape[1] - s_pix]

        self.s_lin = s_lin
        self.s_pix = s_pix
        self.e_lin = self.s_lin + self.shape[0]
        self.e_pix = self.s_pix + self.shape[1]
        self.coordinates = coordinates

        # Load height and create lat/lon or load lat/lon
        self.dem = self.meta.image_load_data_memory('import_DEM', self.s_lin, self.s_pix, self.shape, dem_key)

        if coordinates.grid_type == 'geographic':

            self.lat = np.flip(np.linspace(coordinates.lat0, coordinates.lat0 + coordinates.dlat *
                                           (shape[0] - 1), shape[0]), axis=0)[
                                            self.s_lin:self.s_lin + self.shape[0], None] * np.ones((1, self.shape[1]))

            self.lon = np.linspace(coordinates.lon0 + coordinates.dlon * self.s_pix,
                                   coordinates.lon0 + coordinates.dlon * (self.s_pix + self.shape[1] - 1),
                                   self.shape[1])[None, :] * np.ones((self.shape[0], 1))

        elif coordinates.grid_type == 'projection':

            y = np.linspace(coordinates.y0 + coordinates.dy * self.s_pix,
                                   coordinates.y0 + coordinates.dy * (self.s_pix + self.shape[0] - 1),
                                   self.shape[0])[None, :] * np.ones((self.shape[1], 1))

            x = np.linspace(coordinates.x0 + coordinates.dx * self.s_pix,
                                   coordinates.x0 + coordinates.dx * (self.s_pix + self.shape[1] - 1),
                                   self.shape[1])[None, :] * np.ones((self.shape[0], 1))

            self.lat, self.lon = coordinates.proj2ell(x, y)

        # Initialize the output data
        self.x = []
        self.y = []
        self.z = []
        self.line = []
        self.pixel = []
        self.orbits = OrbitCoordinates(self.meta)

    def __call__(self):

        if len(self.lat) == 0 or len(self.lon) == 0 or len(self.dem) == 0:
            print('Missing input data for inverse geocoding for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Here the actual geocoding is done.

            # Then calculate the x,y,z coordinates
            self.x, self.y, self.z = self.orbits.ell2xyz(np.ravel(self.lat), np.ravel(self.lon), np.ravel(self.dem))

            # Finally calculate the lat/lon coordinates
            self.line, self.pixel = self.orbits.xyz2lp(self.x, self.y, self.z)
            self.x = []
            self.y = []
            self.z = []
            self.line = self.line.reshape(self.shape).astype(np.float32)
            self.pixel = self.pixel.reshape(self.shape).astype(np.float32)

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.line, 'inverse_geocode', self.s_lin, self.s_pix, file_type='DEM_line' + self.coordinates.sample)
            self.meta.image_new_data_memory(self.pixel, 'inverse_geocode', self.s_lin, self.s_pix, file_type='DEM_pixel' + self.coordinates.sample)

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

        if 'inverse_geocode' in meta.processes.keys():
            meta_info = meta.processes['inverse_geocode']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['DEM_line', 'DEM_pixel'],
                                                 ['real4', 'real4'], meta_info)

        meta.image_add_processing_step('inverse_geocode', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='cmaster'):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed Dem, Dem_line and Dem_pixel
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat[meta_type]['import_DEM']['DEM' + coordinates.sample]['file'] = 'DEM_' + coordinates.sample + '.raw'
        input_dat[meta_type]['import_DEM']['DEM' + coordinates.sample]['coordinates'] = coordinates
        input_dat[meta_type]['import_DEM']['DEM' + coordinates.sample]['slice'] = coordinates.slice

        # One output file created radar dem
        output_dat = recursive_dict()
        for t in ['DEM_line', 'DEM_pixel']:
            output_dat[meta_type]['inverse_geocode'][t + coordinates.sample]['files'] = t + coordinates.sample + '.raw'
            output_dat[meta_type]['inverse_geocode'][t + coordinates.sample]['coordinates'] = coordinates
            output_dat[meta_type]['inverse_geocode'][t + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('inverse_geocode', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('inverse_geocode', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('inverse_geocode', file_type, coordinates)
