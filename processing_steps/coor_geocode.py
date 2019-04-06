"""
This function does the geocoding for points with a projection/geographic coordinate system.

Given the coordinate system the following information is known:
- latitude/longitude information per pixel (geographic grid)
Following information is created with a different function:
- latitude/longitude information per pixel (projection)
- height values per pixel
Following information is generated
- Cartesian coordinates of the pixel (X,Y,Z)
- line and pixel values based on the satellite orbit of the coregistration master image

"""

from rippl.image_data import ImageData
from collections import OrderedDict, defaultdict
from rippl.coordinate_system import CoordinateSystem
from rippl.processing_steps.coor_dem import CoorDem
from rippl.processing_steps.projection_coor import ProjectionCoor
from rippl.orbit_resample_functions.orbit_coordinates import OrbitCoordinates
import numpy as np
import logging
import os


class CoorGeocode(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # Load coordinates
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates
        shape = coordinates.shape
        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        self.shape = [l, shape[1] - s_pix]
        self.sample = coordinates.sample

        # Radar coordinates are not possible for this function, so these are excluded.
        if self.coordinates.grid_type == 'radar_coordinates':
            print('Using radar coordinates for the coor_geocode function is not possible. Use the regular geocode function')
            return

        # Load lat/lon values plus dem values
        self.lat, self.lon = ProjectionCoor.load_lat_lon(coordinates, meta, s_lin, s_pix, self.shape)
        self.height = CoorDem.load_dem(coordinates, meta, s_lin, s_pix, self.shape)

        self.no0 = (self.height != 0)
        if self.coordinates.mask_grid:
            mask = self.meta.image_load_data_memory('create_sparse_grid', s_lin, 0, self.shape,
                                                    'mask' + self.coordinates.sample)
            self.no0 *= mask

        # And the output data
        self.lines = np.zeros(self.height.shape).astype(np.float64)
        self.pixels = np.zeros(self.height.shape).astype(np.float64)
        self.x_coordinate = np.zeros(self.height.shape).astype(np.float64)
        self.y_coordinate = np.zeros(self.height.shape).astype(np.float64)
        self.z_coordinate = np.zeros(self.height.shape).astype(np.float64)

    def __call__(self):
        if len(self.height) == 0 or len(self.lat) == 0 or len(self.lon) == 0:
            print('Missing input data for creating radar DEM for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Resample the new grid based on the coordinates of the old grid based on a bilinear approximation.
            self.x_coordinate[self.no0], self.y_coordinate[self.no0], self.z_coordinate[self.no0] = \
                OrbitCoordinates.ell2xyz(self.lat[self.no0], self.lon[self.no0], self.height[self.no0].astype(np.float64))
            orbit = OrbitCoordinates(self.meta)
            self.lines[self.no0], self.pixels[self.no0] = \
                orbit.xyz2lp(self.x_coordinate[self.no0], self.y_coordinate[self.no0], self.z_coordinate[self.no0])

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.lines, 'coor_geocode', self.s_lin, self.s_pix, file_type='line' + self.sample)
            self.meta.image_new_data_memory(self.pixels, 'coor_geocode', self.s_lin, self.s_pix, file_type='pixel' + self.sample)
            self.meta.image_new_data_memory(self.x_coordinate, 'coor_geocode', self.s_lin, self.s_pix, file_type='X' + self.sample)
            self.meta.image_new_data_memory(self.y_coordinate, 'coor_geocode', self.s_lin, self.s_pix, file_type='Y' + self.sample)
            self.meta.image_new_data_memory(self.z_coordinate, 'coor_geocode', self.s_lin, self.s_pix, file_type='Z' + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed coor geocoding for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed coor geocoding for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'coor_geocode' in meta.processes.keys():
            meta_info = meta.processes['coor_geocode']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['line', 'pixel', 'X', 'Y', 'Z'], ['real8', 'real8', 'real8', 'real8', 'real8'], meta_info)
        meta.image_add_processing_step('coor_geocode', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='cmaster'):

        # Three input files needed Dem, Dem_line and Dem_pixel
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()

        input_dat[meta_type]['coor_DEM']['DEM' + coordinates.sample]['file'] = 'DEM' + coordinates.sample + '.raw'
        input_dat[meta_type]['coor_DEM']['DEM' + coordinates.sample]['coordinates'] = coordinates
        input_dat[meta_type]['coor_DEM']['DEM' + coordinates.sample]['slice'] = coordinates.slice
        input_dat[meta_type]['coor_DEM']['DEM' + coordinates.sample]['coor_change'] = 'resample'

        if coordinates.grid_type == 'projection':
            for dat_type in ['lat', 'lon']:
                input_dat[meta_type]['projection_coor'][dat_type + coordinates.sample]['file'] = dat_type + coordinates.sample + '.raw'
                input_dat[meta_type]['projection_coor'][dat_type + coordinates.sample]['coordinates'] = coordinates
                input_dat[meta_type]['projection_coor'][dat_type + coordinates.sample]['slice'] = coordinates.slice
                input_dat[meta_type]['projection_coor'][dat_type + coordinates.sample]['coor_change'] = 'resample'

        # One output file created radar dem
        output_dat = recursive_dict()
        for dat_type in ['line', 'pixel', 'X', 'Y', 'Z']:
            output_dat[meta_type]['coor_geocode'][dat_type + coordinates.sample]['files'] = dat_type + coordinates.sample + '.raw'
            output_dat[meta_type]['coor_geocode'][dat_type + coordinates.sample]['coordinate'] = coordinates
            output_dat[meta_type]['coor_geocode'][dat_type + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 15 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('coor_geocode', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('coor_geocode', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('coor_geocode', file_type, coordinates)

    @staticmethod
    def load_xyz(coordinates, meta, s_lin, s_pix, shape):

        x_key = 'X' + coordinates.sample
        y_key = 'Y' + coordinates.sample
        z_key = 'Z' + coordinates.sample
        if coordinates.grid_type == 'radar_coordinates':
            x = meta.image_load_data_memory('geocode', s_lin, s_pix, shape, x_key)
            y = meta.image_load_data_memory('geocode', s_lin, s_pix, shape, y_key)
            z = meta.image_load_data_memory('geocode', s_lin, s_pix, shape, z_key)
        elif coordinates.grid_type in ['projection', 'geographic']:
            x = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, x_key)
            y = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, y_key)
            z = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, z_key)
        else:
            print('xyz coordinates could not be loaded with this function!')
            return

        return x, y, z

    @staticmethod
    def xyz_processing_info(input_dat, coordinates, type='cmaster', resample=False, slice=None):
        # Load the xyz coordinate information for an xyz location.

        if coordinates.grid_type == 'radar_coordinates':
            step = 'geocode'
        elif coordinates.grid_type in ['projection', 'geographic']:
            step = 'coor_geocode'

        for t in ['X', 'Y', 'Z']:
            input_dat[type][step][t + coordinates.sample]['file'] = t + coordinates.sample + '.raw'
            input_dat[type][step][t + coordinates.sample]['coordinates'] = coordinates
            if resample:
                input_dat[type][step][t + coordinates.sample]['coor_change'] = 'resample'
            if slice != None:
                input_dat[type][step][t + coordinates.sample]['slice'] = slice
            else:
                input_dat[type][step][t + coordinates.sample]['slice'] = coordinates.slice

        return input_dat

    @staticmethod
    def line_pixel_processing_info(input_dat, coordinates, type='cmaster', resample=False, slice=None):
        # Load the xyz coordinate information for an xyz location.

        if coordinates.grid_type == 'radar_coordinates':
            return input_dat
        elif coordinates.grid_type in ['projection', 'geographic']:
            step = 'coor_geocode'

        for t in ['line', 'pixel']:
            input_dat[type][step][t + coordinates.sample]['file'] = t + coordinates.sample + '.raw'
            input_dat[type][step][t + coordinates.sample]['coordinates'] = coordinates
            if slice != None:
                input_dat[type][step][t + coordinates.sample]['slice'] = slice
            else:
                input_dat[type][step][t + coordinates.sample]['slice'] = coordinates.slice
            if resample:
                input_dat[type][step][t + coordinates.sample]['coor_change'] = 'resample'

        return input_dat