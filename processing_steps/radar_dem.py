# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from rippl.image_data import ImageData
from rippl.processing_steps.coor_geocode import CoorGeocode
from collections import OrderedDict, defaultdict
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
import numpy as np
import logging
import os
from shapely.geometry import Polygon
import copy


class RadarDem(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0, coor_in=''):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for meta_data in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # Load coordinates
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates
        self.sample = coordinates.sample
        self.first_pixel = coordinates.first_pixel
        self.first_line = coordinates.first_line
        self.multilook = coordinates.multilook
        self.shape, self.lines, self.pixels = RadarDem.find_coordinates(meta, s_lin, s_pix, lines, coordinates)

        # Load input data and check whether extend of input data is large enough.
        if isinstance(coor_in, CoordinateSystem):
            dem_type = coor_in.sample
        else:
            if 'import_DEM' in self.meta.processes.keys():
                dem_types = [key_str[3:-12] for key_str in self.meta.processes['import_DEM'] if
                             key_str.endswith('output_file') and not key_str.startswith('DEM_q')]
                if len(dem_types) > 0:
                    dem_type = dem_types[0]
                else:
                    print('No imported DEM found. Please rerun import_DEM function')
                    return
            else:
                print('Import a DEM first using the import_DEM function!')
                return

        self.lines = self.lines[:, 0]
        self.pixels = self.pixels[0, :]
        self.dem, self.dem_line, self.dem_pixel = RadarDem.source_dem_extend(self.meta, self.lines,
                                                                             self.pixels, dem_type)
        self.no0 = (self.dem_line != 0) * (self.dem_pixel != 0)
        if self.coordinates.mask_grid:
            mask = self.meta.image_load_data_memory('create_sparse_grid', s_lin, 0, self.shape,
                                                       'mask' + self.coordinates.sample)
            self.no0 *= mask
            # TODO Implement masking for calculating radar DEM.

        self.radar_dem = []

    def __call__(self):
        if len(self.dem) == 0 or len(self.dem_line) == 0 or len(self.dem_pixel) == 0:
            print('Missing input data for creating radar DEM for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Here the actual geocoding is done.
            # First calculate the heights using an external DEM. This generates the self.height grid..
            used_grids, grid_extends = self.dem_pixel2grid(self.dem, self.dem_line, self.dem_pixel, self.lines, self.pixels, self.multilook)

            # Find coordinates and matching interpolation areas
            dem_id, first_triangle = self.radar_in_dem_grid(used_grids, grid_extends, self.lines, self.pixels, self.multilook, self.shape, self.dem_line, self.dem_pixel)
            del used_grids, grid_extends

            # Then do the interpolation
            self.radar_dem = self.dem_barycentric_interpolation(self.dem, self.dem_line, self.dem_pixel, self.shape, first_triangle, dem_id)
            del self.dem, self.dem_line, self.dem_pixel

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.radar_dem, 'radar_DEM', self.s_lin, self.s_pix, file_type='DEM' + self.sample)
            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating radar DEM for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating radar DEM for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def source_dem_extend(meta, lines, pixels, dem_type, buf=3):


    @staticmethod
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'radar_DEM' in meta.processes.keys():
            meta_info = meta.processes['radar_DEM']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['DEM'], ['real4'], meta_info)

        meta.image_add_processing_step('radar_DEM', meta_info)
