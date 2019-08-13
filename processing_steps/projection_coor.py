"""
This function creates a grid of lat/lon values for a projection grid. These values will be needed for further processing
with a projection grid.
"""

from rippl.meta_data.image_data import ImageData
from rippl.meta_data.process_meta import Process
from collections import OrderedDict, defaultdict
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
import numpy as np
import logging
import os


class ProjectionCoor(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0):
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
        shape = coordinates.shape
        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        self.shape = [l, shape[1] - s_pix]
        self.sample = coordinates.sample

        # Radar coordinates are not possible for this function, so these are excluded.
        if self.coordinates.grid_type != 'projection':
            print('This function is only meant for projected grids')
            return

        # Check if we work with a projection and load x, y coordinates if so.
        x0 = self.coordinates.x0 + (self.coordinates.first_line + self.s_pix) * self.coordinates.dx
        y0 = self.coordinates.y0 + (self.coordinates.first_pixel + self.s_lin) * self.coordinates.dy
        x_max = x0 + self.coordinates.dx * (self.shape[1] - 1)
        y_max = y0 + self.coordinates.dy * (self.shape[0] - 1)
        self.x, self.y = np.meshgrid(np.linspace(x0, x_max, self.shape[1]),
                                         np.linspace(y0, y_max, self.shape[0]))


    def __call__(self):
        if len(self.x) == 0 or len(self.y) == 0:
            print('Missing input data for creating radar DEM for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Resample the new grid based on the coordinates of the old grid based on a bilinear approximation.
            lat, lon = self.coordinates.proj2ell(self.x, self.y)

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(lat, 'projection_coor', self.s_lin, self.s_pix, file_type='lat' + self.sample)
            self.meta.image_new_data_memory(lon, 'projection_coor', self.s_lin, self.s_pix, file_type='lon' + self.sample)
            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating projection coordinates for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating projection coordinates for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'projection_coor' in meta.processes.keys():
            meta_info = meta.processes['projection_coor']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['lat', 'lon'], ['real8', 'real8'], meta_info)
        meta.image_add_processing_step('projection_coor', meta_info)
