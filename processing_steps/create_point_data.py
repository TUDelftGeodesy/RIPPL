"""
This function loads sparse grids for processing. There are 2 options:
1. Load a list op pixels and lines to select the right pixels to be processed. Additionally you can add the coordinates
    or azimuth/range time for the first pixel, to adjust for offsets in the coordinate system.
2. Give a shapefile as input. However, the shape should be lat/lon coordinates and if we are working with a radar
    grid, this should be done after geocoding.

Note: This conversion creates either a list of pixels/lines, which is data efficient for image processing but only works
for small datasets, or with a mask, which is very convenient for processing but creates data with empty spaces.

It is possible convert between these two formats. Either use lin_pix_2_mask or mask_2_lin_pix functions.

Finally: Because conversion from sparse datasets to non-sparse datasets is complicated, be sure that this data is not
needed anywhere later on in the processing, as it is non-reversible.
"""

import numpy as np
import fiona
from matplotlib.path import Path
from collections import OrderedDict
import os
import logging
import copy

from rippl.image_data import ImageData
from rippl.coordinate_system import CoordinateSystem
from rippl.processing_steps.projection_coor import ProjectionCoor
from collections import defaultdict


class CreatePointData(object):

    def __init__(self, meta, coordinates, shapefile='', points='', coor_points='', point_data_name='sparse'):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        self.coordinates = coordinates
        self.sample = coordinates.sample

        self.mask = []
        self.lines = []
        self.pixels = []
        self.shape = []

        # Load coordinates
        if shapefile:
            self.shapefile = shapefile
            self.lat, self.lon = ProjectionCoor.load_lat_lon(coordinates, meta, 1, 1, coordinates.shape)
            self.input = 'shape'

        elif len(points) > 0:   # In case we have points instead of a shapefile.

            self.input = 'points'
            if len(coor_points) == 0:
                print('There should be a coordinate system supplied with points if there are any offsets. We assume'
                      'the offsets to be the same in this case.')
                self.coor_points = self.coordinates

            offsets = self.coordinates.get_offset(self.coor_points)

            self.lines = points[:, 0] - offsets[0]
            self.pixels = points[:, 2] - offsets[1]



    def __call__(self):

        try:
            # Calculate the mask using the provided shapefile
            if self.input == 'shape':
                self.mask = self.shape_2_mask(self.lat, self.lon, self.shapefile)
                offsets = [self.coordinates.first_line, self.coordinates.first_pixel]
                self.lines, self.pixels = self.mask_2_lp(self.mask, offsets)

            point_coordinates = copy.copy(self.coordinates)
            point_coordinates.shape = (1, len(self.lines))

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.lines, 'point_data', 1, 1, file_type='line' + self.sample)
            self.meta.image_new_data_memory(self.pixels, 'point_data', 1, 1, file_type='pixel' + self.sample)
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

        if 'point_data' in meta.processes.keys():
            meta_info = meta.processes['point_data']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['line', 'pixel'], ['int4', 'int4'], meta_info)
        meta.image_add_processing_step('point_data', meta_info)

    @staticmethod
    def processing_info(coor_out, meta_type='cmaster'):

        # Lat/lon files as output
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        output_dat = recursive_dict()

        for dat_type in ['line', 'pixel']:
            output_dat[meta_type]['point_data'][dat_type + coor_out.sample]['files'] = dat_type + coor_out.sample + '.raw'
            output_dat[meta_type]['point_data'][dat_type + coor_out.sample]['coordinate'] = coor_out
            output_dat[meta_type]['point_data'][dat_type + coor_out.sample]['slice'] = coor_out.slice

        # Number of times input data is used in ram. Bit difficult here but 15 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('point_data', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('point_data', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('point_data', file_type, coordinates)

    @staticmethod
    def shape_2_mask(lat_grid, lon_grid, shapefile):

        shp_col = fiona.open(shapefile)
        shp = shp_col.next()  # only one feature in the shapefile

        polygon = Path(list(shp.exterior.coords))
        points = [(lat, lon) for lat, lon in zip(np.ravel(lat_grid), np.ravel(lon_grid))]
        mask = polygon.contains_points(points).reshape(lat_grid.shape)

        return mask

    @staticmethod
    def mask_2_lp(mask, offsets=''):

        points = np.argwhere(mask)

        if len(offsets) == 2:
            lines = points[:, 0] + offsets[0]
            pixels = points[:, 1] + offsets[1]
        else:
            lines = points[:, 0]
            pixels = points[:, 1]

        return lines, pixels
