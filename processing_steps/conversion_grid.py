# The following class creates an interferogram from a master and slave image.

from image_data import ImageData
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import numpy as np
import logging
import os

class ConversionGrid(object):

    def __init__(self, meta, coor_in, coor_out):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.
        # When you want to use a certain projection, please give the proj4 string to do the conversion. Most projection
        # descriptions can be found at: spatialreference.org
        # The projection name is used as a shortname for the .res file, to get track of the output files.

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        if isinstance(coor_in, CoordinateSystem) and isinstance(coor_out, CoordinateSystem):
            self.coor_in = coor_in
            self.coor_out = coor_out

        # Load the latitude and longitude information.
        if not self.coor_in.grid_type == 'radar_coordinates' and self.coor_out.grid_type in ['geographic', 'projection']:
            print('Conversion grid can only be created for a conversion between a radar grid to a geographic grid or'
                  'projection.')
            return

        self.lat = self.meta.image_load_data_memory('geocode', 0, 0, self.coor_in.shape, 'lat' + coor_in.sample)
        self.lon = self.meta.image_load_data_memory('geocode', 0, 0, self.coor_in.shape, 'lon' + coor_in.sample)

        # Prepare output
        self.sort_ids = []
        self.sum_ids = []
        self.output_ids = []

    def __call__(self):
        if len(self.lat) == 0 or len(self.lon) == 0:
            print('Missing input data for conversion grid for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Create the conversion grid.
            if self.coor_out.grid_type == 'geographic':
                self.sort_ids, self.sum_ids, self.output_ids = ConversionGrid.geographical_grid(self.lat, self.lon,
                                                                                                self.coor_out)
            elif self.coor_out.grid_type == 'projection':
                self.sort_ids, self.sum_ids, self.output_ids = ConversionGrid.projection_grid(self.lat, self.lon,
                                                                                                self.coor_out)

            # Save the meta data
            convert_sample = self.coor_in.sample + '_' + self.coor_out.sample
            convert_ids_sizes = dict()
            convert_ids_sizes['sort_ids'] = self.sort_ids.shape
            convert_ids_sizes['sum_ids'] = self.sum_ids.shape
            convert_ids_sizes['output_ids'] = self.output_ids.shape

            ConversionGrid.create_meta_data(self.meta, self.coor_in, self.coor_out, convert_ids_sizes)

            # Save the output data
            self.meta.image_new_data_memory(self.sort_ids, 'coor_conversion', 0, 0, file_type='sort_ids' + convert_sample)
            self.meta.image_new_data_memory(self.sum_ids, 'coor_conversion', 0, 0, file_type='sum_ids' + convert_sample)
            self.meta.image_new_data_memory(self.output_ids, 'coor_conversion', 0, 0, file_type='output_ids' + convert_sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating conversion grid for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating conversion grid for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def geographical_grid(radar_lat, radar_lon, coordinates):
        # This method does a geographical multilooking. This means that the multilooking window is defined by the
        # geographical location of the data point.
        # - In first instance we will assume that we convert to a regular lat/lon grid with regular steps in latitude
        #   and longitude. If the projection definition, we assume that y_lim/latlon and y_step/x_step change accordingly
        # - The borders are defined by the chosen latitude and longitude limits. If these are not defined, it will be
        #   based on the extend of the lat/lon grid.
        # To find the needed proj4 string for your projection you could search for it on spatialreference.org

        if not isinstance(coordinates, CoordinateSystem):
            print('In and out coordinate system should be a CoordinateSystem object')
            return

        # Coordinate info based on whether we apply oversampling or not.
        if coordinates.oversample == [1, 1]:
            latlim = [coordinates.lat0, coordinates.lat0 + coordinates.shape[1] * coordinates.dlat]
            lonlim = [coordinates.lon0, coordinates.lon0 + coordinates.shape[0] * coordinates.dlon]
            shape = coordinates.shape
            dlat = coordinates.dlat
            dlon = coordinates.dlon
        else:
            latlim = [coordinates.lat0 - (coordinates.oversample[0] - 1) / 2 * coordinates.dlat,
                      coordinates.lat0 + (
                                  coordinates.shape[0] + (coordinates.oversample[0] - 1) / 2) * coordinates.dlat]
            lonlim = [coordinates.lon0 - (coordinates.oversample[1] - 1) / 2 * coordinates.dlat,
                      coordinates.lon0 + (
                                  coordinates.shape[0] + (coordinates.oversample[1] - 1) / 2) * coordinates.dlon]
            shape = [coordinates.shape[0] * 2 + coordinates.oversample[0] * 2 - 2,
                     coordinates.shape[1] * 2 + coordinates.oversample[1] * 2 - 2]

            # We have to change the stepsize to accommodate the oversampling later on. For example with an oversampling
            # of 2, the new pixel only overlaps 0.5 pixel to all sides. With an odd oversampling factor this is not
            # needed, but still implemented.
            dlat = coordinates.dlat * 0.5
            dlon = coordinates.dlon * 0.5

        inside = (latlim[0] < radar_lat < latlim[1]) * (lonlim[0] < radar_lon < latlim[1])

        # Select all pixels inside boundaries.
        # Calculate the coordinates of the new pixels and find the pixels outside the given boundaries.
        lat_id = np.int32((radar_lat - latlim[0]) / dlat)
        lon_id = np.int32((radar_lon - lonlim[0]) / dlon)
        flat_id = lat_id * shape[0] + lon_id
        del lat_id, lon_id

        # Sort ids and find number of pixels in every grid cell
        sort_ids = np.argsort(np.ravel(flat_id))[np.ravel(inside)]
        [output_ids, no_ids] = np.unique(flat_id[sort_ids], return_counts=True)
        sum_ids = np.cumsum(no_ids) - 1

        return sort_ids, sum_ids, output_ids

    @staticmethod
    def projection_grid(radar_lat, radar_lon, coordinates):
        # This method does a geographical multilooking. This means that the multilooking window is defined by the
        # geographical location of the data point.
        # - In first instance we will assume that we convert to a regular lat/lon grid with regular steps in latitude
        #   and longitude. If the projection definition, we assume that y_lim/latlon and y_step/x_step change accordingly
        # - The borders are defined by the chosen latitude and longitude limits. If these are not defined, it will be
        #   based on the extend of the lat/lon grid.
        # To find the needed proj4 string for your projection you could search for it on spatialreference.org

        if not isinstance(coordinates, CoordinateSystem):
            print('In and out coordinate system should be a CoordinateSystem object')
            return

        # Coordinate info based on whether we apply oversampling or not.
        if coordinates.oversample == [1, 1]:
            ylim = [coordinates.y0, coordinates.y0 + coordinates.shape[1] * coordinates.dy]
            xlim = [coordinates.x0, coordinates.x0 + coordinates.shape[0] * coordinates.dx]
            shape = coordinates.shape
            dy = coordinates.dy
            dx = coordinates.dx
        else:
            ylim = [coordinates.y0 - (coordinates.oversample[0] - 1) / 2 * coordinates.dy,
                    coordinates.y0 + (coordinates.shape[0] + (coordinates.oversample[0] - 1) / 2) * coordinates.dy]
            xlim = [coordinates.x0 - (coordinates.oversample[1] - 1) / 2 * coordinates.dy,
                    coordinates.x0 + (coordinates.shape[0] + (coordinates.oversample[1] - 1) / 2) * coordinates.dx]
            shape = [coordinates.shape[0] * 2 + coordinates.oversample[0] * 2 - 2,
                     coordinates.shape[1] * 2 + coordinates.oversample[1] * 2 - 2]

            # We have to change the stepsize to accommodate the oversampling later on. For example with an oversampling
            # of 2, the new pixel only overlaps 0.5 pixel to all sides. With an odd oversampling factor this is not
            # needed, but still implemented.
            dy = coordinates.dy * 0.5
            dx = coordinates.dx * 0.5

        radar_x, radar_y = coordinates.ell2proj(radar_lat, radar_lon)
        inside = (ylim[0] < radar_y < ylim[1]) * (xlim[0] < radar_x < ylim[1])

        # Select all pixels inside boundaries.
        # Calculate the coordinates of the new pixels and find the pixels outside the given boundaries.
        y_id = np.int32((radar_y - ylim[0]) / dy)
        x_id = np.int32((radar_x - xlim[0]) / dx)
        flat_id = y_id * shape[0] + x_id
        del y_id, x_id

        # Sort ids and find number of pixels in every grid cell
        sort_ids = np.argsort(np.ravel(flat_id))[np.ravel(inside)]
        [output_ids, no_ids] = np.unique(flat_id[sort_ids], return_counts=True)
        sum_ids = np.cumsum(no_ids) - 1

        return sort_ids, sum_ids, output_ids

    @staticmethod
    def input_output_info(coor_in, coor_out, meta_type='master'):
        # Information on this processing step. meta type should be defined here because this method is not directly
        # connected to either slave/master/ifg/coreg_master data type.

        if not isinstance(coor_in, CoordinateSystem) or not isinstance(coor_out, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        sample = coor_in.sample + '_' + coor_out.sample

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()
        for t in ['lat', 'lon']:
            input_dat[meta_type]['geocode'][t]['file'] = [t + coor_in.sample + '.raw']
            input_dat[meta_type]['geocode'][t]['coordinates'] = coor_in
            input_dat[meta_type]['geocode'][t]['slice'] = coor_in.slice

        # line and pixel output files.
        output_dat = defaultdict()
        for t in ['sort_ids', 'sum_ids', 'output_ids']:
            output_dat[meta_type]['conversion_grid'][t]['file'] = [t + sample + '.raw']
            output_dat[meta_type]['conversion_grid'][t]['coordinates'] = coor_out
            output_dat[meta_type]['conversion_grid'][t]['slice'] = coor_out.slice

        # Number of times input data is used in ram.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_meta_data(meta, coor_in, coor_out, convert_id_sizes):
        # Create meta data for the convert ids.
        # Note that the input coordinates are always radar coordinates and the output coordinates are always
        # geographic or projection coordinates.
        # These are the only files which are based on two coordinate systems and are therefore an exception on all the
        # other types, which are created from the CoordinateSystem object directly.

        if 'coor_conversion' in meta.processes.keys():
            meta_info = meta.processes['coor_conversion']
        else:
            meta_info = OrderedDict()

        data_name = coor_in.sample + '_' + coor_out.sample
        meta_info[data_name + '_sort_ids' + '_output_file'] = data_name + '_sort_ids' + '.raw'
        meta_info[data_name + '_sort_ids' + '_output_format'] = 'int32'
        meta_info[data_name + '_sort_ids' + '_lines'] = str(convert_id_sizes['sort_ids'][0])
        meta_info[data_name + '_sort_ids' + '_pixels'] = str(convert_id_sizes['sort_ids'][0][1])

        for file_type in ['sum_ids', 'output_ids']:
            meta_info[data_name + '_' + file_type + '_output_file'] = data_name + '_' + file_type + '.raw'
            meta_info[data_name + '_' + file_type + '_output_format'] = 'int32'
            meta_info[data_name + '_' + file_type + '_lines'] = str(convert_id_sizes[file_type][0])
            meta_info[data_name + '_' + file_type + '_pixels'] = str(convert_id_sizes[file_type][0][1])

        # Save information of input grid
        meta_info[data_name + '_input_grid'] = coor_in.sample
        meta_info[data_name + '_first_line'] = str(coor_in.first_line)
        meta_info[data_name + '_first_pixel'] = str(coor_in.first_pixel)
        meta_info[data_name + '_multilook_azimuth'] = str(coor_in.multilook[0])
        meta_info[data_name + '_multilook_range'] = str(coor_in.multilook[1])
        meta_info[data_name + '_oversampling_azimuth'] = str(coor_in.oversample[0])
        meta_info[data_name + '_oversampling_range'] = str(coor_in.oversample[1])
        meta_info[data_name + '_offset_azimuth'] = str(coor_in.offset[0])
        meta_info[data_name + '_offset_range'] = str(coor_in.offset[1])

        # Save information of output grid
        meta_info[data_name + '_output_grid'] = coor_out.sample
        if coor_out.grid_type == 'geographic':
            meta_info[data_name + '_ellipse_type'] = coor_out.ellipse_type
            meta_info[data_name + '_lat0'] = str(coor_out.lat0)
            meta_info[data_name + '_lon0'] = str(coor_out.lon0)
            meta_info[data_name + '_dlat'] = str(coor_out.dlat)
            meta_info[data_name + '_dlon'] = str(coor_out.dlon)
        elif coor_out.grid_type == 'projection':
            meta_info[data_name + '_projection_type'] = coor_out.projection_type
            meta_info[data_name + '_ellipse_type'] = coor_out.ellipse_type
            meta_info[data_name + '_proj4_str'] = coor_out.proj4_str
            meta_info[data_name + '_x0'] = str(coor_out.lat0)
            meta_info[data_name + '_y0'] = str(coor_out.lon0)
            meta_info[data_name + '_dx'] = str(coor_out.dlat)
            meta_info[data_name + '_dy'] = str(coor_out.dlon)

        meta.image_add_processing_step('coor_conversion', meta_info)

    @staticmethod
    def create_output_files(meta, output_file_steps=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not output_file_steps:
            meta_info = meta.processes['conversion_grid']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_create_disk('conversion_grid', s)

    @staticmethod
    def save_to_disk(meta, output_file_steps=''):

        if not output_file_steps:
            meta_info = meta.processes['conversion_grid']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_memory_to_disk('conversion_grid', s)
