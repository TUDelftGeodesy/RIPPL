# The following class creates an interferogram from a master and slave image.

from rippl.image_data import ImageData
from rippl.coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import numpy as np
import logging
import os

class ConversionGrid(object):

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0, coor_in=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.
        # When you want to use a certain projection, please give the proj4 string to do the conversion. Most projection
        # descriptions can be found at: spatialreference.org
        # The projection name is used as a shortname for the .res file, to get track of the output files.

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        if not isinstance(coor_in, CoordinateSystem):
            coor_in = CoordinateSystem()
            coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
            self.coor_in = coor_in
        else:
            self.coor_in = coor_in

        coor_in.add_res_info(self.meta)
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')
        self.coor_out = coordinates

        # Load the latitude and longitude information.
        if not self.coor_in.grid_type == 'radar_coordinates' and self.coor_out.grid_type in ['geographic', 'projection']:
            print('Conversion grid can only be created for a conversion between a radar grid to a geographic grid or'
                  'projection.')
            return

        self.lat = self.meta.image_load_data_memory('geocode', 0, 0, self.coor_in.shape, 'lat' + coor_in.sample)
        self.lon = self.meta.image_load_data_memory('geocode', 0, 0, self.coor_in.shape, 'lon' + coor_in.sample)
        # self.no0 = (self.lat != 0) * (self.lon != 0)

        # Prepare output ids
        self.sort_ids = []
        self.sum_ids = []
        self.output_ids = []

        # No of looks
        self.looks = []

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
            convert_sample = self.coor_in.sample + self.coor_out.sample

            # Add zeros when we encounter zero sizes
            if self.sort_ids.shape[0] == 0:
                self.sort_ids = np.array([0])
                self.sum_ids = np.array([1])
                self.output_ids = np.array([0])

            # Save the meta data

            convert_ids_sizes = dict()
            convert_ids_sizes['sort_ids'] = [1, self.sort_ids.shape[0]]
            convert_ids_sizes['sum_ids'] = [1, self.sum_ids.shape[0]]
            convert_ids_sizes['output_ids'] = [1, self.output_ids.shape[0]]

            ConversionGrid.add_meta_data(self.meta, self.coor_in, self.coor_out, convert_ids_sizes)

            # Get the number of looks
            self.looks = np.zeros(self.coor_out.shape).astype(np.int32)
            self.looks[self.output_ids // self.coor_out.shape[0], self.output_ids % self.coor_out.shape[1]] = \
                np.diff(np.concatenate((self.sum_ids, [len(self.sort_ids)])))

            # Create output images
            self.meta.images_create_disk('conversion_grid', ['looks', 'sort_ids', 'sum_ids', 'output_ids'], self.coor_out, self.coor_in)

            # Save the output data
            self.meta.image_new_data_memory(self.looks.astype(np.int32), 'conversion_grid', 0, 0, file_type='looks' + convert_sample)
            self.meta.image_new_data_memory(self.sort_ids[None, :], 'conversion_grid', 0, 0, file_type='sort_ids' + convert_sample)
            self.meta.image_new_data_memory(self.sum_ids[None, :], 'conversion_grid', 0, 0, file_type='sum_ids' + convert_sample)
            self.meta.image_new_data_memory(self.output_ids[None, :], 'conversion_grid', 0, 0, file_type='output_ids' + convert_sample)

            self.meta.images_memory_to_disk('conversion_grid', ['looks', 'sort_ids', 'sum_ids', 'output_ids'], self.coor_out, self.coor_in)

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

        inside = (np.min(latlim) < radar_lat) * (radar_lat < np.max(latlim)) * \
                 (np.min(lonlim) < radar_lon) * (radar_lon < np.max(lonlim)) * \
                 ~((radar_lat == 0) * (radar_lon == 0))

        # Select all pixels inside boundaries.
        # Calculate the coordinates of the new pixels and find the pixels outside the given boundaries.
        lat_id = np.int64((radar_lat - latlim[0]) / dlat)
        lon_id = np.int64((radar_lon - lonlim[0]) / dlon)
        flat_id = lat_id * shape[0] + lon_id
        lat_id = []
        lon_id = []

        # Sort ids and find number of pixels in every grid cell
        flat_id[inside == False] = -1
        num_outside = np.sum(~inside)
        sort_ids = np.argsort(np.ravel(flat_id))[num_outside:]
        [output_ids, no_ids] = np.unique(flat_id[np.unravel_index(sort_ids, radar_lat.shape)], return_counts=True)
        sum_ids = np.cumsum(no_ids) - no_ids


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
        inside = (np.min(ylim) < radar_y) * (radar_y < np.max(ylim)) * \
                 (np.min(xlim) < radar_x) * (radar_x < np.max(xlim)) * \
                 ~((radar_lat == 0) * (radar_lon == 0))

        # Select all pixels inside boundaries.
        # Calculate the coordinates of the new pixels and find the pixels outside the given boundaries.
        y_id = np.int64((radar_y - ylim[0]) / dy)
        x_id = np.int64((radar_x - xlim[0]) / dx)
        flat_id = y_id * shape[0] + x_id

        y_id = []
        x_id = []

        # Sort ids and find number of pixels in every grid cell
        flat_id[inside == False] = -1
        num_outside = np.sum(~inside)
        sort_ids = np.argsort(np.ravel(flat_id))[num_outside:]
        [output_ids, no_ids] = np.unique(flat_id[np.unravel_index(sort_ids, radar_x.shape)], return_counts=True)
        sum_ids = np.cumsum(no_ids) - no_ids

        """
        Test for multilook:
        radar_lon[radar_lon < 6] = 0
        radar_lat[radar_lat < 6] = 0
        values_out = np.zeros(coordinates.shape)
        values_out[np.unravel_index(output_ids, coordinates.shape)] = np.add.reduceat(np.ravel(radar_lon)[np.ravel(sort_ids)], np.ravel(sum_ids))
        looks = np.zeros(coordinates.shape)
        looks[np.unravel_index(output_ids, coordinates.shape)] = np.add.reduceat(np.ravel(inside)[np.ravel(sort_ids)], np.ravel(sum_ids))
        import matplotlib.pyplot as plt
        plt.imshow(values_out / looks)
        plt.show()
        """


        return sort_ids, sum_ids, output_ids

    @staticmethod
    def processing_info(coor_out, coor_in='', meta_type='master'):
        # Information on this processing step. meta type should be defined here because this method is not directly
        # connected to either slave/master/ifg/coreg_master data type.

        if not isinstance(coor_in, CoordinateSystem):
            coor_in = CoordinateSystem()
            coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        if not isinstance(coor_out, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        sample = coor_in.sample + coor_out.sample

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        for t in ['lat', 'lon']:
            input_dat[meta_type]['geocode'][t + coor_in.sample]['file'] = [t + coor_in.sample + '.raw']
            input_dat[meta_type]['geocode'][t + coor_in.sample]['coordinates'] = coor_in
            input_dat[meta_type]['geocode'][t + coor_in.sample]['slice'] = coor_in.slice
            input_dat[meta_type]['geocode'][t + coor_in.sample]['coor_change'] = 'resample'

        # line and pixel output files.
        output_dat = recursive_dict()
        for t in ['sort_ids', 'sum_ids', 'output_ids', 'looks']:
            output_dat[meta_type]['conversion_grid'][t + sample]['file'] = [t + sample + '.raw']
            output_dat[meta_type]['conversion_grid'][t + sample]['coordinates'] = coor_out
            output_dat[meta_type]['conversion_grid'][t + sample]['slice'] = coor_out.slice

        # Number of times input data is used in ram.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def add_meta_data(meta, coordinates, coor_in, convert_id_sizes=''):
        # Create meta data for the convert ids.
        # Note that the input coordinates are always radar coordinates and the output coordinates are always
        # geographic or projection coordinates.
        # These are the only files which are based on two coordinate systems and are therefore an exception on all the
        # other types, which are created from the CoordinateSystem object directly.

        if 'conversion_grid' in meta.processes.keys():
            meta_info = meta.processes['conversion_grid']
        else:
            meta_info = OrderedDict()

        if not isinstance(coor_in, CoordinateSystem):
            coor_in = CoordinateSystem()
            coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if len(convert_id_sizes) == 0:
            convert_id_sizes = dict()
            for stp in ['sort_ids', 'sum_ids', 'output_ids']:
                convert_id_sizes[stp] = [0, 0]
        
        data_name = coor_in.sample + coordinates.sample
        meta_info['sort_ids' + data_name + '_output_file'] = 'sort_ids' + data_name + '.raw'
        meta_info['sort_ids' + data_name + '_output_format'] = 'int64'
        meta_info['sort_ids' + data_name + '_lines'] = str(convert_id_sizes['sort_ids'][0])
        meta_info['sort_ids' + data_name + '_pixels'] = str(convert_id_sizes['sort_ids'][1])
        meta_info['sort_ids' + data_name + '_first_line'] = str(1)
        meta_info['sort_ids' + data_name + '_first_pixel'] = str(1)

        for file_type in ['sum_ids', 'output_ids']:
            meta_info[file_type + data_name + '_output_file'] = file_type + data_name + '.raw'
            meta_info[file_type + data_name + '_output_format'] = 'int64'
            meta_info[file_type + data_name + '_lines'] = str(convert_id_sizes[file_type][0])
            meta_info[file_type + data_name + '_pixels'] = str(convert_id_sizes[file_type][1])
            meta_info[file_type + data_name + '_first_line'] = str(1)
            meta_info[file_type + data_name + '_first_pixel'] = str(1)

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

        # Save information of output looks
        meta_info['looks' + coordinates.sample + '_output_file'] = 'looks' + data_name + '.raw'
        meta_info['looks' + coordinates.sample + '_output_format'] = 'int32'
        meta_info['looks' + coordinates.sample + '_lines'] = str(coordinates.shape[0])
        meta_info['looks' + coordinates.sample + '_pixels'] = str(coordinates.shape[1])
        meta_info['looks' + coordinates.sample + '_first_line'] = str(1)
        meta_info['looks' + coordinates.sample + '_first_pixel'] = str(1)

        if coordinates.grid_type == 'geographic':
            meta_info['looks' + coordinates.sample + '_ellipse_type'] = coordinates.ellipse_type
            meta_info['looks' + coordinates.sample + '_lat0'] = str(coordinates.lat0)
            meta_info['looks' + coordinates.sample + '_lon0'] = str(coordinates.lon0)
            meta_info['looks' + coordinates.sample + '_dlat'] = str(coordinates.dlat)
            meta_info['looks' + coordinates.sample + '_dlon'] = str(coordinates.dlon)
        elif coordinates.grid_type == 'projection':
            meta_info['looks' + coordinates.sample + '_projection_type'] = coordinates.projection_type
            meta_info['looks' + coordinates.sample + '_ellipse_type'] = coordinates.ellipse_type
            meta_info['looks' + coordinates.sample + '_proj4_str'] = coordinates.proj4_str
            meta_info['looks' + coordinates.sample + '_x0'] = str(coordinates.lat0)
            meta_info['looks' + coordinates.sample + '_y0'] = str(coordinates.lon0)
            meta_info['looks' + coordinates.sample + '_dx'] = str(coordinates.dlat)
            meta_info['looks' + coordinates.sample + '_dy'] = str(coordinates.dlon)

        meta.image_add_processing_step('conversion_grid', meta_info)

    @staticmethod
    def create_output_files(meta, file_type='', coordinates='', coor_out=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        # We do not know the size, so this part is skipped.
        # meta.images_create_disk('conversion_grid', file_type, coordinates, coor_out)
        pass

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates='', coor_out=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('conversion_grid', file_type, coordinates, coor_out)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates='', coor_out=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('conversion_grid', file_type, coordinates, coor_out)
