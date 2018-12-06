# The following class creates an interferogram from a master and slave image.

from image_data import ImageData
from find_coordinates import FindCoordinates
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import numpy as np
import logging
import os
import copy


class Multilook(object):

    def __init__(self, meta, cmaster_meta, step, file_type, coordinates, coor_in=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.
        # When you want to use a certain projection, please give the proj4 string to do the conversion. Most projection
        # descriptions can be found at: spatialreference.org
        # The projection name is used as a shortname for the .res file, to get track of the output files.

        if isinstance(meta, ImageData) and isinstance(cmaster_meta, ImageData):
            self.meta = meta
            self.cmaster = cmaster_meta
        else:
            return

        # If the in coordinates are not defined, we default to the original radar coordinate system. This is the obvious
        # choiche
        if not isinstance(coor_in, CoordinateSystem):
            self.coor_in = CoordinateSystem()
            self.coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        else:
            self.coor_in = coor_in
        if isinstance(coordinates, CoordinateSystem):
            self.coor_out = coordinates

        # Load input data.
        self.data = self.meta.image_load_data_memory(step, 0, 0, self.coor_in.shape, file_type)
        self.step = step
        if len(file_type) == 0:
            self.file_type = step
        else:
            self.file_type = file_type

        self.sort_ids = []
        self.sum_ids = []
        self.output_ids = []

        # Load data (somewhat complicated for the geographical multilooking.
        if self.coor_in.grid_type == 'radar_coordinates' and self.coor_out.grid_type in ['geographic', 'projection']:

            self.use_ids = True

            # Check additional information on geographical multilooking
            convert_sample = self.coor_in.sample + '_' + self.coor_out.sample
            
            sort_ids_shape = self.cmaster.image_get_data_size(step, 'sort_ids' + convert_sample)
            sum_ids_shape = self.cmaster.image_get_data_size(step, 'sum_ids' + convert_sample)
            output_ids_shape = self.cmaster.image_get_data_size(step, 'output_ids' + convert_sample)

            self.sort_ids = self.cmaster.image_load_data_memory('geocode', 0, 0, sort_ids_shape, file_type='sort_ids' + convert_sample)
            self.sum_ids = self.cmaster.image_load_data_memory('geocode', 0, 0, sum_ids_shape, file_type='sum_ids' + convert_sample)
            self.output_ids = self.cmaster.image_load_data_memory('geocode', 0, 0, output_ids_shape, file_type='output_ids' + convert_sample)

        elif self.coor_out.grid_type != self.coor_in.grid_type:
            print('Conversion from geographic or projection grid type to other grid type not supported. Aborting')
            return
        else:
            self.use_ids = False

        # Prepare output
        self.multilooked = []

    def __call__(self):
        if len(self.data) == 0 or (((len(self.sort_ids) == 0 or len(self.sum_ids) == 0 or len(self.output_ids) == 0))
                                   and self.use_ids):
            print('Missing input data for multilooking for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Calculate the multilooked image.
            if self.coor_in.grid_type == 'radar_coordinates' and self.coor_out.grid_type in ['geographic','projection']:
                if self.coor_out.grid_type == 'geographic':
                    self.multilooked = Multilook.radar2geographical(self.data, self.coor_out, self.sort_ids,
                                                                    self.sum_ids, self.output_ids)
                elif self.coor_out.grid_type == 'projection':
                    self.multilooked = Multilook.radar2projection(self.data, self.coor_out, self.sort_ids,
                                                                    self.sum_ids, self.output_ids)
            # If we use the same coordinate system for input and output.
            elif self.coor_in.grid_type == self.coor_out.grid_type:

                if self.coor_in.grid_type == 'radar_coordinates':
                    self.multilooked = Multilook.radar2radar(self.data, self.coor_in, self.coor_out)
                elif self.coor_in.grid_type == 'projection':
                    self.multilooked = Multilook.projection2projection(self.data, self.coor_in, self.coor_out)
                elif self.coor_in.grid_type == 'geographic':
                    self.multilooked = Multilook.geographic2geographic(self.data, self.coor_in, self.coor_out)
            else:
                print('Conversion from a projection or geographic coordinate system to another system is not possible')

            # Save meta data and results
            self.add_meta_data(self.meta, self.coor_out, self.coor_in, self.step, self.file_type)
            self.meta.image_new_data_memory(self.multilooked, self.step, 0, 0, file_type= self.file_type + self.coor_out.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed multilooking for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed multilooking for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def grid_multilooking(values, lin_in, pix_in, lin_out, pix_out, ml_diff, ovr_out):
        # Multilooking of a grid, where the output grid is possibly

        lin_in = list(lin_in)
        pix_in = list(pix_in)

        lin_id = [lin_in.index(x) for x in lin_out]
        pix_id = [pix_in.index(x) for x in pix_out]
        d_lin = (lin_out[-1] - lin_out[-2])
        d_pix = (pix_out[-1] - pix_out[-2])
        last_lin = lin_id[-1] + d_lin
        last_pix = pix_id[-1] + d_pix

        if ovr_out == [1, 1]:
            values_out = np.add.reduceat(np.add.reduceat(values[:last_lin, :last_pix], lin_id), pix_id, axis=1)
        else:
            ovr_lin_id = []
            ovr_pix_id = []
            for lin, pix in zip(lin_id, pix_id):
                ovr_lin_id.append(lin)
                ovr_lin_id.append(lin + d_lin * ml_diff[0])
                ovr_pix_id.append(pix)
                ovr_pix_id.append(pix + d_pix * ml_diff[0])

            last_lin += d_lin * (ml_diff[0] - 1)
            last_pix += d_pix * (ml_diff[1] - 1)
            values_out = np.add.reduceat(np.add.reduceat(values[:last_lin, :last_pix], ovr_lin_id[:-1])[::2,:], ovr_pix_id[:-1], axis=1)[:, ::2]

        return values_out

    @staticmethod
    def radar2radar(values, coor_in, coor_out):
        # In this function we expect that the needed offsets are already applied so the whole dataset is used.
        # Please apply Multilook.find_coordinates first to avoid processing errors.

        # To catch the two cases where either we have to calculate the exact cutoff of our val matrix or when it is
        # already done before, we use two cases:
        # - When only multilook, oversample and offset are given. The cutoff should still be defined.
        # - When also the original shape and if needed s_lin, s_pix and lines are defined. The cutoff is applied already,
        #       so we need some additional info.

        if not isinstance(coor_out, CoordinateSystem) or not isinstance(coor_in, CoordinateSystem):
            print('In and out coordinate system should be a CoordinateSystem object')
            return
        if coor_in.oversample != [1, 1]:
            print('Multilooking of already oversampled data not allowed')
            return

        # Coordinates of lines in both input and output image.
        smp_in, ml_in, ovr_in, off_in, [lin, pix], [lin_in, pix_in] = FindCoordinates.multilook_lines(
            coor_in.shape, 0, 0, 0, 0, 0, coor_in.multilook, coor_in.oversample, coor_in.offset)
        lin_in += coor_in.first_line
        pix_in += coor_in.first_pixel

        smp_out, ml_out, ovr_out, off_out, [lin, pix], [lin_out, pix_out] = FindCoordinates.multilook_lines(
            coor_in.shape, 0, 0, 0, 0, 0, coor_out.multilook, coor_out.oversample, coor_out.offset)
        lin_out += coor_in.first_line
        pix_out += coor_in.first_pixel
        ml_diff = np.array(ml_out) / np.array(ml_in)

        # Check if multilooking is possible
        if not coor_in.ra_time == coor_out.ra_time or not coor_in.az_time == coor_out.az_time:
            print('Start range and azimuth time should be the same')
            return
        if not all(l_out in lin_in for l_out in lin_out) or not all(p_out in pix_in for p_out in pix_out):
            print('All lines/pixels in output grid should exist in input grid')
            return

        values_out = Multilook.grid_multilooking(values, lin_in, pix_in, lin_out, pix_out, ml_diff, ovr_out)

        return values_out

    @staticmethod
    def radar2geographical(values, coordinates, sort_ids, sum_ids, output_ids):
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
            dlat = coordinates.dlat
            dlon = coordinates.dlon
        else:
            # We have to change the stepsize to accommodate the oversampling later on. For example with an oversampling
            # of 2, the new pixel only overlaps 0.5 pixel to all sides. With an odd oversampling factor this is not
            # needed, but still implemented.
            dlat = coordinates.dlat * 0.5
            dlon = coordinates.dlon * 0.5

        # Preallocate the output grid
        values_out = np.zeros(shape=coordinates.shape)

        # Add to output grids.
        values_out[output_ids] = np.diff(np.concatenate([0], np.cumsum(values[sort_ids])[sum_ids]))

        # To create the oversampling we do a second multilooking step on the regular grid created.
        if coordinates.oversample != [1, 1]:
            coor_in = copy.deepcopy(coordinates)
            coor_in.dlat = dlat
            coor_in.dlon = dlon
            coor_in.shape = coordinates.shape * 2

            values_out = Multilook.geographic2geographic(values_out, coor_in, coordinates)

        return values_out

    @staticmethod
    def radar2projection(values, coordinates, sort_ids, sum_ids, output_ids):
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
            dy = coordinates.dy
            dx = coordinates.dx
        else:
            # We have to change the stepsize to accommodate the oversampling later on. For example with an oversampling
            # of 2, the new pixel only overlaps 0.5 pixel to all sides. With an odd oversampling factor this is not
            # needed, but still implemented.
            dy = coordinates.dy * 0.5
            dx = coordinates.dx * 0.5

        # Preallocate the output grid
        values_out = np.zeros(shape=coordinates.shape)

        # Add to output grids.
        values_out[output_ids] = np.diff(np.concatenate([0], np.cumsum(values[sort_ids])[sum_ids]))

        # To create the oversampling we do a second multilooking step on the regular grid created.
        if coordinates.oversample != [1, 1]:
            coor_in = copy.deepcopy(coordinates)
            coor_in.dy = dy
            coor_in.dx = dx
            coor_in.shape = coordinates.shape * 2

            values_out = Multilook.geographic2geographic(values_out, coor_in, coordinates)

        return values_out

    @staticmethod
    def geographic2geographic(values, coor_in, coor_out):
        # Multilook a geographic to another geographic coordinate system. O

        if not isinstance(coor_out, CoordinateSystem) or not isinstance(coor_in, CoordinateSystem):
            print('In and out coordinate system should be a CoordinateSystem object')
            return
        if coor_in.oversample != [1, 1]:
            print('Multilooking of already oversampled data not allowed')
            return

        # Resolution conversion between in and out coordinates.
        res_change = np.array([int(coor_out.dlat / coor_in.dlat), int(coor_out.dlon / coor_in.dlon)])

        # Coordinates of lines in both input and output image.
        smp_in, ml_in, ovr_in, off_in, [lin, pix], [lin_in, pix_in] = FindCoordinates.multilook_lines(
            coor_in.shape, 0, 0, 0, [1, 1], coor_in.oversample, [0, 0])
        lin_in += coor_in.first_line
        pix_in += coor_in.first_pixel

        smp_out, ml_out, ovr_out, off_out, [lin, pix], [lin_out, pix_out] = FindCoordinates.multilook_lines(
            coor_out.shape, 0, 0, 0, res_change, coor_out.oversample, [0, 0])
        lin_out += coor_out.first_line
        pix_out += coor_out.first_pixel
        ml_diff = np.array(ml_out) / np.array(ml_in)

        # Check if multilooking is possible
        if not coor_in.lat0 == coor_out.lat0 or not coor_in.lon0 == coor_out.lon0:
            print('Start latitude and longitude should be the same')
            return
        if not all(l_out in lin_in for l_out in lin_out) or not all(p_out in pix_in for p_out in pix_out):
            print('All lines/pixels in output grid should exist in input grid')
            return

        values_out = Multilook.grid_multilooking(values, lin_in, pix_in, lin_out, pix_out, ml_diff, ovr_out)

        return values_out

    @staticmethod
    def projection2projection(values, coor_in, coor_out):

        if not isinstance(coor_out, CoordinateSystem) or not isinstance(coor_in, CoordinateSystem):
            print('In and out coordinate system should be a CoordinateSystem object')
            return
        if coor_in.oversample != [1, 1]:
            print('Multilooking of already oversampled data not allowed')
            return

        # Resolution conversion between in and out coordinates.
        res_change = np.array([int(coor_out.dy / coor_in.dy), int(coor_out.dx / coor_in.dx)])

        # Coordinates of lines in both input and output image.
        smp_in, ml_in, ovr_in, off_in, [lin, pix], [lin_in, pix_in] = FindCoordinates.multilook_lines(
            coor_in.shape, 0, 0, 0, [1, 1], coor_in.oversample, [0, 0])
        lin_in += coor_in.first_line
        pix_in += coor_in.first_pixel

        smp_out, ml_out, ovr_out, off_out, [lin, pix], [lin_out, pix_out] = FindCoordinates.multilook_lines(
            coor_out.shape, 0, 0, 0, res_change, coor_out.oversample, [0, 0])
        lin_out += coor_out.first_line
        pix_out += coor_out.first_pixel
        ml_diff = np.array(ml_out) / np.array(ml_in)

        # Check if multilooking is possible
        if not coor_in.y0 == coor_out.y0 or not coor_in.x0 == coor_out.x0:
            print('Start latitude and longitude should be the same')
            return
        if not all(l_out in lin_in for l_out in lin_out) or not all(p_out in pix_in for p_out in pix_out):
            print('All lines/pixels in output grid should exist in input grid')
            return

        values_out = Multilook.grid_multilooking(values, lin_in, pix_in, lin_out, pix_out, ml_diff, ovr_out)

        return values_out

    @staticmethod
    def processing_info(coor_out, coor_in='', step='earth_topo_phase', file_type=''):
        # Information on this processing step. meta type should be defined here because this method is not directly
        # connected to either slave/master/ifg/coreg_master data type.

        if not isinstance(coor_in, CoordinateSystem):
            coor_in = CoordinateSystem()
            coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        if not isinstance(coor_out, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if file_type == '':
            file_type = step

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()

        input_dat['slave'][step][file_type + coor_in.sample]['file'] = file_type + coor_in.sample + '.raw'
        input_dat['slave'][step][file_type + coor_in.sample]['coordinates'] = coor_in
        input_dat['slave'][step][file_type + coor_in.sample]['slice'] = coor_in.slice

        if coor_in.grid_type == 'radar_coordinates' and coor_out.grid_type in ['geographic', 'projection']:

            conv_sample = coor_in.sample + '_' + coor_out.sample

            for t in ['sort_ids', 'sum_ids', 'output_ids']:
                input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle' + coor_out.sample]['file'] = t + conv_sample + '.raw'
                input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle' + coor_out.sample]['coordinates'] = coor_out
                input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle' + coor_out.sample]['slice'] = coor_out.slice

        # line and pixel output files.
        output_dat = recursive_dict()
        input_dat['slave'][step][file_type + coor_out.sample]['file'] = file_type + coor_out.sample + '.raw'
        input_dat['slave'][step][file_type + coor_out.sample]['coordinates'] = coor_out
        input_dat['slave'][step][file_type + coor_out.sample]['slice'] = coor_out.slice

        # Number of times input data is used in ram.
        mem_use = 2

        return input_dat, output_dat, mem_use

    @staticmethod
    def add_meta_data(meta, coordinates, coor_in, step, file_type=''):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem) or not isinstance(coor_in, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if step in meta.processes.keys():
            meta_info = meta.processes[step]
        else:
            meta_info = OrderedDict()

        if not file_type:
            file_type = step

        data_types = [meta.data_types[step][file_type + coor_in.sample]]
        meta_info = coordinates.create_meta_data([file_type], data_types, meta_info)
        meta.image_add_processing_step(step, meta_info)

    @staticmethod
    def create_output_files(meta, step, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk(step, file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, step, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk(step, file_type, coordinates)

    @staticmethod
    def clear_memory(meta, step, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory(step, file_type, coordinates)


