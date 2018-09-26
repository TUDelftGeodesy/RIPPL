# The following class creates an interferogram from a master and slave image.

from image_data import ImageData
import pyproj
from find_coordinates import FindCoordinates
from collections import OrderedDict, defaultdict
import numpy as np
import logging
import os


class Multilook(object):
    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, step, data_type, s_lin=0, s_pix=0, lines=0, multilook='', oversample='', offset='',
                 geo_multilook=False, coreg_meta='', projection='WGS84', proj4='+proj=longlat +ellps=WGS84 +datum=WGS84',
                 y_lim=[0, 0], x_lim=[0, 0], y_step=0.01, x_step=0.01):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.
        # When you want to use a certain projection, please give the proj4 string to do the conversion. Most projection
        # descriptions can be found at: spatialreference.org
        # The projection name is used as a shortname for the .res file, to get track of the output files.

        if isinstance(meta, ImageData):
            self.meta = meta

        # Find input shape
        self.shape = self.meta.image_get_data_size(step, data_type)
        self.geo_multilook = geo_multilook
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.lines = lines
        self.step = step
        self.data_type = data_type

        # Find the coordinates of input and output.
        if self.geo_multilook:
            if isinstance(coreg_meta, str):
                if len(coreg_meta) != 0:
                    self.meta = ImageData(coreg_meta, 'single')
            elif isinstance(coreg_meta, ImageData):
                self.coreg_meta = coreg_meta

            self.y_lim = y_lim
            self.x_lim = x_lim
            self.y_step = y_step
            self.x_step = x_step
            self.projection = projection
            self.proj4 = proj4

            if lines != 0:
                l = np.minimum(lines, self.shape[0] - s_lin)
            else:
                l = self.shape[0] - s_lin
            self.shape = [l, self.shape[1] - s_pix]

        else:
            sample, multilook, oversample, offset, in_shape, out_shape = Multilook.find_coordinates(
                self.shape, s_lin=s_lin, s_pix=s_pix, lines=lines, multilook=multilook, oversample=oversample, offset=offset)
            self.multilook = multilook
            self.offset = offset
            self.oversample = oversample
            self.sample = sample

        # Load data (somewhat complicated for the geographical multilooking.
        if self.geo_multilook:
            self.data = self.meta.image_load_data_memory(step, s_lin, s_pix, self.shape, data_type)
            self.lat = self.coreg_meta.image_load_data_memory('geocode', s_lin, s_pix, self.shape, 'Lat')
            self.lon = self.coreg_meta.image_load_data_memory('geocode', s_lin, s_pix, self.shape, 'Lon')

            # Check additional information on geographical multilooking
            if len(x_lim) == 2 and len(y_lim) == 2:
                out_size = [np.round(np.diff(y_lim)[0]), np.round(np.diff(x_lim)[0])]
                self.sample = '_' + projection + '_' + str(y_lim[0]) + '_' + str(x_lim[1]) + '_' + \
                         str(y_step) + '_' + str(x_step) + '_' + str(out_size[0]) + '_' + str(out_size[1])

                sort_ids_shape = self.meta.image_get_data_size(step, 'Sort_ids' + self.sample)
                output_ids_shape = self.meta.image_get_data_size(step, 'Sum_ids' + self.sample)

                self.sort_ids = self.coreg_meta.image_load_data_memory('geocode', 0, 0, sort_ids_shape, file_type='Sort_ids' + self.sample)
                self.sum_ids = self.coreg_meta.image_load_data_memory('geocode', 0, 0, output_ids_shape, file_type='Sum_ids' + self.sample)
                self.output_ids = self.coreg_meta.image_load_data_memory('geocode', 0, 0, output_ids_shape, file_type='Output_ids' + self.sample)

                self.convert_ids = [self.sort_ids, self.sum_ids, self.output_ids]

        else:
            self.data = self.meta.image_load_data_memory(step, in_shape[0], in_shape[1], in_shape[2], data_type)

        # Prepare output
        self.multilooked = []

    def __call__(self):
        if len(self.data) == 0:
            print('Missing input data for multilooking for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Calculate the multilooked image.
            if self.geo_multilook:

                if len(self.sort_ids) > 0 and len(self.sum_ids) > 0 and len(self.output_ids) > 0:
                    self.multilooked, self.convert_ids, dat_lim = Multilook.multilook_geographical(
                        self.lat, self.lon, self.data, self.y_step, self.x_step, self.y_lim, self.x_lim, self.projection)

                    # Save the convert ids
                    sort_id_size = self.convert_ids[0].shape
                    output_id_size = self.convert_ids[1].shape
                    self.create_geocode_meta_data(self.coreg_meta, dat_lim[0], dat_lim[1], self.y_step,
                                                  self.x_step, projection=self.projection, coor_size=self.shape,
                                                  sort_id_size=sort_id_size, output_id_size=output_id_size)
                    self.meta.image_new_data_memory(self.convert_ids[0], 'geocode', 0, 0, file_type='Sort_ids' + self.sample)
                    self.meta.image_new_data_memory(self.convert_ids[1], 'geocode', 0, 0, file_type='Output_ids' + self.sample)
                    self.meta.image_new_data_memory(self.convert_ids[2], 'geocode', 0, 0, file_type='Sum_ids' + self.sample)

                else:
                    self.multilooked, convert, dat_lim = Multilook.multilook_geographical(self.lat, self.lon,
                        self.data, self.y_step, self.x_step, self.y_lim, self.x_lim, self.projection, self.convert_ids)

                # Save meta data and results
                self.create_geographical_meta_data(self.meta, self.step, dat_lim[0], dat_lim[1], self.y_step,
                                                   self.x_step, projection=self.projection, data_type=self.data_type)
                self.meta.image_new_data_memory(self.multilooked, self.step, 0, 0, file_type=self.data_type + self.sample)


            else:
                self.multilooked = Multilook.multilook_regular(self.data, self.shape, s_lin=self.s_lin,
                                                               s_pix=self.s_pix, lines=self.lines,
                                                               multilook=self.multilook, oversample=self.oversample)

                # Save meta data and results
                self.create_meta_data(self.meta, self.step, self.data_type, self.multilook, self.oversample, self.offset)
                self.meta.image_new_data_memory(self.multilooked, self.step, self.s_lin, 0, file_type=self.data_type + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed multilooking for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed multilooking for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def multilook_regular(val, original_shape='', s_lin=0, s_pix=0, lines=0, multilook='', oversample='', offset=''):
        # In this function we expect that the needed offsets are already applied so the whole dataset is used.
        # Please apply Multilook.find_coordinates first to avoid processing errors.

        # To catch the two cases where either we have to calculate the exact cutoff of our val matrix or when it is
        # already done before, we use two cases:
        # - When only multilook, oversample and offset are given. The cutoff should still be defined.
        # - When also the original shape and if needed s_lin, s_pix and lines are defined. The cutoff is applied already,
        #       so we need some additional info.

        if len(original_shape) == 0:
            dat_shape = val.shape
        else:
            dat_shape = original_shape

        sample, multilook, oversample, offset, in_shape, out_shape = Multilook.find_coordinates(
            dat_shape, s_lin=s_lin, s_pix=s_pix, lines=lines, multilook=multilook, oversample=oversample, offset=offset)

        if len(original_shape) == 0:
            ls = 0
            ps = 0
        else:
            ls = in_shape[0]
            ps = in_shape[1]

        shape = in_shape[2]
        le = ls + shape[0]
        pe = ps + shape[1]

        if oversample == [1, 1]:
            ir = np.arange(0, shape[0], multilook[0])
            jr = np.arange(0, shape[1], multilook[1])

            ml_out = np.add.reduceat(np.add.reduceat(val[ls:le, ps:pe], ir), jr, axis=1)
        else:
            ir_h = np.arange(0, shape[0], multilook[0] / oversample[0])[:-(oversample[0] - 1)]
            ir = [[i, i + multilook[0]] for i in ir_h]
            ir = [item for sublist in ir for item in sublist][:-1]
            jr_h = np.arange(0, shape[1], multilook[1] / oversample[1])[:-(oversample[1] - 1)]
            jr = [[i, i + multilook[1]] for i in jr_h]
            jr = [item for sublist in jr for item in sublist][:-1]

            ml_out = np.add.reduceat(np.add.reduceat(val[ls:le, ps:pe], ir)[::2,:], jr, axis=1)[:, ::2]

        return ml_out

    @staticmethod
    def multilook_geographical(lat, lon, val, y_step, x_step, y_lim=[0, 0], x_lim=[0, 0], projection='',  convert_ids=[]):
        # This method does a geographical multilooking. This means that the multilooking window is defined by the
        # geographical location of the data point.
        # - In first instance we will assume that we convert to a regular lat/lon grid with regular steps in latitude
        #   and longitude. If the projection definition, we assume that y_lim/latlon and y_step/x_step change accordingly
        # - The borders are defined by the chosen latitude and longitude limits. If these are not defined, it will be
        #   based on the extend of the lat/lon grid.
        # To find the needed proj4 string for your projection you could search for it on spatialreference.org

        if len(convert_ids) == 0:

            # First check whether a specific projection is defined.
            if len(projection) > 0:
                # During geocoding we always convert the WGS84 projections
                proj_in = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
                proj_out = pyproj.Proj(projection)

                x, y = pyproj.transform(proj_in, proj_out, lon, lat)
            else:
                x = lon
                y = lat

            # If y_lim and/or x_lim is not defined calculate them.
            # Keep in mind that the y_lim, x_lim, y_step and x_step can mean both distances in degrees or meters,
            # based on the chosen projection.
            if y_lim == [0, 0]:
                y_lim = [np.floor(np.min(y) / y_step) * y_step, np.ceil(np.max(y) / y_step) * y_step]
            if x_lim == [0, 0]:
                x_lim = [np.floor(np.min(x) / x_step) * x_step, np.ceil(np.max(x) / x_step) * x_step]
            out_size = [np.round(np.diff(y_lim)[0]), np.round(np.diff(x_lim)[0])]

            # Select all pixels inside boundaries.
            inside = (y_lim[0] < lat < y_lim[1]) * (x_lim[0] < lon < x_lim[1])

            # Calculate the coordinates of the new pixels and find the pixels outside the given boundaries.
            lat = np.int32((lat - y_lim[0]) / y_step)
            lon = np.int32((lon - x_lim[0]) / x_step)
            flat_id = lat * out_size[1] + lon
            del lat, lon

            # Sort ids and find number of pixels in every grid cell
            sort_ids = np.argsort(np.ravel(flat_id))[np.ravel(inside)]
            [out_ids, no_ids] = np.unique(flat_id[sort_ids], return_counts=True)
            sum_ids = np.cumsum(no_ids) - 1

        else:
            [sort_ids, out_ids, sum_ids] = convert_ids
            out_size = [np.round(np.diff(y_lim)[0]), np.round(np.diff(x_lim)[0])]

        # Preallocate the output grid
        ml_grid = np.zeros(shape=out_size)

        # Add to output grids.
        ml_grid[out_ids] = np.diff(np.concatenate([0], np.cumsum(val[sort_ids])[sum_ids]))

        return ml_grid, [sort_ids, out_ids, sum_ids], [y_lim, x_lim, out_size]

    @staticmethod
    def find_coordinates(meta, s_lin=0, s_pix=0, lines=0, multilook='', oversample='', offset=''):

        sample, ml, ovr, off, in_shape, out_shape = \
            FindCoordinates.multilook_coors(meta, s_lin, s_pix, lines, multilook, oversample, offset)

        return sample, multilook, oversample, offset, in_shape, out_shape

    @staticmethod
    def input_output_info(meta, step, meta_type='slave', data_type='Data', multilook='', oversample='', offset=''):
        # Information on this processing step. meta type should be defined here because this method is not directly
        # connected to either slave/master/ifg/coreg_master data type.

        slice = meta.processes['crop']['Data_slice']
        sample, multilook, oversample, offset = FindCoordinates(multilook, oversample, offset)

        input_dat = defaultdict()
        input_dat[meta_type][step] = dict()
        input_dat[meta_type][step]['files'] = [data_type + '.raw']
        input_dat[meta_type][step]['multilook'] = [1, 1]
        input_dat[meta_type][step]['oversample'] = [1, 1]
        input_dat[meta_type][step]['offset'] = [0, 0]
        input_dat[meta_type][step]['slice'] = slice

        output_dat = defaultdict()
        output_dat[meta_type][step] = dict()
        output_dat[meta_type][step]['files'] = [data_type + sample + '.raw']
        output_dat[meta_type][step]['multilook'] = multilook
        output_dat[meta_type][step]['oversample'] = oversample
        output_dat[meta_type][step]['offset'] = offset
        output_dat[meta_type][step]['slice'] = slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 3

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_meta_data(meta, step, data_type='Data', multilook='', oversample='', offset=''):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.

        if step in meta.processes.keys():
            meta_info = meta.processes[step]
        else:
            meta_info = OrderedDict()

        sample, multilook, oversample, offset, in_shape, out_shape = \
            Multilook.find_coordinates(meta, multilook=multilook, oversample=oversample, offset=offset)

        dat = data_type + sample
        meta_info[dat + '_output_file'] = dat + '.raw'
        meta_info[dat + '_output_format'] = meta.processes[step][data_type + '_output_format']

        meta_info[dat + '_lines'] = str(out_shape[2][0])
        meta_info[dat + '_pixels'] = str(out_shape[2][1])
        meta_info[dat + '_first_line'] = meta.processes[step][data_type + '_first_line']
        meta_info[dat + '_first_pixel'] = meta.processes[step][data_type + '_first_pixel']
        meta_info[dat + '_multilook_azimuth'] = str(multilook[0])
        meta_info[dat + '_multilook_range'] = str(multilook[1])
        meta_info[dat + '_oversampling_azimuth'] = str(oversample[0])
        meta_info[dat + '_oversampling_range'] = str(oversample[1])
        meta_info[dat + '_offset_azimuth'] = str(offset[0])
        meta_info[dat + '_offset_range'] = str(offset[1])

        meta.image_add_processing_step(step, meta_info)

    @staticmethod
    def create_geographical_meta_data(meta, step, y_lim, x_lim, y_step, x_step, data_type='', projection='WGS84'):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing. This means that in the case of geographical processing the latitude and
        # longitude limits and steps should be known beforehand.

        if step in meta.processes.keys():
            meta_info = meta.processes[step]
        else:
            meta_info = OrderedDict()

        out_shape = [np.round(np.diff(y_lim)[0] / y_step), np.round(np.diff(x_lim)[0] / x_step)]
        sample = '_' + projection + '_' + str(y_lim[0]) + '_' + str(x_lim[1]) + '_' + \
                 str(y_step) + '_' + str(x_step) + '_' + str(out_shape[0]) + '_' + str(out_shape[1])

        dat = data_type + sample
        meta_info[dat + '_output_file'] = dat + '.raw'
        meta_info[dat + '_output_format'] = meta.processes[step][data_type + '_output_format']

        meta_info[dat + '_lines'] = str(out_shape[2][0])
        meta_info[dat + '_pixels'] = str(out_shape[2][1])
        meta_info[dat + '_min_latitude'] = str(y_lim[0])
        meta_info[dat + '_min_longitude'] = str(x_lim[0])
        meta_info[dat + '_y_step'] = str(y_step)
        meta_info[dat + '_x_step'] = str(x_step)

        meta.image_add_processing_step(step, meta_info)

    @staticmethod
    def create_geocode_meta_data(meta, y_lim, x_lim, y_step, x_step, projection='WGS84', coor_size=[0,0],
                                 sort_id_size=0, output_id_size=0):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing. This means that in the case of geographical processing the latitude and
        # longitude limits and steps should be known beforehand.

        if 'geocode' in meta.processes.keys():
            meta_info = meta.processes['geocode']
        else:
            meta_info = OrderedDict()

        out_shape = [np.round(np.diff(y_lim)[0] / y_step), np.round(np.diff(x_lim)[0] / x_step)]
        sample = '_' + projection + '_' + str(y_lim[0]) + '_' + str(x_lim[1]) + '_' + \
                 str(y_step) + '_' + str(x_step) + '_' + str(out_shape[0]) + '_' + str(out_shape[1])

        dat = 'Sort_ids' + sample
        meta_info[dat + '_output_file'] = dat + '.raw'
        meta_info[dat + '_output_format'] = 'int32'

        meta_info[dat + '_lines'] = str(1)
        meta_info[dat + '_pixels'] = str(sort_id_size)
        meta_info[dat + '_coverage_percentage'] = str(sort_id_size * 100 / (coor_size[0] * coor_size[1]))
        meta_info['Data_first_line'] = meta.processes['crop']['Data_first_line']
        meta_info['Data_first_pixel'] = meta.processes['crop']['Data_first_pixel']

        for dat in ['Output_ids' + sample, 'Sum_ids' + sample]:
            meta_info[dat + '_output_file'] = dat + '.raw'
            meta_info[dat + '_output_format'] = 'int32'

            meta_info[dat + '_lines'] = str(1)
            meta_info[dat + '_pixels'] = str(output_id_size)
            meta_info[dat + '_coverage_percentage'] = str(output_id_size * 100 / (out_shape[2][0] * out_shape[2][1]))
            meta_info[dat + '_min_latitude'] = str(y_lim[0])
            meta_info[dat + '_min_longitude'] = str(x_lim[0])
            meta_info[dat + '_y_step'] = str(y_step)
            meta_info[dat + '_x_step'] = str(x_step)

        meta.image_add_processing_step('coreg', meta_info)

    @staticmethod
    def create_output_files(meta, step, data_type='Data'):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        meta.image_create_disk(step, data_type)

    @staticmethod
    def create_coreg_output_file():

    @staticmethod
    def save_output_files(meta, ):

    @staticmethod
    def save_coreg_output_files(meta, ):