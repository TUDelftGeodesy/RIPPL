# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from doris_processing.image_data import ImageData
from doris_processing.orbit_dem_functions.interp_dem import InterpDem
from collections import OrderedDict, defaultdict
import numpy as np
import logging
import os


class RadarDem(InterpDem):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, s_lin=0, s_pix=0, lines=0, interval='', buffer='', buf=3, resolution='SRTM3'):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.meta = meta

        if not isinstance(self.meta, ImageData):
            return

        self.sample, self.interval, self.buffer, self.coors, self.in_coors, self.out_coors = self.get_interval_coors(meta, s_lin, s_pix, lines, interval, buffer)
        in_s_lin = self.in_coors[0]
        in_s_pix = self.in_coors[1]
        in_shape = self.in_coors[2]
        in_shape[0] += 1
        self.out_s_lin = self.out_coors[0]
        self.out_s_pix = self.out_coors[1]
        self.shape = self.out_coors[2]
        self.lines_out = self.coors[0]
        self.pixels_out = self.coors[1]
        self.lines = self.lines_out[self.out_s_lin:self.out_s_lin + self.shape[0]]
        self.pixels = self.pixels_out[self.out_s_pix:self.out_s_pix + self.shape[1]]
        self.first_line = self.lines[0]
        self.first_pixel = self.pixels[0]
        self.last_line = self.lines[-1]
        self.last_pixel = self.pixels[-1]

        self.resolution = resolution
        lin_key = 'Dem_line_' + self.resolution
        pix_key = 'Dem_pixel_' + self.resolution
        dem_key = 'Dem_' + self.resolution

        if lin_key not in self.meta.data_memory['inverse_geocode'].keys() or pix_key not in self.meta.data_memory['inverse_geocode'].keys():
            mem_line = self.meta.data_disk['inverse_geocode'][lin_key]
            mem_pixel = self.meta.data_disk['inverse_geocode'][pix_key]

            shp = mem_line.shape
            # Look for the region from the image we have to load, if not whole file is loaded in memory already.
            s_lin_region = np.max(mem_line, 1) < self.lines[0]
            s_pix_region = np.max(mem_pixel, 0) < self.pixels[0]
            e_lin_region = np.min(mem_line, 1) > self.lines[-1]
            e_pix_region = np.min(mem_pixel, 0) > self.pixels[-1]

            region_lin = np.asarray([s == e for s, e in zip(s_lin_region.ravel(), e_lin_region.ravel())])
            region_pix = np.asarray([s == e for s, e in zip(s_pix_region.ravel(), e_pix_region.ravel())])
            self.s_lin_source = np.maximum(0, np.min(np.argwhere(region_lin)) - buf)
            self.s_pix_source = np.maximum(0, np.min(np.argwhere(region_pix)) - buf)
            self.e_lin_source = np.minimum(shp[0], np.max(np.argwhere(region_lin)) + buf)
            self.e_pix_source = np.minimum(shp[1], np.max(np.argwhere(region_pix)) + buf)

            self.shape_source = [self.e_lin_source - self.s_lin_source, self.e_pix_source - self.s_pix_source]
        else:
            self.s_lin_source = self.meta.data_memory_limits['inverse_geocode'][lin_key][0]
            self.s_pix_source = self.meta.data_memory_limits['inverse_geocode'][lin_key][1]

            self.lines_source = self.meta.data_memory_sizes['inverse_geocode'][lin_key][0]
            self.pixels_source = self.meta.data_memory_sizes['inverse_geocode'][lin_key][1]
            self.shape_source = self.meta.data_memory_sizes['inverse_geocode'][lin_key]

        # Load the input data
        self.dem_line = self.meta.image_load_data_memory('inverse_geocode', self.s_lin_source, self.s_pix_source, self.shape_source, lin_key)
        self.dem_pixel = self.meta.image_load_data_memory('inverse_geocode', self.s_lin_source, self.s_pix_source, self.shape_source, pix_key)
        self.dem = self.meta.image_load_data_memory('import_dem', self.s_lin_source, self.s_pix_source,  self.shape_source, dem_key)

        self.dem_id = np.array([])
        self.first_triangle = np.array([])

        # Initialize the results
        self.radar_dem = np.zeros(self.shape)

    def __call__(self):
        if len(self.dem) == 0 or len(self.dem_line) == 0 or len(self.dem_pixel) == 0:
            print('Missing input data for creating radar dem for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Here the actual geocoding is done.
            # First calculate the heights using an external DEM. This generates the self.height grid..
            self.dem_pixel2grid()

            # Find coordinates and matching interpolation areas
            self.radar_in_dem_grid()

            # Then do the interpolation
            self.dem_barycentric_interpolation()

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.sample, [self.lines_out, self.pixels_out], self.interval, self.buffer)
            self.meta.image_new_data_memory(self.radar_dem, 'radar_dem', self.out_s_lin, self.out_s_pix, file_type='Data' + self.sample)
            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating radar dem for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating radar dem for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_output_files(meta, sample, to_disk=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = ['Data' + sample]

        for s in to_disk:
            meta.image_create_disk('radar_dem', s)

    def save_to_disk(self, to_disk=''):

        if not to_disk:
            to_disk = ['Data' + self.sample]

        for s in to_disk:
            self.meta.image_memory_to_disk('radar_dem', s)

    @staticmethod
    def add_meta_data(meta, sample, coors, interval, buffer):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'radar_dem' in meta.processes.keys():
            meta_info = meta.processes['radar_dem']
        else:
            meta_info = OrderedDict()

        dat = 'Data' + sample
        meta_info[dat + '_output_file'] = 'Dem' + sample + '.raw'
        meta_info[dat + '_output_format'] = 'real4'

        meta_info[dat + '_lines'] = len(coors[0])
        meta_info[dat + '_pixels'] = len(coors[1])
        meta_info[dat + '_first_line'] = str(coors[0][0] + 1)
        meta_info[dat + '_first_pixel'] = str(coors[1][0] + 1)
        meta_info[dat + '_interval_range'] = str(interval[1])
        meta_info[dat + '_interval_azimuth'] = str(interval[0])
        meta_info[dat + '_buffer_range'] = str(buffer[1])
        meta_info[dat + '_buffer_azimuth'] = str(buffer[0])

        meta.image_add_processing_step('radar_dem', meta_info)

    @staticmethod
    def get_interval_coors(meta, s_lin=0, s_pix=0, lines=0, interval='', buffer='', warn=False):

        if len(interval) != 2:
            if interval == '':
                interval = [1, 1]
                int_str = ''
            else:
                print('Interval should be a list with 2 values [interval lines, interval pixels]')
                return
        else:
            if warn:
                print('If you choose to use an interval gecoding and dem creation should be done using the same intervals')
            int_str = 'int_' + str(interval[0]) + '_' + str(interval[1])

        if len(buffer) != 2:
            if buffer == '':
                buffer = [1, 1]
                buf_str = ''
            else:
                print('Buffer should be a list with 2 values [interval lines, interval pixels]')
                return
        else:
            if warn:
                print('If you choose to use an interval gecoding and dem creation should be done using the same buffers')
            buf_str = 'buf_' + str(buffer[0]) + '_' + str(buffer[1])
        interval = interval
        buffer = buffer

        if int_str and buf_str:
            sample = '_' + int_str + '_' + buf_str
        elif int_str and not buf_str:
            sample = '_' + int_str
        elif not int_str and buf_str:
            sample = '_' + buf_str
        else:
            sample = ''

        # First load and create the input data
        orig_s_lin = meta.data_limits['crop']['Data'][0] - buffer[0] - 1
        orig_s_pix = meta.data_limits['crop']['Data'][1] - buffer[1] - 1
        orig_lines = (meta.data_sizes['crop']['Data'][0] + buffer[0] * 2)
        orig_pixels = (meta.data_sizes['crop']['Data'][1] + buffer[1] * 2)

        # Find the coordinate of the first pixel, based on which the new_line and new_pixel are calculated.
        first_line = orig_s_lin
        first_pixel = orig_s_pix
        lines_tot = (first_line + np.arange(orig_lines / interval[0] + 2) * interval[0])
        pixels_tot = (first_pixel + np.arange(orig_pixels / interval[1] + 2) * interval[1])

        shape = [len(lines_tot), len(pixels_tot)]
        if lines != 0:
            shape[0] = lines
        e_lin = s_lin + shape[0]
        e_pix = s_pix + shape[1]

        lines = lines_tot[s_lin:e_lin]
        pixels = pixels_tot[s_pix:e_pix]

        in_coors = [np.min(lines), np.min(pixels), [len(lines), len(pixels)]]
        out_coors = [s_lin, s_pix, [len(lines), len(pixels)]]

        return sample, interval, buffer, [lines_tot, pixels_tot], in_coors, out_coors

    @staticmethod
    def processing_info(sample, resolution):

        # Information on this processing step
        input_dat = defaultdict()
        input_dat['slave']['import_dem'] = ['Dem_' + resolution]
        input_dat['slave']['inverse_geocode'] = ['Dem_line_' + resolution, 'Dem_pixel_' + resolution]

        output_dat = defaultdict()
        output_dat['slave']['radar_dem'] = ['Data' + sample]

        # Number of times input data is used in ram. Bit difficult here but 15 times is ok guess.
        mem_use = 18

        return input_dat, output_dat, mem_use
