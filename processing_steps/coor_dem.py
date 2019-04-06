# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from rippl.image_data import ImageData
from rippl.orbit_resample_functions.resample import Resample
from collections import OrderedDict, defaultdict
from rippl.coordinate_system import CoordinateSystem
from rippl.processing_steps.projection_coor import ProjectionCoor
import numpy as np
import logging
import os


class CoorDem(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0, coor_in='', buf=3):
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

        # Check for input DEM type
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

            coor_in = self.meta.read_res_coordinates('import_DEM', [dem_type])[0]

        # Radar coordinates are not possible for this function, so these are excluded.
        if self.coordinates.grid_type == 'radar_coordinates':
            print('Using radar coordinates for the coor_dem function is not possible. Use the coor_DEM function')
            return

        # Check if we work with a projection and load lat/lon coordinates if so.
        self.lat, self.lon = ProjectionCoor.load_lat_lon(self.coordinates, self.meta, self.s_lin, self.s_pix, self.shape)

        # Now load the needed part of the DEM
        lat0 = coor_in.lat0 + coor_in.first_line * coor_in.dlat
        dem_lats = np.linspace(lat0, lat0 + coor_in.dlat * (coor_in.shape[0] - 1), coor_in.shape[0])
        lon0 = coor_in.lon0 + coor_in.first_line * coor_in.dlat
        dem_lons = np.linspace(lon0, lon0 + coor_in.dlat * (coor_in.shape[1] - 1), coor_in.shape[1])

        lat_lim = [np.maximum(0, np.min(np.argwhere(dem_lats >= np.min(self.lat))) - 1 - buf),
                   np.minimum(coor_in.shape[0], np.max(np.argwhere(dem_lats <= np.max(self.lat))) + 1 + buf)]
        lon_lim = [np.maximum(0, np.min(np.argwhere(dem_lons >= np.min(self.lon))) - 1 - buf),
                   np.minimum(coor_in.shape[1], np.max(np.argwhere(dem_lons <= np.max(self.lon))) + 1 + buf)]

        self.dem_in_lat0 = dem_lats[lat_lim[1]]
        self.dem_in_lon0 = dem_lons[lon_lim[0]]
        self.coor_in = coor_in
        self.dem = meta.image_load_data_memory('import_DEM', lat_lim[0], lon_lim[0], [lat_lim[1] - lat_lim[0], lon_lim[1] - lon_lim[0]], 'DEM' + coor_in.sample)

    def __call__(self):
        if len(self.dem) == 0 or len(self.lat) == 0 or len(self.lon) == 0:
            print('Missing input data for creating radar DEM for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Resample the new grid based on the coordinates of the old grid based on a bilinear approximation.
            lines = (self.lat - self.dem_in_lat0) / -self.coor_in.dlat
            pixels = (self.lon - self.dem_in_lon0) / self.coor_in.dlon

            self.coor_DEM = Resample.resample_grid(self.dem, lines, pixels, 'linear').astype(np.float32)

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.coor_DEM, 'coor_DEM', self.s_lin, self.s_pix, file_type='DEM' + self.sample)
            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating coor DEM for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating coor DEM for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'coor_DEM' in meta.processes.keys():
            meta_info = meta.processes['coor_DEM']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['DEM'], ['real4'], meta_info)

        meta.image_add_processing_step('coor_DEM', meta_info)

    @staticmethod
    def processing_info(coor_out, coor_in='', meta_type='cmaster'):

        # Three input files needed Dem, Dem_line and Dem_pixel
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()

        input_dat[meta_type]['import_DEM']['DEM' + coor_in.sample]['file'] = 'DEM' + coor_in.sample + '.raw'
        input_dat[meta_type]['import_DEM']['DEM' + coor_in.sample]['coordinates'] = coor_in
        input_dat[meta_type]['import_DEM']['DEM' + coor_in.sample]['slice'] = coor_in.slice
        input_dat[meta_type]['import_DEM']['DEM' + coor_in.sample]['coor_change'] = 'resample'

        if coor_out.grid_type == 'projection':
            for dat_type in ['lat', 'lon']:
                input_dat[meta_type]['projection_coor'][dat_type + coor_out.sample]['file'] = dat_type + coor_out.sample + '.raw'
                input_dat[meta_type]['projection_coor'][dat_type + coor_out.sample]['coordinates'] = coor_out
                input_dat[meta_type]['projection_coor'][dat_type + coor_out.sample]['slice'] = coor_out.slice
                input_dat[meta_type]['projection_coor'][dat_type + coor_out.sample]['coor_change'] = 'resample'

        # One output file created radar dem
        output_dat = recursive_dict()
        output_dat[meta_type]['coor_DEM']['DEM' + coor_out.sample]['files'] = 'DEM' + coor_out.sample + '.raw'
        output_dat[meta_type]['coor_DEM']['DEM' + coor_out.sample]['coordinate'] = coor_out
        output_dat[meta_type]['coor_DEM']['DEM' + coor_out.sample]['slice'] = coor_out.slice

        # Number of times input data is used in ram. Bit difficult here but 15 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('coor_DEM', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('coor_DEM', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('coor_DEM', file_type, coordinates)

    @staticmethod
    def load_dem(coordinates, meta, s_lin, s_pix, shape):

        dem_key = 'DEM' + coordinates.sample
        if coordinates.grid_type == 'radar_coordinates':
            height = meta.image_load_data_memory('radar_DEM', s_lin, s_pix, shape, dem_key)
        elif coordinates.grid_type in ['projection', 'geographic']:
            height = meta.image_load_data_memory('coor_DEM', s_lin, s_pix, shape, dem_key)
        else:
            print('DEM values could not be loaded with this function!')
            return

        return height

    @staticmethod
    def dem_processing_info(input_dat, coordinates, type='cmaster', resample=False):
        # Load the lat/lon info for the processing step.

        if coordinates.grid_type == 'radar_coordinates':
            step = 'radar_DEM'
        elif coordinates.grid_type in ['projection', 'geographic']:
            step = 'coor_DEM'

        # For multiprocessing this information is needed to define the selected area to deramp.
        input_dat[type][step]['DEM' + coordinates.sample]['file'] = 'DEM' + coordinates.sample + '.raw'
        input_dat[type][step]['DEM' + coordinates.sample]['coordinates'] = coordinates
        input_dat[type][step]['DEM' + coordinates.sample]['slice'] = coordinates.slice
        if resample:
            input_dat[type][step]['DEM' + coordinates.sample]['coor_change'] = 'resample'

        return input_dat