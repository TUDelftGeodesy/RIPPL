# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from image_data import ImageData
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from find_coordinates import FindCoordinates
from collections import OrderedDict
import os
import logging


class Geocode(object):

    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, s_lin=0, s_pix=0, lines=0, interval='', buffer=''):
        # There are three options for processing:

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.meta = meta

        if not isinstance(self.meta, ImageData):
            return

        self.sample, self.interval, self.buffer, self.coors, self.in_coors, self.out_coors = FindCoordinates.interval_coors(meta, s_lin, s_pix, lines, interval, buffer)
        self.out_s_lin = self.out_coors[0]
        self.out_s_pix = self.out_coors[1]
        self.out_shape = self.out_coors[2]
        self.lines_out = self.coors[0]
        self.pixels_out = self.coors[1]
        self.lines = self.lines_out[self.out_s_lin:self.out_s_lin + self.out_shape[0]]
        self.pixels = self.pixels_out[self.out_s_pix:self.out_s_pix + self.out_shape[1]]

        dat_key = 'Data' + self.sample
        self.height = self.meta.image_load_data_memory('radar_dem',self.out_s_lin, self.out_s_pix, self.out_shape, dat_key)
        self.orbits = OrbitCoordinates(self.meta)
        self.orbits.height = self.height

        self.orbits.lp_time(lines=self.lines, pixels=self.pixels)

        # And the output data
        self.lat = []
        self.lon = []
        self.x_coordinate = []
        self.y_coordinate = []
        self.z_coordinate = []

    def __call__(self):
        # Here the actual geocoding is done.
        # Check if needed data is loaded
        if len(self.height) == 0:
            print('Missing input data for geocoding for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Then calculate the x,y,z coordinates
            self.orbits.lph2xyz()
            self.x_coordinate = self.orbits.x
            self.y_coordinate = self.orbits.y
            self.z_coordinate = self.orbits.z

            # Finally calculate the lat/lon coordinates
            self.orbits.xyz2ell()
            self.lat = self.orbits.lat
            self.lon = self.orbits.lon

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.sample, [self.lines_out, self.pixels_out], self.interval, self.buffer)
            self.meta.image_new_data_memory(self.lat, 'geocode', self.out_s_lin, self.out_s_pix, file_type='Lat' + self.sample)
            self.meta.image_new_data_memory(self.lon, 'geocode', self.out_s_lin, self.out_s_pix, file_type='Lon' + self.sample)
            self.meta.image_new_data_memory(self.x_coordinate, 'geocode', self.out_s_lin, self.out_s_pix, file_type='X' + self.sample)
            self.meta.image_new_data_memory(self.y_coordinate, 'geocode', self.out_s_lin, self.out_s_pix, file_type='Y' + self.sample)
            self.meta.image_new_data_memory(self.z_coordinate, 'geocode', self.out_s_lin, self.out_s_pix, file_type='Z' + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed geocoding for ' +
                              self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed geocoding for ' +
                  self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_output_files(meta, sample, to_disk=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = ['Lat' + sample, 'Lon' + sample, 'X' + sample, 'Y' + sample,
                       'Z' + sample]

        for s in to_disk:
            meta.image_create_disk('geocode', s)

    def save_to_disk(self, to_disk=''):

        if not to_disk:
            to_disk = ['Lat' + self.sample, 'Lon' + self.sample, 'X' + self.sample, 'Y' + self.sample,
                       'Z' + self.sample]

        for s in to_disk:
            self.meta.image_memory_to_disk('geocode', s, )

    @staticmethod
    def add_meta_data(meta, sample, coors, interval, buffer):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'geocode' in meta.processes.keys():
            meta_info = meta.processes['geocode']
        else:
            meta_info = OrderedDict()

        for dat, t in zip(['Lat' + sample, 'Lon' + sample, 'X' + sample, 'Y' + sample,
                           'Z' + sample], ['real4', 'real4', 'real8', 'real8', 'real8']):
            meta_info[dat + '_output_file'] = dat + '.raw'
            meta_info[dat + '_output_format'] = t
            meta_info[dat + '_lines'] = len(coors[0])
            meta_info[dat + '_pixels'] = len(coors[1])
            meta_info[dat + '_first_line'] = str(coors[0][0] + 1)
            meta_info[dat + '_first_pixel'] = str(coors[1][0] + 1)
            meta_info[dat + '_interval_range'] = str(interval[1])
            meta_info[dat + '_interval_azimuth'] = str(interval[0])
            meta_info[dat + '_buffer_range'] = str(buffer[1])
            meta_info[dat + '_buffer_azimuth'] = str(buffer[0])

        meta.image_add_processing_step('geocode', meta_info)

    def processing_info(self):

        # Information on this processing step
        input_dat = dict()
        input_dat['slave']['radar_dem'] = ['Data' + self.sample]

        output_dat = dict()
        output_dat['slave']['geocode'] = ['Lat' + self.sample, 'Lon' + self.sample, 'X' + self.sample,
                                          'Y' + self.sample, 'Z' + self.sample]

        # Number of times input data is used in ram. Bit difficult here but 15 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use
