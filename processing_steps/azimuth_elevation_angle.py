# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from image_data import ImageData
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from collections import OrderedDict
from find_coordinates import FindCoordinates
import logging
import os


class AzimuthElevationAngle(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, input_dem='', dem_database='', s_lin=0, s_pix=0, lines=0, interval='', buffer='', orbit=False, scatterer=True):
        # There are three options for processing:

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.meta = meta

        if not isinstance(self.meta, ImageData):
            return

        self.sample, self.interval, self.buffer, self.coors, self.in_coors, self.out_coors = FindCoordinates.interval_coors(meta, s_lin, s_pix, lines, interval, buffer)
        self.s_lin = self.out_coors[0]
        self.s_pix = self.out_coors[1]
        self.shape = self.out_coors[2]
        self.lines_out = self.coors[0]
        self.pixels_out = self.coors[1]
        self.lines = self.lines_out[self.s_lin:self.s_lin + self.shape[0]]
        self.pixels = self.pixels_out[self.s_pix:self.s_pix + self.shape[1]]
        self.first_line = self.lines[0]
        self.first_pixel = self.pixels[0]
        self.last_line = self.lines[-1]
        self.last_pixel = self.pixels[-1]

        # Load data
        x_key = 'X' + self.sample
        y_key = 'Y' + self.sample
        z_key = 'Z' + self.sample
        dem_key = 'Data' + self.sample
        self.x = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, x_key)
        self.y = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, y_key)
        self.z = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, z_key)
        self.height = self.meta.image_load_data_memory('radar_dem', self.s_lin, self.s_pix, self.shape, dem_key)

        self.orbits = OrbitCoordinates(self.meta)
        self.orbits.x = self.x
        self.orbits.y = self.y
        self.orbits.z = self.z
        self.orbits.height = self.height

        self.orbits.lp_time(lines=self.lines, pixels=self.pixels)

        # And the output data
        self.orbit = orbit
        self.scatterer = scatterer
        if self.scatterer:
            self.az_angle = []
            self.elev_angle = []
        if self.orbit:
            self.heading = []
            self.off_nadir_angle = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.x) == 0 or len(self.y) == 0 or len(self.z) == 0 or len(self.height) == 0:
            print('Missing input data for processing azimuth_elevation_angle for ' + self.meta.folder + '. Aborting..')
            return False

        # Here the actual geocoding is done.
        try:
            self.add_meta_data(self.meta, self.sample, [self.lines_out, self.pixels_out], self.interval, self.buffer, self.scatterer, self.orbit)

            # Then calculate the x,y,z coordinates
            if self.scatterer:
                self.orbits.xyz2scatterer_azimuth_elevation()
                self.az_angle = self.orbits.azimuth_angle
                self.elev_angle = self.orbits.elevation_angle

                # Data can be saved using the create output files and add meta data function.
                self.meta.image_new_data_memory(self.az_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='Azimuth_angle' + self.sample)
                self.meta.image_new_data_memory(self.elev_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='Elevation_angle' + self.sample)
            if self.orbit:
                self.orbits.xyz2ell(orbit=True)
                self.orbits.xyz2orbit_heading_off_nadir()
                self.heading = self.orbits.heading[:, None]
                self.off_nadir_angle = self.orbits.off_nadir_angle

                # Data can be saved using the create output files and add meta data function.
                self.meta.image_new_data_memory(self.heading, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='Heading' + self.sample)
                self.meta.image_new_data_memory(self.off_nadir_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='Off_nadir_angle' + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing azimuth_elevation_angle for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing azimuth_elevation_angle for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_output_files(meta, sample, to_disk='', scatterer=True, orbit=False):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = []
            if scatterer:
                to_disk.extend(['Azimuth_angle' + sample, 'Elevation_angle' + sample])
            if orbit:
                to_disk.extend(['Heading' + sample, 'Off_nadir_angle' + sample])

        for s in to_disk:
            meta.image_create_disk('azimuth_elevation_angle', s)

    def save_to_disk(self, to_disk=''):

        if not to_disk:
            to_disk = []
            if self.scatterer:
                to_disk.extend(['Azimuth_angle' + self.sample, 'Elevation_angle' + self.sample])
            if self.orbit:
                to_disk.extend(['Heading' + self.sample, 'Off_nadir_angle' + self.sample])

        for s in to_disk:
            self.meta.image_memory_to_disk('azimuth_elevation_angle', s)

    @staticmethod
    def add_meta_data(meta, sample, coors, interval, buffer, scatterer=True, orbit=False):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'azimuth_elevation_angle' in meta.processes.keys():
            meta_info = meta.processes['azimuth_elevation_angle']
        else:
            meta_info = OrderedDict()

        files = []
        file_type = []
        if scatterer:
            files.extend(['Azimuth_angle' + sample, 'Elevation_angle' + sample])
            file_type.extend(['real4', 'real4'])
        if orbit:
            files.extend(['Heading' + sample, 'Off_nadir_angle' + sample])
            file_type.extend(['real4', 'real4'])

        for dat, t in zip(files, file_type):
            meta_info[dat + '_output_file'] = dat + '.raw'
            meta_info[dat + '_output_format'] = t

            if not dat.startswith('Heading'):
                meta_info[dat + '_first_line'] = str(coors[0][0] + 1)
                meta_info[dat + '_first_pixel'] = str(coors[1][0] + 1)
                meta_info[dat + '_lines'] = str(len(coors[0]))
                meta_info[dat + '_pixels'] = str(len(coors[1]))
            else:
                meta_info[dat + '_first_line'] = str(coors[0][0] + 1)
                meta_info[dat + '_first_pixel'] = str(1)
                meta_info[dat + '_lines'] = str(len(coors[0]))
                meta_info[dat + '_pixels'] = str(1)
            meta_info[dat + '_interval_range'] = str(interval[1])
            meta_info[dat + '_interval_azimuth'] = str(interval[0])
            meta_info[dat + '_buffer_range'] = str(buffer[1])
            meta_info[dat + '_buffer_azimuth'] = str(buffer[0])

        meta.image_add_processing_step('azimuth_elevation_angle', meta_info)

    def processing_info(self):

        # Information on this processing step
        input_dat = dict()
        input_dat['slave']['geocode'] = ['X' + self.sample, 'Y' + self.sample, 'Z' + self.sample]

        output_dat = dict()
        output_dat['slave']['azimuth_elevation_angle'] = ['Azimuth_angle' + self.sample, 'Elevation_angle' + self.sample,
                                                          'Heading' + self.sample, 'Off_nadir_angle' + self.sample]

        # Number of times input data is used in ram. Bit difficult here but 15 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use
