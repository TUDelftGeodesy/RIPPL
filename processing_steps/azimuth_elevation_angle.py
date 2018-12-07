# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from image_data import ImageData
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from collections import OrderedDict, defaultdict
from find_coordinates import FindCoordinates
from coordinate_system import CoordinateSystem
from processing_steps.radar_dem import RadarDem
import numpy as np
import logging
import os


class AzimuthElevationAngle(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0, orbit=True, scatterer=True):
        # There are three options for processing:

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # Load coordinates
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates
        self.shape, self.lines, self.pixels = RadarDem.find_coordinates(meta, s_lin, s_pix, lines, coordinates)
        self.sample = coordinates.sample

        # Load data
        x_key = 'X' + self.sample
        y_key = 'Y' + self.sample
        z_key = 'Z' + self.sample
        dem_key = 'radar_DEM' + self.sample
        self.x = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, x_key)
        self.y = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, y_key)
        self.z = self.meta.image_load_data_memory('geocode', self.s_lin, self.s_pix, self.shape, z_key)
        self.height = self.meta.image_load_data_memory('radar_DEM', self.s_lin, self.s_pix, self.shape, dem_key)

        self.no0 = ((self.x != 0) * (self.y != 0) * (self.z != 0))

        self.orbits = OrbitCoordinates(self.meta)
        self.orbits.x = self.x[self.no0]
        self.orbits.y = self.y[self.no0]
        self.orbits.z = self.z[self.no0]
        self.orbits.height = self.height[self.no0]

        # And the output data
        self.orbit = orbit
        self.scatterer = scatterer
        if self.scatterer:
            self.az_angle = np.zeros(self.height.shape).astype(np.float32)
            self.elev_angle = np.zeros(self.height.shape).astype(np.float32)
        if self.orbit:
            self.heading = np.zeros((self.height.shape)).astype(np.float32)
            self.off_nadir_angle = np.zeros(self.height.shape).astype(np.float32)

    def __call__(self):
        # Check if needed data is loaded
        if len(self.x) == 0 or len(self.y) == 0 or len(self.z) == 0 or len(self.height) == 0:
            print('Missing input data for processing azimuth_elevation_angle for ' + self.meta.folder + '. Aborting..')
            return False

        # Here the actual geocoding is done.
        try:
            self.add_meta_data(self.meta, self.coordinates, self.orbit, self.scatterer)

            if np.sum(self.no0) > 0:
                l_id, p_id = np.where(self.no0)
                self.orbits.lp_time(lines=self.lines[l_id], pixels=self.pixels[p_id], regular=False)

                # Then calculate the x,y,z coordinates
                if self.scatterer:
                    self.orbits.xyz2scatterer_azimuth_elevation()
                    self.az_angle[self.no0] = self.orbits.azimuth_angle
                    self.elev_angle[self.no0] = self.orbits.elevation_angle

                if self.orbit:
                    self.orbits.xyz2ell(orbit=True, pixel=False)
                    self.orbits.xyz2orbit_heading_off_nadir()
                    self.heading[self.no0] = self.orbits.heading
                    self.off_nadir_angle[self.no0] = self.orbits.off_nadir_angle

            if self.scatterer:
                # Data can be saved using the create output files and add meta data function.
                self.meta.image_new_data_memory(self.az_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='azimuth_angle' + self.sample)
                self.meta.image_new_data_memory(self.elev_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='elevation_angle' + self.sample)
            if self.orbit:
                # Data can be saved using the create output files and add meta data function.
                self.meta.image_new_data_memory(self.heading, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='heading' + self.sample)
                self.meta.image_new_data_memory(self.off_nadir_angle, 'azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                file_type='off_nadir_angle' + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing azimuth_elevation_angle for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing azimuth_elevation_angle for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates, scatterer=True, orbit=True):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'azimuth_elevation_angle' in meta.processes.keys():
            meta_info = meta.processes['azimuth_elevation_angle']
        else:
            meta_info = OrderedDict()

        if scatterer:
            meta_info = coordinates.create_meta_data(['azimuth_angle', 'elevation_angle'], ['real4', 'real4'], meta_info)
        if orbit:
            meta_info = coordinates.create_meta_data(['heading', 'off_nadir_angle'], ['real4', 'real4'], meta_info)

        meta.image_add_processing_step('azimuth_elevation_angle', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='cmaster', scatterer=True, orbit=True):

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        for t in ['X', 'Y', 'Z']:
            input_dat[meta_type]['geocode'][t + coordinates.sample]['file'] = t + coordinates.sample + '.raw'
            input_dat[meta_type]['geocode'][t + coordinates.sample]['coordinates'] = coordinates
            input_dat[meta_type]['geocode'][t + coordinates.sample]['slice'] = coordinates.slice

        input_dat[meta_type]['radar_DEM']['radar_DEM' + coordinates.sample]['file'] = 'radar_DEM' + coordinates.sample + '.raw'
        input_dat[meta_type]['radar_DEM']['radar_DEM' + coordinates.sample]['coordinates'] = coordinates
        input_dat[meta_type]['radar_DEM']['radar_DEM' + coordinates.sample]['slice'] = coordinates.slice

        file_type = []
        if scatterer:
            file_type.extend(['azimuth_angle', 'elevation_angle'])
        if orbit:
            file_type.extend(['heading', 'off_nadir_angle'])

        # 2 or 4 output files.
        output_dat = recursive_dict()
        for t in file_type:
            output_dat[meta_type]['azimuth_elevation_angle'][t + coordinates.sample]['file'] = t + coordinates.sample + '.raw'
            output_dat[meta_type]['azimuth_elevation_angle'][t + coordinates.sample]['multilook'] = coordinates
            output_dat[meta_type]['azimuth_elevation_angle'][t + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('azimuth_elevation_angle', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('azimuth_elevation_angle', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('azimuth_elevation_angle', file_type, coordinates)
