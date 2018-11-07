# The following class does the coregistration and resampling of a slave image to the geometry of a master image
# The coregistration includes the following main steps:
#   - Calculate the geometry of the master image and if needed the dem of the master based on the master
#   - Calculate the line and pixel coordinates of the master image in the slave line and pixel coordinates
#   - (optional) Create a mask for the areas used to check shift using cross-correlation
#   - (optional) Calculate the window shifts for large or small windows to verify the geometrical shift
#   - (optional) Adapt calculated geometrical shift using windows.
#   - Resample the slave grid to master coordinates
#   - (optional) Calculate the baselines between the two images

from image_data import ImageData
from orbit_dem_functions.orbit_interpolate import OrbitInterpolate
from radar_dem import RadarDem
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import numpy as np
import os
import logging
import datetime


class Baseline(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, cmaster_meta, meta, coordinates, s_lin=0, s_pix=0, lines=0,
                 perpendicular=True, parallel=False, horizontal=False, vertical=False, total_baseline=False, angle=False):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, ImageData) and isinstance(cmaster_meta, ImageData):
            self.slave = meta
            self.cmaster = cmaster_meta
        else:
            return

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates
        self.sample = self.coordinates.sample
        self.shape, self.lines, self.pixels = RadarDem.find_coordinates(self.cmaster, s_lin, s_pix, lines, coordinates)

        # Information on conversion from range/azimuth to distances.
        sol = 299792458  # speed of light [m/s]
        self.ra2m = 1 / float(self.cmaster.processes['readfiles']['Range_sampling_rate (computed, MHz)']) / 1000000 * sol

        az_time = self.cmaster.processes['readfiles']['First_pixel_azimuth_time (UTC)']
        az_seconds = (datetime.datetime.strptime(az_time, '%Y-%m-%dT%H:%M:%S.%f') -
                           datetime.datetime.strptime(az_time[:10], '%Y-%m-%d'))
        self.master_az_time = az_seconds.seconds + az_seconds.microseconds / 1000000.0
        self.master_az_step = 1 / float(self.cmaster.processes['readfiles']['Pulse_Repetition_Frequency (computed, Hz)'])

        az_time = self.slave.processes['readfiles']['First_pixel_azimuth_time (UTC)']
        az_seconds = (datetime.datetime.strptime(az_time, '%Y-%m-%dT%H:%M:%S.%f') -
                           datetime.datetime.strptime(az_time[:10], '%Y-%m-%d'))
        self.slave_az_time = az_seconds.seconds + az_seconds.microseconds / 1000000.0
        self.slave_az_step = 1 / float(self.slave.processes['readfiles']['Pulse_Repetition_Frequency (computed, Hz)'])

        # Load data
        self.ra_shift = self.cmaster.image_load_data_memory('geometrical_coreg', self.s_lin, self.s_pix, self.shape, 'new_pixel' + self.sample)
        self.az_shift = self.cmaster.image_load_data_memory('geometrical_coreg', self.s_lin, self.s_pix, self.shape, 'new_line' + self.sample)

        if horizontal or vertical or angle:
            self.incidence = (90 - self.cmaster.image_load_data_memory('azimuth_elevation_angle', self.s_lin, self.s_pix,
                                                                       self.shape, 'Elevation_angle' + self.sample)) / 180 * np.pi

        # Initialize output
        self.perpendicular = perpendicular
        self.parallel = parallel
        self.horizontal = horizontal
        self.vertical = vertical
        self.total_baseline = total_baseline
        self.angle = angle

        self.baseline = np.zeros(self.ra_shift.shape).astype(np.float32)

        if perpendicular:
            self.perp_b = np.zeros(self.ra_shift.shape).astype(np.float32)
        if self.parallel:
            self.parallel_b = np.zeros(self.ra_shift.shape).astype(np.float32)
        if horizontal:
            self.horizontal_b = np.zeros(self.ra_shift.shape).astype(np.float32)
        if vertical:
            self.vertical_b = np.zeros(self.ra_shift.shape).astype(np.float32)
        if angle:
            self.angle_b = np.zeros(self.ra_shift.shape).astype(np.float32)
        if total_baseline:
            self.total_b = np.zeros(self.ra_shift.shape).astype(np.float32)

        # Prepare orbit interpolation
        self.master_orbit = OrbitInterpolate(self.cmaster)
        self.master_orbit.fit_orbit_spline(vel=False, acc=False)
        self.slave_orbit = OrbitInterpolate(self.cmaster)
        self.slave_orbit.fit_orbit_spline(vel=False, acc=False)

    def __call__(self):
        # Calculate the new line and pixel coordinates based orbits / geometry
        if len(self.ra_shift) == 0 or len(self.az_shift) == 0:
            print('Missing input data for geometrical coregistration for ' + self.slave.folder + '. Aborting..')
            return False

        try:
            no0 = (self.ra_shift != 0) * (self.az_shift != 0)

            if np.sum(no0) > 0:
                # First get the master and slave positions.
                master_az_times = self.master_az_time + self.master_az_step * (self.lines - 1)
                [m_x, m_y, m_z], v, a = self.master_orbit.evaluate_orbit_spline(master_az_times)
                slave_az_times = self.slave_az_time + self.slave_az_step * (self.lines - 1)
                [s_x, s_y, s_z], v, a = self.slave_orbit.evaluate_orbit_spline(self.az_shift[no0] * self.slave_az_step + slave_az_times[:, None])

                # Then calculate the parallel and perpendicular baseline.
                self.baseline_2 = np.zeros(self.ra_shift.shape)
                self.baseline_2[no0] = ((s_x - m_x[:, None])**2 + (s_y - m_y[:, None])**2 + (s_z - m_z[:, None])**2).astype(np.float32)
                del m_x, m_y, m_z, s_x, s_y, s_z

                self.parallel_b[no0] = self.ra_shift[no0] * self.ra2m
                self.perpendicular_b = np.zeros(self.ra_shift.shape)
                self.perpendicular_b[no0] = np.sqrt(self.parallel**2 + self.baseline_2[no0])

            # Save meta data
            self.add_meta_data(self.slave, self.coordinates, self.perpendicular,
                               self.parallel, self.horizontal, self.vertical, self.angle, self.total_baseline)

            # Save perpendicular and/or parallel baseline.
            if self.perpendicular:
                self.slave.image_new_data_memory(self.perpendicular_b, 'baseline', self.s_lin, self.s_pix,
                                                file_type='perpendicular_baseline' + self.sample)
            if self.parallel:
                self.slave.image_new_data_memory(self.parallel_b, 'baseline', self.s_lin, self.s_pix,
                                                file_type='parallel_baseline' + self.sample)

            # Create and save the other baseline types if needed.
            if self.horizontal:
                self.horizontal_b[no0] = self.perpendicular_b[no0] * np.cos(self.incidence[no0]) \
                                                 + self.parallel_b[no0] * np.sin(self.incidence[no0])
                self.slave.image_new_data_memory(self.horizontal_b, 'baseline', self.s_lin, self.s_pix,
                                            file_type='horizontal_baseline' + self.sample)
            if self.vertical:
                self.vertical_b[no0] = self.perpendicular_b[no0] * np.sin(self.incidence[no0])\
                                               - self.parallel_b[no0] * np.cos(self.incidence[no0])
                self.slave.image_new_data_memory(self.vertical_b, 'baseline', self.s_lin, self.s_pix,
                                                file_type='vertical_baseline' + self.sample)
            if self.angle:
                self.angle_b[no0] = (self.incidence[no0] - np.arctan(self.parallel_b[no0] / self.perpendicular_b[no0])) / np.pi * 180
                self.slave.image_new_data_memory(self.angle_b, 'baseline', self.s_lin, self.s_pix,
                                                file_type='horizontal_baseline' + self.sample)
            if self.total_baseline:
                self.total_b[no0] = np.sqrt(self.baseline_2[no0])
                self.slave.image_new_data_memory(self.angle_b, 'baseline', self.s_lin, self.s_pix,
                                                file_type='total_baseline' + self.sample)

            return True

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed baseline calculation for ' +
                              self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed baseline calculation for ' +
                  self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def baseline_types(perpendicular=True, parallel=False, horizontal=False, vertical=False, total_baseline=False, angle=False):

        baseline_types = []

        if perpendicular:
            baseline_types.append('perpendicular_baseline')
        if parallel:
            baseline_types.append('parallel_baseline')
        if horizontal:
            baseline_types.append('horizontal_baseline')
        if vertical:
            baseline_types.append('vertical_baseline')
        if angle:
            baseline_types.append('angle_baseline')
        if total_baseline:
            baseline_types.append('total_baseline')

        return baseline_types

    @staticmethod
    def add_meta_data(meta, coordinates,
                 perpendicular=True, parallel=False, horizontal=False, vertical=False, total_baseline=False, angle=False):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'baseline' in meta.processes.keys():
            meta_info = meta.processes['baseline']
        else:
            meta_info = OrderedDict()

        data_names = Baseline.baseline_types(perpendicular, parallel, horizontal, vertical, total_baseline, angle)
        data_types = ['real4' for i in data_names]

        meta_info = coordinates.create_meta_data(data_names, data_types, meta_info)

        meta.image_add_processing_step('baseline', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='', perpendicular=True, parallel=False, horizontal=False, vertical=False,
                        total_baseline=False, angle=False):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Information on this processing step
        data_names = Baseline.baseline_types(perpendicular, parallel, horizontal, vertical, total_baseline, angle)

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        for t in ['new_line', 'new_pixel']:
            input_dat['slave']['geometrical_coreg'][t]['file'] = [t + coordinates.sample + '.raw']
            input_dat['slave']['geometrical_coreg'][t]['coordinates'] = coordinates
            input_dat['slave']['geometrical_coreg'][t]['slice'] = coordinates.slice

        if horizontal or vertical or angle:
            input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle']['file'] = [t + coordinates.sample + '.raw']
            input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle']['coordinates'] = coordinates
            input_dat['cmaster']['azimuth_elevation_angle']['Elevation_angle']['slice'] = coordinates.slice

        # line and pixel output files.
        output_dat = recursive_dict()
        for name in data_names:
            output_dat['slave']['baseline'][name]['file'] = [name + coordinates.sample + '.raw']
            output_dat['slave']['baseline'][name]['coordinates'] = coordinates
            output_dat['slave']['baseline'][name]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('baseline', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('baseline', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('baseline', file_type, coordinates)
