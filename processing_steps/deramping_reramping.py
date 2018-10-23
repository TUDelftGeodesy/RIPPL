# Function to do deramping and reramping on individual bursts.
import datetime
import numpy as np
from image_data import ImageData
from orbit_dem_functions.orbit_interpolate import OrbitInterpolate
from processing_steps.resample import Resample
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import logging
import os


class GetDopplerRamp(OrbitInterpolate):
    # Read information from resfile
    def __init__(self, meta=''):

        if isinstance(meta, str):
            if len(meta) != 0:
                self.slave = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.slave = meta

        # Read information from res data
        ################################################################################
        self.orbit = OrbitInterpolate(self.slave)

        # FM
        self.t_fm = float(self.slave.processes['readfiles']['FM_reference_range_time'])
        self.c_fm = [0, 0, 0]
        self.c_fm[0] = float(self.slave.processes['readfiles']['FM_polynomial_constant_coeff (Hz, early edge)'])
        self.c_fm[1] = float(self.slave.processes['readfiles']['FM_polynomial_linear_coeff (Hz/s, early edge)'])
        self.c_fm[2] = float(self.slave.processes['readfiles']['FM_polynomial_quadratic_coeff (Hz/s/s, early edge)'])

        # DC
        az_time_dc = self.slave.processes['readfiles']['DC_reference_azimuth_time']
        az_time_dc = (datetime.datetime.strptime(az_time_dc, '%Y-%m-%dT%H:%M:%S.%f') -
                      datetime.datetime.strptime(az_time_dc[:10], '%Y-%m-%d'))
        self.az_time_dc = az_time_dc.seconds + az_time_dc.microseconds / 1000000.0

        self.t_dc = float(self.slave.processes['readfiles']['DC_reference_range_time'])
        self.c_dc = [0, 0, 0]
        self.c_dc[0] = float(self.slave.processes['readfiles']['Xtrack_f_DC_constant (Hz, early edge)'])
        self.c_dc[1] = float(self.slave.processes['readfiles']['Xtrack_f_DC_linear (Hz/s, early edge)'])
        self.c_dc[2] = float(self.slave.processes['readfiles']['Xtrack_f_DC_quadratic (Hz/s/s, early edge)'])

        self.ks = float(self.slave.processes['readfiles']['Azimuth_steering_rate (deg/s)'])

        # Image sampling parameters
        t_az_start = self.slave.processes['readfiles']['First_pixel_azimuth_time (UTC)']
        t_az_start = (datetime.datetime.strptime(t_az_start, '%Y-%m-%dT%H:%M:%S.%f') -
                      datetime.datetime.strptime(t_az_start[:10], '%Y-%m-%d'))
        self.t_az_start = t_az_start.seconds + t_az_start.microseconds / 1000000.0
        self.l_num = int(self.slave.processes['readfiles']['Number_of_lines_original'])

        self.t_rg_start = float(self.slave.processes['readfiles']['Range_time_to_first_pixel (2way) (ms)']) * 1e-3
        self.fs_rg = float(self.slave.processes['readfiles']['Range_sampling_rate (computed, MHz)'])

        self.dt_az = float(self.slave.processes['readfiles']['Azimuth_time_interval (s)'])
        self.dt_rg = 1 / self.fs_rg / 1e6
    
        # Initialize the needed orbit variables
        self.orbit_time = []
        self.orbit_spline = []
        self.orbit_fit = []

    def calc_ramp(self, az_time, ra_time, demodulation=False):

        self.orbit.fit_orbit_spline()
        mid_orbit_time = self.t_az_start + self.dt_az * (float(self.l_num) / 2)
        orbit_vel = self.orbit.evaluate_orbit_spline(np.asarray([mid_orbit_time]), pos=False, vel=True, acc=False)[1]

        orbit_velocity = np.sqrt(orbit_vel[0] ** 2 + orbit_vel[1] ** 2 + orbit_vel[2] ** 2)

        # Compute Nominal DC for the whole burst
        # Compute FM rate along range
        k_fm = self.c_fm[0] + self.c_fm[1] * (ra_time - self.t_fm) + self.c_fm[2] * (ra_time - self.t_fm) ** 2
        k_fm_0 = (self.c_fm[0] + self.c_fm[1] * (self.t_rg_start - self.t_fm) + self.c_fm[2] * 
                  (self.t_rg_start - self.t_fm) ** 2)

        # Compute DC along range at reference azimuth time (azimuthTime)
        df_az_ctr = self.c_dc[0] + self.c_dc[1] * (ra_time - self.t_dc) + self.c_dc[2] * (ra_time - self.t_dc) ** 2
        f_dc_ref_0 = (self.c_dc[0] + self.c_dc[1] * (self.t_rg_start - self.t_dc) + self.c_dc[2] * 
                      (self.t_rg_start - self.t_dc) ** 2)
        del ra_time

        # From S-1 steering rate and orbit information %
        # Computes sensor velocity from orbits
        c_lambda = float(self.slave.processes['readfiles']['Radar_wavelength (m)'])

        # Frequency rate
        ks_hz = 2 * np.mean(orbit_velocity) / c_lambda * self.ks / 180 * np.pi

        # Time ratio
        alpha_nom = 1 - ks_hz / k_fm

        # DC Azimuth rate [Hz/s]
        dr_est = ks_hz / alpha_nom
        del ks_hz, alpha_nom

        # Reference time
        az_dc = -(df_az_ctr / k_fm) + (f_dc_ref_0 / k_fm_0)
        del k_fm, k_fm_0
        t_az_vec = az_time - az_dc
        del az_dc

        # % Generate inverse chirp %
        if demodulation:
            ramp = np.exp(-1j * np.pi * dr_est * t_az_vec ** 2 + -1j * np.pi * 2 * df_az_ctr * t_az_vec)
        else:
            ramp = np.exp(-1j * np.pi * dr_est * t_az_vec**2)

        return ramp


class Deramp(GetDopplerRamp):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, res_coordinates, s_lin=0, s_pix=0, lines=0, buf=5):
        # Most important for deramping is that we use the resampling grid as a basis. This is because this is the only
        # script that uses the slave geometry, which would make it very difficult to add in a full processing chain.
        # Therefore, we define the area that should be deramped based on the region defined by the resampling.
        # Main consequence is that deramping can only be done after coregistration.

        if isinstance(meta, ImageData):
            self.slave = meta
        else:
            return

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        shape = self.slave.data_sizes['resample']['resample']
        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        shape = [l, shape[1] - s_pix]

        new_line = self.slave.image_load_data_memory('combined_coreg', s_lin, s_pix, shape,
                                                     'new_line' + res_coordinates.sample)
        new_pixel = self.slave.image_load_data_memory('combined_coreg', s_lin, s_pix, shape,
                                                      'new_pixel' + res_coordinates.sample)

        # Load needed data from crop
        self.s_lin, self.s_pix, self.shape, self.res_s_lin, self.res_s_pix = \
            Resample.select_region_resampling(self.slave, new_line, new_pixel, buf)
        self.crop_ramped = self.slave.image_load_data_memory('crop', self.s_lin, self.s_pix, self.shape)

        GetDopplerRamp.__init__(self, self.slave)
        self.crop = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.crop_ramped) == 0:
            print('Missing input data for processing deramping for ' + self.slave.folder + '. Aborting..')
            return False

        try:
            # First calculate the azimuth and range times we are referring to
            az_time, ra_time = self.get_az_time()

            # Then calculate the ramp
            ramp = self.calc_ramp(az_time, ra_time)

            # Finally correct the data
            self.crop = (self.crop_ramped * ramp).astype('complex64')

            # And save the data
            self.add_meta_data(self.slave)
            self.slave.image_new_data_memory(self.crop, 'deramp', self.s_lin, self.s_pix)

            return True

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing azimuth_elevation_angle for ' + self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing azimuth_elevation_angle for ' + self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    def get_az_time(self):

        first_pixel = int(self.slave.processes['crop']['Data_first_pixel']) - 1 + self.s_pix
        first_line = int(self.slave.processes['crop']['Data_first_line']) - 1 + self.s_lin

        t_vectrg = self.t_rg_start + np.arange(first_pixel, first_pixel + self.shape[1]) * self.dt_rg
        t_vectaz = (np.arange(first_line, first_line + self.shape[0]) * self.dt_az - 
                    (float(self.l_num) / 2 * self.dt_az))[:, None]

        ra_time = np.tile(t_vectrg, (self.shape[0], 1))
        az_time = np.tile(t_vectaz, (1, self.shape[1]))

        return az_time, ra_time

    @staticmethod
    def add_meta_data(meta):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.

        if 'deramp' in meta.processes.keys():
            meta_info = meta.processes['deramp']
        else:
            meta_info = OrderedDict()

        coordinates = CoordinateSystem()
        coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        meta_info = coordinates.create_meta_data(['reramped'], ['complex_int'], meta_info)

        meta.image_add_processing_step('reramp', meta_info)

    @staticmethod
    def processing_info():

        coordinates = CoordinateSystem()
        coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])

        input_dat = defaultdict()
        input_dat['slave']['crop']['crop']['file'] = ['crop.raw']
        input_dat['slave']['crop']['crop']['coordinates'] = coordinates
        input_dat['slave']['crop']['crop']['slice'] = 'True'

        # For multiprocessing this information is needed to define the selected area to deramp.
        for t in ['new_line', 'new_pixel']:
            input_dat['slave']['combined_coreg'][t]['file'] = [t + coordinates.sample + '.raw']
            input_dat['slave']['combined_coreg'][t]['coordinates'] = coordinates
            input_dat['slave']['combined_coreg'][t]['slice'] = 'True'

        output_dat = defaultdict()
        output_dat['slave']['deramp']['deramped']['file'] = ['deramped.raw']
        output_dat['slave']['deramp']['deramped']['coordinates'] = coordinates
        output_dat['slave']['deramp']['deramped']['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('deramp', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_create_disk('deramp', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('deramp', file_type, coordinates)

class Reramp(GetDopplerRamp):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(meta, ImageData):
            self.slave = meta
        else:
            return

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        shape = self.slave.data_sizes['resample']['resample']
        if lines != 0:
            l = np.minimum(lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        self.shape = [l, shape[1] - s_pix]

        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates

        GetDopplerRamp.__init__(self, self.slave)
        self.resampled = []

        # Load data
        self.resampled_deramped = self.slave.image_load_data_memory('resample', self.s_lin, self.s_pix, self.shape)
        self.new_pixel = self.slave.image_load_data_memory('combined_coreg', self.s_lin, self.s_pix, self.shape, 'new_pixel' + coordinates.sample)
        self.new_line = self.slave.image_load_data_memory('combined_coreg', self.s_lin, self.s_pix, self.shape, 'new_line' + coordinates.sample)

    def __call__(self):
        # Check if needed data is loaded
        if len(self.resampled_deramped) == 0 or len(self.new_pixel) == 0 or len(self.new_pixel) == 0:
            print('Missing input data for processing reramping for ' + self.slave.folder + '. Aborting..')
            return False

        try:
            # First calculate the azimuth and range times we are referring to
            az_time, ra_time = self.get_az()

            # Then calculate the ramp
            ramp = self.calc_ramp(az_time, ra_time)

            # Finally correct the data
            self.reramped_resampled = (self.resampled_deramped * ramp.conj()).astype('complex64')

            self.add_meta_data(self.slave, self.coordinates)
            self.slave.image_new_data_memory(self.reramped_resampled, 'reramp', self.s_lin, self.s_pix)

            return True

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing azimuth_elevation_angle for ' +
                              self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing azimuth_elevation_angle for ' +
                  self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    def get_az(self):

        ra_time = self.new_pixel * self.dt_rg + self.t_rg_start
        az_time = self.new_line * self.dt_az - (float(self.l_num)/2 * self.dt_az)

        return az_time, ra_time

    @staticmethod
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.

        if 'reramp' in meta.processes.keys():
            meta_info = meta.processes['reramp']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['reramped'], ['complex_int'], meta_info)

        meta.image_add_processing_step('reramp', meta_info)

    @staticmethod
    def processing_info(coordinates):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        input_dat = defaultdict()
        for t in ['new_line', 'new_pixel']:
            input_dat['slave']['combined_coreg'][t]['file'] = [t + coordinates.sample + '.raw']
            input_dat['slave']['combined_coreg'][t]['coordinates'] = coordinates
            input_dat['slave']['combined_coreg'][t]['slice'] = 'True'

        input_dat['slave']['resample']['resampled']['file'] = ['resampled.raw']
        input_dat['slave']['resample']['resampled']['coordinates'] = coordinates
        input_dat['slave']['resample']['resampled']['slice'] = 'True'

        output_dat = defaultdict()
        output_dat['slave']['reramp']['reramped']['file'] = ['reramped' + coordinates.sample + '.raw']
        output_dat['slave']['reramp']['reramped']['coordinates'] = coordinates
        output_dat['slave']['reramp']['reramped']['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('reramp', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_create_disk('reramp', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('reramp', file_type, coordinates)
