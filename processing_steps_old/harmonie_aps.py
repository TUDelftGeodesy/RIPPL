# The following class creates an interferogram from a master and slave image.

import os
import numpy as np
import datetime
import logging
from collections import OrderedDict, defaultdict

from rippl.meta_data.image_processing_data import ImageData
from rippl.processing_steps_old.resample_dem import ResampleDem
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

from rippl.NWP_simulations.harmonie.harmonie_database import HarmonieDatabase
from rippl.NWP_simulations.harmonie.harmonie_load_file import HarmonieData
from rippl.NWP_simulations.model_ray_tracing import ModelRayTracing
from rippl.NWP_simulations.model_to_delay import ModelToDelay
from rippl.NWP_simulations.model_interpolate_delays import ModelInterpolateDelays
from rippl.processing_steps_old.projection_coor import ProjectionCoor
from rippl.processing_steps_old.coor_dem import CoorDem
from rippl.processing_steps_old.coor_geocode import CoorGeocode


class HarmonieAps(object):
    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, cmaster_meta, in_coor, coordinates, s_lin=0, s_pix=0, lines=0,
                 weather_data_archive='', h_type='all', time_interp='nearest', split=False, t_step=1, t_offset=0):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, ImageData) and isinstance(cmaster_meta, ImageData):
            self.slave = meta
            self.cmaster = cmaster_meta
        else:
            return

        # The weather data archive
        if time_interp in ['nearest', 'linear']:
            self.time_interp = time_interp
        else:
            print('time_interp should be nearest of linear. Using nearest')
            self.time_interp = 'nearest'

        if len(weather_data_archive) == 0 or not os.path.exists(weather_data_archive):
            self.weather_data_archive = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.cmaster.folder))), 'weather_models')
        else:
            self.weather_data_archive = weather_data_archive
        self.weather_data_folder = os.path.join(self.weather_data_archive, 'harmonie_data')

        self.t_step = t_step
        self.t_offset = t_offset
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.out_coor = coordinates
        self.in_coor = in_coor
        shape = coordinates.shape
        if lines != 0:
            self.line = np.minimum(lines, shape[0] - s_lin)
        else:
            self.line = shape[0] - s_lin
        self.shape = [self.line, shape[1] - s_pix]

        # The in_coor grid is a course grid to find the height delay dependence of our Harmonie data.
        # The out_coor grid is the final interpolation grid that is generated as an output.
        # Normally the in_coor grid can be of more or less the same resolution as the Harmonie data, which is 2 km.
        # For Sentinel-1 data this means a multilooking of about [50, 200]
        self.shape_in, self.coarse_lines, self.coarse_pixels = ResampleDem.find_coordinates(self.cmaster, 0, 0, 0, in_coor)
        self.shape_out, self.lines, self.pixels = ResampleDem.find_coordinates(self.cmaster, self.s_lin, self.s_pix, self.line, self.out_coor)

        # Load data input grid
        self.lat, self.lon, self.height, self.azimuth_angle, self.elevation_angle, self.mask, self.out_height = \
            self.load_aps_data(self.in_coor, self.out_coor, self.cmaster, self.shape_in, self.shape_out, self.s_lin, self.s_pix)
        self.mask *= ~((self.lines == 0) * (self.pixels == 0))

        self.simulated_delay = np.zeros(self.shape_out)
        self.split = split
        if self.split:
            self.hydrostatic_delay = np.zeros(self.shape_out)
            self.wet_delay = np.zeros(self.shape_out)

    def __call__(self):
        # Check if needed data is loaded
        if len(self.lat) == 0 or len(self.lon) == 0 or len(self.height) == 0 or len(self.azimuth_angle) == 0 or len(self.elevation_angle) == 0:
            print('Missing input data for ray tracing weather model ' + self.cmaster.folder +
                  '. Check whether you are using the right reference image. If so, you can run the geocode image '
                  'function to calculate the needed values. Aborting..')
            return False

        try:
            # Define date we need weather data.
            overpass = datetime.datetime.strptime(self.slave.processes['readfile.py']['First_pixel_azimuth_time (UTC)'], '%Y-%m-%dT%H:%M:%S.%f')
            harmonie_archive = HarmonieDatabase(database_folder=self.weather_data_folder)
            filename, date = harmonie_archive(overpass)

            # Load the Harmonie data if available
            if filename[0]:
                data = HarmonieData()
                data.load_harmonie(date[0], filename[0])
            else:
                print('No harmonie data available for ' + date[0].strftime('%Y-%m-%dT%H:%M:%S.%f'))
                return True

            proc_date = date[0].strftime('%Y%m%dT%H%M')
            # Run the ray tracing
            self.ray_tracing(data, proc_date)

            # TODO Add a loop to run this method for different time steps. So the wind vectors will be used to create
            # new delay images on different time scales.

            # Save meta data
            self.add_meta_data(self.slave, self.out_coor, self.split)

            # Save the data itself
            self.slave.image_new_data_memory(self.simulated_delay.astype(np.float32), 'harmonie_aps', self.s_lin, self.s_pix,
                                             'harmonie_aps' + self.out_coor.sample)
            if self.split:
                self.slave.image_new_data_memory(self.hydrostatic_delay.astype(np.float32), 'harmonie_aps', self.s_lin, self.s_pix,
                                                'harmonie_hydrostatic' + self.out_coor.sample)
                self.slave.image_new_data_memory(self.wet_delay.astype(np.float32), 'harmonie_aps', self.s_lin, self.s_pix,
                                                'harmonie_wet' + self.out_coor.sample)

            return True

        except ValueError:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating aps from Harmonie for ' +
                              self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating aps from Harmonie for ' +
                  self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    def ray_tracing(self, data, proc_date):
        # The ray tracing methods which are independent of the input source.

        # Load the geometry
        ray_delays = ModelRayTracing(split_signal=self.split)
        ray_delays.load_geometry(self.coarse_lines[:, 0], self.coarse_pixels[0, :],
                                 self.azimuth_angle, self.elevation_angle,
                                 self.lat, self.lon, self.height)

        # And convert the data to delays

        geoid_file = os.path.join(self.weather_data_archive, 'egm96.raw')

        model_delays = ModelToDelay(data.levels, geoid_file)
        model_delays.load_model_delay(data.model_data)
        model_delays.model_to_delay()

        # Convert model delays to delays over specific rays
        ray_delays.load_delay(model_delays.delay_data)
        ray_delays.calc_cross_sections()
        ray_delays.find_point_delays()
        model_delays.remove_delay(proc_date)

        # Finally resample to the full grid
        point_delays = ModelInterpolateDelays(self.coarse_lines[:, 0], self.coarse_pixels[0, :], split=self.split)
        point_delays.add_interp_points(np.ravel(self.lines[self.mask]), np.ravel(self.pixels[self.mask]),
                                       np.ravel(self.out_height[self.mask]))

        point_delays.add_delays(ray_delays.spline_delays)
        point_delays.interpolate_points()

        # Save the file
        shp = self.mask.shape
        self.simulated_delay[self.mask] = point_delays.interp_delays['total'][proc_date].astype(np.float32)
        if self.split:
            self.hydrostatic_delay[self.mask] = point_delays.interp_delays['hydrostatic'][proc_date].astype(np.float32)
            self.wet_delay[self.mask] = point_delays.interp_delays['wet'][proc_date].astype(np.float32)
        ray_delays.remove_delay(proc_date)

    @staticmethod
    def load_aps_data(in_coor, out_coor, cmaster, shape_in, shape_out, s_lin, s_pix):

        lat, lon = ProjectionCoor.load_lat_lon(in_coor, cmaster, 0, 0, shape_in)
        height = CoorDem.load_dem(in_coor, cmaster, 0, 0, shape_in)

        azimuth_angle = cmaster.image_load_data_memory('azimuth_elevation_angle', 0, 0, shape_in,
                                                                 'azimuth_angle' + in_coor.sample)
        elevation_angle = cmaster.image_load_data_memory('azimuth_elevation_angle', 0, 0, shape_in,
                                                                   'elevation_angle' + in_coor.sample)

        # Load height of multilook grid (from an interval, buffer grid, which makes it a bit complicated...)
        out_height = CoorDem.load_dem(out_coor, cmaster, s_lin, 0, shape_out)

        if out_coor.mask_grid:
            mask = cmaster.image_load_data_memory('create_sparse_grid', s_lin, 0, shape_out,
                                                            'mask' + out_coor.sample)
        else:
            mask = np.ones(out_height.shape).astype(np.bool)

        return lat, lon, height, azimuth_angle, elevation_angle, mask, out_height

    @staticmethod
    def add_meta_data(meta, coordinates, split=False):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.

        if 'harmonie_aps' in meta.processes.keys():
            meta_info = meta.processes['harmonie_aps']
        else:
            meta_info = OrderedDict()

        if split:
            aps_types = ['harmonie_aps', 'harmonie_wet', 'harmonie_hydrostatic']
            data_types = ['real4', 'real4', 'real4']
        else:
            aps_types = ['harmonie_aps']
            data_types = ['real4']

        meta_info = coordinates.create_meta_data(aps_types, data_types, meta_info)

        meta.image_add_processing_step('harmonie_aps', meta_info)

    @staticmethod
    def processing_info(coordinates, split=False):

        # Fix the input coordinate system to fit the harmonie grid...
        # TODO adapt the in_coor when the output is not in radar coordinates
        in_coor = CoordinateSystem(over_size=True)
        in_coor.create_radar_coordinates(multilook=[50, 200])

        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat = CoorDem.dem_processing_info(input_dat, coordinates, 'cmaster', False)
        input_dat = CoorDem.dem_processing_info(input_dat, in_coor, 'cmaster', True)
        input_dat = ProjectionCoor.lat_lon_processing_info(input_dat, in_coor, 'cmaster', True)
        input_dat = CoorGeocode.line_pixel_processing_info(input_dat, coordinates, 'cmaster', False)

        # For multiprocessing this information is needed to define the selected area to deramp.
        for t in ['azimuth_angle', 'elevation_angle']:
            input_dat['cmaster']['azimuth_elevation_angle'][t + in_coor.sample]['file'] = t + in_coor.sample + '.raw'
            input_dat['cmaster']['azimuth_elevation_angle'][t + in_coor.sample]['coordinates'] = in_coor
            input_dat['cmaster']['azimuth_elevation_angle'][t + in_coor.sample]['slice'] = in_coor.slice
            input_dat['cmaster']['azimuth_elevation_angle'][t + in_coor.sample]['coor_change'] = 'resample'

        if split:
            aps_types = ['harmonie_aps', 'harmonie_wet', 'harmonie_hydrostatic']
        else:
            aps_types = ['harmonie_aps']

        output_dat = recursive_dict()
        for t in aps_types:
            output_dat['slave']['harmonie_aps'][t + coordinates.sample]['file'] = t + coordinates.sample + '.raw'
            output_dat['slave']['harmonie_aps'][t + coordinates.sample]['coordinates'] = coordinates
            output_dat['slave']['harmonie_aps'][t + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('harmonie_aps', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('harmonie_aps', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('harmonie_aps', file_type, coordinates)
