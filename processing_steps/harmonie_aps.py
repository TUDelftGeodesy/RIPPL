# The following class creates an interferogram from a master and slave image.

import os
import numpy as np
import datetime
import logging
from collections import OrderedDict, defaultdict

from image_data import ImageData
from processing_steps.radar_dem import RadarDem
from coordinate_system import CoordinateSystem

from NWP_functions.harmonie.harmonie_database import HarmonieDatabase
from NWP_functions.harmonie.harmonie_load_file import HarmonieData
from NWP_functions.model_ray_tracing import ModelRayTracing
from NWP_functions.model_to_delay import ModelToDelay
from NWP_functions.model_interpolate_delays import ModelInterpolateDelays


class HarmonieAps(object):
    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, cmaster_meta, coor_in, coor_out, s_lin=0, s_pix=0, lines=0,
                 weather_data_archive='', h_type='h_38', time_interp='nearest', split=False, t_step=1, t_offset=0):
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

        if h_type not in ['h_37', 'h_38', 'h_40']:
            print('harmonie type should be h_37, h_38, h_40. Switching to default h_38')
            self.harmonie_type = 'h_38'
        else:
            self.harmonie_type = h_type

        if len(weather_data_archive) == 0 or not os.path.exists(weather_data_archive):
            self.weather_data_archive = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.cmaster.folder))), 'weather_models')
        else:
            self.weather_data_archive = weather_data_archive
        self.weather_data_folder = os.path.join(self.weather_data_archive, 'harmonie_data')

        self.t_step = t_step
        self.t_offset = t_offset
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coor_out = coor_out
        self.coor_in = coor_in

        # The coor_in grid is a course grid to find the height delay dependence of our Harmonie data.
        # The coor_out grid is the final interpolation grid that is generated as an output.
        # Normally the coor_in grid can be of more or less the same resolution as the Harmonie data, which is 2 km.
        # For Sentinel-1 data this means a multilooking of about [50, 200]
        self.shape_in, self.coarse_lines, self.coarse_pixels = RadarDem.find_coordinates(meta, s_lin, s_pix, lines, coor_in)
        self.shape_out, self.lines, self.pixels = RadarDem.find_coordinates(meta, s_lin, s_pix, lines, coor_out)

        # Load data input grid
        self.lat = self.cmaster.image_load_data_memory('geocode', s_lin, 0, self.shape_in, 'Lat' + coor_in.sample)
        self.lon = self.cmaster.image_load_data_memory('geocode', s_lin, 0, self.shape_in, 'Lon' + coor_in.sample)
        self.height = self.cmaster.image_load_data_memory('radar_dem', s_lin, 0, self.shape_in, 'Data' + coor_in.sample)
        self.azimuth_angle = self.cmaster.image_load_data_memory('azimuth_elevation_angle', s_lin, 0, self.shape_in,
                                                                 'Azimuth_angle' + coor_in.sample)
        self.elevation_angle = self.cmaster.image_load_data_memory('azimuth_elevation_angle', s_lin, 0, self.shape_in,
                                                                   'Elevation_angle' + coor_in.sample)

        # Load height of multilook grid (from an interval, buffer grid, which makes it a bit complicated...)
        self.out_height = self.cmaster.image_load_data_memory('radar_dem', s_lin, 0, self.shape_in, 'Data' + coor_out.sample)
        self.simulated_delay = []
        self.split = split
        if self.split:
            self.hydrostatic_delay = []
            self.wet_delay = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.lat) == 0 or len(self.lon) == 0 or len(self.height) == 0 or len(self.azimuth_angle) == 0 or len(self.elevation_angle) == 0:
            print('Missing input data for ray tracing weather model ' + self.cmaster.folder +
                  '. Check whether you are using the right reference image. If so, you can run the geocode image '
                  'function to calculate the needed values. Aborting..')
            return False

        try:
            # Load the geometry
            ray_delays = ModelRayTracing(split_signal=self.split)
            ray_delays.load_geometry(self.coarse_lines, self.coarse_pixels,
                                     self.azimuth_angle, self.elevation_angle,
                                     self.lat, self.lon, self.height)

            # Define date we need weather data.
            overpass = datetime.datetime.strptime(self.slave.processes['readfiles']['First_pixel_azimuth_time (UTC)'], '%Y-%m-%dT%H:%M:%S.%f')
            harmonie_archive = HarmonieDatabase(database_folder=self.weather_data_folder)
            filename, date = harmonie_archive(overpass)

            # Load the Harmonie data if available
            if filename[0]:
                data = HarmonieData()
                data.load_harmonie(date[0], filename[0])
            else:
                print('No harmonie data available for ' + date[0].strftime('%Y-%m-%dT%H:%M:%S.%f'))
                return

            # TODO Add a loop to run this method for different time steps. So the wind vectors will be used to create
            # new delay images on different time scales.

            # And convert the data to delays
            proc_date = date[0].strftime('%Y%m%dT%H%M')
            geoid_file = os.path.join(self.weather_data_archive, 'egm96.raw')

            model_delays = ModelToDelay(65, geoid_file)
            model_delays.load_model_delay(data.model_data)
            model_delays.model_to_delay()
            data.remove_harmonie(proc_date)

            # Convert model delays to delays over specific rays
            ray_delays.load_delay(model_delays.delay_data)
            ray_delays.calc_cross_sections()
            ray_delays.find_point_delays()
            model_delays.remove_delay(proc_date)

            # Finally resample to the full grid
            pixel_points, lines_points = np.meshgrid(self.pixels, self.lines)
            point_delays = ModelInterpolateDelays(self.coarse_lines, self.coarse_pixels, split=self.split)
            point_delays.add_interp_points(np.ravel(lines_points), np.ravel(pixel_points), np.ravel(self.out_height))

            point_delays.add_delays(ray_delays.spline_delays)
            point_delays.interpolate_points()

            # Save the file
            shp = (len(self.lines), len(self.pixels))
            self.simulated_delay = point_delays.interp_delays['total'][proc_date].reshape(shp).astype(np.float32)
            if self.split:
                self.hydrostatic_delay = point_delays.interp_delays['hydrostatic'][proc_date].reshape(shp).astype(np.float32)
                self.wet_delay = point_delays.interp_delays['wet'][proc_date].reshape(shp).astype(np.float32)
            ray_delays.remove_delay(proc_date)

            # Save meta data
            self.add_meta_data(self.slave, self.coor_out, self.harmonie_type, self.split)

            # Save the data itself
            self.slave.image_new_data_memory(self.simulated_delay, 'NWP_phase', self.s_lin, self.s_pix, 'harmonie_' +
                                             self.harmonie_type + '_aps' + self.coor_out.sample)
            if self.split:
                self.slave.image_new_data_memory(self.hydrostatic_delay, 'NWP_phase', self.s_lin, self.s_pix,
                                                'harmonie_' + self.harmonie_type + '_hydrostatic' + self.coor_out.sample)
                self.slave.image_new_data_memory(self.wet_delay, 'NWP_phase', self.s_lin, self.s_pix,
                                                'harmonie_' + self.harmonie_type + '_wet' + self.coor_out.sample)

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating aps from Harmonie for ' +
                              self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating aps from Harmonie for ' +
                  self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates, h_type='h38', split=False):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.

        if 'NWP_phase' in meta.processes.keys():
            meta_info = meta.processes['NWP_phase']
        else:
            meta_info = OrderedDict()

        if split:
            aps_types = ['harmonie_' + h_type + '_aps', 'harmonie_' + h_type + '_wet', 'harmonie_' + h_type + '_hydrostatic']
            data_types = ['real4', 'real4', 'real4']
        else:
            aps_types = ['harmonie_' + h_type + '_aps']
            data_types = ['real4']

        meta_info = coordinates.create_meta_data(aps_types, data_types, meta_info)

        meta.image_add_processing_step('NWP_phase', meta_info)

    @staticmethod
    def processing_info(coordinates, h_type='h38', split=False):

        # Fix the input coordinate system to fit the harmonie grid...
        # TODO adapt the coor_in when the output is not in radar coordinates
        coor_in = CoordinateSystem()
        coor_in.create_radar_coordinates(multilook=[200, 50])

        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat['cmaster']['radar_dem']['radar_dem']['file'] = ['radar_dem' + coordinates.sample]
        input_dat['cmaster']['radar_dem']['radar_dem']['coordinates'] = coordinates
        input_dat['cmaster']['radar_dem']['radar_dem']['slice'] = True

        # For multiprocessing this information is needed to define the selected area to deramp.
        for t in ['azimuth_angle', 'elevation_angle']:
            input_dat['cmaster']['azimuth_elevation_angle'][t]['file'] = [t + coor_in.sample + '.raw']
            input_dat['cmaster']['azimuth_elevation_angle'][t]['coordinates'] = coor_in
            input_dat['cmaster']['azimuth_elevation_angle'][t]['slice'] = True
            input_dat['cmaster']['azimuth_elevation_angle'][t]['coor_change'] = 'resample'

        # For multiprocessing this information is needed to define the selected area to deramp.
        for t in ['lat', 'lon']:
            input_dat['cmaster']['geocode'][t]['file'] = [t + coor_in.sample + '.raw']
            input_dat['cmaster']['geocode'][t]['coordinates'] = coor_in
            input_dat['cmaster']['geocode'][t]['slice'] = True
            input_dat['cmaster']['geocode'][t]['coor_change'] = 'resample'

        if split:
            aps_types = ['harmonie_' + h_type + '_aps', 'harmonie_' + h_type + '_wet', 'harmonie_' + h_type + '_hydrostatic']
        else:
            aps_types = ['harmonie_' + h_type + '_aps']

        output_dat = recursive_dict()
        for t in aps_types:
            output_dat['slave']['NWP_phase'][t]['file'] = [t + coor_out.sample + '.raw']
            output_dat['slave']['NWP_phase'][t]['coordinates'] = coor_out
            output_dat['slave']['NWP_phase'][t]['slice'] = coor_out.slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('NWP_phase', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('NWP_phase', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('NWP_phase', file_type, coordinates)
