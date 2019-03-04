# The following class creates an interferogram from a master and slave image.

import os
import numpy as np
import datetime
import logging
from collections import OrderedDict, defaultdict

from image_data import ImageData
from coordinate_system import CoordinateSystem

from processing_steps.radar_dem import RadarDem

from NWP_functions.ECMWF.ecmwf_type import ECMWFType
from NWP_functions.ECMWF.ecmwf_download import ECMWFdownload
from NWP_functions.ECMWF.ecmwf_load_file import ECMWFData
from NWP_functions.model_ray_tracing import ModelRayTracing
from NWP_functions.radar_data import RadarData
from NWP_functions.model_to_delay import ModelToDelay
from NWP_functions.model_interpolate_delays import ModelInterpolateDelays


class EcmwfInterimAps(object):
    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, cmaster_meta, coor_in, coordinates, s_lin=0, s_pix=0, lines=0,
                 weather_data_archive='', ecmwf_type='interim', time_interp='nearest', split=False, download=True):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, ImageData) and isinstance(cmaster_meta, ImageData):
            self.slave = meta
            self.cmaster = cmaster_meta
        else:
            return

        # The weather data archive
        if ecmwf_type in ['oper', 'interim', 'era5']:
            self.ecmwf_type = ecmwf_type
        else:
            print('ecmwf_type should be oper, interim or era5. Using interim')
            self.ecmwf_type = 'interim'
        if time_interp in ['nearest', 'linear']:
            self.time_interp = time_interp
        else:
            print('time_interp should be nearest of linear. Using nearest')
            self.time_interp = 'nearest'

        if ecmwf_type not in ['interim', 'era5', 'oper']:
            print('ecmwf type should be interim, era5, oper. Switching to default era5')
            self.ecmwf_type = 'interim'
        else:
            self.ecmwf_type = ecmwf_type

        if len(weather_data_archive) == 0 or not os.path.exists(weather_data_archive):
            self.weather_data_archive = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.cmaster.folder))), 'weather_models')
        else:
            self.weather_data_archive = weather_data_archive
        self.weather_data_folder = os.path.join(self.weather_data_archive, 'ecmwf_data')

        self.download = download
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coor_out = coordinates
        self.coor_in = coor_in

        # The coor_in grid is a course grid to find the height delay dependence of our Harmonie data.
        # The coor_out grid is the final interpolation grid that is generated as an output.
        # Normally the coor_in grid can be of more or less the same resolution as the Harmonie data, which is 2 km.
        # For Sentinel-1 data this means a multilooking of about [50, 200]
        self.shape_in, self.coarse_lines, self.coarse_pixels = RadarDem.find_coordinates(cmaster_meta, 0, 0, 0, coor_in)
        self.shape_out, self.lines, self.pixels = RadarDem.find_coordinates(cmaster_meta, s_lin, s_pix, lines, self.coor_out)

        # Load data input grid
        self.lat = self.cmaster.image_load_data_memory('geocode', 0, 0, self.shape_in, 'lat' + coor_in.sample)
        self.lon = self.cmaster.image_load_data_memory('geocode', 0, 0, self.shape_in, 'lon' + coor_in.sample)
        self.height = self.cmaster.image_load_data_memory('radar_DEM', 0, 0, self.shape_in, 'radar_DEM' + coor_in.sample)
        self.azimuth_angle = self.cmaster.image_load_data_memory('azimuth_elevation_angle', 0, 0, self.shape_in,
                                                                 'azimuth_angle' + coor_in.sample)
        self.elevation_angle = self.cmaster.image_load_data_memory('azimuth_elevation_angle', 0, 0, self.shape_in,
                                                                   'elevation_angle' + coor_in.sample)

        # Load height of multilook grid (from an interval, buffer grid, which makes it a bit complicated...)
        self.out_height = self.cmaster.image_load_data_memory('radar_DEM', s_lin, 0, self.shape_out, 'radar_DEM' + self.coor_out.sample)
        self.simulated_delay = []
        self.split = split
        if self.split:
            self.hydrostatic_delay = []
            self.wet_delay = []
    def __call__(self, latlim='', lonlim=''):
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
            date = datetime.datetime.strptime(self.slave.processes['readfiles']['First_pixel_azimuth_time (UTC)'], '%Y-%m-%dT%H:%M:%S.%f')

            ecmwf_type = ECMWFType(self.ecmwf_type)
            radar_data = RadarData(self.time_interp, ecmwf_type.t_step, 0)
            radar_data.match_overpass_weather_model(date)

            # Load the ECMWF data
            # Download the needed files
            # For simplicity we limit ourselves here to europe if the lat/lon limits are not given.
            if len(latlim) != 2:
                latlim = [45, 56]
            if len(lonlim) != 2:
                lonlim = [-2, 12]

            down = ECMWFdownload(latlim, lonlim, self.weather_data_folder, data_type=self.ecmwf_type)
            down.prepare_download(radar_data.date_times)
            if self.download:
                down.download()
            else:
                # Check if file is already available.
                if not os.path.exists(down.filenames[0] + '_atmosphere.grb') or not os.path.exists(down.filenames[0] + '_surface.grb'):
                    print('ECMWF file ' + down.filenames[0] + '_atmosphere.grb not found. Skipping because download is switched of')
                    return

            # Load the data files
            data = ECMWFData(ecmwf_type.levels)
            data.load_ecmwf(down.dates[0], down.filenames[0])

            date = down.dates[0].strftime('%Y%m%dT%H%M')

            # And convert the data to delays
            geoid_file = os.path.join(self.weather_data_archive, 'egm96.raw')

            model_delays = ModelToDelay(ecmwf_type.levels, geoid_file)
            model_delays.load_model_delay(data.model_data)
            model_delays.model_to_delay()
            data.remove_ecmwf(date)

            # Convert model delays to delays over specific rays
            ray_delays.load_delay(model_delays.delay_data)
            ray_delays.calc_cross_sections()
            ray_delays.find_point_delays()
            model_delays.remove_delay(date)

            # Finally resample to the full grid
            pixel_points, lines_points = np.meshgrid(self.pixels, self.lines)
            point_delays = ModelInterpolateDelays(self.coarse_lines, self.coarse_pixels, split=self.split)
            point_delays.add_interp_points(np.ravel(lines_points), np.ravel(pixel_points), np.ravel(self.out_height))

            point_delays.add_delays(ray_delays.spline_delays)
            point_delays.interpolate_points()

            # Save the file
            shp = (len(self.lines), len(self.pixels))
            self.simulated_delay = point_delays.interp_delays['total'][date].reshape(shp).astype(np.float32)
            if self.split:
                self.hydrostatic_delay = point_delays.interp_delays['hydrostatic'][date].reshape(shp).astype(np.float32)
                self.wet_delay = point_delays.interp_delays['wet'][date].reshape(shp).astype(np.float32)

            ray_delays.remove_delay(date)

            # If needed do the multilooking step
            self.slave.image_new_data_memory(self.simulated_delay, 'NWP_phase', self.s_lin, self.s_pix, 'harmonie_' +
                                             self.ecmwf_type + '_aps' + self.coor_out.sample)
            if self.split:
                self.slave.image_new_data_memory(self.hydrostatic_delay, 'NWP_phase', self.s_lin, self.s_pix,
                                                'harmonie_' + self.ecmwf_type + '_hydrostatic' + self.coor_out.sample)
                self.slave.image_new_data_memory(self.wet_delay, 'NWP_phase', self.s_lin, self.s_pix,
                                                'harmonie_' + self.ecmwf_type + '_wet' + self.coor_out.sample)

            return True

        except Exception:
            log_file = os.path.join(self.slave.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating aps from ECMWF for ' +
                              self.slave.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating aps from ECMWF for ' +
                  self.slave.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates, e_type='interim', split=False):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.

        if 'ecmwf_aps' in meta.processes.keys():
            meta_info = meta.processes['ecmwf_aps']
        else:
            meta_info = OrderedDict()

        if split:
            aps_types = ['ecmwf_' + e_type + '_aps', 'ecmwf_' + e_type + '_wet', 'ecmwf_' + e_type + '_hydrostatic']
            data_types = ['real4', 'real4', 'real4']
        else:
            aps_types = ['ecmwf_' + e_type + '_aps']
            data_types = ['real4']

        meta_info = coordinates.create_meta_data(aps_types, data_types, meta_info)

        meta.image_add_processing_step('ecmwf_aps', meta_info)

    @staticmethod
    def processing_info(coordinates, e_type='interim', split=False):

        # Fix the input coordinate system to fit the ecmwf grid... (1 km grid more or less)
        # TODO adapt the coor_in when the output is not in radar coordinates
        coor_in = CoordinateSystem(over_size=True)
        coor_in.create_radar_coordinates(multilook=[50, 200])

        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat['cmaster']['radar_DEM']['radar_DEM' + coordinates.sample]['file'] = 'radar_dem' + coordinates.sample
        input_dat['cmaster']['radar_DEM']['radar_DEM' + coordinates.sample]['coordinates'] = coordinates
        input_dat['cmaster']['radar_DEM']['radar_DEM' + coordinates.sample]['slice'] = coordinates.slice

        input_dat['cmaster']['radar_DEM']['radar_DEM' + coor_in.sample]['file'] = 'radar_dem' + coordinates.sample
        input_dat['cmaster']['radar_DEM']['radar_DEM' + coor_in.sample]['coordinates'] = coordinates
        input_dat['cmaster']['radar_DEM']['radar_DEM' + coor_in.sample]['slice'] = coordinates.slice

        # For multiprocessing this information is needed to define the selected area to deramp.
        for t in ['azimuth_angle', 'elevation_angle']:
            input_dat['cmaster']['azimuth_elevation_angle'][t + coor_in.sample]['file'] = t + coor_in.sample + '.raw'
            input_dat['cmaster']['azimuth_elevation_angle'][t + coor_in.sample]['coordinates'] = coor_in
            input_dat['cmaster']['azimuth_elevation_angle'][t + coor_in.sample]['slice'] = coor_in.slice
            input_dat['cmaster']['azimuth_elevation_angle'][t + coor_in.sample]['coor_change'] = 'resample'

        # For multiprocessing this information is needed to define the selected area to deramp.
        for t in ['lat', 'lon']:
            input_dat['cmaster']['geocode'][t + coor_in.sample]['file'] = t + coor_in.sample + '.raw'
            input_dat['cmaster']['geocode'][t + coor_in.sample]['coordinates'] = coor_in
            input_dat['cmaster']['geocode'][t + coor_in.sample]['slice'] = coor_in.slice
            input_dat['cmaster']['geocode'][t + coor_in.sample]['coor_change'] = 'resample'

        if split:
            aps_types = ['ecmwf_' + e_type + '_aps', 'ecmwf_' + e_type + '_wet', 'ecmwf_' + e_type + '_hydrostatic']
        else:
            aps_types = ['ecmwf_' + e_type + '_aps']

        output_dat = recursive_dict()
        for t in aps_types:
            output_dat['slave']['ecmwf_aps'][t + coordinates.sample]['file'] = t + coordinates.sample + '.raw'
            output_dat['slave']['ecmwf_aps'][t + coordinates.sample]['coordinates'] = coordinates
            output_dat['slave']['ecmwf_aps'][t + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('ecmwf_aps', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('ecmwf_aps', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('ecmwf_aps', file_type, coordinates)
