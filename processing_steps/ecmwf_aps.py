# The following class creates an interferogram from a master and slave image.

import os
import numpy as np
import datetime
import logging
from collections import OrderedDict

from doris_processing.image_data import ImageData
from doris_processing.find_coordinates import FindCoordinates

from slant_delay.weather_models.ECMWF.ecmwf_type import ECMWFType
from slant_delay.weather_models.ECMWF.ecmwf_download import ECMWFdownload
from slant_delay.weather_models.ECMWF.ecmwf_load_file import ECMWFData
from slant_delay.weather_models.model_ray_tracing import ModelRayTracing
from slant_delay.weather_models.radar_data import RadarData
from slant_delay.weather_models.model_to_delay import ModelToDelay
from slant_delay.weather_models.model_interpolate_delays import ModelInterpolateDelays


class EcmwfAps(object):
    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, ref_meta, s_lin=0, s_pix=0, lines=0, multilook_coarse='', multilook_fine='', offset='',
                 weather_data_archive='', ecmwf_type='era5', time_interp='nearest', split=False, download=True):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.meta = meta

        # Load the reference image, used for geocoding and resampling of the other images
        if isinstance(ref_meta, str):
            if len(ref_meta) != 0:
                self.ref_meta = ImageData(ref_meta, 'single')
        elif isinstance(ref_meta, ImageData):
            self.ref_meta = ref_meta

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
            self.ecmwf_type = 'era5'
        else:
            self.ecmwf_type = ecmwf_type

        if len(weather_data_archive) == 0 or not os.path.exists(weather_data_archive):
            self.weather_data_archive = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.ref_meta.folder))), 'weather_models')
        else:
            self.weather_data_archive = weather_data_archive
        self.weather_data_folder = os.path.join(self.weather_data_archive, 'ecmwf_data')

        self.download = download
        self.s_lin = s_lin
        self.s_pix = s_pix

        # Define the output grid
        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop

        # Information is loaded for:
        # 1. the output multilooking factor and offset
        # 2. the output multilooking factor and offset transformed to an interval and buffer for geocoding.
        self.str_ml, self.multilook_fine, self.offset, self.ml_shape, self.fine_lines, self.fine_pixels, \
        str_int_coarse, interval_coarse, buffer_coarse, offset_coarse, coarse_shape, self.coarse_lines, self.coarse_pixels, \
        str_int_fine, interval_fine, buffer_fine, offset_fine, fine_shape \
            = FindCoordinates.interval_multilook_coors(self.meta, s_lin, s_pix, lines, multilook_coarse=multilook_coarse,
                                                       multilook_fine=multilook_fine, offset=offset)

        # Load data input grid
        self.lat = self.ref_meta.image_load_data_memory('geocode', 0, 0, coarse_shape, 'Lat' + str_int_coarse)
        self.lon = self.ref_meta.image_load_data_memory('geocode', 0, 0, coarse_shape, 'Lon' + str_int_coarse)
        self.height = self.ref_meta.image_load_data_memory('radar_dem', 0, 0, coarse_shape, 'Data' + str_int_coarse)
        self.azimuth_angle = self.ref_meta.image_load_data_memory('azimuth_elevation_angle', 0, 0, coarse_shape, 'Azimuth_angle' + str_int_coarse)
        self.elevation_angle = self.ref_meta.image_load_data_memory('azimuth_elevation_angle', 0, 0, coarse_shape, 'Elevation_angle' + str_int_coarse)

        # Load height of multilook grid (from an interval, buffer grid, which makes it a bit complicated...)
        self.out_height = self.ref_meta.image_load_data_memory('radar_dem', offset_fine[0], offset_fine[1], fine_shape, 'Data' + str_int_fine)
        self.simulated_delay = []
        self.split = split
        if self.split:
            self.hydrostatic_delay = []
            self.wet_delay = []

    def __call__(self, latlim='', lonlim=''):
        # Check if needed data is loaded
        if len(self.lat) == 0 or len(self.lon) == 0 or len(self.height) == 0 or len(self.azimuth_angle) == 0 or len(self.elevation_angle) == 0:
            print('Missing input data for ray tracing weather model ' + self.ref_meta.folder +
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
            date = datetime.datetime.strptime(self.meta.processes['readfiles']['First_pixel_azimuth_time (UTC)'], '%Y-%m-%dT%H:%M:%S.%f')

            ecmwf_type = ECMWFType(self.ecmwf_type)
            radar_data = RadarData(self.time_interp, ecmwf_type.t_step, 0)
            radar_data.match_overpass_weather_model(date)

            # Load the ECMWF data
            # Download the needed files
            # For simplicity we limit ourselves here to europe is the lat/lon limits are not given.
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
            pixel_points, lines_points = np.meshgrid(self.fine_pixels, self.fine_lines)
            point_delays = ModelInterpolateDelays(self.coarse_lines, self.coarse_pixels, split=self.split)
            point_delays.add_interp_points(np.ravel(lines_points), np.ravel(pixel_points), np.ravel(self.out_height))

            point_delays.add_delays(ray_delays.spline_delays)
            point_delays.interpolate_points()

            # Save the file
            shp = (len(self.fine_lines), len(self.fine_pixels))
            self.simulated_delay = point_delays.interp_delays['total'][date].reshape(shp).astype(np.float32)
            if self.split:
                self.hydrostatic_delay = point_delays.interp_delays['hydrostatic'][date].reshape(shp).astype(np.float32)
                self.wet_delay = point_delays.interp_delays['wet'][date].reshape(shp).astype(np.float32)

            ray_delays.remove_delay(date)

            # If needed do the multilooking step
            if self.split:
                self.add_meta_data(self.meta, self.str_ml, self.ml_shape, self.multilook_fine, self.offset,
                                   self.ecmwf_type, ['total', 'hydrostatic', 'wet'])
            else:
                self.add_meta_data(self.meta, self.str_ml, self.ml_shape, self.multilook_fine, self.offset,
                                   self.ecmwf_type, ['total'])

            self.meta.image_new_data_memory(self.simulated_delay, 'NWP_phase', self.s_lin, self.s_pix, 'ECMWF_' + self.ecmwf_type + '_total' + self.str_ml)
            if self.split:
                self.meta.image_new_data_memory(self.hydrostatic_delay, 'NWP_phase', self.s_lin, self.s_pix,
                                                'ECMWF_' + self.ecmwf_type + '_hydrostatic' + self.str_ml)
                self.meta.image_new_data_memory(self.wet_delay, 'NWP_phase', self.s_lin, self.s_pix,
                                                'ECMWF_' + self.ecmwf_type + '_wet' + self.str_ml)

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating aps from ECMWF for ' +
                              self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating aps from ECMWF for ' +
                  self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_output_files(meta, to_disk):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        for s in to_disk:
            meta.image_create_disk('NWP_phase', s)

    def save_to_disk(self, to_disk=''):

        if not to_disk:
            if self.split:
                to_disk = ['ECMWF_' + self.ecmwf_type + '_total' + self.str_ml,
                           'ECMWF_' + self.ecmwf_type + '_wet' + self.str_ml,
                           'ECMWF_' + self.ecmwf_type + '_hydrostatic' + self.str_ml]
            else:
                to_disk = ['ECMWF_' + self.ecmwf_type + '_total' + self.str_ml]

        for s in to_disk:
            self.meta.image_memory_to_disk('NWP_phase', s)



    @staticmethod
    def add_meta_data(meta, sample, shape, multilook, offset, ecmwf_type, data_types):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'NWP_phase' in meta.processes.keys():
            meta_info = meta.processes['NWP_phase']
        else:
            meta_info = OrderedDict()

        if 'coreg_crop' in meta.processes.keys():
            step = 'coreg_crop'
        else:
            step = 'crop'
        for data_type in data_types:
            dat = 'ECMWF_' + ecmwf_type + '_' + data_type + sample
            meta_info[dat + '_output_file'] = dat + '.raw'
            meta_info[dat + '_output_format'] = 'real4'

            meta_info[dat + '_lines'] = str(shape[0])
            meta_info[dat + '_pixels'] = str(shape[1])
            meta_info[dat + '_first_line'] = meta.processes[step]['Data_first_line']
            meta_info[dat + '_first_pixel'] = meta.processes[step]['Data_first_pixel']
            meta_info[dat + '_multilook_azimuth'] = str(multilook[0])
            meta_info[dat + '_multilook_range'] = str(multilook[1])
            meta_info[dat + '_offset_azimuth'] = str(offset[0])
            meta_info[dat + '_offset_range'] = str(offset[1])

        meta.image_add_processing_step('NWP_phase', meta_info)
