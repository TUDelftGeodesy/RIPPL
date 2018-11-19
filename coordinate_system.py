from find_coordinates import FindCoordinates
from collections import OrderedDict
import pyproj
import datetime
import numpy as np
import os


class CoordinateSystem():

    def __init__(self):

        self.grid_type = ''
        self.slice = False
        self.coor_str = ''

        # Characteristics for all images
        self.shape = [0, 0]
        self.first_line = 1
        self.first_pixel = 1
        self.oversample = [1, 1]
        self.sample = ''
        self.meta_name = ''
        self.res_path = ''

        # Characteristics for radar type
        self.multilook = [1, 1]
        self.offset = [0, 0]
        self.interval_lines = []
        self.interval_pixels = []
        self.ml_lines_in = []
        self.ml_pixels_in = []
        self.ml_lines_out = []
        self.ml_pixels_out = []

        # Next values are all in seconds (azimuth time from start of day to max of 25 hours)
        self.ra_time = 0
        self.az_time = 0
        self.az_step = 0
        self.ra_step = 0

        # Characteristics for geographic type
        self.ellipse_type = ''
        self.shape = [0, 0]
        self.lat0 = 0
        self.lon0 = 0
        self.dlat = 0
        self.dlon = 0

        # Characteristics for projection type
        self.projection_type = ''
        self.proj4_str = ''
        self.shape = [0, 0]
        self.x0 = 0
        self.y0 = 0
        self.dx = 0
        self.dy = 0

        # Init other variables
        self.meta_data = OrderedDict()
        self.geo = ''
        self.proj = ''

    def create_radar_coordinates(self, res_info='', multilook='', oversample='', offset=''):

        self.sample, self.multilook, self.oversample, self.offset, in_dat, out_dat = \
            FindCoordinates.multilook_coors([0, 0], oversample=oversample, multilook=multilook, offset=offset)
        self.grid_type = 'radar_coordinates'

        self.sample = FindCoordinates.multilook_str(self.multilook, self.oversample, self.offset)[0]

        if res_info:
            self.add_res_info(res_info)

    def create_geographic(self, dlat, dlon, res_info='', ellipse_type='WGS84', shape='', lat0='', lon0='', oversample=''):

        self.ellipse_type = ellipse_type
        self.shape = shape
        self.lat0 = lat0
        self.lon0 = lon0
        self.dlat = dlat
        self.dlon = dlon
        self.grid_type = 'geographic'

        if len(oversample) != 2:
            self.oversample = [1, 1]
        else:
            self.oversample = oversample

        self.sample = '_' + ellipse_type + '_stp_' + str(int(np.round(dlat * 3600))) + '_' + str(int(np.round(dlon * 3600)))
        if not self.oversample == [1, 1]:
            self.sample = self.sample + '_ovr_' + str(self.oversample[0]) + '_' + str(self.oversample[1])

        self.geo = pyproj.Geod(ellps=ellipse_type)

        if res_info:
            self.add_res_info(res_info)

    def create_projection(self, dx, dy, res_info='', projection_type='', ellipse_type='WGS84', proj4_str='',
                          shape='', x0='', y0='', oversample=''):
        # Define projection. For specific projections visit https://proj4.org

        self.ellipse_type = ellipse_type
        self.projection_type = projection_type
        self.proj4_str = proj4_str  # Any additional information if needed..
        self.shape = shape
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.grid_type = 'projection'

        if len(oversample) != 2:
            self.oversample = [1, 1]
        else:
            self.oversample = oversample

        self.sample = '_' + projection_type + '_stp_' + str(dx).zfill(0) + '_' + str(dy).zfill(0)
        if not self.oversample == [1, 1]:
            self.sample = self.sample + '_ovr_' + str(self.oversample[0]) + '_' + str(self.oversample[1])

        self.geo = pyproj.Geod(ellps=ellipse_type)
        if proj4_str:
            self.proj = pyproj.Proj(proj4_str)
        else:
            self.proj = pyproj.Proj(proj=projection_type, ellps=ellipse_type)

        if res_info:
            self.add_res_info(res_info)

    def add_res_info(self, res_info, buf=0.1, round=1, change_ref=True, coreg_grid=True, old_coor=''):
        # Here we add extra information to our radar coordinates to get the first line/pixel and original image size.
        # This also generates some extra info on line and pixel numbers in the new configuration.

        if not res_info:
            print('res_info should be an ImageData object')
            return

        # Add the .res name
        meta_name = os.path.basename(os.path.dirname(res_info.res_path))
        self.res_path = res_info.res_path
        if meta_name.startswith('slice'):
            self.meta_name = meta_name
        else:
            self.meta_name = 'full'

        if self.grid_type == 'radar_coordinates':

            if not change_ref and self.az_time != 0 and self.ra_time != 0:
                self.az_time, self.ra_time, self.az_step, self.ra_step, self.first_line, self.first_pixel, orig_shape = \
                    CoordinateSystem.res_pixel_spacing(res_info, self.az_time, self.ra_time, coreg_grid=coreg_grid)
            else:
                self.az_time, self.ra_time, self.az_step, self.ra_step, self.first_line, self.first_pixel, orig_shape = \
                    CoordinateSystem.res_pixel_spacing(res_info, coreg_grid=coreg_grid)

            if old_coor == '':
                # Lines and pixels in case of intervals
                self.sample, self.multilook, self.oversample, self.offset, [self.interval_lines, self.interval_pixels] = \
                    FindCoordinates.interval_lines(orig_shape, multilook=self.multilook, oversample=self.oversample,
                                                          offset=self.offset)

                # Lines and pixels in case of multilook
                self.sample, self.multilook, self.oversample, self.offset, [self.ml_lines_in, self.ml_pixels_in], \
                [self.ml_lines_out, self.ml_pixels_out] = FindCoordinates.multilook_lines(orig_shape, multilook=self.multilook,
                                                                                  oversample=self.oversample,
                                                                                  offset=self.offset)
                self.shape = [len(self.ml_lines_out), len(self.ml_pixels_out)]
            else:
                self.sample = old_coor.sample
                self.shape = old_coor.shape
                self.multilook = old_coor.multilook
                self.oversample = old_coor.oversample
                self.offset = old_coor.offset


        elif self.grid_type == 'geographic':

            if old_coor == '':
                lat_lim = res_info.lat_lim + np.array([-buf, buf])
                lon_lim = res_info.lon_lim + np.array([-buf, buf])
                first_lat = np.floor((lat_lim[0] % round) / self.dlat) * self.dlat + np.floor(lat_lim[0] / round) * round
                first_lon = np.floor((lon_lim[0] % round) / self.dlon) * self.dlon + np.floor(lon_lim[0] / round) * round

                self.shape = np.array([np.round((lat_lim[1] - first_lat) / self.dlat),
                                       np.round((lon_lim[1] - first_lon) / self.dlon)]).astype(np.int32)
            else:
                first_lat = old_coor.lat0
                first_lon = old_coor.lon0
                self.shape = old_coor.shape

            if not change_ref and self.lat0 != 0 and self.lon0 != 0:
                self.first_line = int((first_lat - self.lat0) / self.dlat) + 1
                self.first_pixel = int((first_lon - self.lon0) / self.dlon) + 1
            else:
                self.lat0 = first_lat
                self.lon0 = first_lon

        elif self.grid_type == 'projection':

            if old_coor == '':
                lat = [l[0] for l in res_info.polygon.coords]
                lon = [l[1] for l in res_info.polygon.coords]
                x, y = self.ell2proj(lat, lon)

                x_lim = [np.min(x), np.max(x)] + np.array([-buf, buf])
                y_lim = [np.min(y), np.max(y)] + np.array([-buf, buf])
                first_x = np.floor((x_lim[0] % round) / self.dx) * self.dx + np.floor(x_lim[0] / round) * round
                first_y = np.floor((y_lim[0] % round) / self.dy) * self.dy + np.floor(y_lim[0] / round) * round

                self.shape = np.array([np.round((x_lim[1] - self.x0) / self.dx),
                                       np.round((y_lim[1] - self.y0) / self.dy)]).astype(np.int32)

            else:
                first_x = old_coor.x0
                first_y = old_coor.y0
                self.shape = old_coor.shape

            if not change_ref and self.lat0 != '' and self.lon0 != 0:
                self.first_line = int((first_x - self.x0) / self.dx) + 1
                self.first_pixel = int((first_y - self.y0) / self.dy) + 1
            else:
                self.x0 = first_x
                self.y0 = first_y

        if 'readfiles' in res_info.processes:
            self.slice = res_info.processes['readfiles']['slice'] == 'True'
        else:
            self.slice = res_info.processes['coreg_readfiles']['slice'] == 'True'

    @staticmethod
    def res_pixel_spacing(res_info, az_time=0.0, ra_time=0.0, coreg_grid=True):

        if res_info.process_control['coreg_readfiles'] == '1' and coreg_grid:
            step_meta = 'coreg_readfiles'
            step_crop = 'coreg_crop'
        else:
            step_meta = 'readfiles'
            step_crop = 'crop'

        az_datetime = res_info.processes[step_meta]['First_pixel_azimuth_time (UTC)']
        az_seconds = (datetime.datetime.strptime(az_datetime, '%Y-%m-%dT%H:%M:%S.%f') -
                      datetime.datetime.strptime(az_datetime[:10], '%Y-%m-%d'))
        az_step = 1 / float(res_info.processes[step_meta]['Pulse_Repetition_Frequency (computed, Hz)'])
        ra_step = 1 / float(res_info.processes[step_meta]['Range_sampling_rate (computed, MHz)']) / 1000000

        new_az_time = az_seconds.seconds + az_seconds.microseconds / 1000000.0
        new_ra_time = float(res_info.processes[step_meta]['Range_time_to_first_pixel (2way) (ms)']) / 1000

        if az_time != 0.0:
            first_line = int(res_info.processes[step_crop]['crop_first_line']) + int((new_az_time - az_time) / az_step)
        else:
            az_time = new_az_time
            first_line = int(res_info.processes[step_crop]['crop_first_line'])

        if ra_time != 0.0:
            first_pixel = int(res_info.processes[step_crop]['crop_first_pixel']) + int((new_ra_time - ra_time) / ra_step)
        else:
            ra_time = new_ra_time
            first_pixel = int(res_info.processes[step_crop]['crop_first_pixel'])

        shape = [int(res_info.processes[step_crop]['crop_lines']), int(res_info.processes[step_crop]['crop_pixels'])]

        return az_time, ra_time, az_step, ra_step, first_line, first_pixel, shape

    def create_meta_data(self, data_names, data_types, meta_info=''):
        # Create meta data information to save to .res files.

        if not isinstance(meta_info, OrderedDict):
            meta_info = OrderedDict()

        for data_name, data_type in zip(data_names, data_types):

            data_name = data_name + self.sample
            meta_info[data_name + '_output_file'] = data_name + '.raw'
            meta_info[data_name + '_output_format'] = data_type
            meta_info[data_name + '_lines'] = str(self.shape[0])
            meta_info[data_name + '_pixels'] = str(self.shape[1])
            meta_info[data_name + '_first_line'] = str(self.first_line)
            meta_info[data_name + '_first_pixel'] = str(self.first_pixel)

            if self.grid_type == 'radar_coordinates':
                meta_info[data_name + '_multilook_azimuth'] = str(self.multilook[0])
                meta_info[data_name + '_multilook_range'] = str(self.multilook[1])
                meta_info[data_name + '_oversample_azimuth'] = str(self.oversample[0])
                meta_info[data_name + '_oversample_range'] = str(self.oversample[1])
                meta_info[data_name + '_offset_azimuth'] = str(self.offset[0])
                meta_info[data_name + '_offset_range'] = str(self.offset[1])

            elif self.grid_type == 'geographic':
                meta_info[data_name + '_ellipse_type'] = self.ellipse_type
                meta_info[data_name + '_lat0'] = str(self.lat0)
                meta_info[data_name + '_lon0'] = str(self.lon0)
                meta_info[data_name + '_dlat'] = str(self.dlat)
                meta_info[data_name + '_dlon'] = str(self.dlon)

            elif self.grid_type == 'projection':
                meta_info[data_name + '_projection_type'] = self.projection_type
                meta_info[data_name + '_ellipse_type'] = self.ellipse_type
                meta_info[data_name + '_proj4_str'] = self.proj4_str
                meta_info[data_name + '_x0'] = str(self.x0)
                meta_info[data_name + '_y0'] = str(self.y0)
                meta_info[data_name + '_dx'] = str(self.dx)
                meta_info[data_name + '_dy'] = str(self.dy)

        return meta_info

    # Create a single identifier for every coordinate system
    def coor_str(self, coor):
        # Compare other metadata to check whether they are similar.

        if self.grid_type == 'radar_coordinates':
            self.coor_str = self.grid_type + '_' + 'ovr'

    def ell2proj(self, lat, lon):

        if isinstance(self.proj, pyproj.Proj) and isinstance(self.geo, pyproj.Geod):
            x, y = self.proj(lat, lon, inverse=False)
        else:
            print('Either the projection or geographic coordinate system is not loaded as pyproj class')
            return

        return x, y

    def proj2ell(self, x, y):

        if isinstance(self.proj, pyproj.Proj) and isinstance(self.geo, pyproj.Geod):
            lat, lon = self.proj(x, y, inverse=True)
        else:
            print('Either the projection or geographic coordinate system is not loaded as pyproj class')
            return

        return lat, lon
