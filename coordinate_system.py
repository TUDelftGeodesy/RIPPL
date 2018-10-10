from find_coordinates import FindCoordinates
from collections import OrderedDict
import pyproj
import datetime
from image_data import ImageData
import numpy as np


class CoordinateSystem():

    def __init__(self):

        self.grid_type = ''
        self.slice = 'False'

        # Characteristics for all images
        self.shape = [0, 0]
        self.first_line = 1
        self.first_pixel = 1
        self.oversample = [1, 1]
        self.sample = ''

        # Characteristics for radar type
        self.multilook = [1, 1]
        self.factor = [1, 1]
        self.offset = [0, 0]

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

        if isinstance(res_info, ImageData):
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

        if oversample == [1, 1]:
            self.sample = ellipse_type + '_stp_' + str(dlat * 3600).zfill(0) + '_' + str(dlon * 3600).zfill(0)
        else:
            self.sample = self.sample + '_ovr_' + str(self.oversample[0]) + '_' + str(self.oversample[1])

        self.geo = pyproj.Geod(ellps=ellipse_type)

        if isinstance(res_info, ImageData):
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

        if oversample == [1, 1]:
            self.sample = projection_type + '_stp_' + str(dx).zfill(0) + '_' + str(dy).zfill(0)
        else:
            self.sample = self.sample + '_ovr_' + str(self.oversample[0]) + '_' + str(self.oversample[1])

        self.geo = pyproj.Geod(ellps=ellipse_type)
        if proj4_str:
            self.proj = pyproj.Proj(proj4_str)
        else:
            self.proj = pyproj.Proj(proj=projection_type, ellps=ellipse_type)

        if isinstance(res_info, ImageData):
            self.add_res_info(res_info)

    def add_res_info(self, res_info, buf=0.01, round=1, change_ref=True):
        # Here we add extra information to our radar coordinates to get the first line/pixel and original image size.
        # This also generates some extra info on line and pixel numbers in the new configuration.

        if not isinstance(res_info, ImageData):
            print('res_info should be an ImageData object')
            return

        if self.grid_type == 'radar_coordinates':

            if not change_ref and self.az_time != 0 and self.ra_time != 0:
                orig_shape, self.az_time, self.ra_time, self.az_step, self.ra_step, self.first_line, self.first_pixel = \
                    CoordinateSystem.res_pixel_spacing(res_info, self.az_time, self.ra_time)
            else:
                orig_shape, self.az_time, self.ra_time, self.az_step, self.ra_step, self.first_line, self.first_pixel = \
                    CoordinateSystem.res_pixel_spacing(res_info)

            if self.factor != [1, 1] or self.factor == '':
                self.sample, self.multilook, self.oversample, self.offset, [s_lin, s_pix, self.shape] = \
                    FindCoordinates.interval_sparse_coors(orig_shape, multilook=self.multilook, oversample=self.oversample,
                                                          offset=self.offset, factor=self.factor)
            else:
                self.sample, self.multilook, self.oversample, self.offset, [s_lin, s_pix, self.shape] = \
                    FindCoordinates.interval_coors(orig_shape, multilook=self.multilook, oversample=self.oversample,
                                                   offset=self.offset)

        elif self.grid_type == 'geographic':

            lat_lim = res_info.lat_lim + np.array([-buf, buf])
            lon_lim = res_info.lon_lim + np.array([-buf, buf])
            first_lat = np.floor((lat_lim[0] % round) / self.dlat) * self.dlat + np.floor(lat_lim[0] / round) * round
            first_lon = np.floor((lon_lim[0] % round) / self.dlon) * self.dlon + np.floor(lon_lim[0] / round) * round

            if not change_ref and self.lat0 != '' and self.lon0 != 0:
                self.first_line = int((first_lat - self.lat0) / self.dlat) + 1
                self.first_pixel = int((first_lon - self.lon0) / self.dlon) + 1
            else:
                self.lat0 = first_lat
                self.lon0 = first_lon
            self.shape = np.array([np.ceil((lat_lim[1] - first_lat) / self.dlat),
                                   np.ceil((lon_lim[1] - first_lon) / self.dlon)])

        elif self.grid_type == 'projection':

            lat = [l[0] for l in res_info.polygon.coords]
            lon = [l[1] for l in res_info.polygon.coords]
            x, y = self.ell2proj(lat, lon)

            x_lim = [np.min(x), np.max(x)] + np.array([-buf, buf])
            y_lim = [np.min(y), np.max(y)] + np.array([-buf, buf])
            first_x = np.floor((x_lim[0] % round) / self.dx) * self.dx + np.floor(x_lim[0] / round) * round
            first_y = np.floor((y_lim[0] % round) / self.dy) * self.dy + np.floor(y_lim[0] / round) * round

            if not change_ref and self.lat0 != '' and self.lon0 != 0:
                self.first_line = int((first_x - self.x0) / self.dx) + 1
                self.first_pixel = int((first_y - self.y0) / self.dy) + 1
            else:
                self.x0 = first_x
                self.y0 = first_y

            self.shape = np.array([np.ceil((x_lim[1] - self.x0) / self.dx),
                                   np.ceil((y_lim[1] - self.y0) / self.dy)])

        self.slice = res_info.processes['crop']['Data_slice']

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
            first_line = res_info.processes[step_crop]['Data_first_line'] + int((new_az_time - az_time) / az_step)
        else:
            az_time = new_az_time
            first_line = res_info.processes[step_crop]['Data_first_line']

        if ra_time != 0.0:
            first_pixel = res_info.processes[step_crop]['Data_first_pixel'] + int((new_ra_time - ra_time) / ra_step)
        else:
            ra_time = new_ra_time
            first_pixel = res_info.processes[step_crop]['Data_first_pixel']

        shape = int(res_info.processes[step_crop]['Data_lines'], int(res_info.processes[step_crop]['Data_pixels']))

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

            if self.grid_type == 'radar_coordinates':
                meta_info[data_name + '_first_line'] = str(self.first_line)
                meta_info[data_name + '_first_pixel'] = str(self.first_pixel)
                meta_info[data_name + '_multilook_azimuth'] = str(self.multilook[0])
                meta_info[data_name + '_multilook_range'] = str(self.multilook[1])
                meta_info[data_name + '_oversampling_azimuth'] = str(self.oversample[0])
                meta_info[data_name + '_oversampling_range'] = str(self.oversample[1])
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
                meta_info[data_name + '_x0'] = str(self.lat0)
                meta_info[data_name + '_y0'] = str(self.lon0)
                meta_info[data_name + '_dx'] = str(self.dlat)
                meta_info[data_name + '_dy'] = str(self.dlon)

        return meta_info

    def compare_meta_data(self):
        # Compare other metadata to check whether they are similar.
        print('Not needed in project yet!')

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
