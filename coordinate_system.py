import xml.etree.cElementTree as ET
from collections import OrderedDict
import pyproj
import datetime
import numpy as np
import os
import osr

from rippl.find_coordinates import FindCoordinates


class CoordinateSystem():

    def __init__(self, over_size=False):

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
        self.over_size = over_size
        self.sparse_grid = False
        self.sparse_name = ''
        self.mask_grid = False
        self.mask_name = ''

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
        self.lat0 = 0
        self.lon0 = 0
        self.dlat = 0
        self.dlon = 0

        # Characteristics for projection type
        self.projection_type = ''
        self.proj4_str = ''
        self.x0 = 0
        self.y0 = 0
        self.dx = 0
        self.dy = 0

        # Init other variables
        self.meta_data = OrderedDict()
        self.geo = ''
        self.proj = ''
        self.tree = []

    def create_xml(self):

        root = ET.Element("root")

        radar = ET.SubElement(root, "radar")
        ET.SubElement(radar, 'multilook').text = str(self.multilook[0]) + '_' + str(self.multilook[1])
        ET.SubElement(radar, 'offset').text = str(self.offset[0]) + '_' + str(self.offset[1])
        ET.SubElement(radar, 'interval_lines').text = str(self.interval_lines)
        ET.SubElement(radar, 'interval_pixels').text = str(self.interval_pixels)
        ET.SubElement(radar, 'ml_lines_in').text = str(self.ml_lines_in)
        ET.SubElement(radar, 'ml_pixels_in').text = str(self.ml_pixels_in)
        ET.SubElement(radar, 'ml_lines_out').text = str(self.ml_lines_out)
        ET.SubElement(radar, 'ml_pixels_out').text = str(self.ml_pixels_out)

        geographic = ET.SubElement(root, "geographic")
        ET.SubElement(geographic, 'ellipse_type').text = self.ellipse_type
        ET.SubElement(geographic, 'lat0').text = str(self.lat0)
        ET.SubElement(geographic, 'lon0').text = str(self.lon0)
        ET.SubElement(geographic, 'dlat').text = str(self.dlat)
        ET.SubElement(geographic, 'dlon').text = str(self.dlon)

        projection = ET.SubElement(root, "projections")
        ET.SubElement(projection, 'ellipse_type').text = self.ellipse_type
        ET.SubElement(projection, 'projection_type').text = self.projection_type
        ET.SubElement(projection, 'proj4_str').text = self.proj4_str
        ET.SubElement(projection, 'x0').text = str(self.x0)
        ET.SubElement(projection, 'y0').text = str(self.y0)
        ET.SubElement(projection, 'dx').text = str(self.dx)
        ET.SubElement(projection, 'dy').text = str(self.dy)

        timing = ET.SubElement(root, "timing")
        ET.SubElement(timing, 'ra_time').text = str(self.ra_time)
        ET.SubElement(timing, 'az_time').text = str(self.az_time)
        ET.SubElement(timing, 'ra_step').text = str(self.ra_step)
        ET.SubElement(timing, 'az_step').text = str(self.az_step)

        grid_spec = ET.SubElement(root, "grid_spec")
        ET.SubElement(grid_spec, 'shape').text = str(self.ra_time)
        ET.SubElement(grid_spec, 'first_line').text = str(self.first_line)
        ET.SubElement(grid_spec, 'first_pixel').text = str(self.first_pixel)
        ET.SubElement(grid_spec, 'az_step').text = str(self.az_step)
        ET.SubElement(grid_spec, 'sample').text = self.sample
        ET.SubElement(grid_spec, 'meta_name').text = self.meta_name
        ET.SubElement(grid_spec, 'res_path').text = self.res_path
        ET.SubElement(grid_spec, 'over_size').text = str(self.over_size)
        ET.SubElement(grid_spec, 'sparse_grid').text = str(self.sparse_grid)
        ET.SubElement(grid_spec, 'sparse_name').text = str(self.sparse_name)
        ET.SubElement(grid_spec, 'mask_grid').text = str(self.mask_grid)
        ET.SubElement(grid_spec, 'mask_name').text = str(self.mask_name)

        grid_type = ET.SubElement(root, "grid_type").text = self.grid_type
        slice = ET.SubElement(root, "slice").text = str(self.slice)
        coor_str = ET.SubElement(root, "coor_str").text = self.coor_str

        self.xml = ET.ElementTree(root)

    def read_xml(self):
        # Load information from xml tree data.

        print('Working on this')

    def create_radar_coordinates(self, res_info='', multilook='', oversample='', offset='', sparse_name=''):

        if self.over_size:
            offset = [self.offset[0] - multilook[0] * 2, self.offset[1] - multilook[1] * 2]

        self.sample, self.multilook, self.oversample, self.offset, in_dat, out_dat = \
            FindCoordinates.multilook_coors([0, 0], oversample=oversample, multilook=multilook, offset=offset, interval=self.over_size)
        self.grid_type = 'radar_coordinates'

        self.sample = FindCoordinates.multilook_str(self.multilook, self.oversample, self.offset)[0]
        if sparse_name:
            self.sparse_grid = True
            self.sparse_name = sparse_name
            self.sample = self.sample + '_' + self.sparse_name

        if res_info:
            self.add_res_info(res_info)

    def create_geographic(self, dlat, dlon, res_info='', ellipse_type='WGS84', shape='', lat0='', lon0='',
                          oversample='', sparse_name=''):

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
        if sparse_name:
            self.sparse_grid = True
            self.sparse_name = sparse_name
            self.sample = self.sample + '_' + self.sparse_name

        self.geo = pyproj.Geod(ellps=ellipse_type)

        if res_info:
            self.add_res_info(res_info)

    def create_projection(self, dx, dy, res_info='', projection_type='', ellipse_type='WGS84', proj4_str='',
                          shape='', x0='', y0='', oversample='', sparse_name=''):
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
        if sparse_name:
            self.sparse_grid = True
            self.sparse_name = sparse_name
            self.sample = self.sample + '_' + self.sparse_name

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
                                                          offset=self.offset, interval=self.over_size)

                # Lines and pixels in case of multilook
                self.sample, self.multilook, self.oversample, self.offset, [self.ml_lines_in, self.ml_pixels_in], \
                [self.ml_lines_out, self.ml_pixels_out] = FindCoordinates.multilook_lines(orig_shape, multilook=self.multilook,
                                                                                  oversample=self.oversample,
                                                                                  offset=self.offset, interval=self.over_size)
                self.shape = [len(self.ml_lines_out), len(self.ml_pixels_out)]
                if self.sparse_grid and len(self.sparse_name) > 0:
                    self.sample = self.sample + '_' + self.sparse_name
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
                lat = [l[0] for l in res_info.polygon.exterior.coords]
                lon = [l[1] for l in res_info.polygon.exterior.coords]
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

    def create_gdal_projection(self, res_info):

        # Check if res_info is an ImageData dataset
        if not res_info:
            print('res_info should be an ImageData object')
            return

        # If geometry is not calculated yet.
        if len(res_info.lat_lim) == 0:
            res_info.geometry()

        geo_transform = np.zeros(6)
        projection = osr.SpatialReference()

        if self.grid_type == 'radar_coordinates':
            # Get the coordinates from the .res file
            total_n_s = ((res_info.corner_coordinates[0][0] - res_info.corner_coordinates[3][0]) * 0.5 +
                        (res_info.corner_coordinates[1][0] - res_info.corner_coordinates[2][0]) * 0.5)
            total_w_e = ((res_info.corner_coordinates[0][1] - res_info.corner_coordinates[1][1]) * 0.5 +
                        (res_info.corner_coordinates[3][1] - res_info.corner_coordinates[2][1]) * 0.5)
            skew_n_s = ((res_info.corner_coordinates[0][1] - res_info.corner_coordinates[3][1]) * 0.5 +
                        (res_info.corner_coordinates[1][1] - res_info.corner_coordinates[2][1]) * 0.5)
            skew_w_e = ((res_info.corner_coordinates[0][0] - res_info.corner_coordinates[1][0]) * 0.5 +
                        (res_info.corner_coordinates[3][0] - res_info.corner_coordinates[2][0]) * 0.5)

            geo_transform[0] = res_info.corner_coordinates[0][1]
            geo_transform[1] = - total_w_e / self.shape[1]
            geo_transform[2] = - skew_n_s / self.shape[0]
            geo_transform[3] = res_info.corner_coordinates[0][0]
            geo_transform[4] = - skew_w_e / self.shape[1]
            geo_transform[5] = - total_n_s / self.shape[0]

            # Assume that the projection is WGS84
            projection = osr.SpatialReference()
            projection.ImportFromEPSG(4326)

        elif self.grid_type == 'geographic':
            # Create geo transform based on lat/lon
            geo_transform[0] = self.lon0
            geo_transform[1] = self.dlon
            geo_transform[2] = 0
            geo_transform[3] = self.lat0
            geo_transform[4] = 0
            geo_transform[5] = self.dlat

            # Import projection from pyproj
            projection = osr.SpatialReference()
            projection.ImportFromEPSG(4326)

        elif self.grid_type == 'projection':
            # Create geo transform based on projection steps
            geo_transform[0] = self.x0
            geo_transform[1] = self.dx
            geo_transform[2] = 0
            geo_transform[3] = self.y0
            geo_transform[4] = 0
            geo_transform[5] = self.dy

            # Import projection from pyproj
            projection = osr.SpatialReference()
            projection.ImportFromProj4(self.proj.srs)

        else:
            print('Grid type ' + self.grid_type + ' is ')

        return projection, geo_transform

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
            meta_info[data_name + '_sparse_grid'] = str(self.sparse_grid)
            meta_info[data_name + '_sparse_name'] = str(self.sparse_name)

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

    def get_offset(self, offset_coor, ml_off=True):
        # This function gets the offset in lines and pixels with another coordinate system.

        if not self.grid_type == offset_coor.grid_type:
            print('The offset grid should be the same grid type.')
            return

        first_line_off = offset_coor.first_line - self.first_line
        first_pixel_off = offset_coor.first_pixel - self.first_pixel

        if self.grid_type == 'radar_coordinates':

            if offset_coor.ra_step != self.ra_step or offset_coor.az_step != self.az_step \
                 or offset_coor.multilook != self.multilook:
                print('Pixel spacing should be the same')
                return

            ra_time_offset = np.int(np.round((offset_coor.ra_time - self.ra_time) / self.ra_step))
            az_time_offset = np.int(np.round((offset_coor.az_time - self.az_time) / self.az_step))

            if ml_off:
                line_offset = az_time_offset + first_line_off + (offset_coor.offset[0] - self.offset[0])
                pixel_offset = ra_time_offset + first_pixel_off + (offset_coor.offset[1] - self.offset[1])
            else:
                line_offset = az_time_offset + first_line_off
                pixel_offset = ra_time_offset + first_pixel_off

        elif self.grid_type == 'geographic':

            if offset_coor.dlat != self.dlat or offset_coor.dlon != self.dlon:
                print('Pixel spacing should be the same')
                return

            lat_offset = np.int(np.round((offset_coor.lat0 - self.lat0) / self.dlat))
            lon_offset = np.int(np.round((offset_coor.lon0 - self.lon0) / self.dlon))

            line_offset = first_line_off + lat_offset
            pixel_offset = first_pixel_off + lon_offset

        elif self.grid_type == 'projection':

            if offset_coor.dy != self.dy or offset_coor.dx != self.dx:
                print('Pixel spacing should be the same')
                return

            y_offset = np.int(np.round((offset_coor.y0 - self.y0) / self.dy))
            x_offset = np.int(np.round((offset_coor.x0 - self.x0) / self.dx))

            line_offset = first_line_off + y_offset
            pixel_offset = first_pixel_off + x_offset
        else:
            return

        return [line_offset, pixel_offset]

    # Create a single identifier for every coordinate system
    def coor_str(self, coor):
        # Compare other metadata to check whether they are similar.

        if self.grid_type == 'radar_coordinates':
            self.coor_str = self.grid_type + '_' + 'ovr'

    def ell2proj(self, lat, lon):

        if isinstance(self.proj, pyproj.Proj) and isinstance(self.geo, pyproj.Geod):
            x, y = self.proj(lon, lat, inverse=False)
        else:
            print('Either the projection or geographic coordinate system is not loaded as pyproj class')
            return

        return x, y

    def proj2ell(self, x, y):

        if isinstance(self.proj, pyproj.Proj) and isinstance(self.geo, pyproj.Geod):
            lon, lat = self.proj(x, y, inverse=True)
        else:
            print('Either the projection or geographic coordinate system is not loaded as pyproj class')
            return

        return lat, lon
