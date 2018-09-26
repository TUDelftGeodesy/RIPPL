from find_coordinates import FindCoordinates
from collections import OrderedDict
import pyproj
from image_data import ImageData


class CoordinateSystem():

    def __init__(self, coor_dat='', res_info=''):

        if coor_dat:
            self.coor_dat = coor_dat
        else:
            self.coor_dat = dict()

        # Find the grid type (radar, geographic, projection)
        self.grid_type = ''
        if coor_dat and isinstance(coor_dat, dict):
            self.grid_type = coor_dat['type']

        # Characteristics for radar type
        self.multilook = [1, 1]
        self.oversample = [1, 1]
        self.offset = [0, 0]
        self.factor = [1, 1]
        self.first_line = 0
        self.first_pixel = 0
        self.slice = 'False'

        if self.grid_type == 'radar_coordinates':
            self.multilook = coor_dat['multilook']
            self.oversample = coor_dat['oversample']
            self.offset = coor_dat['offset']
            self.factor = coor_dat['factor']
            if coor_dat['shape_info'] == True:
                self.shape = coor_dat['shape']
                self.first_line = coor_dat['first_line']
                self.first_pixel = coor_dat['first_pixel']
            elif isinstance(res_info, ImageData):
                self.add_res_info(res_info)

        # Characteristics for geographic type
        self.ellipse_type = ''
        self.shape = [0, 0]
        self.lat0 = 0
        self.lon0 = 0
        self.dlat = 0
        self.dlon = 0
        if self.grid_type == 'geographic':
            self.shape = coor_dat['shape']
            self.lat0 = coor_dat['lat0']
            self.lon0 = coor_dat['lon0']
            self.dlat= coor_dat['dlat']
            self.dlon = coor_dat['dlat']
            self.ellipse_type = coor_dat['ellipse_type']

            self.geo = pyproj.Geod(ellps=self.ellipse_type)

        # Characteristics for projection type
        self.projection_type = ''
        self.proj_str = ''
        self.shape = [0, 0]
        self.x0 = 0
        self.y0 = 0
        self.dx = 0
        self.dy = 0
        if self.grid_type == 'projection':
            self.shape = coor_dat['shape']
            self.x0 = coor_dat['x0']
            self.y0 = coor_dat['y0']
            self.dx= coor_dat['dx']
            self.dy = coor_dat['dy']
            self.proj4_str = coor_dat['proj4_str']
            self.ellipse_type = coor_dat['ellipse_type']
            self.projection_type = coor_dat['projection_type']

            self.geo = pyproj.Geod(ellps=self.ellipse_type)
            if self.proj4_str:
                self.proj = pyproj.Proj(self.proj4_str)
            else:
                self.proj = pyproj.Proj(proj=self.projection_type, ellps=self.ellipse_type)

        # Init other variables
        self.meta_data = OrderedDict()
        self.geo = ''
        self.proj = ''

    def create_radar_coordinates(self, res_info='', multilook='', oversample='', offset='', factor=''):

        self.sample, self.multilook, self.oversample, self.offset, in_dat, out_dat = \
            FindCoordinates.multilook_coors([0, 0], oversample=oversample, multilook=multilook, offset=offset)

        if isinstance(res_info, ImageData):
            self.add_res_info(res_info)

        self.coor_dat['type'] = 'radar_coordinates'
        self.coor_dat['multilook'] = self.multilook
        self.coor_dat['oversample'] = self.oversample
        self.coor_dat['offset'] = self.offset
        self.coor_dat['sample'] = self.sample

    def add_res_info(self, res_info):
        # Here we add extra information to our radar coordinates to get the first line/pixel and original image size.
        # This also generates some extra info on line and pixel numbers in the new configuration.

        orig_shape = [res_info.processes['crop']['Data_lines'], res_info.processes['crop']['Data_pixels']]

        if self.factor != [1, 1] or self.factor == '':
            self.sample, self.multilook, self.oversample, self.offset, [s_lin, s_pix, self.shape] = \
                FindCoordinates.interval_sparse_coors(orig_shape, multilook=self.multilook, oversample=self.oversample,
                                                      offset=self.offset, factor=self.factor)
        else:
            self.sample, self.multilook, self.oversample, self.offset, [s_lin, s_pix, self.shape] = \
                FindCoordinates.interval_coors(orig_shape, multilook=self.multilook, oversample=self.oversample,
                                               offset=self.offset)

        self.coor_dat['shape'] = self.shape
        self.coor_dat['multilook'] = self.multilook
        self.coor_dat['oversample'] =  self.oversample
        self.coor_dat['offset'] = self.offset
        self.coor_dat['factor'] = self.factor
        self.coor_dat['first_line'] = res_info.processes['crop']['Data_first_line']
        self.coor_dat['first_pixel'] = res_info.processes['crop']['Data_first_pixel']
        self.coor_dat['slice'] = res_info.processes['crop']['Data_slice']
        self.slice = self.coor_dat['slice']

    def create_geographic_coordinates(self, ellipse_type='WGS84', shape=[0, 0], lat0=0, lon0=0, dlat=0, dlon=0):

        self.ellipse_type = ellipse_type
        self.shape = shape
        self.lat_0 = lat0
        self.lon_0 = lon0
        self.dlat = dlat
        self.dlon = dlon

        self.coor_dat['type'] = 'geographic'
        self.coor_dat['shape'] = shape
        self.coor_dat['lat0'] = lat0
        self.coor_dat['lon0'] = lon0
        self.coor_dat['dlat'] = dlat
        self.coor_dat['dlat'] = dlon
        self.coor_dat['ellipse_type'] = ellipse_type

        self.sample = ellipse_type + '_org_' + str(lat0).zfill(0) + '_' + str(lon0).zfill(0) + \
                      '_stp_' + str(dlat * 3600).zfill(0) + '_' + str(dlon * 3600).zfill(0)
        self.coor_dat['sample'] = self.sample

        self.geo = pyproj.Geod(ellps=ellipse_type)

    def create_projection(self, projection_type='', ellipse_type='WGS84', proj4_str='', shape=[0, 0], x0=0, y0=0, dx=0, dy=0):
        # Define projection. For specific projections visit https://proj4.org

        self.ellipse_type = ellipse_type
        self.projection_type = projection_type
        self.proj4_str = proj4_str  # Any additional information if needed..
        self.shape = shape
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy

        self.coor_dat['type'] = 'projection'
        self.coor_dat['shape'] = shape
        self.coor_dat['x0'] = x0
        self.coor_dat['y0'] = y0
        self.coor_dat['dx'] = dx
        self.coor_dat['dy'] = dy
        self.coor_dat['ellipse_type'] = ellipse_type
        self.coor_dat['projection_type'] = projection_type
        self.coor_dat['proj4_str'] = proj4_str

        self.sample = projection_type + '_org_' + str(x0).zfill(0) + '_' + str(y0).zfill(0) + \
                      '_stp_' + str(dx).zfill(0) + '_' + str(dy).zfill(0)
        self.coor_dat['sample'] = self.sample

        self.geo = pyproj.Geod(ellps=ellipse_type)
        if proj4_str:
            self.proj = pyproj.Proj(proj4_str)
        else:
            self.proj = pyproj.Proj(proj=projection_type, ellps=ellipse_type)

    def create_meta_data(self, data_names, data_types, meta_info=''):
        # Create meta data information to save to .res files.

        if not isinstance(meta_info, OrderedDict):
            meta_info = OrderedDict()

        for data_name, data_type in zip(data_names, data_types):

            meta_info[data_name + '_output_file'] = data_name + self.sample + '.raw'
            meta_info[data_name + '_output_format'] = data_type
            meta_info[data_name + '_lines'] = str(self.shape[0])
            meta_info[data_name + '_pixels'] = str(self.shape[1])

            if self.coor_dat['type'] == 'radar_coordinates':
                meta_info[data_name + '_first_line'] = str(self.first_line)
                meta_info[data_name + '_first_pixel'] = str(self.first_pixel)
                meta_info[data_name + '_multilook_azimuth'] = str(self.multilook[0])
                meta_info[data_name + '_multilook_range'] = str(self.multilook[1])
                meta_info[data_name + '_oversampling_azimuth'] = str(self.oversample[0])
                meta_info[data_name + '_oversampling_range'] = str(self.oversample[1])
                meta_info[data_name + '_offset_azimuth'] = str(self.offset[0])
                meta_info[data_name + '_offset_range'] = str(self.offset[1])

            elif self.coor_dat['type'] == 'geographic':
                meta_info[data_name + '_ellipse_type'] = self.ellipse_type
                meta_info[data_name + '_lat0'] = str(self.lat0)
                meta_info[data_name + '_lon0'] = str(self.lon0)
                meta_info[data_name + '_dlat'] = str(self.dlat)
                meta_info[data_name + '_dlon'] = str(self.dlon)

            elif self.coor_dat['type'] == 'projection':
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
