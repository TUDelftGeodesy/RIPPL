import json
from collections import OrderedDict
import pyproj
import numpy as np
import osr

from rippl.meta_data.readfile import Readfile
from rippl.meta_data.orbit import Orbit


class CoordinateSystem():

    """
    :type readfile = Readfile | None
    :type orbit = Orbit | None
    :type geo = pyproj.Geod | None
    :type proj = pyproj.Proj | None
    """

    def __init__(self, json_data=''):

        self.json_dict = OrderedDict()

        self.grid_type = ''
        self.slice = True
        self.coor_str = ''
        self.short_id_str = ''
        self.id_str = ''

        # Characteristics for all images
        self.shape = [0, 0]
        self.first_line = 0
        self.first_pixel = 0
        self.sample_name = ''
        self.sparse_name = ''
        self.mask_name = ''

        # Characteristics for radar type
        self.multilook = [1, 1]
        self.oversample = [1, 1]

        # Information from readfiles if available.
        # Next values are all in seconds (azimuth time from start of day to max of 25 hours)
        self.date = ''
        self.ra_time = 0
        self.az_time = 0
        self.az_step = 0
        self.ra_step = 0
        self.center_lon = 0
        self.center_lat = 0
        self.swath = 0

        # Characteristics for geographic type
        self.ellipse_type = ''
        self.lat0 = 0
        self.lon0 = 0
        self.dlat = 0
        self.dlon = 0

        # Characteristics for projection type
        self.projection_type = ''
        self.ellipse_type = ''
        self.proj4_str = ''
        self.x0 = 0
        self.y0 = 0
        self.dx = 0
        self.dy = 0
        
        # Information on line and pixel coordinates.
        self.ml_lines = []
        self.ml_pixels = []
        self.interval_lines = []
        self.interval_pixels = []
        
        # Pyproj variables
        self.geo = None
        self.proj = None

        # Information on orbits (loaded as an orbit object)
        self.orbit = None
        self.readfile = None

        if json_data == '':
            self.json_dict = OrderedDict()
        else:
            self.load_json(json_data=json_data)

    def load_orbit(self, orbit):
        # Load orbit object

        if not isinstance(orbit, Orbit):
            print('Input should be Orbit object')
            return

        self.orbit = orbit

    def load_readfile(self, readfile):
        # type: (CoordinateSystem, Readfile) -> None

        self.ra_time = readfile.ra_first_pix_time
        self.az_time = readfile.az_first_pix_time
        self.ra_step = readfile.ra_time_step
        self.az_step = readfile.az_time_step
        self.center_lat = readfile.center_lat
        self.center_lon = readfile.center_lon
        self.date = readfile.date
        self.swath = readfile.swath

        if self.shape == '' and self.grid_type == 'radar_coordinates':
            self.shape = readfile.size
        # To define the origin of the readfile we assume it is always from the same track. So it can be defined using
        # the date only.
        self.radar_grid_date = readfile.date

        self.readfile = readfile

    def manual_radar_timing(self, az_time, ra_time, az_step, ra_step, date='1900-01-01'):
        # type: (CoordinateSystem, float, float, float, float, str) -> None
        # When no readfiles is available from an actual radar dataset. All variables should be in seconds

        if not isinstance(ra_time, float) or not isinstance(ra_time, float) \
            or not isinstance(ra_time, float) or not isinstance(ra_time, float):
            print('The needed range/azimuth start times and steps should be in seconds! (from midnight UTM for azimuth)')
            return

        self.ra_time = ra_time
        self.az_time = az_time
        self.ra_step = ra_step
        self.az_step = az_step

        # To define the origin of the readfile we assume it is always from the same track. So it can be defined using
        # the date only.
        self.radar_grid_date = date

    def add_crop_info(self, shape, first_line, first_pixel):
        # Add information on the crop from the original file. This is done to create the first crop file and is used
        # as a reference for the creation of further multilooked/offsetted/oversampled images.

        self.orig_shape = shape
        self.orig_first_line = first_line
        self.orig_first_pixel = first_pixel

    def get_line_pixels(self, postings=True, meta=''):
        # type: (CoordinateSystem, bool, Readfile) -> (list, list)
    
        if postings:
            return self.interval_lines, self.interval_pixels
        else:
            return self.ml_lines, self.ml_pixels

    def update_json(self, save_orbit=False, save_readfile=False):
        # type: (CoordinateSystem, str) -> None
        # Save to json format.

        # Define wether it is a slice and what the coordinate system name is
        self.json_dict['slice'] = self.slice
        self.json_dict['coor_str'] = self.coor_str

        # First the general information for all grid types.
        self.json_dict['shape'] = [int(s) for s in self.shape]
        self.json_dict['first_line'] = int(self.first_line)
        self.json_dict['first_pixel'] = int(self.first_pixel)
        self.json_dict['oversample'] = [int(s) for s in self.oversample]
        self.json_dict['sparse_name'] = self.sparse_name
        self.json_dict['mask_name'] = self.mask_name

        # Grid type dependent information
        self.json_dict['grid_type'] = self.grid_type

        # Now add information based on the grid type
        if self.grid_type == 'radar_coordinates':
            self.json_dict['date'] = self.date
            self.json_dict['az_time'] = float(self.az_time)
            self.json_dict['ra_time'] = float(self.ra_time)
            self.json_dict['az_step'] = float(self.az_step)
            self.json_dict['ra_step'] = float(self.ra_step)
            self.json_dict['center_lat'] = float(self.center_lat)
            self.json_dict['center_lon'] = float(self.center_lon)
            self.json_dict['multilook'] = [int(s) for s in self.multilook]
            self.json_dict['date'] = self.date
            self.json_dict['swath'] = int(self.swath)
        elif self.grid_type == 'geographic':
            self.json_dict['ellipse_type'] = self.ellipse_type
            self.json_dict['lat0'] = float(self.lat0)
            self.json_dict['lon0'] = float(self.lon0)
            self.json_dict['dlat'] = float(self.dlat)
            self.json_dict['dlon'] = float(self.dlon)
        elif self.grid_type == 'geographic':
            self.json_dict['ellipse_type'] = self.ellipse_type
            self.json_dict['projection_type'] = self.projection_type
            self.json_dict['proj4_str'] = self.proj4_str
            self.json_dict['x0'] = float(self.x0)
            self.json_dict['y0'] = float(self.y0)
            self.json_dict['dx'] = float(self.dx)
            self.json_dict['dy'] = float(self.dy)

        if not self.readfile or not save_readfile:
            self.json_dict['readfile'] = OrderedDict()
        else:
            self.json_dict['readfile'] = self.readfile.json_dict
        if not self.orbit or not save_orbit:
            self.json_dict['orbit'] = OrderedDict()
        else:
            self.json_dict['orbit'] = self.orbit.json_dict

    def save_json(self, json_path, save_orbit=False, save_readfile=False):
        # Save .json file
        self.update_json(save_orbit, save_readfile)

        file = open(json_path, 'w+')
        json.dump(self.json_dict, file, indent=3)
        file.close()

    def load_json(self, json_data, json_path=''):
        # type: (CoordinateSystem, OrderedDict, str) -> None
        # Load from json data source
        if json_path:
            self.json_dict = json.load(json_path, object_pairs_hook=OrderedDict)
        else:
            self.json_dict = json_data

        # Define wether it is a slice and what the coordinate system name is
        self.slice = self.json_dict['slice']
        self.coor_str = self.json_dict['coor_str']

        # First the general information for all grid types.
        self.shape = self.json_dict['shape']
        self.first_line = self.json_dict['first_line']
        self.first_pixel = self.json_dict['first_pixel']
        self.oversample = self.json_dict['oversample']
        self.sparse_name = self.json_dict['sparse_name']
        self.mask_name = self.json_dict['mask_name']

        # Grid type dependent information
        self.grid_type = self.json_dict['grid_type']

        # Now add information based on the grid type
        if self.grid_type == 'radar_coordinates':
            self.date = self.json_dict['date']
            self.az_time = self.json_dict['az_time']
            self.ra_time = self.json_dict['ra_time']
            self.az_step = self.json_dict['az_step']
            self.ra_step = self.json_dict['ra_step']
            self.center_lat = self.json_dict['center_lat']
            self.center_lon = self.json_dict['center_lon']
            self.multilook = self.json_dict['multilook']
            self.date = self.json_dict['date']
            self.swath = self.json_dict['swath']
        elif self.grid_type == 'geographic':
            self.ellipse_type = self.json_dict['ellipse_type']
            self.lat0 = self.json_dict['lat0']
            self.lon0 = self.json_dict['lon0']
            self.dlat = self.json_dict['dlat']
            self.dlon = self.json_dict['dlon']
        elif self.grid_type == 'geographic':
            self.ellipse_type = self.json_dict['ellipse_type']
            self.projection_type = self.json_dict['projection_type']
            self.proj4_str = self.json_dict['proj4_str']
            self.x0 = self.json_dict['x0']
            self.y0 = self.json_dict['y0']
            self.dx = self.json_dict['dx']
            self.dy = self.json_dict['dy']

        # Add the readfile and orbit
        if len(self.json_dict['readfile']) > 0:
            self.readfile = Readfile(self.json_dict['readfile'])
        if len(self.json_dict['orbit']) > 0:
            self.orbit = Orbit(self.json_dict['orbit'])

    def add_sparse_name(self, sparse_name):

        if sparse_name:
            self.sparse_name = sparse_name
            self.sample = self.sample + '_' + self.sparse_name

    def add_mask_name(self, mask_name):

        if mask_name:
            self.mask_name = mask_name
            self.sample = self.sample + '_' + self.mask_name

    def create_radar_coordinates(self, multilook='', oversample='', shape='', first_line=0, first_pixel=0,
                                 sparse_name='', mask_name=''):

        self.grid_type = 'radar_coordinates'
        self.shape = shape
        self.first_line = first_line
        self.first_pixel = first_pixel
        self.multilook = multilook
        if multilook == '':
            self.multilook = [1, 1]
        self.oversample = oversample
        if oversample == '':
            self.oversample = [1, 1]

        if shape:
            self.create_radar_lines()

        self.add_mask_name(mask_name)
        self.add_sparse_name(sparse_name)

    def create_radar_lines(self):
        # Create a list of radar lines/pixels bases on multilook/oversample

        steps = np.array(self.multilook) / np.array(self.oversample)
        self.ml_lines = self.first_line + np.arange(self.shape[0]) * steps[0]
        self.ml_pixels = self.first_pixel + np.arange(self.shape[1]) * steps[1]

        self.interval_lines = self.ml_lines + (steps[0] - 1) / 2
        self.interval_pixels = self.ml_pixels + (steps[1] - 1) / 2

    def create_geographic(self, dlat='', dlon='', ellipse_type='WGS84', shape='', lat0=-90, lon0=-180,
                          sparse_name='', mask_name='', geo_transform=[]):

        self.ellipse_type = ellipse_type
        self.shape = shape
        if not geo_transform:
            self.lat0 = float(lat0)
            self.lon0 = float(lon0)
            if dlat == '' or dlon == '':
                raise TypeError('dlat and dlon should be given as an input to create a geographic grid.')
            self.dlat = float(dlat)
            self.dlon = float(dlon)
        else:
            self.dlat = float(geo_transform[5])
            self.dlon = float(geo_transform[1])
            self.lat0 = float(geo_transform[3]) + 0.5 * self.dlat
            self.lon0 = float(geo_transform[0]) + 0.5 * self.dlon

        self.grid_type = 'geographic'
        self.sample = '_' + ellipse_type + '_stp_' + str(int(np.round(self.dlat * 3600))) + '_' + str(int(np.round(self.dlon * 3600)))

        self.add_mask_name(mask_name)
        self.add_sparse_name(sparse_name)

        self.geo = pyproj.Geod(ellps=ellipse_type)

    def create_geographic_extend(self, buf_pix=10, lat0='', lon0=''):
        # type: (CoordinateSystem, int, float, float) -> None
        # If both lat0 and lon0 are set and the ellipse type is chosen, we can select an appropriate bounding box
        # based on the information for the original image. Because the coordinates between the corners can be irregular
        # we add an extra buffer of 10 pixels.
        # If lat0 and lon0 are defined, the first line and first pixel are shifted to include only the needed area.
        # This is especially convenient with concatenating images.

        lon_lim = [self.readfile.polygon.bounds[0], self.readfile.polygon.bounds[2]]
        lat_lim = [self.readfile.polygon.bounds[1], self.readfile.polygon.bounds[3]]

        self.lon0, self.lat0, self.first_line, self.first_pixel, self.shape = self.get_coordinate_extend(
            lon_lim, lat_lim, self.dlon, self.dlat, lon0, lat0, buf_pix)

    def create_projection(self, dx, dy, projection_type='', ellipse_type='WGS84', proj4_str='',
                          shape='', x0=0, y0=0, sparse_name='', mask_name=''):
        # Define projection. For specific projections visit https://proj4.org

        self.ellipse_type = ellipse_type
        self.projection_type = projection_type
        self.proj4_str = proj4_str  # Any additional information if needed..
        self.shape = shape
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.dx = float(dx)
        self.dy = float(dy)
        self.grid_type = 'projection'

        self.sample = '_' + projection_type + '_stp_' + str(dx).zfill(0) + '_' + str(dy).zfill(0)

        self.add_mask_name(mask_name)
        self.add_sparse_name(sparse_name)
        self.geo = pyproj.Geod(ellps=ellipse_type)

        if proj4_str:
            self.proj = pyproj.Proj(proj4_str)
        else:
            self.proj = pyproj.Proj(proj=projection_type, ellps=ellipse_type)

    def create_projection_extend(self, buf_pix=10, x0='', y0=''):
        # type: (CoordinateSystem, int, float, float) -> None
        # If both x0 and y0 are set and the projection type is chosen, we can select an appropriate bounding box
        # based on the information for the original image. Because the coordinates between the corners can be irregular
        # we add an extra buffer of 10 pixels.

        lat = [l[0] for l in self.readfile.polygon.exterior.coords]
        lon = [l[1] for l in self.readfile.polygon.exterior.coords]
        x, y = self.ell2proj(lat, lon)
        x_lim = [np.min(x), np.max(x)]
        y_lim = [np.min(y), np.max(y)]

        self.x0, self.y0, self.first_line, self.first_pixel, self.shape = self.get_coordinate_extend(
            x_lim, y_lim, self.dx, self.dy, x0, y0, buf_pix)

    @staticmethod
    def get_coordinate_extend(x_lim, y_lim, dx, dy, x0='', y0='', buf_pix=10):

        if dy > 0:
            y_ids = [1, 3]
        else:
            y_ids = [3, 1]

        if dx > 0:
            x_ids = [0, 2]
        else:
            x_ids = [2, 0]

        # Update y0 and x0 if needed.
        if len(y0) == 0 or len(x0) == 0:
            y0 = np.floor(y_lim[y_ids[0]] / dy) * dy - buf_pix * dy
            x0 = np.floor(x_lim[x_ids[0]] / dx) * dx - buf_pix * dx

        # Update first line and last line.
        first_line = np.floor((y_lim[y_ids[0]] - y0) / dy) - buf_pix
        first_pixel = np.floor((x_lim[x_ids[0]] - x0) / dx) - buf_pix
        if first_pixel < 0 or first_line < 0:
            print('Use of original coordinates is not possible! Returning negative first pixel/line')

        # Update shape.
        y_max = np.ceil(y_lim[y_ids[1]] / dy) * dy + buf_pix * dy
        x_max = np.ceil(x_lim[x_ids[1]] / dx) * dx + buf_pix * dx

        shape = np.array([np.round((y_max - y0) / dy).astype(int),
                               np.round((x_max - x0) / dx).astype(int)])

        return x0, y0, first_line, first_pixel, shape

    def create_gdal_projection(self):
        """
        Create the projection characteristics to save data as .tiff file using gdal.
        Note that the values for radar coordinates are an approximation only!

        :return: osr.SpatialReference, list
        """

        geo_transform = np.zeros(6)
        projection = osr.SpatialReference()
        flipped = False

        if self.grid_type == 'radar_coordinates':
            # Get the coordinates from the .res file
            total_n_s = ((self.readfile.poly_coor[0][0] - self.readfile.poly_coor[3][0]) * 0.5 +
                        (self.readfile.poly_coor[1][0] - self.readfile.poly_coor[2][0]) * 0.5)
            total_w_e = ((self.readfile.poly_coor[0][1] - self.readfile.poly_coor[1][1]) * 0.5 +
                        (self.readfile.poly_coor[3][1] - self.readfile.poly_coor[2][1]) * 0.5)
            skew_n_s = ((self.readfile.poly_coor[0][1] - self.readfile.poly_coor[3][1]) * 0.5 +
                        (self.readfile.poly_coor[1][1] - self.readfile.poly_coor[2][1]) * 0.5)
            skew_w_e = ((self.readfile.poly_coor[0][0] - self.readfile.poly_coor[1][0]) * 0.5 +
                        (self.readfile.poly_coor[3][0] - self.readfile.poly_coor[2][0]) * 0.5)

            geo_transform[0] = self.readfile.poly_coor[0][1]
            geo_transform[1] = - total_w_e / self.shape[1]
            geo_transform[2] = - skew_n_s / self.shape[0]
            geo_transform[3] = self.readfile.poly_coor[0][0]
            geo_transform[4] = - skew_w_e / self.shape[1]
            geo_transform[5] = - total_n_s / self.shape[0]

            # Assume that the projection is WGS84
            projection = osr.SpatialReference()
            projection.ImportFromEPSG(4326)

        elif self.grid_type == 'geographic':
            # Create geo transform based on lat/lon
            geo_transform[0] = self.lon0 + self.dlon * self.first_pixel
            geo_transform[1] = self.dlon
            geo_transform[2] = 0
            if self.dlat < 0:
                geo_transform[3] = self.lat0 + self.dlat * self.first_line
                geo_transform[4] = 0
                geo_transform[5] = self.dlat
            else:
                geo_transform[3] = self.lat0 + self.dlat * (self.first_line + self.shape[0])
                geo_transform[4] = 0
                geo_transform[5] = -self.dlat
                flipped = True

            # Import projection from pyproj
            projection = osr.SpatialReference()
            projection.ImportFromEPSG(4326)

        elif self.grid_type == 'projection':
            # Create geo transform based on projection steps
            geo_transform[0] = self.x0 + self.dx * self.first_pixel
            geo_transform[1] = self.dx
            geo_transform[2] = 0
            geo_transform[3] = self.y0 + self.dy * self.first_line
            geo_transform[4] = 0
            geo_transform[5] = self.dy

            # Import projection from pyproj
            projection = osr.SpatialReference()
            projection.ImportFromProj4(self.proj.srs)

        else:
            print('Grid type ' + self.grid_type + ' is an unknown type.')

        return projection, geo_transform, flipped

    def get_offset(self, offset_coor):
        # type: (CoordinateSystem, CoordinateSystem) -> (int, int)
        # This function gets the offset in lines and pixels with another coordinate system.

        if not self.grid_type == offset_coor.grid_type:
            print('The offset grid should be the same grid type.')
            return

        first_line_off = offset_coor.first_line - self.first_line
        first_pixel_off = offset_coor.first_pixel - self.first_pixel

        if self.grid_type == 'radar_coordinates':
            ra_time_offset = np.int(np.round((offset_coor.ra_time - self.ra_time) / self.ra_step))
            az_time_offset = np.int(np.round((offset_coor.az_time - self.az_time) / self.az_step))

            orig_line_offset = az_time_offset + first_line_off
            orig_pixel_offset = ra_time_offset + first_pixel_off

            line_offset = orig_line_offset // self.multilook[0]
            pixel_offset = orig_pixel_offset // self.multilook[1]

            if orig_line_offset % self.multilook[0] != 0 or orig_pixel_offset % self.multilook[1]:
                print('Pixels do not align due to multilooking!')

        elif self.grid_type == 'geographic':
            lat_offset = np.int(np.round((offset_coor.lat0 - self.lat0) / self.dlat))
            lon_offset = np.int(np.round((offset_coor.lon0 - self.lon0) / self.dlon))

            if ((offset_coor.lon0 - self.lon0) / self.dlon) % 1 > 0.001 or \
                    ((offset_coor.lat0 - self.lat0) / self.dlat) % 1 > 0.001:
                print('Geographic grids do not align!')

            line_offset = first_line_off + lat_offset
            pixel_offset = first_pixel_off + lon_offset

        elif self.grid_type == 'projection':
            y_offset = np.int(np.round((offset_coor.y0 - self.y0) / self.dy))
            x_offset = np.int(np.round((offset_coor.x0 - self.x0) / self.dx))

            if ((offset_coor.x0 - self.x0) / self.dx) % 1 > 0.001 or \
                    ((offset_coor.y0 - self.y0) / self.dy) % 1 > 0.001:
                print('Projected grids do not align!')

            line_offset = first_line_off + y_offset
            pixel_offset = first_pixel_off + x_offset

        return [line_offset, pixel_offset]

    # Create a single identifier for every coordinate system
    def create_coor_id(self):

        if self.grid_type == 'radar_coordinates':
            self.id_str = ('radar_' + self.date + '_' + str(self.az_time) + '_' + str(self.ra_time) + '_' +
                           str(self.shape[0]) + '_' + str(self.shape[1]) + '_' +
                           str(self.first_line) + '_' + str(self.first_pixel) + '_' +
                           str(self.az_step) + '_' + str(self.ra_step) + '_' +
                           str(self.multilook[0]) + '_' + str(self.multilook[1]) + '_' +
                           str(self.oversample[0]) + '_' + str(self.oversample[1]) + '_' +
                           self.sparse_name + self.mask_name)
        elif self.grid_type == 'geographic':
            self.id_str = ('geo_' + self.ellipse_type + '_' + str(self.shape[0]) + '_' + str(self.shape[1]) + '_' +
                           str(self.first_line) + '_' + str(self.first_pixel) + '_' +
                           str(self.lon0) + '_' + str(self.lat0) + '_' +
                           str(self.dlon) + '_' + str(self.dlat) + '_' +
                           self.sparse_name + self.mask_name)
        elif self.grid_type == 'projection':
            self.id_str = ('proj_' + self.proj4_str + '_' + self.ellipse_type + '_' +
                           str(self.shape[0]) + '_' + str(self.shape[1]) + '_' +
                           str(self.first_line) + '_' + str(self.first_pixel) + '_' +
                           str(self.y0) + '_' + str(self.x0) + '_' +
                           str(self.dy) + '_' + str(self.dx) + '_' +
                           self.sparse_name + self.mask_name)

        if self.id_str.endswith('_'):
            self.id_str = self.id_str[:-1]

    # Create a basic coordinate identifier. (Only include the basic coordinate settings)
    # We assume that during processing most other parameters will stay constant, so it should not matter to give a
    # short id in most cases. The long ID will only be used when files are exactly the same.
    def create_short_coor_id(self):

        if self.grid_type == 'radar_coordinates':
            if self.multilook == [1, 1]:
                ml_str = ''
            else:
                ml_str = 'ml_' + str(self.multilook[0]) + '_' + str(self.multilook[1]) + '_'
            if self.oversample == [1, 1]:
                ovr_str = ''
            else:
                ovr_str = 'ovr_' + str(self.oversample[0]) + '_' + str(self.oversample[1]) + '_'
            if self.date:
                date_str = self.date[:4] + self.date[5:7] + self.date[8:10] + '_'

            self.short_id_str = 'radar_' + (ml_str + ovr_str + self.sparse_name + self.mask_name)
        elif self.grid_type == 'geographic':
            self.short_id_str = 'geo_' + (self.ellipse_type + '_' + str(int(self.dlon * 3600)) + '_' + str(int(self.dlat * 3600))) + '_' + self.sparse_name + self.mask_name
        elif self.grid_type == 'projection':
            self.short_id_str = 'proj_' + (self.projection_type + '_' + str(self.dy) + '_' + str(self.dx)) + '_' + self.sparse_name + self.mask_name

        if self.short_id_str.endswith('_'):
            self.short_id_str = self.short_id_str[:-1]

    def same_coordinates(self, coor, strict=False):
        # type: (CoordinateSystem, CoordinateSystem) -> bool

        if strict:
            self.create_coor_id()
            coor.create_coor_id()
        else:
            self.create_short_coor_id()
            coor.create_short_coor_id()

        if strict:
            if self.id_str == coor.id_str:
                return True
        elif not strict:
            if self.short_id_str == coor.short_id_str:
                return True

        return False

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

    def create_xy_grid(self, x_interval=1, y_interval=1):
        # Creates the xy grid for a projection.
        if self.grid_type != 'projection':
            print('xy grid can only be created for a projection')
            return

        y_vals = self.y0 + np.arange(self.shape[0]) * self.dy + self.first_line * self.dy
        x_vals = self.x0 + np.arange(self.shape[1]) * self.dx + self.first_pixel * self.dx

        x, y = np.meshgrid(x_vals, y_vals)

        return x, y

    def create_latlon_grid(self, lat_interval=1, lon_interval=1):
        # Creates the lat/lon grid for a projection
        if self.grid_type != 'geographic':
            print('lat/lon grid can only be created for a geographic coordinate system')
            return

        lat_vals = self.lat0 + np.arange(self.shape[0]) * self.dlat + self.first_line * self.dlat
        lon_vals = self.lon0 + np.arange(self.shape[1]) * self.dlon + self.first_pixel * self.dlon

        lon, lat = np.meshgrid(lon_vals, lat_vals)

        return lat, lon
