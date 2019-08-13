import json
from collections import OrderedDict
import pyproj
import numpy as np
import osr

from rippl.orbit_geometry.sparse_coordinates import SparseCoordinates
from rippl.meta_data.image_metadata import ImageMetadata
from rippl.meta_data.readfile import Readfiles
from rippl.meta_data.orbit import Orbit


class CoordinateSystem():

    """
    :type readfiles = Readfiles | None
    :type orbit = Orbit | None
    :type geo = pyproj.Geod | None
    :type proj = pyproj.Proj | None
    """

    def __init__(self):


        self.json_dict = OrderedDict()

        self.grid_type = ''
        self.slice = False
        self.coor_str = ''

        # Characteristics for all images
        self.shape = [0, 0]
        self.first_line = 1
        self.first_pixel = 1
        self.sample_name = ''
        self.over_size = False
        self.sparse_grid = ''
        self.mask_grid = ''

        # Characteristics for radar type
        self.multilook = [1, 1]
        self.offset = [0, 0]
        self.oversample = [1, 1]

        # Information on the original crop size
        self.orig_shape = [0, 0]
        self.orig_first_line = 1
        self.orig_first_pixel = 1

        # Information from readfiles if available.
        # Next values are all in seconds (azimuth time from start of day to max of 25 hours)
        self.ra_time = 0
        self.az_time = 0
        self.az_step = 0
        self.ra_step = 0
        self.radar_grid_date = ''   # The origin of the master/slave or coregistration master grid and the orbits that
                                    # come with it.

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
        self.ml_lines
        self.ml_pixels 
        self.interval_lines 
        self.interval_pixels
        
        # Pyproj variables
        self.geo = None
        self.proj = None

        # Information on orbits (loaded as an orbit object)
        self.json_dict = OrderedDict()
        self.orbit = None
        self.readfiles = None

    def load_orbit(self, orbit):
        # Load orbit object

        if not isinstance(orbit, Orbit):
            print('Input should be Orbit object')
            return

        self.orbit = orbit

    def load_readfiles(self, readfiles):
        # type: (CoordinateSystem, Readfiles) -> None

        self.ra_time = readfiles.ra_first_pix_time
        self.az_time = readfiles.az_first_pix_time
        self.ra_step = readfiles.ra_time_step
        self.az_step = readfiles.az_time_step

        # To define the origin of the readfile we assume it is always from the same track. So it can be defined using
        # the date only.
        self.radar_grid_date = readfiles.date

        self.readfiles = readfiles

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
        # type: (CoordinateSystem, bool, ImageMetadata) -> (list, list)
    
        if postings:
            return self.interval_lines, self.interval_pixels
        else:
            return self.ml_lines, self.ml_pixels


    def update_json(self, json_path=''):
        # type: (CoordinateSystem, str) -> None
        # Save to json format.

        # Define wether it is a slice and what the coordinate system name is
        self.json_dict['slice'] = self.slice
        self.json_dict['coor_str'] = self.coor_str

        # First the general information for all grid types.
        self.json_dict['shape'] = self.shape
        self.json_dict['first_line'] = self.first_line
        self.json_dict['first_pixel'] = self.first_pixel
        self.json_dict['oversample'] = self.oversample
        self.json_dict['over_size'] = self.over_size
        self.json_dict['sparse_grid'] = self.sparse_grid
        self.json_dict['mask_grid'] = self.mask_grid

        # Grid type dependent information
        self.json_dict['grid_type'] = self.grid_type

        # Now add information based on the grid type
        if self.grid_type == 'radar_coordinates':
            self.json_dict['az_time'] = self.az_time
            self.json_dict['ra_time'] = self.ra_time
            self.json_dict['az_step'] = self.az_step
            self.json_dict['ra_step'] = self.ra_step
            self.json_dict['multilook'] = self.multilook
            self.json_dict['offset'] = self.offset
            self.json_dict['orig_shape'] = self.orig_shape
            self.json_dict['orig_first_line'] = self.orig_first_line
            self.json_dict['orig_first_pixel'] = self.orig_first_pixel
        elif self.grid_type == 'geographic':
            self.json_dict['ellipse_type'] = self.ellipse_type
            self.json_dict['lat0'] = self.lat0
            self.json_dict['lon0'] = self.lon0
            self.json_dict['dlat'] = self.dlat
            self.json_dict['dlon'] = self.dlon
        elif self.grid_type == 'geographic':
            self.json_dict['ellipse_type'] = self.ellipse_type
            self.json_dict['projection_type'] = self.projection_type
            self.json_dict['proj4_str'] = self.proj4_str
            self.json_dict['x0'] = self.x0
            self.json_dict['y0'] = self.y0
            self.json_dict['dx'] = self.dx
            self.json_dict['dy'] = self.dy

        if json_path:
            json.dump(json_path, self.json_dict)

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
        self.over_size = self.json_dict['over_size']
        self.sparse_grid = self.json_dict['sparse_grid']
        self.mask_grid = self.json_dict['mask_grid']

        # Grid type dependent information
        self.grid_type = self.json_dict['grid_type']

        # Now add information based on the grid type
        if self.grid_type == 'radar_coordinates':
            self.az_time = self.json_dict['az_time']
            self.ra_time = self.json_dict['ra_time']
            self.az_step = self.json_dict['az_step']
            self.ra_step = self.json_dict['ra_step']
            self.multilook = self.json_dict['multilook']
            self.offset = self.json_dict['offset']
            self.orig_shape = self.json_dict['orig_shape']
            self.orig_first_pixel = self.json_dict['orig_first_pixel']
            self.orig_first_line = self.json_dict['orig_first_line']
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

    def add_sparse_name(self, sparse_name):

        if sparse_name:
            self.sparse_name = sparse_name
            self.sample = self.sample + '_' + self.sparse_name

    def add_mask_name(self, mask_name):

        if mask_name:
            self.mask_name = mask_name
            self.sample = self.sample + '_' + self.mask_name

    def create_radar_coordinates(self, multilook='', oversample='', offset='', sparse_name='', mask_name=''):

        self.grid_type = 'radar_coordinates'

        # If we want to extend the boundaries of our image outside of the original radar image we should add an extra
        # offset. If offset is already defined, we leave it as it is.
        if self.over_size and len(offset) == 0:
            offset = [self.offset[0] - multilook[0] * 2, self.offset[1] - multilook[1] * 2]

        # What are are the first lines/pixels of the multilooking blocks.
        self.sample, self.multilook, self.oversample, self.offset, self.ml_lines, self.ml_pixels = \
            SparseCoordinates.multilook_lines(self.orig_shape, oversample=oversample, multilook=multilook,
                                              offset=offset, interval=self.over_size)

        # If we are only interested in the center of every multilooking block of x times y pixels, what are the
        # coordinates then. (Usefull for geometrical calculations of lat/lon, x/y/z, height calculations)
        self.sample, self.multilook, self.oversample, self.offset, self.interval_lines, self.interval_pixels = \
            SparseCoordinates.interval_lines(self.orig_shape, oversample=oversample, multilook=multilook,
                                              offset=offset, interval=self.over_size)

        self.ml_lines += self.first_line
        self.ml_pixels += self.first_pixel

        self.interval_lines += self.first_line
        self.interval_pixels += self.first_pixel

        self.add_mask_name(mask_name)
        self.add_sparse_name(sparse_name)

    def create_geographic(self, dlat, dlon, ellipse_type='WGS84', shape='', lat0='', lon0='',
                          sparse_name='', mask_name=''):

        self.ellipse_type = ellipse_type
        self.shape = shape
        self.lat0 = lat0
        self.lon0 = lon0
        self.dlat = dlat
        self.dlon = dlon
        self.grid_type = 'geographic'
        self.sample = '_' + ellipse_type + '_stp_' + str(int(np.round(dlat * 3600))) + '_' + str(int(np.round(dlon * 3600)))

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

        lon_lim = [self.readfiles.polygon.bounds[0], self.readfiles.polygon.bounds[2]]
        lat_lim = [self.readfiles.polygon.bounds[1], self.readfiles.polygon.bounds[3]]

        self.lon0, self.lat0, self.first_line, self.first_pixel, self.shape = self.get_coordinate_extend(
            lon_lim, lat_lim, self.dlon, self.dlat, lon0, lat0, buf_pix)

    def create_projection(self, dx, dy, projection_type='', ellipse_type='WGS84', proj4_str='',
                          shape='', x0='', y0='', sparse_name='', mask_name=''):
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

        lat = [l[0] for l in self.readfiles.polygon.exterior.coords]
        lon = [l[1] for l in self.readfiles.polygon.exterior.coords]
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

    def create_gdal_projection(self, readfiles):
        # type: (CoordinateSystem, Readfiles) -> (osr.SpatialReference, list)

        geo_transform = np.zeros(6)
        projection = osr.SpatialReference()

        if self.grid_type == 'radar_coordinates':
            # Get the coordinates from the .res file
            total_n_s = ((self.readfiles.poly_coor[0][0] - self.readfiles.poly_coor[3][0]) * 0.5 +
                        (self.readfiles.poly_coor[1][0] - self.readfiles.poly_coor[2][0]) * 0.5)
            total_w_e = ((self.readfiles.poly_coor[0][1] - self.readfiles.poly_coor[1][1]) * 0.5 +
                        (self.readfiles.poly_coor[3][1] - self.readfiles.poly_coor[2][1]) * 0.5)
            skew_n_s = ((self.readfiles.poly_coor[0][1] - self.readfiles.poly_coor[3][1]) * 0.5 +
                        (self.readfiles.poly_coor[1][1] - self.readfiles.poly_coor[2][1]) * 0.5)
            skew_w_e = ((self.readfiles.poly_coor[0][0] - self.readfiles.poly_coor[1][0]) * 0.5 +
                        (self.readfiles.poly_coor[3][0] - self.readfiles.poly_coor[2][0]) * 0.5)

            geo_transform[0] = self.readfiles.poly_coor[0][1]
            geo_transform[1] = - total_w_e / self.shape[1]
            geo_transform[2] = - skew_n_s / self.shape[0]
            geo_transform[3] = self.readfiles.poly_coor[0][0]
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
            print('Grid type ' + self.grid_type + ' is an unknown type.')

        return projection, geo_transform

    def get_offset(self, offset_coor):
        # type: (CoordinateSystem, CoordinateSystem) -> (int, int)
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

            orig_line_offset = az_time_offset + first_line_off + (offset_coor.offset[0] - self.offset[0])
            orig_pixel_offset = ra_time_offset + first_pixel_off + (offset_coor.offset[1] - self.offset[1])

            line_offset = orig_line_offset // self.multilook[0]
            pixel_offset = orig_pixel_offset // self.multilook[1]

            if orig_line_offset % self.multilook[0] != 0 or orig_pixel_offset % self.multilook[1]:
                print('Pixels do not align with multilooking!')

        elif self.grid_type == 'geographic':

            if offset_coor.dlat != self.dlat or offset_coor.dlon != self.dlon:
                print('Pixel spacing should be the same')
                return
            
            lat_offset = np.int(np.round((offset_coor.lat0 - self.lat0) / self.dlat))
            lon_offset = np.int(np.round((offset_coor.lon0 - self.lon0) / self.dlon))

            if ((offset_coor.lon0 - self.lon0) / self.dlon) % 1 > 0.001 or \
                    ((offset_coor.lat0 - self.lat0) / self.dlat) % 1 > 0.001:
                print('Geographic grids do not align!')

            line_offset = first_line_off + lat_offset
            pixel_offset = first_pixel_off + lon_offset

        elif self.grid_type == 'projection':

            if offset_coor.dy != self.dy or offset_coor.dx != self.dx:
                print('Pixel spacing should be the same')
                return
            if offset_coor.proj4_str != self.proj4_str:
                print('Projection should be the same')
                return

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
            self.id_str = (self.readfiles.date + '_' + str(self.az_time) + '_' + str(self.ra_time) + '_' +
                           str(self.shape[0]) + '_' + str(self.shape[1]) + '_' +
                           str(self.first_line) + '_' + str(self.first_pixel) + '_' +
                           str(self.az_step) + '_' + str(self.ra_step) + '_' +
                           str(self.multilook[0]) + '_' + str(self.multilook[1]) + '_' +
                           str(self.offset[0]) + '_' + str(self.offset[1]) + '_' +
                           str(self.oversample[0]) + '_' + str(self.oversample[1]) + '_' +
                           self.sparse_name + self.mask_name)
        elif self.grid_type == 'geographic':
            self.id_str = (self.ellipse_type + '_' + str(self.shape[0]) + '_' + str(self.shape[1]) + '_' +
                           str(self.first_line) + '_' + str(self.first_pixel) + '_' +
                           str(self.lon0) + '_' + str(self.lat0) + '_' +
                           str(self.dlon) + '_' + str(self.dlat))
        elif self.grid_type == 'projection':
            self.id_str = (self.proj4_str + '_' + self.ellipse_type + '_' +
                           str(self.shape[0]) + '_' + str(self.shape[1]) + '_' +
                           str(self.first_line) + '_' + str(self.first_pixel) + '_' +
                           str(self.y0) + '_' + str(self.x0) + '_' +
                           str(self.dy) + '_' + str(self.dx))

    # Create a basic coordinate identifier. (Only include the basic coordinate settings)
    # We assume that during processing most other parameters will stay constant, so it should not matter to give a
    # short id in most cases. The long ID will only be used when files are exactly the same.
    def create_short_coor_id(self):

        if self.grid_type == 'radar_coordinates':
            self.short_id_str = ('ml_' + str(self.multilook[0]) + '_' + str(self.multilook[1]) +
                           '_off_' + str(self.offset[0]) + '_' + str(self.offset[1]) +
                           '_ovr_' + str(self.oversample[0]) + '_' + str(self.oversample[1]) +
                           '_' + self.sparse_name + self.mask_name)
        elif self.grid_type == 'geographic':
            self.short_id_str = (self.ellipse_type + '_' + str(int(self.dlon * 3600)) + '_' + str(int(self.dlat * 3600)))
        elif self.grid_type == 'projection':
            self.short_id_str = (self.projection_type + '_' + str(self.dy) + '_' + str(self.dx))

    def same_coordinates(self, coor):
        # type: (CoordinateSystem, CoordinateSystem) -> bool

        self.create_coor_id()
        coor.create_coor_id()

        if self.id_str == coor.id_str:
            return True
        else:
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
