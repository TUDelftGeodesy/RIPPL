"""
This class consists of three functions that help to load height, lat, lon, radar pixels, radar lines and Cartesian
coordinates in X,Y,Z for different types of coordinate systems.

For the different datasets they can be generated easily as they are constant. These cases are:
- radar pixels/lines for a radar coordinate system
- lat/lon values for a geographic grid

Other information has to be loaded from former processing steps. The respective functions will throw an error if this
is not possible.
"""

from rippl.meta_data.image_processing_data import ImageData
import numpy as np
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
import copy

class LoadImageGeometry():

    @staticmethod
    def load_xyz(coordinates, meta, s_lin, s_pix, shape):

        x_key = 'X' + coordinates.sample
        y_key = 'Y' + coordinates.sample
        z_key = 'Z' + coordinates.sample
        if coordinates.grid_type == 'radar_coordinates':
            x = meta.image_load_data_memory('geocode', s_lin, s_pix, shape, x_key)
            y = meta.image_load_data_memory('geocode', s_lin, s_pix, shape, y_key)
            z = meta.image_load_data_memory('geocode', s_lin, s_pix, shape, z_key)
        elif coordinates.grid_type in ['projection', 'geographic']:
            x = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, x_key)
            y = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, y_key)
            z = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, z_key)
        else:
            print('xyz coordinates could not be loaded with this function!')
            return

        return x, y, z

    @staticmethod
    def load_dem(coordinates, meta, s_lin, s_pix, shape):

        dem_key = 'dem' + coordinates.sample
        if coordinates.grid_type == 'radar_coordinates':
            height = meta.image_load_data_memory('radar_dem', s_lin, s_pix, shape, dem_key)
        elif coordinates.grid_type in ['projection', 'geographic']:
            height = meta.image_load_data_memory('coor_dem', s_lin, s_pix, shape, dem_key)
        else:
            print('dem values could not be loaded with this function!')
            return

        return height

    @staticmethod
    def load_lat_lon(coordinates, meta, s_lin, s_pix, shape):

        if coordinates.grid_type == 'radar_coordinates':
            lat = meta.image_load_data_memory('geocode', s_lin, s_pix, shape, 'lat' + coordinates.sample)
            lon = meta.image_load_data_memory('geocode', s_lin, s_pix, shape, 'lon' + coordinates.sample)
        elif coordinates.grid_type == 'projection':
            lat_str = 'lat' + coordinates.sample
            lat = meta.image_load_data_memory('projection_coor', s_lin, s_pix, shape, lat_str)
            lon_str = 'lon' + coordinates.sample
            lon = meta.image_load_data_memory('projection_coor', s_lin, s_pix, shape, lon_str)
        elif coordinates.grid_type == 'geographic':
            lat0 = coordinates.lat0 + (coordinates.first_line + s_lin) * coordinates.dlat
            lon0 = coordinates.lon0 + (coordinates.first_pixel + s_pix) * coordinates.dlon
            lat_max = lat0 + coordinates.dlat * (shape[0] - 1)
            lon_max = lon0 + coordinates.dlon * (shape[1] - 1)
            lat, lon = np.meshgrid(np.linspace(lat0, lat_max, shape[0]),
                                             np.linspace(lon0, lon_max, shape[1]))
        else:
            print('lat/lon coordinates could not be loaded with this function!')
            return

        return lat, lon

    @staticmethod
    def load_lines_pixels(meta, s_lin, s_pix, n_lines, coordinates, ml_coors=False):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')
            return

        if len(
                coordinates.interval_lines) == 0 and coordinates.grid_type == 'radar_coordinates' and not coordinates.sparse_grid:
            coordinates.add_res_info(meta)
        shape = copy.copy(coordinates.shape)
        if n_lines != 0:
            l = np.minimum(n_lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        shape = [l, shape[1] - s_pix]

        if coordinates.grid_type == 'radar_coordinates':
            if coordinates.sparse_grid:
                pixels = meta.image_load_data_memory('point_data', s_lin, s_pix, shape,
                                                     'pixel' + coordinates.sample)
                lines = meta.image_load_data_memory('point_data', s_lin, s_pix, shape, 'line' + coordinates.sample)
            else:
                line = coordinates.interval_lines[s_lin: s_lin + shape[0]] + coordinates.first_line
                pixel = coordinates.interval_pixels[s_pix: s_pix + shape[1]] + coordinates.first_pixel
                pixels, lines = np.meshgrid(pixel, line)

        else:
            lines = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, 'line' + coordinates.sample)
            pixels = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, 'pixel' + coordinates.sample)

        return shape, lines, pixels
