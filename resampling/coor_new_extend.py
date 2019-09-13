'''
Before converting to a new coordinate system we will have to define the coordinates of the new coordinate system using
a different coordinate definition. To find the appropriate size of such a new coordinate system it is good to know
expected coverage of the old product in the new. Therefore we estimate:
- The minimum size to include all the information from the old coordinate system
- The maximum size to fill all values in the new grid.

Because the conversion between a radar grid is dependent on height while we do not always know the height, we use a
minimum and maximum height of the region. This can be adjusted by users based on knowledge about the region they are
working on.
'''

import numpy as np

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class CoorNewExtend(object):

    def __init__(self, coor_in, coor_out, min_height=0, max_height=500, full_coverage='in', buffer=0, rounding=0):
        # type: (CoorNewExtend, CoordinateSystem, CoordinateSystem, int, int, str) -> None
        # We assume that the x0/y0/lat0/lon0/az_time/ra_time are either 0 or already defined.

        self.coor_in = coor_in
        self.min_height = min_height
        self.max_height = max_height
        self.full_coverage = full_coverage

        if coor_in.grid_type == 'radar_coordinates':
            self.coor_out = CoorNewExtend.radar2new(coor_in, coor_out, min_height, max_height, full_coverage, buffer,
                                                    rounding)
        else:
            self.coor_out = CoorNewExtend.geographic_projection2new(coor_in, coor_out, min_height, max_height,
                                                                    full_coverage, buffer, rounding)

    @staticmethod
    def radar_convex_hull(coor_in):
        # type: (CoordinateSystem) -> (list, list)
        # What are the line/pixel coordinates of the outside grid?

        coor_in.create_radar_lines()
        lins = np.array(coor_in.interval_lines).astype(np.int32)
        pixs = np.array(coor_in.interval_pixels).astype(np.int32)
        lines, pixels = CoorNewExtend.concat_coors(lins, pixs)

        return lines, pixels

    @staticmethod
    def geographic_convex_hull(coor_in):
        # type: (CoordinateSystem) -> (list, list)
        # Outside lats/lons of image border

        lat = (coor_in.lat0 + coor_in.first_line * coor_in.dlat) + np.arange(coor_in.shape[0]) * coor_in.dlat
        lon = (coor_in.lon0 + coor_in.first_pixel * coor_in.dlon) + np.arange(coor_in.shape[1]) * coor_in.dlon
        lats, lons = CoorNewExtend.concat_coors(lat, lon)

        return lats, lons

    @staticmethod
    def projection_convex_hull(coor_in):
        # type: (CoordinateSystem) -> (list, list)
        # Outside x/y of image border

        y = (coor_in.y0 + coor_in.first_line * coor_in.dy) + np.arange(coor_in.shape[0]) * coor_in.dy
        x = (coor_in.x0 + coor_in.first_pixel * coor_in.dx) + np.arange(coor_in.shape[1]) * coor_in.dx
        y_coors, x_coors = CoorNewExtend.concat_coors(y, x)

        return y_coors, x_coors

    @staticmethod
    def radar2new(coor_in, coor_out, min_height=0, max_height=500, full_coverage='in', buffer=0, rounding=0):
        # type: (CoordinateSystem, CoordinateSystem, int, int) -> CoordinateSystem
        # If the input coordinate system is a radar coordinate system.

        # We get all the pixels from the convex hull with a varying height.
        lines, pixels = CoorNewExtend.radar_convex_hull(coor_in)
        heights = np.concatenate((np.ones(pixels.shape) * min_height, np.ones(pixels.shape) * max_height))
        lines = np.concatenate((lines, lines))
        pixels = np.concatenate((pixels, pixels))

        orbit_in = OrbitCoordinates(coor_in)
        orbit_in.manual_line_pixel_height(lines, pixels, heights)
        orbit_in.lph2xyz()

        if coor_out.grid_type == 'radar':
            orbit_out = OrbitCoordinates(coor_out)
            lines_out, pixels_out = orbit_out.xyz2lp(orbit_in.xyz)
            coor_out = CoorNewExtend.update_coor(coor_out, lines_out, pixels_out, full_coverage, buffer, rounding)
        elif coor_out.grid_type == 'geographic':
            orbit_in.xyz2ell()
            coor_out = CoorNewExtend.update_coor(coor_out, orbit_in.lat, orbit_in.lon, full_coverage, buffer, rounding)
        elif coor_out.grid_type == 'projection':
            orbit_in.xyz2ell()
            x, y = coor_out.ell2proj(orbit_in.lat, orbit_in.lon)
            coor_out = CoorNewExtend.update_coor(coor_out, y, x, full_coverage, buffer, rounding)

        return coor_out

    @staticmethod
    def geographic_projection2new(coor_in, coor_out, min_height=0, max_height=500, full_coverage='in', buffer=0, rounding=0):
        # type: (CoordinateSystem, CoordinateSystem, int, int) -> CoordinateSystem
        # If the input coordinate system is a geographic coordinate system.

        # We get all the pixels from the convex hull with a varying height.
        if coor_in.grid_type == 'geographic':
            lat, lon = CoorNewExtend.geographic_convex_hull(coor_in)
        elif coor_in.grid_type == 'projection':
            x_coors, y_coors = CoorNewExtend.projection_convex_hull(coor_in)
            lat, lon = coor_in.proj2ell(x_coors, y_coors)
        heights = np.concatenate((np.ones(lat.shape) * min_height, np.ones(lat.shape) * max_height))
        lat_d = np.concatenate((lat, lat))
        lon_d = np.concatenate((lon, lon))

        if coor_out.grid_type == 'radar':
            orbit_out = OrbitCoordinates(coor_out)
            xyz = orbit_out.ell2xyz(lat_d, lon_d, heights)
            lines_out, pixels_out = orbit_out.xyz2lp(xyz)
            coor_out = CoorNewExtend.update_coor(coor_out, lines_out, pixels_out, full_coverage, buffer, rounding)
        elif coor_out.grid_type == 'geographic':
            coor_out = CoorNewExtend.update_coor(coor_out, lat, lon, full_coverage, buffer, rounding)
        elif coor_out.grid_type == 'projection':
            x, y = coor_out.ell2proj(lat, lon)
            coor_out = CoorNewExtend.update_coor(coor_out, y, x, full_coverage, buffer, rounding)

        return coor_out

    @staticmethod
    def update_coor(coor, new_y, new_x, full_coverage='in', buffer=0, rounding=0):
        # type: (CoordinateSystem, list, list, str) -> (CoordinateSystem)
        
        if coor.grid_type == 'radar_coordinates':
            az_time, ra_time, coor.first_line, coor.first_pixel, coor.shape = CoorNewExtend.get_bounding_coor(
                coor.az_time / coor.az_step, coor.ra_time / coor.ra_step, coor.multilook[0], coor.multilook[1],
                new_y, new_x, full_coverage, buffer, rounding)
            coor.az_time = az_time * coor.az_step
            coor.ra_time = ra_time * coor.ra_step
        elif coor.grid_type == 'geographic':
            coor.lat0, coor.lon0, coor.first_line, coor.first_pixel, coor.shape = CoorNewExtend.get_bounding_coor(
                coor.lat0, coor.lon0, coor.dlat, coor.dlon, new_y, new_x, full_coverage, buffer, rounding)
        elif coor.grid_type == 'projection':
            coor.y0, coor.x0, coor.first_line, coor.first_pixel, coor.shape = CoorNewExtend.get_bounding_coor(
                coor.y0, coor.x0, coor.dy, coor.dx, new_y, new_x, full_coverage, buffer, rounding)
        
        return coor

    @staticmethod
    def concat_coors(y, x):
        # type: (list, list) -> (np.ndarray, np.ndarray)
        # Concatenate the convex hull of the the image.
        y_coors = np.concatenate((np.ones(len(x) - 1) * y[0], y[:-1],
                               np.ones(len(x) - 1) * y[-1], np.flip(y)[:-1])).astype(np.int32)
        x_coors = np.concatenate((x[:-1], np.ones(len(y) - 1) * x[-1],
                                np.flip(x)[:-1], np.ones(len(y) - 1) * x[0])).astype(np.int32)

        return y_coors, x_coors

    @staticmethod
    def get_bounding_coor(y_orig, x_orig, dy, dx, new_y, new_x, full_coverage='in', buffer=0, rounding=0):
        # type: (float, float, float, float, np.ndarray, np.ndarray, str) -> (float, float, int, int, list)

        if full_coverage == 'in':
            if rounding != 0:
                min_x = np.floor((np.min(new_x) - buffer) / rounding) * rounding
                max_x = np.ceil((np.max(new_x) + buffer) / rounding) * rounding
                min_y = np.floor((np.min(new_y) - buffer) / rounding) * rounding
                max_y = np.ceil((np.max(new_y) + buffer) / rounding) * rounding
            else:
                min_x = np.min(new_x) - buffer
                max_x = np.max(new_x) + buffer
                min_y = np.min(new_y) - buffer
                max_y = np.max(new_y) + buffer
        else:
            print('For now only the minimum size to get all data is in is used. Alternatives seem to be quite '
                  'difficult to determine.')
            return

        if x_orig == '':
            x_orig = 0
        if y_orig == '':
            y_orig = 0

        shape = [0, 0]
        # Based on the minimum find the first/line pixel and determine the shape.
        # If the first/pixel line is less than 0 the origin is shifted.
        first_pixel = np.int(np.floor((min_x - x_orig) / dx))
        shape[1] = int(np.round((max_x - x_orig) / dx) - first_pixel + 1)        # Both begin and end point are included.
        if first_pixel < 0:
            x_orig -= np.abs(first_pixel) * dx
            first_pixel = 0
        first_line = np.int(np.floor((min_y- y_orig) / dy))
        shape[0] = int(np.round((max_y - y_orig) / dy) - first_line + 1)       # Both begin and end point are included.
        if first_line < 0:
            y_orig -= np.abs(first_line) * dy
            first_line = 0

        return x_orig, y_orig, first_line, first_pixel, shape
