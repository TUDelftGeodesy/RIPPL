'''
Before converting to a new coordinate system we will have to define the coordinates of the new coordinate system using
a different coordinate definition. To find the appropriate size of such a new coordinate system it is good to know
expected coverage of the old product in the new. Therefore, we estimate:
- The minimum size to include all the information from the old coordinate system
- The maximum size to fill all values in the new grid.

Because the conversion between a radar grid is dependent on height while we do not always know the height, we use a
minimum and maximum height of the region. This can be adjusted by users based on knowledge about the region they are
working on.
'''
import copy

import numpy as np

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class CoorNewExtend(object):

    def __init__(self, in_coor, out_coor, min_height=0, max_height=5000, buffer=0, rounding=0, out_coor_limits=True,
                 dx=100, dy=100, dlat=0.1, dlon=0.1, corners_only=False, corners_midpoints=False, spacing=1):
        """
        Here we calculate the extent of a new grid based on the extent of an old grid. This is used for:
        1. Defining the size of an output grid in a new coordinate system
        2. Defining the needed input of block of a grid in parallel processing to define the input grid mininmal needed
            to run the parallel block

        The input variables are different for different coordinate system types. We assume that all the needed information
        about the extent of the grid in the in_coor grid are already defined. The other information depends on the
        out_coor grid
        min_height,max_height > if either the input or output grid is in radar coordinates. The heights do have a strong
            influence on the lat/lon coordinates of the converted grid.

        For the out_coor grid there are 3 short codes for often used grids.
        'UTM' > Creates a UTM grid
        'oblique_mercator' > Also creates a UTM grid, but now using the heading of the satellite on the main axis. This
            creates a grid which is projected but is still more or less aligned to the range/azimuth directions
        'geographic' > Creates a geographic grid based on the WGS84 ellipsoid
        If these short codes are used the following extra information is needed.
        dx,dy > for a projected grid only
        dlat,dlon > for a geographic grid only
        buffer,rounding > for all grids, allthough translated to meters (projected), degrees (geographic), radar grid
            cells (radar_coordinates)

        :param CoordinateSystem in_coor: Input coordinate system (origin, spacing should be set)
        :param CoordinateSystem out_coor: Output coordinate system (spacing should be set or given with short code)
        :param int min_height: To convert radar coordinates to geographic or projected coordinates the height of the
                    pixels is relevant. This gives the expected minimum height over the region.
        :param int max_height: To convert radar coordinates to geographic or projected coordinates the height of the
                    pixels is relevant. This gives the expected maximum height over the region.
        :param float buffer: Buffer around the input grid that should also be covered by the output grid in radar pixels,
                    degrees or meters
        :param float rounding: Value that the output grid should be rounded, for example full degrees or 1000 meters.
        :param float dx: Grid size in x direction for projection (meters)
        :param float dy: Grid size in y direction for projectiom (meters)
        :param float dlat: Grid size in lat direction (degrees)
        :param float dlon: Grid size in lon direction (degrees)
                :param bool corners_only: If we only want to check the corners of the input image in the output image
        :param bool corners_midpoints: If we only want to check the corners of the input image in the output image and
                        the midpoints between those corners
        :param spacing: If we do not focus on corners or midpoints only, the sub-sampling all the convex hull pixels
                        can be given here to save processing time
        """

        self.in_coor = copy.deepcopy(in_coor)
        self.out_coor = copy.deepcopy(out_coor)
        self.min_height = min_height
        self.max_height = max_height

        if in_coor.grid_type == 'radar_coordinates':
            if out_coor == 'UTM':
                out_coor = self.get_mercator_projection(in_coor, True, dx, dy)
            elif out_coor == 'oblique_mercator':
                out_coor = self.get_mercator_projection(in_coor, False, dx, dy)
            elif out_coor == 'geographic':
                out_coor = CoordinateSystem()
                out_coor.create_geographic(dlat=dlat, dlon=dlon)

            self.out_coor = CoorNewExtend.radar2new(self.in_coor, self.out_coor, min_height, max_height, buffer,
                                                    rounding, out_coor_limits=out_coor_limits)
        else:
            self.out_coor = CoorNewExtend.geographic_projection2new(self.in_coor, self.out_coor, min_height, max_height,
                                                                    buffer, rounding, out_coor_limits=out_coor_limits)

    @staticmethod
    def get_mercator_projection(in_coor, UTM, dx, dy):
        """
        Create the default mercator projections based on information from the input coordinate system

        :param CoordinateSystem in_coor: Input coordinate system
        :param bool UTM: Should the coordinate system be a normal UTM? Otherwise, oblique mercator is used.
        :param float dx: stepsize in x direction
        :param float dy: stepsize in y direction
        """

        orbit_in = OrbitCoordinates(in_coor)
        proj_string = orbit_in.create_mercator_projection(UTM)

        out_coor = CoordinateSystem()
        if UTM:
            out_coor.create_projection(dx, dy, proj4_str=proj_string, projection_type='UTM')
        else:
            out_coor.create_projection(dx, dy, proj4_str=proj_string, projection_type='oblique_mercator')

        return out_coor

    @staticmethod
    def radar_convex_hull(in_coor, corners_only=False, corners_midpoints=True, spacing=1):
        """
        Method to determine the line/pixel coordinates of the border of the grid for a radar grid (convex hull)

        :param CoordinateSystem in_coor: Input coordinate system
        :param bool corners_only: If we only want to check the corners of the input image in the output image
        :param bool corners_midpoints: If we only want to check the corners of the input image in the output image and
                        the midpoints between those corners
        :param spacing: If we do not focus on corners or midpoints only, the sub-sampling all the convex hull pixels
                        can be given here to save processing time (default is 0)
        """

        in_coor.create_radar_lines()
        lins = np.array(in_coor.interval_lines).astype(np.int32)
        pixs = np.array(in_coor.interval_pixels).astype(np.int32)
        lines, pixels = CoorNewExtend.concat_coors(lins, pixs, corners_only, corners_midpoints, spacing)

        return lines, pixels

    @staticmethod
    def geographic_convex_hull(in_coor, corners_only=False, corners_midpoints=True, spacing=1):
        """
        Method to determine the lats/lons coordinates of the border of the grid for a geographic grid (convex hull)

        :param CoordinateSystem in_coor: Input coordinate system
        :param bool corners_only: If we only want to check the corners of the input image in the output image
        :param bool corners_midpoints: If we only want to check the corners of the input image in the output image and
                        the midpoints between those corners
        :param spacing: If we do not focus on corners or midpoints only, the sub-sampling all the convex hull pixels
                        can be given here to save processing time (default is 0)
        """

        lat = (in_coor.lat0 + in_coor.first_line * in_coor.dlat) + np.arange(in_coor.shape[0]) * in_coor.dlat
        lon = (in_coor.lon0 + in_coor.first_pixel * in_coor.dlon) + np.arange(in_coor.shape[1]) * in_coor.dlon
        lats, lons = CoorNewExtend.concat_coors(lat, lon, corners_only, corners_midpoints, spacing)

        return lats, lons

    @staticmethod
    def projection_convex_hull(in_coor, corners_only=False, corners_midpoints=True, spacing=1):
        """
        Method to determine the x/y coordinates of the border of the grid for a projected grid (convex hull)

        :param CoordinateSystem in_coor: Input coordinate system
        :param bool corners_only: If we only want to check the corners of the input image in the output image
        :param bool corners_midpoints: If we only want to check the corners of the input image in the output image and
                        the midpoints between those corners
        :param spacing: If we do not focus on corners or midpoints only, the sub-sampling all the convex hull pixels
                        can be given here to save processing time (default is 0)
        """

        y = (in_coor.y0 + in_coor.first_line * in_coor.dy) + np.arange(in_coor.shape[0]) * in_coor.dy
        x = (in_coor.x0 + in_coor.first_pixel * in_coor.dx) + np.arange(in_coor.shape[1]) * in_coor.dx
        y_coors, x_coors = CoorNewExtend.concat_coors(y, x, corners_only, corners_midpoints, spacing)

        return y_coors, x_coors

    @staticmethod
    def concat_coors(y, x, corners_only=False, corners_midpoints=False, spacing=1):
        # type: (list, list) -> (np.ndarray, np.ndarray)
        """
        Concatenate the convex hull of the image based on:
        1. The corners of the grid only
        2. The corners and the midpoints
        3. The full convex hull but sub-sampled by the spacing parameter.

        :param list(float) y: Y-coordinates in ascending order
        :param list(float) x: X-coordinates in ascending order
        :param bool corners_only: If we only want to check the corners of the input image in the output image
        :param bool corners_midpoints: If we only want to check the corners of the input image in the output image and
                        the midpoints between those corners
        :param spacing: If we do not focus on corners or midpoints only, the sub-sampling all the convex hull pixels
                        can be given here to save processing time (default is 0)
        """

        if corners_only:
            y_coors = np.array([y[0], y[-1], y[-1], y[0]])
            x_coors = np.array([x[0], x[0], x[-1], x[-1]])
        elif corners_midpoints:
            mid_x = np.int32(np.floor(len(x) / 2))
            mid_y = np.int32(np.floor(len(y) / 2))
            y_coors = np.array([y[0], y[mid_y], y[-1], y[-1], y[-1], y[mid_y], y[0], y[0]])
            x_coors = np.array([x[0], x[0], x[0], x[mid_x], x[-1], x[-1], x[-1], x[mid_x]])
        else:
            y_coors = np.concatenate((np.ones(len(x) - 1) * y[0], y[:-1],
                                   np.ones(len(x) - 1) * y[-1], np.flip(y)[:-1])).astype(np.int32)
            x_coors = np.concatenate((x[:-1], np.ones(len(y) - 1) * x[-1],
                                    np.flip(x)[:-1], np.ones(len(y) - 1) * x[0])).astype(np.int32)

            y_coors = y_coors[:spacing:]
            x_coors = x_coors[:spacing:]

        return y_coors, x_coors

    @staticmethod
    def radar2new(in_coor, out_coor, min_height=0, max_height=500, buffer=0, rounding=0, corners_only=False,
                  corners_midpoints=False, spacing=1, out_coor_limits=False):
        """
        This processing step uses a radar coordinate system as a basis and determines the shape of the output coordinate
        system based on a different radar grid / geographic grid / projection grid.

        :param CoordinateSystem in_coor: Input coordinate system (should be in radar coordinates!)
        :param CoordinateSystem out_coor: Output coordinate system. Resolution of this coordinate system should be set
        :param int min_height: To convert radar coordinates to geographic or projected coordinates the height of the
                    pixels is relevant. This gives the expected minimum height over the region.
        :param int max_height: To convert radar coordinates to geographic or projected coordinates the height of the
                    pixels is relevant. This gives the expected maximum height over the region.
        :param float buffer: Buffer around the input grid that should also be covered by the output grid in radar pixels,
                    degrees or meters
        :param float rounding: Value that the output grid should be rounded, for example full degrees or 1000 meters.
        """

        # We get all the pixels from the convex hull with a varying height.
        lines, pixels = CoorNewExtend.radar_convex_hull(in_coor)
        heights = np.concatenate((np.ones(pixels.shape) * min_height, np.ones(pixels.shape) * max_height))
        lines = np.concatenate((lines, lines))
        pixels = np.concatenate((pixels, pixels))

        orbit_in = OrbitCoordinates(in_coor)
        orbit_in.manual_line_pixel_height(lines, pixels, heights)
        orbit_in.lph2xyz()

        if out_coor.grid_type == 'radar_coordinates':
            orbit_out = OrbitCoordinates(out_coor)
            lines_out, pixels_out = orbit_out.xyz2lp(orbit_in.xyz)
            out_coor = CoorNewExtend.update_coor(out_coor, lines_out, pixels_out, buffer, rounding, out_coor_limits)
        elif out_coor.grid_type == 'geographic':
            orbit_in.xyz2ell()
            out_coor = CoorNewExtend.update_coor(out_coor, orbit_in.lat, orbit_in.lon, buffer, rounding, out_coor_limits)
        elif out_coor.grid_type == 'projection':
            orbit_in.xyz2ell()
            x, y = out_coor.ell2proj(orbit_in.lat, orbit_in.lon)
            out_coor = CoorNewExtend.update_coor(out_coor, y, x, buffer, rounding)

        return out_coor

    @staticmethod
    def geographic_projection2new(in_coor, out_coor, min_height=0, max_height=500, buffer=0, rounding=0,
                                  corners_only=False, corners_midpoints=False, spacing=1, out_coor_limits=False):
        """
        This processing step uses a radar coordinate system as a basis and determines the shape of the output coordinate
        system based on a different radar grid / geographic grid / projection grid.

        :param CoordinateSystem in_coor: Input coordinate system (should be in radar coordinates!)
        :param CoordinateSystem out_coor: Output coordinate system. Resolution of this coordinate system should be set
        :param int min_height: To convert radar coordinates to geographic or projected coordinates the height of the
                    pixels is relevant. This gives the expected minimum height over the region.
        :param int max_height: To convert radar coordinates to geographic or projected coordinates the height of the
                    pixels is relevant. This gives the expected maximum height over the region.
        :param float buffer: Buffer around the input grid that should also be covered by the output grid in radar pixels,
                    degrees or meters
        :param float rounding: Value that the output grid should be rounded, for example full degrees or 1000 meters.
        """

        # We get all the pixels from the convex hull with a varying height.
        if in_coor.grid_type == 'geographic':
            lat, lon = CoorNewExtend.geographic_convex_hull(in_coor)
        elif in_coor.grid_type == 'projection':
            y_coors, x_coors = CoorNewExtend.projection_convex_hull(in_coor)
            lat, lon = in_coor.proj2ell(x_coors, y_coors)
        heights = np.concatenate((np.ones(lat.shape) * min_height, np.ones(lat.shape) * max_height))
        lat_d = np.concatenate((lat, lat))
        lon_d = np.concatenate((lon, lon))

        if out_coor.grid_type == 'radar_coordinates':
            orbit_out = OrbitCoordinates(out_coor)
            xyz = orbit_out.ell2xyz(lat_d, lon_d, heights)
            lines_out, pixels_out = orbit_out.xyz2lp(xyz)
            out_coor = CoorNewExtend.update_coor(out_coor, lines_out, pixels_out, buffer, rounding, out_coor_limits)
        elif out_coor.grid_type == 'geographic':
            out_coor = CoorNewExtend.update_coor(out_coor, lat, lon, buffer, rounding, out_coor_limits)
        elif out_coor.grid_type == 'projection':
            x, y = out_coor.ell2proj(lat, lon)
            out_coor = CoorNewExtend.update_coor(out_coor, y, x, buffer, rounding, out_coor_limits)

        return out_coor

    @staticmethod
    def update_coor(coor, new_y, new_x, buffer=0, rounding=0, out_coor_limits=False):
        """
        Update the new coordinate system based on the found bounding box data from the input coordinates grid.

        :param CoordinateSystem coor: Input coordinate system
        :param float new_x: The x coordinates for the output grid based on the borders of the input grid
        :param float new_y: The y coordinates for the output grid based on the borders of the input grid
        :param float buffer: Buffer around the input grid that should also be covered by the output grid in radar pixels,
                    degrees or meters
        :param float rounding: Value that the output grid should be rounded, for example full degrees or 1000 meters.
        """
        
        if coor.grid_type == 'radar_coordinates':
            az_time, ra_time, coor.first_line, coor.first_pixel, coor.shape = CoorNewExtend.get_bounding_coor(
                0, 0, coor.multilook[0], coor.multilook[1],
                coor.first_line, coor.first_pixel, coor.shape, new_y, new_x, buffer, rounding, out_coor_limits)
            coor.az_time += az_time * coor.az_step
            coor.ra_time += ra_time * coor.ra_step
        elif coor.grid_type == 'geographic':
            coor.lat0, coor.lon0, coor.first_line, coor.first_pixel, coor.shape = CoorNewExtend.get_bounding_coor(
                coor.lat0, coor.lon0, coor.dlat, coor.dlon, coor.first_line, coor.first_pixel, coor.shape,
                new_y, new_x, buffer, rounding, out_coor_limits)
        elif coor.grid_type == 'projection':
            coor.y0, coor.x0, coor.first_line, coor.first_pixel, coor.shape = CoorNewExtend.get_bounding_coor(
                coor.y0, coor.x0, coor.dy, coor.dx, coor.first_line, coor.first_pixel, coor.shape,
                new_y, new_x, buffer, rounding, out_coor_limits)
        
        return coor

    @staticmethod
    def get_bounding_coor(y_orig, x_orig, dy, dx, s_lin_orig, s_pix_orig, shape_orig,
                          new_y, new_x, buffer=0, rounding=0, out_coor_limits=False):
        """
        Method to create the bounding box of a grid based on:
        1 The minimum and maximum values of the new x and y coordinates (new_x, new_y)
        2 The buffer that should be included (radar pixels, degrees, meters)
        3 The value that the coordinate values should be rounded to (radar pixels, degrees, meters)

        :param float x_orig: X-coordinate of the origin of the grid (defaults to 0)
        :param float y_orig: Y-coordinate of the origin of the grid (defaults to 0)
        :param float dx: Grid size in x direction
        :param float dy: Grid size in y direction
        :param float new_x: The x coordinates for the output grid based on the borders of the input grid
        :param float new_y: The y coordinates for the output grid based on the borders of the input grid
        :param float buffer: Buffer around the input grid that should also be covered by the output grid in radar pixels,
                    degrees or meters
        :param float rounding: Value that the output grid should be rounded, for example full degrees or 1000 meters.

        """

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

        if out_coor_limits and (x_orig == '' or y_orig == '' or s_pix_orig == '' or s_lin_orig == '' or not shape_orig):
            out_coor_limits = False

        if x_orig == '':
            x_orig = 0
        if y_orig == '':
            y_orig = 0

        shape = [0, 0]
        # Based on the minimum find the first/line pixel and determine the shape.
        # If the first/pixel line is less than 0 the origin is shifted.
        first_pixel = np.int32(np.floor((min_x - x_orig) / dx))
        if out_coor_limits:
            first_pixel = np.maximum(first_pixel, s_pix_orig)

        shape[1] = int(np.round((max_x - x_orig) / dx) - first_pixel + 1) # Both begin and end point are included.
        if out_coor_limits:
            shape[1] = np.minimum(shape[1], shape_orig[1] - (first_pixel - s_pix_orig))
        # If the first pixel is lower than zero, adjust the origin
        if first_pixel < 0:
            x_orig -= np.abs(first_pixel) * dx
            first_pixel = 0

        first_line = np.int32(np.floor((min_y- y_orig) / dy))
        if out_coor_limits:
            first_line = np.maximum(first_line, s_lin_orig)

        shape[0] = int(np.round((max_y - y_orig) / dy) - first_line + 1) # Both begin and end point are included.
        if out_coor_limits:
            shape[0] = np.minimum(shape[0], shape_orig[0] - (first_line - s_lin_orig))
        # If the first line is lower than zero, adjust the origin
        if first_line < 0:
            y_orig -= np.abs(first_line) * dy
            first_line = 0

        # In case of resulting zore or negative shape values.
        if shape[0] <= 0:
            shape = [0, 0]
        if shape[1] <= 0:
            shape = [0, 0]

        return y_orig, x_orig, first_line, first_pixel, shape
