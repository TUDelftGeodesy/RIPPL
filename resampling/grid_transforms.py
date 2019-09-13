'''
This class holds a number of functions which gives a transform between different coordinate systems.

'''

import numpy as np

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.orbit import Orbit
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates
from rippl.meta_data.image_processing_data import ImageData


class GridTransforms(object):

    def __init__(self, coor_in, coor_out):

        if not isinstance(coor_in, CoordinateSystem) or not isinstance(coor_out, CoordinateSystem):
            print('Input and output coordinatesystem should be an CoordinateSystem object!')
            return False

        # Check whether orbits are loaded.
        if (coor_in.grid_type == 'radar_coordinates' and not isinstance(coor_in.orbit, Orbit)) or \
                (coor_out.grid_type == 'radar_coordinates' and not isinstance(coor_out.orbit, Orbit)):
            print('The orbits of radar coordinate systems should be loaded to convert grids')
            return False
        # Also check the timing
        if (coor_in.grid_type == 'radar_coordinates' and
            (coor_in.ra_step == 0 or coor_in.az_step == 0 or coor_in.ra_time == 0 or coor_in.az_time == 0)) or \
                (coor_out.grid_type == 'radar_coordinates' and
                 (coor_out.ra_step == 0 or coor_out.az_step == 0 or coor_out.ra_time == 0 or coor_out.az_time == 0)):
            print('First define the range and azimuth timing before create a transform.')
            return False

        self.coor_in = coor_in
        self.coor_out = coor_out

        return True

    def __call__(self, meta=[]):

        if self.coor_in.grid_type == 'radar_coordinates' or self.coor_out.grid_type == 'radar_coordinates':
            if not isinstance(meta, ImageData):
                print('meta should be an ImageData object')

        self.meta = meta

        # Load the xyz, height data needed. If it is missing, throw an error.
        if self.coor_in.grid_type == 'radar_coordinates':
            if self.coor_out.grid_type == 'radar_coordinates':
                xyz = self.meta.load_data(self.coor_out, 'geocode', 'xyz')
                height = self.meta.load_data(self.coor_out, 'geocode', 'xyz')
            elif self.coor_out.grid_type == 'geographic':
                xyz = self.meta.load_data(self.coor_out, 'coor_geocode', 'xyz')
                height = self.meta.load_data(self.coor_out, 'coor_dem', 'dem')
            elif self.coor_out.grid_type == 'projection':
                xyz = self.meta.load_data(self.coor_out, 'coor_geocode', 'xyz')
                height = self.meta.load_data(self.coor_out, 'coor_dem', 'dem')
                lat = self.meta.load_data(self.coor_out, 'proj_coor', 'lat')
                lon = self.meta.load_data(self.coor_out, 'proj_coor', 'lon')
        elif self.coor_in.grid_type in ['geographic', 'projection']:
            if self.coor_out.grid_type == 'radar_coordinates':
                xyz = self.meta.load_data(self.coor_out, 'geocode', 'xyz')
                lat = self.meta.load_data(self.coor_out, 'geocode', 'lat')
                lon = self.meta.load_data(self.coor_out, 'geocode', 'lon')
            elif self.coor_out.grid_type == 'projection':
                lat = self.meta.load_data(self.coor_out, 'proj_coor', 'lat')
                lon = self.meta.load_data(self.coor_out, 'proj_coor', 'lon')

        # Call the the right transform function and return it as an output.
        if self.coor_in.grid_type == 'radar_coordinates':
            if self.coor_out.grid_type == 'radar_coordinates':
                lines, pixels = GridTransforms.radar2radar(self.coor_in, self.coor_out, xyz)
            if self.coor_out.grid_type == 'geographic':
                lines, pixels = GridTransforms.radar2geographic(self.coor_in, self.coor_out, xyz, height)
            if self.coor_out.grid_type == 'projection':
                lines, pixels = GridTransforms.radar2projection(self.coor_in, self.coor_out, xyz, height, lat, lon)
        if self.coor_in.grid_type == 'geographic':
            if self.coor_out.grid_type == 'radar_coordinates':
                lines, pixels = GridTransforms.geographic2radar(self.coor_in, self.coor_out, xyz, lat, lon)
            if self.coor_out.grid_type == 'geographic':
                lines, pixels = GridTransforms.geographic2geographic(self.coor_in, self.coor_out)
            if self.coor_out.grid_type == 'projection':
                lines, pixels = GridTransforms.geographic2projection(self.coor_in, self.coor_out, lat, lon)
        if self.coor_in.grid_type == 'projection':
            if self.coor_out.grid_type == 'radar_coordinates':
                lines, pixels = GridTransforms.geographic2radar(self.coor_in, self.coor_out, xyz, lat, lon)
            if self.coor_out.grid_type == 'geographic':
                lines, pixels = GridTransforms.geographic2geographic(self.coor_in, self.coor_out)
            if self.coor_out.grid_type == 'projection':
                lines, pixels = GridTransforms.geographic2projection(self.coor_in, self.coor_out, lat, lon)

        return lines, pixels

    # The next part consists of all the different functions to convert between the different coordinate systems.
    # From radar to radar/geographic/projection systems.
    @staticmethod
    def radar2radar(coor_in, coor_out, out_xyz='', out_height=''):
        # First check if the xyz data is there.
        if len(out_xyz) == 0 and len(out_height) == 0:
            print('Either the xyz cartesian coordinates or the grid heights should be available.')
            return
        elif len(out_xyz) == 0 and len(out_height) != 0:
            orbit_out = OrbitCoordinates(coor_out.orbit, coor_out)
            orbit_out.lp_time()
            orbit_out.height = out_height
            orbit_out.lph2xyz()
            out_xyz = orbit_out.xyz

        # Now estimate the orbit of the input image and calculate the output coordinates.
        orbit_in = OrbitCoordinates(coor_in.orbit, coor_in)
        orbit_in.lp_time()
        lines, pixels = orbit_in.xyz2lp(out_xyz)
        lines, pixels = GridTransforms.correct_radar_pixel_spacing(lines, pixels, orbit_in)

        return lines, pixels

    @staticmethod
    def radar2geographic(coor_in, coor_out, out_xyz='', out_height=''):
        # Get the input orbit
        orbit_in = OrbitCoordinates(coor_in.orbit, coor_in)
        orbit_in.lp_time()

        # First check if the xyz or out data is there.
        if len(out_xyz) == 0 and len(out_height) == 0:
            print('Either the xyz cartesian coordinates or the grid heights should be available.')
            return
        elif len(out_xyz) == 0 and len(out_height) != 0:
            lat, lon = coor_out.create_latlon_grid()
            out_xyz = OrbitCoordinates.ell2xyz(lat, lon, out_height)

        lines, pixels = orbit_in.xyz2lp(out_xyz)
        lines, pixels = GridTransforms.correct_radar_pixel_spacing(lines, pixels, orbit_in)

        return lines, pixels

    @staticmethod
    def radar2projection(coor_in, coor_out, out_xyz='', out_height='', out_lat='', out_lon=''):
        # Get the input orbit
        orbit_in = OrbitCoordinates(coor_in.orbit, coor_in)
        orbit_in.lp_time()

        # First check if the xyz or out data is there.
        if len(out_xyz) == 0 and len(out_height) == 0:
            print('Either the xyz cartesian coordinates or the grid heights should be available.')
            return
        elif len(out_xyz) == 0 and len(out_height) != 0:
            if len(out_lat) == 0 or len(out_lon) == 0:
                x, y = coor_out.create_xy_grid()
                lat, lon = coor_out.proj2ell(x, y)

            out_xyz = OrbitCoordinates.ell2xyz(lat, lon, out_height)

        lines, pixels = orbit_in.xyz2lp(out_xyz)
        lines, pixels = GridTransforms.correct_radar_pixel_spacing(lines, pixels, orbit_in)

        return lines, pixels

    # From geographic to radar/geographic/projection
    @staticmethod
    def geographic2geographic(coor_in, coor_out):
        # This is the most basic, as it cannot really be defined in different ways. We only throw a warning if the
        # ellipsoid is not the same, otherwise we do nothing.
        if not GridTransforms.check_ellipsoid(coor_in, coor_out):
            return

        line_vals = ((coor_in.lat0 - coor_out.lat0) + np.arange(coor_out.shape[0]) * coor_out.dlat) / coor_in.dlat
        pixel_vals = ((coor_in.lon0 - coor_out.lon0) + np.arange(coor_out.shape[1]) * coor_out.dlon) / coor_in.dlon

        pixels, lines = np.meshgrid((pixel_vals, line_vals))

        return lines, pixels

    @staticmethod
    def geographic2radar(coor_in, coor_out, out_xyz='', out_height='', out_lat='', out_lon=''):
        # Works only lat/lon or xyz or height is available.
        if (len(out_lat) == 0 or len(out_lon) == 0):
            # If lat/lon coordinates are not available...
            orbit_out = OrbitCoordinates(coor_out.orbit, coor_out)
            orbit_out.lp_time()

            if len(out_xyz) != 0:
                # If xyz data is available it can be used to find lat/lon values
                orbit_out.xyz = out_xyz
            elif len(out_height) != 0:
                # If height data is available it can be used to find the xyz first if not available.
                orbit_out.height = out_height
                orbit_out.lph2xyz()
            else:
                print('Either xyz, lat/lon or heigth data should be available.')
                return

            orbit_out.xyz2ell()
            out_lat = orbit_out.lat
            out_lon = orbit_out.lon

        lines = (out_lat - coor_in.lat0) / coor_in.dlat
        pixels = (out_lon - coor_in.lon0) / coor_in.dlon

        return lines, pixels

    @staticmethod
    def geographic2projection(coor_in, coor_out, out_lat='', out_lon=''):
        # Check ellipsoids
        if not GridTransforms.check_ellipsoid(coor_in, coor_out):
            return

        if (len(out_lat) == 0 or len(out_lon) == 0):
            # If coordinates are missing calculate them.
            x, y = coor_out.create_xy_grid()
            out_lat, out_lon = coor_out.proj2ell(x, y)

        # Now calculate the line and pixel coordinates with respect to the input dataset
        lines = (out_lat - coor_in.lat0) / coor_in.dlat
        pixels = (out_lon - coor_in.lon0) / coor_in.dlon

        return lines, pixels

    # From projection to projection/geographic/radar coordinates.
    @staticmethod
    def projection2projection(coor_in, coor_out, out_lat='', out_lon=''):
        # Check ellipsoid
        if not GridTransforms.check_ellipsoid(coor_in, coor_out):
            return

        # We assume that the two coordinate systems are different.
        if (len(out_lat) == 0 or len(out_lon) == 0):
            # If coordinates are missing calculate them.
            x, y = coor_out.create_xy_grid()
            out_lat, out_lon = coor_out.proj2ell(x, y)

        # Now calculate x,y coordinates in the other projection.
        out_x, out_y = coor_in.ell2proj(out_lat, out_lon)

        # Now calculate the line and pixel coordinates with respect to the input dataset
        lines = (out_y - coor_in.y0) / coor_in.dy
        pixels = (out_x - coor_in.x0) / coor_in.dx

        return lines, pixels

    @staticmethod
    def projection2radar(coor_in, coor_out, out_xyz='', out_height='', out_lat='', out_lon=''):
        # To get the projection coordinates we have to convert to x and y coordinates.

        if (len(out_lat) == 0 or len(out_lon) == 0):
            # If coordinates are missing calculate them.
            orbit_out = OrbitCoordinates(coor_out.orbit, coor_out)
            orbit_out.lp_time()

            if len(out_xyz) != 0:
                # If xyz data is available it can be used to find lat/lon values
                orbit_out.xyz = out_xyz
            elif len(out_height) != 0:
                # If height data is available it can be used to find the xyz first if not available.
                orbit_out.height = out_height
                orbit_out.lph2xyz()
            else:
                print('Either xyz, lat/lon or heigth data should be available.')
                return

            orbit_out.xyz2ell()
            out_lat = orbit_out.lat
            out_lon = orbit_out.lon

        # Calculate the coordinates using the derived lat/lon values.
        out_x, out_y = coor_in.ell2proj(out_lat, out_lon)
        lines = (out_y - coor_in.y0) / coor_in.dy
        pixels = (out_x - coor_in.x0) / coor_in.dx

        return lines, pixels

    @staticmethod
    def projection2geographic(coor_in, coor_out):
        # Check ellipsoid.
        if not GridTransforms.check_ellipsoid(coor_in, coor_out):
            return

        # Calculate lat/lon values
        out_lat, out_lon = coor_out.create_latlon_grid()

        # Calculate the coordinates using the derived lat/lon values.
        out_x, out_y = coor_in.ell2proj(out_lat, out_lon)
        lines = (out_y - coor_in.y0) / coor_in.dy
        pixels = (out_x - coor_in.x0) / coor_in.dx

        return lines, pixels

    @staticmethod
    def check_ellipsoid(coor_in, coor_out):

        if coor_in.ellipse_type != coor_out.ellipse_type:
            print('Ellipsoid types should be the same.')
            return False
        else:
            return True

    @staticmethod
    def correct_radar_pixel_spacing(lines, pixels, orbit_in):
        # Correct for the line pixel spacing of the input dataset.
        lines = (lines - orbit_in.lines[0]) / (orbit_in.lines[1] - orbit_in.lines[0])
        pixels = (pixels - orbit_in.pixels[0]) / (orbit_in.pixels[1] - orbit_in.pixels[0])

        return lines, pixels
