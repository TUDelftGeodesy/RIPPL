'''
This class holds a number of functions which gives a transform between different coordinate systems.

'''

import numpy as np
import os
import logging

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.orbit import Orbit
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates
from rippl.user_settings import UserSettings

class GridTransforms(object):

    def __init__(self, in_coor, out_coor):

        if not isinstance(in_coor, CoordinateSystem) or not isinstance(out_coor, CoordinateSystem):
            raise TypeError('Input and output coordinatesystem should be an CoordinateSystem object!')

        # Check whether orbits are loaded.
        if (in_coor.grid_type == 'radar_coordinates' and not isinstance(in_coor.orbit, Orbit)) or \
                (out_coor.grid_type == 'radar_coordinates' and not isinstance(out_coor.orbit, Orbit)):
            raise AttributeError('The orbits of radar coordinate systems should be loaded to convert grids')

        # Also check the timing
        if (in_coor.grid_type == 'radar_coordinates' and
            (in_coor.ra_step == 0 or in_coor.az_step == 0 or in_coor.ra_time == 0 or in_coor.az_time == 0)) or \
                (out_coor.grid_type == 'radar_coordinates' and
                 (out_coor.ra_step == 0 or out_coor.az_step == 0 or out_coor.ra_time == 0 or out_coor.az_time == 0)):
            raise AttributeError('First define the range and azimuth timing before create a transform.')

        self.in_coor = in_coor
        self.out_coor = out_coor

        self.dem = ''
        self.xyz = ''
        self.lat = ''
        self.lon = ''

    def add_dem(self, dem):
        # Add dem to dataset
        if not dem.shape == tuple(self.out_coor.shape):
            raise AssertionError('Shape of input DEM and output coordinate system are not the same')

        self.dem = np.ravel(dem)

    def add_xyz(self, X, Y, Z):
        # Add xyz data to dataset.
        for dat in [X, Y, Z]:
            if not dat.shape == tuple(self.out_coor.shape):
                raise AssertionError('Shape of input coordinate data and output coordinate system are not the same')

        self.xyz = np.concatenate((np.ravel(X)[None, :], np.ravel(Y)[None, :], np.ravel(Z)[None, :]))

    def add_lat_lon(self, lat, lon):
        # Add xyz data to dataset.
        for dat in [lat, lon]:
            if not dat.shape == tuple(self.out_coor.shape):
                raise AssertionError('Shape of input coordinate data and output coordinate system are not the same')

        self.lat = np.ravel(lat)
        self.lon = np.ravel(lon)

    def __call__(self):

        # Load the xyz, height data needed. If it is missing, throw an error.
        if self.in_coor.grid_type == 'radar_coordinates':
            if self.out_coor.grid_type == 'radar_coordinates':
                if not isinstance(self.xyz, str):
                    out_xyz = self.xyz
                elif not isinstance(self.dem, str) and not isinstance(self.lon, str) and not isinstance(self.lat, str):
                    out_xyz = self.get_radar_xyz(lat=self.lat, lon=self.lon, dem=self.dem)
                elif not isinstance(self.dem, str):
                    out_xyz = self.get_radar_xyz(dem=self.dem)
                elif not isinstance(self.lon, str) and not isinstance(self.lat, str):
                    out_xyz = self.get_radar_xyz(lat=self.lat, lon=self.lon)
                else:
                    raise ValueError('When converting radar to radar grid either xyz, lat/lon/dem, lat/lon or dem '
                                     'should be available')
            else:
                if isinstance(self.dem, str):
                    lat, lon = self.get_projection_lat_lon(self.out_coor)
                    self.dem = self.get_geoid(lat, lon)

                out_xyz = self.get_projection_xyz(self.out_coor)
        elif self.out_coor.grid_type == 'radar_coordinates':
            if not isinstance(self.dem, str):
                orbit = OrbitCoordinates(self.out_coor)
                orbit.xyz = self.get_radar_xyz(dem=self.dem)
                orbit.xyz2ell()
                out_lat = orbit.lat
                out_lon = orbit.lon
            elif not isinstance(self.lon, str) and not isinstance(self.lat, str):
                out_lat = self.lat
                out_lon = self.lon
            else:
                raise ValueError('When converting to radar grid  lat/lon or dem should be available')

        if self.out_coor.shape == [0,0]:
            lines = np.zeros((0, 0))
            pixels = np.zeros((0, 0))

            return lines, pixels

        # Call the right transform function and return it as an output.
        if self.in_coor.grid_type == 'radar_coordinates':
            if self.out_coor.grid_type == 'radar_coordinates':
                lines, pixels = GridTransforms.radar2radar(self.in_coor, self.out_coor, out_xyz)
            elif self.out_coor.grid_type == 'geographic':
                lines, pixels = GridTransforms.radar2geographic(self.in_coor, self.out_coor, out_xyz)
            elif self.out_coor.grid_type == 'projection':
                lines, pixels = GridTransforms.radar2projection(self.in_coor, self.out_coor, out_xyz)
        elif self.in_coor.grid_type == 'geographic':
            if self.out_coor.grid_type == 'radar_coordinates':
                lines, pixels = GridTransforms.geographic2radar(self.in_coor, self.out_coor, out_lat=out_lat, out_lon=out_lon)
            elif self.out_coor.grid_type == 'geographic':
                lines, pixels = GridTransforms.geographic2geographic(self.in_coor, self.out_coor)
            elif self.out_coor.grid_type == 'projection':
                lines, pixels = GridTransforms.geographic2projection(self.in_coor, self.out_coor)
        elif self.in_coor.grid_type == 'projection':
            if self.out_coor.grid_type == 'radar_coordinates':
                lines, pixels = GridTransforms.projection2radar(self.in_coor, self.out_coor, out_lat=out_lat, out_lon=out_lon)
            elif self.out_coor.grid_type == 'geographic':
                lines, pixels = GridTransforms.projection2geographic(self.in_coor, self.out_coor)
            elif self.out_coor.grid_type == 'projection':
                lines, pixels = GridTransforms.projection2projection(self.in_coor, self.out_coor)

        lines = np.reshape(lines, self.out_coor.shape)
        pixels = np.reshape(pixels, self.out_coor.shape)

        return lines, pixels

    @staticmethod
    def get_projection_lat_lon(coordinates):
        """
        This method creates a lat/lon grid based on geographic or projection coordinate system.

        :param CoordinateSystem coordinates: Input coordinate system
        :return: lat/lon coordinate grids
        """

        if coordinates.grid_type == 'geographic':
            lat, lon = coordinates.create_latlon_grid()
        elif coordinates.grid_type == 'projection':
            X, Y = coordinates.create_xy_grid()
            lat, lon = coordinates.proj2ell(X, Y)

        return np.ravel(lat), np.ravel(lon)

    @staticmethod
    def get_projection_xyz(coordinates, dem):
        """
        This method calculates the xyz coordinates. As a by-product it provides also the lat/lon grids.

        :param coordinates:
        :param dem:
        :return:
        """

        lat, lon = GridTransforms.get_projection_lat_lon(coordinates)
        xyz = OrbitCoordinates.ell2xyz(lat, lon, np.ravel(dem))

        return xyz

    @staticmethod
    def get_radar_xyz(coordinates, dem=[], lat=[], lon=[]):
        """
        Get the radar cartesian coordinates based on lat/lon and DEM. If only DEM is available, the xyz coordinates are
        estimated using the OrbitCoordinates class. If only lat/lon are available, we assume that the DEM follows the
        geoid and the DEM is calculated based on the lat/lon combinations.

        If all three DEM and lat/lon are available the xyz coordinates are calculated using the transformation between
        geographic to cartesian coordinates

        :param coordinates:
        :param dem:
        :param lat:
        :param lon:
        :return:
        """

        if not isinstance(dem, str) and not isinstance(lon, str) and not isinstance(lat, str):
            xyz = OrbitCoordinates.ell2xyz(np.ravel(lat), np.ravel(lon), np.ravel(dem))
        elif not isinstance(lon, str) and not isinstance(lat, str):
            geoid = GridTransforms.get_geoid(np.ravel(lat), np.ravel(lon))
            xyz = OrbitCoordinates.ell2xyz(np.ravel(lat), np.ravel(lon), geoid)
        elif not isinstance(dem, str):
            orbit = OrbitCoordinates(coordinates=coordinates)
            orbit.height = np.ravel(dem)
            xyz = orbit.lph2xyz()
        else:
            raise ValueError('Either the DEM or the lat/lon combinations should be available to estimate xyz values.')

        return xyz

    @staticmethod
    def get_geoid(coordinates, lat, lon):
        """
        Get the radar geoid information assuming that this is close to the actual DEM

        :param coordinates:
        :param lat:
        :param lon:
        :return:
        """

        grid_lats = np.ravel(lat)
        grid_lons = np.ravel(lon)
        from rippl.external_dems.geoid import GeoidInterp
        settings = UserSettings()
        settings.load_settings()
        egm_96_file = os.path.join(settings.settings['paths']['DEM_database'], 'geoid', 'egm96.dat')

        geoid = GeoidInterp.create_geoid(egm_96_file=egm_96_file, lat=grid_lats, lon=grid_lons,
                                         download=False)

        return geoid

    # The next part consists of all the different functions to convert between the different coordinate systems.
    # From radar to radar/geographic/projection systems.
    @staticmethod
    def radar2radar(in_coor, out_coor, out_xyz='', out_height=''):
        # First check if the xyz data is there.
        if not isinstance(out_xyz, (np.ndarray, list)) or not isinstance(out_height, (np.ndarray, list)):
            logging.info('Either the xyz cartesian coordinates or the grid heights should be available.')
            return
        elif isinstance(out_xyz, (np.ndarray, list)) and isinstance(out_height, (np.ndarray, list)):
            orbit_out = OrbitCoordinates(out_coor)
            orbit_out.lp_time()
            orbit_out.height = out_height
            orbit_out.lph2xyz()
            out_xyz = orbit_out.xyz

        # Now estimate the orbit of the input image and calculate the output coordinates.
        orbit_in = OrbitCoordinates(in_coor)
        orbit_in.lp_time()
        lines, pixels = orbit_in.xyz2lp(out_xyz)
        lines, pixels = GridTransforms.correct_radar_pixel_spacing(lines, pixels, orbit_in)

        return lines, pixels

    @staticmethod
    def radar2geographic(in_coor, out_coor, out_xyz='', out_height=''):
        # Get the input orbit
        orbit_in = OrbitCoordinates(in_coor)
        orbit_in.lp_time()

        # First check if the xyz or out data is there.
        if not isinstance(out_xyz, (np.ndarray, list)) and not isinstance(out_height, (np.ndarray, list)):
            logging.info('Either the xyz cartesian coordinates or the grid heights should be available.')
            return
        elif not isinstance(out_xyz, (np.ndarray, list)) and isinstance(out_height, (np.ndarray, list)):
            lat, lon = out_coor.create_latlon_grid()
            out_xyz = OrbitCoordinates.ell2xyz(lat, lon, out_height)

        lines, pixels = orbit_in.xyz2lp(out_xyz)
        lines, pixels = GridTransforms.correct_radar_pixel_spacing(lines, pixels, orbit_in)

        return lines, pixels

    @staticmethod
    def radar2projection(in_coor, out_coor, out_xyz='', out_height='', out_lat='', out_lon=''):
        # Get the input orbit
        orbit_in = OrbitCoordinates(in_coor)
        orbit_in.lp_time()

        # First check if the xyz or out data is there.
        if not isinstance(out_xyz, (np.ndarray, list)) and not isinstance(out_height, (np.ndarray, list)):
            logging.info('Either the xyz cartesian coordinates or the grid heights should be available.')
            return
        elif not isinstance(out_xyz, (np.ndarray, list)) and isinstance(out_height, (np.ndarray, list)):
            if not isinstance(out_lat, (np.ndarray, list)) or not isinstance(out_lon, (np.ndarray, list)):
                x, y = out_coor.create_xy_grid()
                lat, lon = out_coor.proj2ell(x, y)

            out_xyz = OrbitCoordinates.ell2xyz(lat, lon, out_height)

        lines, pixels = orbit_in.xyz2lp(out_xyz)
        lines, pixels = GridTransforms.correct_radar_pixel_spacing(lines, pixels, orbit_in)

        return lines, pixels

    # From geographic to radar/geographic/projection
    @staticmethod
    def geographic2geographic(in_coor, out_coor):
        # This is the most basic, as it cannot really be defined in different ways. We only throw a warning if the
        # ellipsoid is not the same, otherwise we do nothing.
        if not GridTransforms.check_ellipsoid(in_coor, out_coor):
            return

        start_lat_diff = -(in_coor.lat0 - out_coor.lat0)
        start_lon_diff = -(in_coor.lon0 - out_coor.lon0)

        line_vals = (start_lat_diff + (np.arange(out_coor.shape[0]) + out_coor.first_line) * out_coor.dlat) / in_coor.dlat
        pixel_vals = (start_lon_diff + (np.arange(out_coor.shape[1]) + out_coor.first_pixel) * out_coor.dlon) / in_coor.dlon

        pixels, lines = np.meshgrid(pixel_vals, line_vals)

        return lines, pixels

    @staticmethod
    def geographic2radar(in_coor, out_coor, out_xyz='', out_height='', out_lat='', out_lon=''):
        # Works only lat/lon or xyz or height is available.
        if (not isinstance(out_lat, (np.ndarray, list)) or not isinstance(out_lon, (np.ndarray, list))):
            # If lat/lon coordinates are not available...
            orbit_out = OrbitCoordinates(out_coor)
            orbit_out.lp_time()

            if isinstance(out_xyz, (np.ndarray, list)):
                # If xyz data is available it can be used to find lat/lon values
                orbit_out.xyz = out_xyz
            elif isinstance(out_height, (np.ndarray, list)):
                # If height data is available it can be used to find the xyz first if not available.
                orbit_out.height = out_height
                orbit_out.lph2xyz()
            else:
                logging.info('Either xyz, lat/lon or height data should be available.')
                return

            orbit_out.xyz2ell()
            out_lat = orbit_out.lat
            out_lon = orbit_out.lon

        lines = (out_lat - in_coor.lat0) / in_coor.dlat
        pixels = (out_lon - in_coor.lon0) / in_coor.dlon

        return lines, pixels

    @staticmethod
    def geographic2projection(in_coor, out_coor, out_lat='', out_lon=''):
        # Check ellipsoids
        if not GridTransforms.check_ellipsoid(in_coor, out_coor):
            return

        if (not isinstance(out_lat, (np.ndarray, list)) or not isinstance(out_lon, (np.ndarray, list))):
            # If coordinates are missing calculate them.
            x, y = out_coor.create_xy_grid()
            out_lat, out_lon = out_coor.proj2ell(x, y)

        # Now calculate the line and pixel coordinates with respect to the input dataset
        lines = (out_lat - in_coor.lat0) / in_coor.dlat
        pixels = (out_lon - in_coor.lon0) / in_coor.dlon

        return lines, pixels

    # From projection to projection/geographic/radar coordinates.
    @staticmethod
    def projection2projection(in_coor, out_coor, out_lat='', out_lon=''):
        # Check ellipsoid
        if not GridTransforms.check_ellipsoid(in_coor, out_coor):
            return

        # We assume that the two coordinate systems are different.
        if (not isinstance(out_lat, (np.ndarray, list)) or not isinstance(out_lon, (np.ndarray, list))):
            # If coordinates are missing calculate them.
            x, y = out_coor.create_xy_grid()
            out_lat, out_lon = out_coor.proj2ell(x, y)

        # Now calculate x,y coordinates in the other projection.
        out_x, out_y = in_coor.ell2proj(out_lat, out_lon)

        # Now calculate the line and pixel coordinates with respect to the input dataset
        lines = (out_y - in_coor.y0) / in_coor.dy
        pixels = (out_x - in_coor.x0) / in_coor.dx

        return lines, pixels

    @staticmethod
    def projection2radar(in_coor, out_coor, out_xyz='', out_height='', out_lat='', out_lon=''):
        # To get the projection coordinates we have to convert to x and y coordinates.

        if (not isinstance(out_lat, (np.ndarray, list)) or not isinstance(out_lon, (np.ndarray, list))):
            # If coordinates are missing calculate them.
            orbit_out = OrbitCoordinates(out_coor)
            orbit_out.lp_time()

            if isinstance(out_xyz, (np.ndarray, list)):
                # If xyz data is available it can be used to find lat/lon values
                orbit_out.xyz = out_xyz
            elif isinstance(out_height, (np.ndarray, list)):
                # If height data is available it can be used to find the xyz first if not available.
                orbit_out.height = out_height
                orbit_out.lph2xyz()
            else:
                logging.info('Either xyz, lat/lon or heigth data should be available.')
                return

            orbit_out.xyz2ell()
            out_lat = orbit_out.lat
            out_lon = orbit_out.lon

        # Calculate the coordinates using the derived lat/lon values.
        out_x, out_y = in_coor.ell2proj(out_lat, out_lon)
        lines = (out_y - in_coor.y0) / in_coor.dy
        pixels = (out_x - in_coor.x0) / in_coor.dx

        return lines, pixels

    @staticmethod
    def projection2geographic(in_coor, out_coor):
        # Check ellipsoid.
        if not GridTransforms.check_ellipsoid(in_coor, out_coor):
            return

        # Calculate lat/lon values
        out_lat, out_lon = out_coor.create_latlon_grid()

        # Calculate the coordinates using the derived lat/lon values.
        out_x, out_y = in_coor.ell2proj(out_lat, out_lon)
        lines = (out_y - in_coor.y0) / in_coor.dy
        pixels = (out_x - in_coor.x0) / in_coor.dx

        return lines, pixels

    @staticmethod
    def check_ellipsoid(in_coor, out_coor):

        if in_coor.ellipse_type != out_coor.ellipse_type:
            logging.info('Ellipsoid types should be the same.')
            return False
        else:
            return True

    @staticmethod
    def correct_radar_pixel_spacing(lines, pixels, orbit_in):
        # Correct for the line pixel spacing of the input dataset.
        lines = (lines - orbit_in.lines[0]) / (orbit_in.lines[1] - orbit_in.lines[0])
        pixels = (pixels - orbit_in.pixels[0]) / (orbit_in.pixels[1] - orbit_in.pixels[0])

        return lines, pixels
