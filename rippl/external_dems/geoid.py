import numpy as np
from scipy.interpolate import RectBivariateSpline
import os

from rippl.download_login import DownloadLogin

class GeoidInterp():

    @staticmethod
    def create_geoid(coordinates=None, egm_96_file='', download=True, lat=None, lon=None):
        """
        Creates a egm96 geoid to correct input dem data to ellipsoid value.

        :param CoordinateSystem coordinates: Coordinate system of grid
        :param str egm_96_file: Filename of egm96 geoid grid. If not available it will be downloaded.
        :return: geoid grid for coordinate system provided
        :rtype: np.ndarray
        """

        # (For this purpose the grid is downloaded from:
        # http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/binarygeoid.html

        if not os.path.exists(egm_96_file):
            # Download egm96 file
            if download:
                if os.name == 'nt':
                    download_data = DownloadLogin('', '', '')
                    url = 'http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/WW15MGH.DAC'
                    print('downloading from ' + url + ' to ' + egm_96_file)
                    download_data.download_file(url[1:-1], egm_96_file, 3)
                else:
                    command = 'wget http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/WW15MGH.DAC -O ' + '"' + egm_96_file + '"'
                    print(command)
                    os.system(command)
            else:
                raise LookupError('No geoid file can be found. Please set download to True and try again')

            if not os.path.exists(egm_96_file):
                raise ConnectionError('Failed to download http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/WW15MGH.DAC to '
                      + egm_96_file + ' ,please try to do it manually and try again. \n Run the following on the command line \n ' + command)
            else:
                print('Succesfully downloaded the EGM96 geoid file to ' + egm_96_file)

        # Load data
        egm96 = np.fromfile(egm_96_file, dtype='>i2').reshape((721, 1440)).astype('float32')
        egm96 = np.concatenate((egm96[:, 721:], egm96[:, :721], egm96[:, 721][:, None]), axis=1)
        lats = np.linspace(-90, 90, 721)
        lons = np.linspace(-180, 180, 1441)
        egm96_interp = RectBivariateSpline(lats, lons, egm96)

        if not coordinates:
            if not lat or not lon:
                egm96_grid = egm96
            else:
                egm96_grid = egm96_interp(lon, lat)

        elif coordinates.grid_type == 'geographic':
            lats = coordinates.lat0 + (np.arange(coordinates.shape[0]) + coordinates.first_line) * coordinates.dlat
            lons = coordinates.lon0 + (np.arange(coordinates.shape[1]) + coordinates.first_pixel) * coordinates.dlon

            lons[lons < -180] = lons[lons < -180] + 360
            lons[lons > 180] = lons[lons > 180] - 360

            if coordinates.dlat < 0:
                lats = np.flip(lats)
            if coordinates.dlon < 0:
                lons = np.flip(lons)

            egm96_grid = np.transpose(egm96_interp(lons, lats))
            if coordinates.dlat < 0:
                egm96_grid = np.flip(egm96_grid, 0)
            if coordinates.dlon < 0:
                egm96_grid = np.flip(egm96_grid, 1)

        elif coordinates.grid_type == 'projection':

            ys = coordinates.y0 + (np.arange(coordinates.shape[0]) + coordinates.first_line) * coordinates.dy
            xs = coordinates.x0 + (np.arange(coordinates.shape[1]) + coordinates.first_pixel) * coordinates.dx
            x, y = np.meshgrid(xs, ys)
            lats, lons = coordinates.proj2ell(np.ravel(x), np.ravel(y))
            del x, y

            lons[lons < -180] = lons[lons < -180] + 360
            lons[lons > 180] = lons[lons > 180] - 360

            egm96_grid = np.reshape(egm96_interp(lons, lats, grid=False), coordinates.shape)
        else:
            raise TypeError('Radar grid inputs cannot be used to interpolate geoid')

        return egm96_grid / 100
