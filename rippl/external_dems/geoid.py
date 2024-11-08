import numpy as np
from scipy.interpolate import RectBivariateSpline
import os
import zipfile
import shutil
import logging

class GeoidInterp():

    @staticmethod
    def download_geoid(egm_96_file=''):
        # Download egm96 file
        if not os.path.exists(egm_96_file + '.zip'):
            url = 'https://earth-info.nga.mil/php/download.php?file=egm-96interpolation'
            logging.info('downloading from ' + url + ' to ' + egm_96_file + '.zip')
            command = 'wget ' + url + ' -O ' + '"' + egm_96_file + '.zip'
            os.system(command)

        # Perform unzip operation
        with zipfile.ZipFile(egm_96_file + '.zip', 'r') as egm_96_zip:
            egm_96_zip.extract('WW15MGH.GRD', egm_96_file + '_unzipped')
        shutil.move(os.path.join(egm_96_file + '_unzipped', 'WW15MGH.GRD'), egm_96_file)
        shutil.rmtree(egm_96_file + '_unzipped')

        if not os.path.exists(egm_96_file):
            raise ConnectionError(
                'Failed to download https://earth-info.nga.mil/php/download.php?file=egm-96interpolation to '
                + egm_96_file + ' ,please try to do it manually and try again.')
        else:
            logging.info('Succesfully downloaded the EGM96 geoid file to ' + egm_96_file)

    @staticmethod
    def create_geoid(coordinates=None, egm_96_file='', lat=None, lon=None):
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
            # Download file
            GeoidInterp.download_geoid(egm_96_file=egm_96_file)
        elif os.path.getsize(egm_96_file) < 5000000:
            # For backward compatability. Old file was 2 MB and new 9 MB. Forces to download new file.
            # Remove old file and download new file
            os.remove(egm_96_file)
            GeoidInterp.download_geoid(egm_96_file=egm_96_file)

        # Load data
        with open(egm_96_file) as f:
            dat_str = " ".join(line.strip('\n') for line in f)
        egm96 = np.fromstring(dat_str, sep=' ')[6:].reshape((721, 1441)).astype('float32')
        egm96 = np.concatenate((egm96[:, 721:-1], egm96[:, :721], egm96[:, 721][:, None]), axis=1)
        lats = np.linspace(-90, 90, 721)
        lons = np.linspace(-180, 180, 1441)
        egm96_interp = RectBivariateSpline(lats, lons, np.flipud(egm96))

        if not coordinates:
            egm96_grid = np.transpose(egm96_interp.ev(lat, lon))

        elif coordinates.grid_type == 'geographic':
            lats = coordinates.lat0 + (np.arange(coordinates.shape[0]) + coordinates.first_line) * coordinates.dlat
            lons = coordinates.lon0 + (np.arange(coordinates.shape[1]) + coordinates.first_pixel) * coordinates.dlon

            lons[lons < -180] = lons[lons < -180] + 360
            lons[lons > 180] = lons[lons > 180] - 360

            if coordinates.dlat < 0:
                lats = np.flip(lats)
            if coordinates.dlon < 0:
                lons = np.flip(lons)

            egm96_grid = egm96_interp(lats, lons)
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

            egm96_grid = np.reshape(egm96_interp(lats, lons, grid=False), coordinates.shape)
        else:
            raise TypeError('Radar grid inputs cannot be used to interpolate geoid')

        return egm96_grid
