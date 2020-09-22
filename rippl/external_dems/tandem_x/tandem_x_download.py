"""
Function created by Gert Mulder
Institute TU Delft
Date 29-10-2019
Part of RIPPL SAR processing software

This class download tandem-x data to create DEMs for InSAR processing. This data is freely available worldwide and can
be found at:
https://geoservice.dlr.de/web/dataguide/tdm90/#access

The data itself is given in a geographic grid that changes based on latitude. Therefore, to create consistent DEM's
a part of the data has to be resampled. It is left to the user to decide the resolution in arc seconds in the longitu-
donal direction. The resolution in latitude is always 3 arc seconds on a WGS84 ellipsoid.

As a guideline it is best to use the native resolution for the largest part of your area of interest, which is:
3 arc-seconds for lat < 50
4.5 arc-seconds for lat < 60
6 arc-seconds for lat < 70
9 arc-seconds for lat < 80
15 arc-seconds for lat < 85
30 arc-seconds for lat < 90
These resolutions hold only for the longitude direction. For example, if you want to look at an area around the equator
you should go for a 3 arc-seconds resolution, but at mid latitudes around for example 55 degrees north, go for the 4.5
arc-seconds resolution. If your region overlaps different regions, it is better to go for the highest resolution in your
region of interest to prevent information loss.
"""

import os
import pickle
import shutil
import zipfile
import numpy as np
import ftplib
from scipy.interpolate import RectBivariateSpline
from multiprocessing import get_context
import gdal

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.image_processing_meta import ImageProcessingMeta
from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.external_dems.geoid import GeoidInterp
from rippl.user_settings import UserSettings


class TandemXDownloadTile(object):
    # To enable parallel processing we create another class for the actual processing.

    def __init__(self, tandem_x_folder='', username='', password='', lon_resolution=3):

        settings = UserSettings()
        settings.load_settings()

        # TanDEM-X folder
        if not tandem_x_folder:
            self.tandem_x_folder = os.path.join(settings.DEM_database, 'TanDEM-X')
        else:
            self.tandem_x_folder = tandem_x_folder

        # credentials
        if not username:
            self.username = settings.DLR_username
        else:
            self.username = username
        if not password:
            self.password = settings.DLR_password
        else:
            self.password = password

        self.lon_resolution = lon_resolution

    def __call__(self, input):

        url = input[0]
        file_zip = input[1]
        file_unzip = input[2]
        lat = input[3]
        lon = input[4]

        if not os.path.exists(file_unzip):
            self.download_dem_file(url, file_zip, file_unzip)
        self.resize_dem_file(file_unzip, self.lon_resolution)

    def download_dem_file(self, ftp_path, file_zip, file_unzip):
        # This function downloads data for 1 or 3 arc second dem.

        # Download and unzip
        try:
            if not os.path.exists(file_zip):
                # Make FTP connection
                server = 'tandemx-90m.dlr.de'

                ftp = ftplib.FTP_TLS()
                ftp.connect(host=server)
                ftp.login(user=self.username, passwd=self.password)
                ftp.prot_p()

                # Download file
                ftp.cwd(os.path.dirname(ftp_path))
                with open(file_zip, 'wb') as file:
                    ftp.retrbinary('RETR %s' % os.path.basename(ftp_path), file.write)

            if not os.path.exists(file_unzip):
                zip_data = zipfile.ZipFile(file_zip)
                source = zip_data.open([path for path in zip_data.namelist() if
                                        os.path.basename(os.path.dirname(path)) == 'DEM' and path.endswith('.tif')][0])
                target = open(file_unzip, 'wb')
                shutil.copyfileobj(source, target)
                target.close()
        except:
            raise ConnectionError('Failed to download ' + ftp_path)

    def resize_dem_file(self, file_unzip, lon_resolution=3):
        """
        Here we resize the DEM file to get 3arc seconds grids in both latitude and longitude direction.
        For larger latitudes this will result in very high sampling in the longitudonal direction but this will not
        affect the final product results.
        If we would like to process data over the poles directly, this approach maybe sub-optimal but other implemen-
        tation would be very cumbersome. In those cases we would advise to use a different projection.

        :return:
        """

        if not lon_resolution in [3, 4.5, 6, 9, 15, 30]:
            raise TypeError('Lon size is not one of the default resolutions 3, 4.5, 6, 9, 15 or 30 arc seconds. '
                            'Aborting...')

        dem_size_folder = os.path.join(self.tandem_x_folder, str(lon_resolution).zfill(2) + '_arc_seconds')

        # Load the data using gdal
        data_file = gdal.Open(file_unzip)
        band = data_file.GetRasterBand(1)
        data = band.ReadAsArray()

        # Check how the data should be split in 1 by 1 degree tiles.
        size = data.shape
        geo_transform = data_file.GetGeoTransform()
        lons = (np.arange(size[1]) + 0.5) * geo_transform[1] + geo_transform[0]
        lats = (np.arange(size[0]) + 0.5) * geo_transform[5] + geo_transform[3]

        # Fill empty values with geoid values (most likely water/sea)
        coor = CoordinateSystem()
        coor.create_geographic(shape=size, geo_transform=geo_transform)
        egm96_file = os.path.join(os.path.dirname(self.tandem_x_folder), 'geoid', 'egm96.dat')
        egm96 = GeoidInterp.create_geoid(coor, egm96_file, True)
        data[data == -32767] = - egm96[data == -32767]

        int_lons = np.arange(int(np.round(lons[0])), int(np.round(lons[-1])))
        degree_size = (size[1] - 1) / len(int_lons) + 1

        # Resample if needed and write the data to disk.
        new_file_names = []
        new_int_lons = []
        lat = np.round(lats[-1])
        if lat < 0:
            lat_str = 'S' + str(np.abs(int(lat))).zfill(2)
        else:
            lat_str = 'N' + str(int(lat)).zfill(2)

        for lon in int_lons:
            if lon < 0:
                lon_str = 'W' + str(np.abs(int(lon))).zfill(3)
            else:
                lon_str = 'E' + str(int(lon)).zfill(3)

            new_file_name = os.path.join(dem_size_folder, 'TDM_DEM_' + lat_str + lon_str + '.raw')
            if not os.path.exists(new_file_name):
                new_file_names.append(new_file_name)
                new_int_lons.append(lon)

        lats = np.flip(lats)
        data = np.flipud(data)

        tdx_dem_interp = RectBivariateSpline(lats, lons, data)

        for new_int_lon, new_file_name in zip(new_int_lons, new_file_names):
            new_lons = new_int_lon + np.arange(np.int(np.round(3600.0 / lon_resolution)) + 1) * (lon_resolution / 3600.0)
            print('Saved DEM file ' + new_file_name)
            new_data = np.memmap(new_file_name, np.float32, 'w+', shape=(size[0], len(new_lons)))
            new_data[:, :] = np.flipud(tdx_dem_interp(lats, new_lons))
            new_data.flush()

class TandemXDownload(object):

    def __init__(self, tandem_x_folder=None, username=None, password=None, lon_resolution=3, n_processes=4):
        # tandem_x_folder

        settings = UserSettings()
        settings.load_settings()

        # TanDEM-X folderƒ
        if not tandem_x_folder:
            self.tandem_x_folder = os.path.join(settings.DEM_database, 'TanDEM-X')
        else:
            self.tandem_x_folder = tandem_x_folder

        # credentials
        if not username:
            self.username = settings.DLR_username
        else:
            self.username = username
        if not password:
            self.password = settings.DLR_password
        else:
            self.password = password

        if not os.path.exists(self.tandem_x_folder):
            raise FileExistsError('Path to tandem-x folder does not exist')
        zip_data_folder = os.path.join(self.tandem_x_folder, 'orig_data')
        tiff_data_folder = os.path.join(self.tandem_x_folder, 'geotiff_data')
        if not os.path.exists(zip_data_folder):
            os.mkdir(zip_data_folder)
        if not os.path.exists(tiff_data_folder):
            os.mkdir(tiff_data_folder)

        if not lon_resolution in [3, 4.5, 6, 9, 15, 30]:
            raise TypeError('Lon size is not one of the default resolutions 3, 4.5, 6, 9, 15 or 30 arc seconds. '
                            'Aborting...')
        resolution_folder = os.path.join(self.tandem_x_folder, str(lon_resolution).zfill(2) + '_arc_seconds')
        if not os.path.exists(resolution_folder):
            os.mkdir(resolution_folder)

        # List of files to be downloaded
        self.filelist = self.tandem_x_listing(self.tandem_x_folder, username, password)

        # shapes and limits of these shapes
        self.shapes = []
        self.latlims = []
        self.lonlims = []

        # meta and polygons
        self.meta = ''
        self.polygon = ''

        # Resolution of files (either tandem_x1, tandem_x3 or STRM30)
        self.lon_resolution = lon_resolution

        # processes
        self.n_processes = n_processes

    def __call__(self, meta, buffer=1.0, rounding=1.0):

        if isinstance(meta, ImageProcessingData):
            self.meta = meta.meta
        elif isinstance(meta, ImageProcessingMeta):
            self.meta = meta
        else:
            raise TypeError('Input meta data should be an ImageProcessingData or ImageProcessingMeta object.')

        # In first instance we assume a step size of 3 seconds in both latitude and longitude direction.
        # The step size in longitude will be updates later on.
        lat_step = 1.0 / 3600 * 3
        lon_step = 1.0 / 3600 * self.lon_resolution

        # Create output coordinates.
        radar_coor = CoordinateSystem()
        radar_coor.create_radar_coordinates()
        radar_coor.load_readfile(self.meta.readfiles['original'])
        radar_coor.orbit = self.meta.find_best_orbit('original')
        self.coordinates = CoordinateSystem()
        self.coordinates.create_geographic(dlat=lat_step, dlon=lon_step)
        new_coor = CoorNewExtend(radar_coor, self.coordinates, buffer=buffer, rounding=rounding)
        self.coordinates = new_coor.out_coor

        resampled_tiles, tiles, download_tiles, [tile_lats, tile_lons], ftp_paths, tiles_zip = \
            self.select_tiles(self.filelist, self.coordinates, self.tandem_x_folder, lon_resolution=self.lon_resolution)

        # First create a download class.
        tile_download = TandemXDownloadTile(self.tandem_x_folder, self.username, self.password, self.lon_resolution)

        # Loop over all images
        download_dat = [[ftp_path, file_zip, file_unzip, lat, lon] for
                     ftp_path, file_zip, file_unzip, lat, lon in
                     zip(ftp_paths, tiles_zip, download_tiles, tile_lats, tile_lons)]
        if self.n_processes > 1:
            with get_context("spawn").Pool(processes=self.n_processes, maxtasksperchild=5) as pool:
                # Process in blocks of 25
                block_size = 25
                for i in range(int(np.ceil(len(download_dat) / block_size))):
                    last_dat = np.minimum((i + 1) * block_size, len(download_dat))
                    pool.map(tile_download, download_dat[i*block_size:last_dat])
        else:
            for download_info in download_dat:
                tile_download(download_info)

    @staticmethod
    def select_tiles(filelist, coordinates, tandem_x_folder, lon_resolution=3):
        # Adds tandem_x files to the list of files to be downloaded

        settings = UserSettings()
        settings.load_settings()

        # TanDEM-X folder
        if not tandem_x_folder:
            tandem_x_folder = os.path.join(settings.DEM_database, 'TanDEM-X')

        # Check coordinates
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')
            return
        elif coordinates.grid_type != 'geographic':
            print('only geographic coordinate systems can be used to download TanDEM-X data')
            return

        tiles_zip = []
        tiles = []
        download_tiles = []
        tile_lats = []
        tile_lons = []
        ftp_path = []
        resampled_tiles = []

        lat0 = coordinates.lat0 + coordinates.dlat * coordinates.first_line
        lon0 = coordinates.lon0 + coordinates.dlon * coordinates.first_pixel
        lats = np.arange(np.floor(lat0), np.ceil(lat0 + coordinates.shape[0] * coordinates.dlat)).astype(np.int32)
        resolution_folder = os.path.join(tandem_x_folder, str(lon_resolution).zfill(2) + '_arc_seconds')

        for lat in lats:
            # Depending on latitude there is a different spacing.
            lon_lim = [int(np.floor(lon0)), int(np.ceil(lon0 + coordinates.shape[1] * coordinates.dlon))]
            grid_size, possible_lons = TandemXDownload.get_lon_spacing(lat)
            tile_size = possible_lons[1] - possible_lons[0]
            lons = possible_lons[(possible_lons + 2 >= np.min(lon_lim)) * (possible_lons <= np.max(lon_lim))]

            for lon in lons:

                lat = int(lat)
                lon = int(lon)

                if lat < 0:
                    latstr = 'S' + str(abs(lat)).zfill(2)
                else:
                    latstr = 'N' + str(lat).zfill(2)
                if lon < 0:
                    lonstr = 'W' + str(abs(lon)).zfill(3)
                else:
                    lonstr = 'E' + str(lon).zfill(3)

                # Check if file exists in filelist
                if latstr not in filelist.keys():
                    continue
                elif lonstr not in filelist[latstr].keys():
                    continue

                for lon_tile_val in np.arange(tile_size) + lon:

                    if lon_tile_val < 0:
                        tile_lonstr = 'W' + str(abs(lon_tile_val)).zfill(3)
                    else:
                        tile_lonstr = 'E' + str(lon_tile_val).zfill(3)

                    if lon_lim[0] < (lon_tile_val + 1) and lon_lim[1] > lon_tile_val:
                        resampled_tile = os.path.join(resolution_folder, 'TDM_DEM_' + latstr + tile_lonstr + '.raw')
                        resampled_tiles.append(resampled_tile)

                unzip = os.path.join(tandem_x_folder, 'geotiff_data', 'TDM_DEM_' + latstr + lonstr + '.tiff')
                tiles.append(unzip)

                tiles_zip.append(os.path.join(tandem_x_folder, 'orig_data', 'TDM_DEM_' + latstr + lonstr + '.zip'))

                download_tiles.append(unzip)
                tile_lats.append(lat)
                tile_lons.append(lon)
                ftp_path.append(filelist[latstr][lonstr])

        return resampled_tiles, tiles, download_tiles, [tile_lats, tile_lons], ftp_path, tiles_zip

    @staticmethod
    def tandem_x_listing(tandem_x_folder, username='', password=''):
        # This script makes a list of all the available 1,3 and 30 arc second datafiles.
        # This makes it easier to detect whether files do or don't exist.

        settings = UserSettings()
        settings.load_settings()

        # TanDEM-X folder
        if not tandem_x_folder:
            tandem_x_folder = os.path.join(settings.DEM_database, 'TanDEM-X')

        # credentials
        if not username:
            username = settings.DLR_username
        if not password:
            password = settings.DLR_password

        data_file = os.path.join(tandem_x_folder, 'filelist')
        if os.path.exists(data_file):
            dat = open(data_file, 'rb')
            filelist = pickle.load(dat)
            dat.close()
            return filelist

        server = 'tandemx-90m.dlr.de'
        ftp = ftplib.FTP_TLS()
        ftp.connect(host=server)
        ftp.login(user=username, passwd=password)
        ftp.prot_p()

        dem_folders = ftp.nlst('90mdem/DEM')
        dem_files = []
        for dem_folder in dem_folders:
            sub_dem_folders = ftp.nlst(dem_folder)
            print('indexing ' + dem_folder)
            for sub_dem_folder in sub_dem_folders:
                dem_files.extend(ftp.nlst(sub_dem_folder))

        # Index based on latitude and longitude values
        filelist = dict()
        for dem_file in dem_files:
            lon_str = dem_file[-8:-4]
            lat_str = dem_file[-11:-8]

            if lat_str not in filelist.keys():
                filelist[lat_str] = dict()
            filelist[lat_str][lon_str] = dem_file

        # Get the tiff file path in the zip
        file_list = open(os.path.join(tandem_x_folder, 'filelist'), 'wb')
        pickle.dump(filelist, file_list)
        file_list.close()

        return filelist

    @staticmethod
    def get_lon_spacing(lat):
        """
        This function gets the spacing and coordinates of TanDEM-X tiles based on latitude value.

        :param int lat: Latitude coordinate of DEM
        :return:
        """

        if lat < 0:
            abs_lat = np.abs(lat) - 1
        else:
            abs_lat = lat

        if abs_lat < 50:
            lons = np.arange(-180, 180, 1)
            grid_size = 3
        elif abs_lat < 60:
            lons = np.arange(-180, 180, 1)
            grid_size = 4.5
        elif abs_lat < 70:
            lons = np.arange(-180, 180, 2)
            grid_size = 6
        elif abs_lat < 80:
            lons = np.arange(-180, 180, 2)
            grid_size = 9
        elif abs_lat < 85:
            lons = np.arange(-180, 180, 4)
            grid_size = 15
        elif abs_lat < 90:
            lons = np.arange(-180, 180, 4)
            grid_size = 30

        return grid_size, lons
