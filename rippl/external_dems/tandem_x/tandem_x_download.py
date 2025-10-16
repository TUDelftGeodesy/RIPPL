"""
Function created by Gert Mulder
Institute TU Delft
Date 29-10-2019
Part of RIPPL SAR processing software

This class download tandem-x data to create DEMs for InSAR processing. This data is freely available worldwide and can
be found at:
https://geoservice.dlr.de/web/dataguide/tdm30/
https://geoservice.dlr.de/web/dataguide/tdm90/

The data itself is given in a geographic grid that changes based on latitude. Therefore, to create consistent DEM's
a part of the data has to be resampled. It is left to the user to decide the resolution in arc seconds in the longitu-
donal direction. The resolution in latitude is 1 or 3 arc seconds on a WGS84 ellipsoid.

As a guideline it is best to use the native resolution for the largest part of your area of interest, which is:
1 or 3 arc-seconds for lat < 50
1.5 or 4.5 arc-seconds for lat 50-60
2 or 6 arc-seconds for lat 60-70
3 or 9 arc-seconds for lat 70-80
5 or 15 arc-seconds for lat 80-85
10 or 30 arc-seconds for lat 85-90
These resolutions hold only for the longitude direction. For example, if you want to look at an area around the equator
you should go for a 3 arc-seconds resolution, but at mid-latitudes around for example 55 degrees north, go for the 4.5
arc-seconds resolution. If your region overlaps different regions, it is better to go for the highest resolution in your
region of interest to prevent information loss.
"""

import zipfile
import os
import numpy as np
import requests
import logging
import itertools
import json
from html.parser import HTMLParser
from scipy.interpolate import RectBivariateSpline
from multiprocessing import get_context
from osgeo import gdal
gdal.DontUseExceptions()
import time
from tqdm import tqdm

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.external_dems.geoid import GeoidInterp
from rippl.user_settings import UserSettings

"""
# Test download of TandemX DEM 
from rippl.external_dems.tandem_x.tandem_x_download import TandemXDownload
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

for tandem_x_type, arc_sec in zip(['TDM30', 'TDM90'], [1, 3]):
    download_folder = '/mnt/external/rippl_tutorial_test/DEM_database/tdx' 
    download = TandemXDownload(n_processes=1, tandem_x_folder=download_folder, tandem_x_type=tandem_x_type, lon_resolution=arc_sec)
    
    coordinates = CoordinateSystem()
    coordinates.create_geographic(dlat=arc_sec/3600, dlon=arc_sec/3600, lat0=45, lon0=4, shape=(1000, 1000))
    download(coordinates=coordinates)

"""


class TandemXDownloadTile(object):
    # To enable parallel processing we create another class for the actual processing.

    def __init__(self, tandem_x_folder='', username='', password='', lon_resolution=3, tandem_x_type='TDM90', filelist_folder=''):

        settings = UserSettings()
        settings.load_settings()
        self.filelist_folder = filelist_folder

        # TanDEM-X folder
        if not tandem_x_folder:
            self.tandem_x_folder = os.path.join(settings.settings['paths']['DEM_database'], settings.settings['DEM']['TanDEM-X'])
        else:
            self.tandem_x_folder = tandem_x_folder

        # credentials
        if not username:
            self.username = settings.settings['accounts']['DLR'][tandem_x_type]['username']
        else:
            self.username = username
        if not password:
            self.password = settings.settings['accounts']['DLR'][tandem_x_type]['password']
        else:
            self.password = password

        self.tandem_x_type = tandem_x_type
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

    def download_dem_file(self, url, file, file_unzip):
        # This function downloads data for 1 or 3 arc second dem.

        # Download
        try:
            if os.path.exists(file) and os.stat(file).st_size == 0:
                os.remove(file)

            if not os.path.exists(file):
                # Download file requests
                print('Downloading:', file)
                response = requests.get(url, auth=(self.username, self.password), stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(file, 'wb') as data_file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        data_file.write(data)
                progress_bar.close()

        except Exception as e:
            if os.path.exists(file):
                os.remove(file)
            raise ConnectionError('Failed to download ' + url + '. ' + str(e))

        # Unzip file
        if not os.path.exists(file_unzip):
            zipdata = zipfile.ZipFile(file)

            if self.tandem_x_type == 'TDM30':
                coor = os.path.basename(file).split('_')[-1][:7]
                zip_tiff = 'TDM1_EDEM_10_' + coor + '_V01_C/EDEM/TDM1_EDEM_10_' + coor + '_EDEM_W84.tif'
                zipdata.getinfo(zip_tiff).filename = os.path.basename(file_unzip)
                zipdata.extract(zip_tiff, os.path.dirname(file_unzip))
            elif self.tandem_x_type == 'TDM90':
                coor = os.path.basename(file).split('_')[-1][:7]
                zip_tiff = 'TDM1_DEM__30_' + coor + '_V01_C/DEM/TDM1_DEM__30_' + coor + '_DEM.tif'
                zipdata.getinfo(zip_tiff).filename = os.path.basename(file_unzip)
                zipdata.extract(zip_tiff, os.path.dirname(file_unzip))

    def resize_dem_file(self, file_unzip, lon_resolution=3):
        """
        Here we resize the DEM file to get 3arc seconds grids in both latitude and longitude direction.
        For larger latitudes this will result in very high sampling in the longitudonal direction but this will not
        affect the final product results.
        If we would like to process data over the poles directly, this approach maybe sub-optimal but other implemen-
        tation would be very cumbersome. In those cases we would advise to use a different projection.

        :return:
        """

        if self.tandem_x_type == 'TDM90':
            if not lon_resolution in [3, 6, 9, 15, 30]:
                raise TypeError('Lon size is not one of the default resolutions 3, 6, 9, 15 or 30 arc seconds. '
                                'Aborting...')
        elif self.tandem_x_type == 'TDM30':
            if not lon_resolution in [1, 2, 3, 5, 10]:
                raise TypeError('Lon size is not one of the default resolutions 1, 2, 3, 5 or 10 arc seconds. '
                                'Aborting...')

        dem_size_folder = os.path.join(self.tandem_x_folder, self.tandem_x_type, str(lon_resolution).zfill(2) + '_arc_seconds')

        # Load the data using gdal
        data_file = gdal.Open(file_unzip)
        band = data_file.GetRasterBand(1)
        data = band.ReadAsArray()

        # Check how the data should be split in 1 by 1 degree tiles.
        size = data.shape
        geo_transform = data_file.GetGeoTransform()
        lons = (np.arange(size[1]) + 0.5) * geo_transform[1] + geo_transform[0]
        lats = (np.arange(size[0]) + 0.5) * geo_transform[5] + geo_transform[3]
        del data_file

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
            new_lons = new_int_lon + np.arange(np.int32(np.round(3600.0 / lon_resolution)) + 1) * (lon_resolution / 3600.0)
            logging.info('Saved DEM file ' + new_file_name)
            new_data = np.memmap(new_file_name, np.float32, 'w+', shape=(size[0], len(new_lons)))
            new_data[:, :] = np.flipud(tdx_dem_interp(lats, new_lons))
            logging.info('Average TanDEM-X DEM value is ' + str(np.mean(new_data)) + ' for tile ' + new_file_name)
            new_data.flush()

class TandemXDownload:

    def __init__(self, tandem_x_folder=None, username=None, password=None, lon_resolution=3, n_processes=4, tandem_x_type='TDM90', filelist_folder=''):
        # tandem_x_folder

        settings = UserSettings()
        settings.load_settings()

        # TanDEM-X type
        if tandem_x_type not in ['TDM90', 'TDM30']:
            raise ValueError('TanDEM-X type should either be TDM90 or TDM30!')
        else:
            self.tandem_x_type = tandem_x_type

        # TanDEM-X folder
        if not tandem_x_folder:
            self.tandem_x_folder = os.path.join(settings.settings['paths']['DEM_database'], settings.settings['DEM']['TanDEM-X'])
        else:
            self.tandem_x_folder = tandem_x_folder

        # credentials
        if not username:
            self.username = settings.settings['accounts']['DLR'][tandem_x_type]['username']
        else:
            self.username = username
        if not password:
            self.password = settings.settings['accounts']['DLR'][tandem_x_type]['password']
        else:
            self.password = password

        if not os.path.exists(self.tandem_x_folder):
            raise FileExistsError('Path to tandem-x folder does not exist')
        self.type_folder = os.path.join(self.tandem_x_folder, self.tandem_x_type)
        if not os.path.exists(self.type_folder):
            os.mkdir(self.type_folder)

        zip_data_folder = os.path.join(self.type_folder, 'orig_data')
        tiff_data_folder = os.path.join(self.type_folder, 'geotiff_data')
        if not os.path.exists(zip_data_folder):
            os.mkdir(zip_data_folder)
        if not os.path.exists(tiff_data_folder):
            os.mkdir(tiff_data_folder)

        if tandem_x_type == 'TDM90':
            if not lon_resolution in [3, 4.5, 6, 9, 15, 30]:
                raise TypeError('Lon size is not one of the default resolutions 3, 4.5, 6, 9, 15 or 30 arc seconds. '
                                'Aborting...')
        elif tandem_x_type == 'TDM30':
            if not lon_resolution in [1, 1.5, 2, 3, 5, 10]:
                raise TypeError('Lon size is not one of the default resolutions 1, 1.5, 2, 3, 5 or 10 arc seconds. '
                                'Aborting...')

        resolution_folder = os.path.join(self.type_folder, str(lon_resolution).zfill(2) + '_arc_seconds')
        if not os.path.exists(resolution_folder):
            os.mkdir(resolution_folder)

        # List of files to be downloaded
        self.filelist = self.tandem_x_listing(filelist_folder)

        # shapes and limits of these shapes
        self.shapes = []
        self.latlims = []
        self.lonlims = []

        # meta and polygons
        self.meta = ''
        self.polygon = ''

        # Resolution of files (either TDM30 or TDM90)
        self.lon_resolution = lon_resolution
        if self.tandem_x_type == 'TDM30':
            self.lat_resolution = 1
        elif self.tandem_x_type == 'TDM90':
            self.lat_resolution = 3

        # processes
        self.n_processes = n_processes

    def __call__(self, coordinates):

        # In first instance we assume a step size of 3 seconds in both latitude and longitude direction.
        # The step size in longitude will be updates later on.
        lat_step = 1.0 / 3600 * self.lat_resolution
        lon_step = 1.0 / 3600 * self.lon_resolution

        # Create output coordinates.
        # Create output coordinates.
        self.coordinates = coordinates      # type: CoordinateSystem
        if not np.abs(self.coordinates.dlat - lat_step) < 0.0000000001 or not np.abs(self.coordinates.dlon - lon_step) < 0.0000000001:
            raise ValueError('Value of dlat and dlon of input coordinate system should be ' + str(lat_step) + ' for '
                             'latitude and ' + str(lon_step) + ' for longitude.')
        if self.coordinates.shape == '' or self.coordinates.shape == [0, 0]:
            raise ValueError('Coordinate shape size is missing to determine DEM download')

        resampled_tiles, tiles, download_tiles, [tile_lats, tile_lons], ftp_paths, tiles_zip = \
            self.select_tiles(self.filelist, self.coordinates, self.tandem_x_folder, self.tandem_x_type,
                              lon_resolution=self.lon_resolution)
        if len(ftp_paths) == 0:
            logging.info('All needed TanDEM-X DEM files already downloaded.')
            return

        # Tiles to be downloaded
        logging.info('TanDEM-X DEM tiles to be downloaded to ' + os.path.dirname(download_tiles[0]))
        logging.info('If downloading fails you can try again later or try and download the files yourself over FTP using your '
              'DLR credentials')
        for ftp_path in ftp_paths:
            logging.info('tandemx-90m.dlr.de' + ftp_path)

        # First create a download class.
        tile_download = TandemXDownloadTile(self.tandem_x_folder, self.username, self.password, self.lon_resolution,
                                            tandem_x_type=self.tandem_x_type)

        # Loop over all images
        download_dat = [[ftp_path, file_zip, file_unzip, lat, lon] for
                     ftp_path, file_zip, file_unzip, lat, lon in
                     zip(ftp_paths, tiles_zip, download_tiles, tile_lats, tile_lons)]
        download_dat = np.array(download_dat)
        if self.n_processes > 1:
            with get_context("spawn").Pool(processes=self.n_processes, maxtasksperchild=5) as pool:
                # Process in blocks of 25
                block_size = 25
                for i in range(int(np.ceil(len(download_dat) / block_size))):
                    last_dat = np.minimum((i + 1) * block_size, len(download_dat))
                    pool.map(tile_download, list(download_dat[i*block_size:last_dat]))
        else:
            for download_info in download_dat:
                tile_download(download_info)

    @staticmethod
    def select_tiles(filelist, coordinates, tandem_x_folder, tandem_x_type='TDM90', lon_resolution=3):
        # Adds tandem_x files to the list of files to be downloaded

        settings = UserSettings()
        settings.load_settings()

        # TanDEM-X folder
        if not tandem_x_folder:
            tandem_x_folder = os.path.join(settings.settings['paths']['DEM_database'], settings.settings['DEM']['TanDEM-X'])

        # Check coordinates
        if not isinstance(coordinates, CoordinateSystem):
            logging.info('coordinates should be an CoordinateSystem object')
            return
        elif coordinates.grid_type != 'geographic':
            logging.info('only geographic coordinate systems can be used to download TanDEM-X data')
            return

        tiles_zip = []
        tiles = []
        download_tiles = []
        tile_lats = []
        tile_lons = []
        url_path = []
        resampled_tiles = []

        lat0 = coordinates.lat0 + coordinates.dlat * coordinates.first_line
        lon0 = coordinates.lon0 + coordinates.dlon * coordinates.first_pixel
        lats = np.arange(np.floor(lat0), np.ceil(lat0 + coordinates.shape[0] * coordinates.dlat)).astype(np.int32)
        resolution_folder = os.path.join(tandem_x_folder, tandem_x_type, str(lon_resolution).zfill(2) + '_arc_seconds')

        for lat in lats:
            # Depending on latitude there is a different spacing.
            lon_lim = [int(np.floor(lon0)), int(np.ceil(lon0 + coordinates.shape[1] * coordinates.dlon))]
            grid_size, possible_lons = TandemXDownload.get_lon_spacing(lat, tandem_x_type=tandem_x_type)
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

                # Check if file exists in srtm_tiles_list.pkl
                if str(lat) not in filelist[tandem_x_type].keys():
                    continue
                elif str(lon) not in filelist[tandem_x_type][str(lat)].keys():
                    continue

                for lon_tile_val in np.arange(tile_size) + lon:

                    if lon_tile_val < 0:
                        tile_lonstr = 'W' + str(abs(lon_tile_val)).zfill(3)
                    else:
                        tile_lonstr = 'E' + str(lon_tile_val).zfill(3)

                    if lon_lim[0] < (lon_tile_val + 1) and lon_lim[1] > lon_tile_val:
                        resampled_tile = os.path.join(resolution_folder, 'TDM_DEM_' + latstr + tile_lonstr + '.raw')
                        resampled_tiles.append(resampled_tile)

                unzip = os.path.join(tandem_x_folder, tandem_x_type, 'geotiff_data', 'TDM_DEM_' + latstr + lonstr + '.tiff')
                tiles.append(unzip)
                tiles_zip.append(os.path.join(tandem_x_folder, tandem_x_type, 'orig_data', 'TDM_DEM_' + latstr + lonstr + '.zip'))

                download_tiles.append(unzip)
                tile_lats.append(lat)
                tile_lons.append(lon)
                url_path.append(filelist[tandem_x_type][str(lat)][str(lon)])

        return resampled_tiles, tiles, download_tiles, [tile_lats, tile_lons], url_path, tiles_zip

    @staticmethod
    def tandem_x_listing(filelist_folder=''):
        # This script makes a list of all the available 1,3 and 30 arc second datafiles.
        # This makes it easier to detect whether files do or don't exist.

        settings = UserSettings()
        settings.load_settings()

        # TanDEM-X folder
        if not filelist_folder:
            filelist_folder = os.path.join(settings.settings['paths']['rippl'], 'rippl', 'external_dems', 'tandem_x')

        data_file = os.path.join(filelist_folder, 'tandem_x_tiles_list.json')
        if os.path.exists(data_file):
            with open(data_file, 'rb') as dat:
                filelist = json.load(dat)
            return filelist

        filelist = {'TDM90': {}, 'TDM30': {}}
        urls = ['https://download.geoservice.dlr.de/TDM90/files/', 'https://download.geoservice.dlr.de/TDM30_EDEM/files/']
        tandem_x_types = ['TDM90', 'TDM30']

        for url, tandem_x_type in zip(urls, tandem_x_types):
            # credentials
            username = settings.settings['accounts']['DLR'][tandem_x_type]['username']
            password = settings.settings['accounts']['DLR'][tandem_x_type]['password']

            html_data = requests.get(url, auth=(username, password)).text
            parser = LinkParser()
            links = parser.feed(html_data)
            dem_folders = [file for file in parser.links if (file.startswith('TDM1') or file.startswith('N') or file.startswith('S'))]

            dem_files = []
            for dem_folder in dem_folders:
                print('Indexing ' + dem_folder)
                succeeded = False
                while not succeeded:
                    try:
                        html_data = requests.get(url + dem_folder, auth=(username, password)).text
                        parser = LinkParser()
                        links = parser.feed(html_data)
                        sub_dem_folders = [dem_folder + file for file in parser.links if
                                           (file.startswith('TDM1') or file.startswith('W') or file.startswith('E'))]
                        for sub_dem_folder in sub_dem_folders:
                            html_data = requests.get(url + sub_dem_folder, auth=(username, password)).text
                            parser = LinkParser()
                            links = parser.feed(html_data)
                            if tandem_x_type == 'TDM30':
                                dem_files.extend([url + sub_dem_folder + file + file[:-1] + '.zip' for file in parser.links
                                                  if file.startswith('TDM1')])
                            elif tandem_x_type == 'TDM90':
                                dem_files.extend([url + sub_dem_folder + file + file[:-7] + '.zip' for file in parser.links
                                                  if file.startswith('TDM1')])
                        succeeded = True
                    except:
                        # If it fails try to log in again after 10 seconds
                        logging.info('Lost connection. Reconnecting...')
                        time.sleep(10)

            # Logout
            logout = requests.get('https://sso.eoc.dlr.de/eoc/auth/logout')
            time.sleep(10)

            # Index based on latitude and longitude values
            for dem_file in dem_files:
                coors = os.path.basename(dem_file).replace('__', '_').split('_')[3]
                lon_str = coors[4:7]
                lat_str = coors[1:3]

                if coors[0] == 'S':
                    lat_str = str(int(lat_str) * -1)
                else:
                    lat_str = str(int(lat_str) * 1)
                if coors[3] == 'W':
                    lon_str = str(int(lon_str) * -1)
                else:
                    lon_str = str(int(lon_str) * 1)

                if lat_str not in filelist[tandem_x_type].keys():
                    filelist[tandem_x_type][lat_str] = dict()
                filelist[tandem_x_type][lat_str][lon_str] = dem_file

        # Save list of files
        logging.info('Saving list of TanDEM-X files at ' + data_file)
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(filelist, f, ensure_ascii=False, indent=4)

        return filelist

    @staticmethod
    def get_lon_spacing(lat, tandem_x_type):
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
            grid_size = 1
        elif abs_lat < 60:
            lons = np.arange(-180, 180, 1)
            grid_size = 1.5
        elif abs_lat < 70:
            lons = np.arange(-180, 180, 2)
            grid_size = 2
        elif abs_lat < 80:
            lons = np.arange(-180, 180, 2)
            grid_size = 3
        elif abs_lat < 85:
            lons = np.arange(-180, 180, 4)
            grid_size = 5
        elif abs_lat < 90:
            lons = np.arange(-180, 180, 4)
            grid_size = 10

        if tandem_x_type == 'TDM90':
            grid_size *= 3

        return grid_size, lons

class LinkParser(HTMLParser):
    def reset(self):
        super().reset()
        self.links = iter([])

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for (name, value) in attrs:
                if name == 'href':
                    self.links = itertools.chain(self.links, [value])