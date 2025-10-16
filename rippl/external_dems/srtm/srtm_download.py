# Function created by Gert Mulder
# Institute TU Delft
# Date 9-11-2016
# Part of RIPPL

# This function creates a dem based on either a shape/kml file or a given bounding box. If a shape/kml file is given a
# minimum offset of about 0.1 degrees is used.
# All grids are based on the WGS84 projection.
# Downloaded data is based on SRTM void filled data:
# Documentation: https://lpdaac.usgs.gov/sites/default/files/public/measures/docs/NASA_SRTM_V3.pdf

# Description srtm data: https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/SRTMGL1_v003
# Description srtm q data: https://lpdaac.usgs.gov/node/505

import os
import json
import shutil
import zipfile
from typing import Optional

import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from html.parser import HTMLParser
from multiprocessing import get_context
import logging
from tqdm import tqdm

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.user_settings import UserSettings

"""
# Test download of SRTM download
from rippl.external_dems.srtm.srtm_download import SrtmDownload
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

for srtm_type, arc_sec in zip(['SRTM1', 'SRTM3'], [1, 3]):
    download_folder = '/mnt/external/rippl_tutorial_test/DEM_database/srtm'
    download = SrtmDownload(n_processes=1, srtm_folder=download_folder, srtm_type=srtm_type)
    
    coordinates = CoordinateSystem()
    coordinates.create_geographic(dlat=arc_sec/3600, dlon=arc_sec/3600, lat0=45, lon0=4, shape=(1000, 1000))
    download(coordinates=coordinates)

"""


class SrtmDownloadTile:
    # To enable parallel processing we create another class for the actual processing.

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):

        settings = UserSettings()
        settings.load_settings()

        if not username:
            self.username = settings.settings['accounts']['EarthData']['username']
        else:
            self.username = username
        if not password:
            self.password = settings.settings['accounts']['EarthData']['password']
        else:
            self.password = password

    def __call__(self, input):

        url = input[0]
        file_zip = input[1]
        file_unzip = input[2]
        lat = input[3]
        lon = input[4]

        if not os.path.exists(file_unzip):
            success = self.download_dem_file(url, file_zip, file_unzip)
        else:
            return

    def download_dem_file(self, url, file_zip, file_unzip):
        # This function downloads data for 1 or 3 arc second dem.

        # Download and unzip
        try:
            if os.path.exists(file_zip) and os.stat(file_zip).st_size == 0:
                os.remove(file_zip)

            if not os.path.exists(file_zip):
                print('Downloading:', file_zip)

                with requests.Session() as session:
                    session.auth = (self.username, self.password)
                    r1 = session.request('get', url)
                    response = session.get(r1.url, auth=(self.username, self.password))
                    total_size_in_bytes = int(response.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kibibyte

                    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                    with open(file_zip, 'wb') as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)
                    progress_bar.close()

            if not os.path.exists(file_unzip):
                zip_data = zipfile.ZipFile(file_zip)
                source = zip_data.open(zip_data.namelist()[0])
                with open(file_unzip, 'wb') as target:
                    shutil.copyfileobj(source, target, length=-1)
                zip_data.close()
                source.close()
        except Exception as e:
            # Remove erroneous files
            if os.path.exists(file_unzip):
                os.remove(file_unzip)
            if os.path.exists(file_zip):
                os.remove(file_zip)
            raise ConnectionError('Failed to download ' + url + '. ' + str(e))

        return True


class SrtmDownload:

    def __init__(self, srtm_folder='', username='', password='', srtm_type='SRTM3', quality=False, n_processes=4, filelist_folder=''):
        # srtm_folder

        settings = UserSettings()
        settings.load_settings()

        # SRTM folder
        if not srtm_folder:
            self.srtm_folder = os.path.join(settings.settings['paths']['DEM_database'], settings.settings['path_names']['DEM']['SRTM'])
        else:
            self.srtm_folder = srtm_folder

        if not os.path.exists(self.srtm_folder):
            os.mkdir(self.srtm_folder)

        for srtm_type_folder in ['srtm1', 'srtm3']:
            path = os.path.join(self.srtm_folder, srtm_type_folder)
            if not os.path.isdir(path):
                os.mkdir(path)

        # credentials
        if not username:
            self.username = settings.settings['accounts']['EarthData']['username']
        else:
            self.username = username
        if not password:
            self.password = settings.settings['accounts']['EarthData']['password']
        else:
            self.password = password

        self.srtm_folder = srtm_folder
        self.quality = quality

        # List of files to be downloaded
        if not filelist_folder:
            filelist_folder = os.path.join(settings.settings['paths']['rippl'], 'rippl', 'external_dems', 'srtm')
        self.filelist = self.srtm_listing(filelist_folder, self.username, self.password)

        # shapes and limits of these shapes
        self.shapes = []
        self.latlims = []
        self.lonlims = []

        # meta and polygons
        self.meta = ''
        self.polygon = ''

        # Resolution of files (either SRTM1, SRTM3 or STRM30)
        self.srtm_type = srtm_type

        # EGM96
        self.egm96_interp = []

        # processes
        self.n_processes = n_processes

    def __call__(self, coordinates, parallel=True):

        if self.srtm_type == 'SRTM3':
            step = 1.0 / 3600 * 3
        elif self.srtm_type == 'SRTM1':
            step = 1.0 / 3600
        else:
            logging.info('Unkown SRTM type' + self.srtm_type)
            return

        # Create output coordinates.
        self.coordinates = coordinates      # type: CoordinateSystem
        if not np.abs(self.coordinates.dlat - step) < 0.0000000001 or not np.abs(self.coordinates.dlon - step) < 0.0000000001:
            raise ValueError('Value of dlat and dlon of input coordinate system should be ' + str(step))
        if self.coordinates.shape == '' or self.coordinates.shape == [0, 0]:
            raise ValueError('Coordinate shape size is missing to determine DEM download')

        tiles, download_tiles, [tile_lats, tile_lons], urls, tiles_zip = \
            self.select_tiles(self.filelist, self.coordinates, self.srtm_folder, self.srtm_type, self.quality)
        if len(urls) == 0:
            logging.info('All needed SRTM DEM files already downloaded.')
            return

        # First create a download class.
        tile_download = SrtmDownloadTile(self.username, self.password)

        # Tiles to be downloaded
        logging.info('SRTM tiles to be downloaded to ' + os.path.dirname(download_tiles[0]))
        logging.info('If downloading fails you can try again later or try and download the files yourself using your EarthData '
              'credentials')
        for url in urls:
            logging.info(url)

        # Loop over all images
        download_dat = [[url, file_zip, file_unzip, lat, lon] for
                     url, file_zip, file_unzip, lat, lon in
                     zip(urls, tiles_zip, download_tiles, tile_lats, tile_lons)]
        np.array(download_dat)
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
    def select_tiles(filelist, coordinates, srtm_folder, srtm_type='SRTM3', quality=True, download=True):
        # Adds SRTM files to the list of files to be downloaded

        # Check coordinates
        if not isinstance(coordinates, CoordinateSystem):
            logging.info('coordinates should be an CoordinateSystem object')
            return
        elif coordinates.grid_type != 'geographic':
            logging.info('only geographic coordinate systems can be used to download SRTM data')
            return

        tiles_zip = []
        tiles = []
        download_tiles = []
        tile_lats = []
        tile_lons = []
        url = []

        lat0 = coordinates.lat0 + coordinates.dlat * coordinates.first_line
        lon0 = coordinates.lon0 + coordinates.dlon * coordinates.first_pixel
        # We add and subtract 0.0001 to the total value to prevent that rounding errors add a full degree to the total value.
        d_small_lat = 0.0001 * np.sign(coordinates.dlat)
        lats = np.arange(np.floor(lat0 + d_small_lat), np.ceil(lat0 + (coordinates.shape[0] - 1) * coordinates.dlat - d_small_lat)).astype(np.int32)
        d_small_lon = 0.0001 * np.sign(coordinates.dlon)
        lons = np.arange(np.floor(lon0 + d_small_lon), np.ceil(lon0 + (coordinates.shape[1] - 1) * coordinates.dlon - d_small_lon)).astype(np.int32)

        for lat in lats:
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

                # Check if file exists in srtm_tiles_list.json
                if str(lat) not in filelist[srtm_type]:
                    continue
                elif str(lon) not in filelist[srtm_type][str(lat)]:
                    continue
                
                if os.path.join(srtm_folder, latstr + lonstr + 'SRTMGL3.hgt.zip') not in tiles_zip or not download:
                    unzip = os.path.join(srtm_folder, srtm_type.lower(), srtm_type + '__' + latstr + lonstr + '.hgt')
                    tiles.append(unzip)

                    if not os.path.exists(unzip) or not download:
                        tiles_zip.append(os.path.join(srtm_folder, srtm_type.lower(), latstr + lonstr + 'SRTMGL3.hgt.zip'))

                        download_tiles.append(unzip)
                        tile_lats.append(lat)
                        tile_lons.append(lon)
                        url.append(filelist[srtm_type][str(lat)][str(lon)])

                    if quality:
                        unzip = os.path.join(srtm_folder, srtm_type.lower(), srtm_type + '__' + latstr + lonstr + '.q')
                        tiles.append(unzip)

                        if not os.path.exists(unzip) or not download:
                            tiles_zip.append(os.path.join(srtm_folder, srtm_type.lower(), latstr + lonstr + 'SRTMGL3.q.zip'))

                            download_tiles.append(unzip)
                            tile_lats.append(lat)
                            tile_lons.append(lon)
                            url.append(filelist[srtm_type][str(lat)][str(lon)][:-7] + 'num.zip')

        return tiles, download_tiles, [tile_lats, tile_lons], url, tiles_zip

    @staticmethod
    def srtm_listing(filelist_folder='', username='', password=''):
        # This script makes a list of all the available 1,3 and 30 arc second datafiles.
        # This makes it easier to detect whether files do or don't exist.

        settings = UserSettings()
        settings.load_settings()

        # File list folder
        if not filelist_folder:
            filelist_folder = os.path.join(settings.settings['paths']['rippl'], 'rippl', 'external_dems', 'srtm')

        # credentials
        if not username:
            username = settings.settings['accounts']['EarthData']['username']
        if not password:
            password = settings.settings['accounts']['EarthData']['password']

        data_file = os.path.join(filelist_folder, 'srtm_tiles_list.json')
        if os.path.exists(data_file):
            with open(data_file, 'rb') as dat:
                filelist = json.load(dat)

            return filelist

        server = "https://e4ftl01.cr.usgs.gov"

        folder = ['MEASURES/SRTMGL1.003/2000.02.11/', 'MEASURES/SRTMGL3.003/2000.02.11/', 'MEASURES/SRTMGL30.002/2000.02.11/']
        sub_folder = ['SRTMGL1_page_', 'SRTMGL3_page_', 'SRTMGL30_page_']
        key = ['SRTM1', 'SRTM3', 'SRTM30']
        folders = []
        keys = []

        for f, s_f, k in zip(folder[:2], sub_folder[:2], key[:2]):
            for i in range(1, 7):
                folders.append(f + s_f + str(i) + '.html')
                keys.append(k)

        folders.append(folder[2] + sub_folder[2] + str(1) + '.html')
        keys.append(key[2])

        filelist = dict()
        filelist['SRTM1'] = dict()
        filelist['SRTM3'] = dict()
        filelist['SRTM30'] = dict()

        logging.info('Indexing SRTM tiles...')
        total_no = 12
        num = 0

        for folder, key_value in zip(folders, keys):

            if len(username) == 0 or len(password) == 0:
                logging.info('username and or password is missing for downloading. If you get this message when processing an '
                      'InSAR stack, be sure you download the SRTM files first to solve this. (download_srtm function in '
                      'stack class)')

            conn = requests.get(server + '/' + folder)
            if conn.status_code == 200:
                logging.info(str(int(num / total_no * 100)) + '% finished')
                logging.info("Indexing tiles " + str(num * 10000) + ' to ' + str((num + 1) * 10000) + ' out of ' + str(total_no * 10000) + ' tiles.')
                num += 1
            else:
                logging.info("an error occurred during connection")

            data = conn.text
            parser = ParseHTMLDirectoryListing()
            parser.feed(data)
            files = parser.getDirListing()

            if key_value == 'SRTM1' or key_value == 'SRTM3':
                files = [str(os.path.basename(f)) for f in files if f.endswith('hgt.zip')]
                north = [int(filename[1:3]) for filename in files]
                east = [int(filename[4:7]) for filename in files]
                for i in [i for i, filename in enumerate(files) if filename[0] == 'S']:
                    north[i] *= -1
                for i in [i for i, filename in enumerate(files) if filename[3] == 'W']:
                    east[i] *= -1
            else:
                files = [str(os.path.basename(f)) for f in files if f.endswith('dem.zip')]
                north = [int(filename[5:7]) for filename in files]
                east = [int(filename[1:4]) for filename in files]
                for i in [i for i, filename in enumerate(files) if filename[4] == 's']:
                    north[i] *= -1
                for i in [i for i, filename in enumerate(files) if filename[0] == 'w']:
                    east[i] *= -1

            for filename, n, e in zip(files, north, east):
                if not str(n) in filelist[key_value]:
                    filelist[key_value][str(n)] = dict()
                filelist[key_value][str(n)][str(e)] = server + '/' + os.path.dirname(folder) + '/' + filename

        logging.info('Saving list of SRTM files at ' + data_file)
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(filelist, f, ensure_ascii=False, indent=4)

        return filelist


# Following code is adapted from srtm-1.py > downloaded from
# https://svn.openstreetmap.org/applications/utils/import/srtm2wayinfo/python/srtm.py
class ParseHTMLDirectoryListing(HTMLParser):
    def __init__(self):
        # print "parseHTMLDirectoryListing.__init__"
        HTMLParser.__init__(self)
        self.title = "Undefined"
        self.isDirListing = False
        self.dirList = []
        self.inTitle = False
        self.inHyperLink = False
        self.currAttrs = ""
        self.currHref = ""

    def handle_starttag(self, tag, attrs):
        # print "Encountered the beginning of a %s tag" % tag
        if tag == "title":
            self.inTitle = True
        if tag == "a":
            self.inHyperLink = True
            self.currAttrs = attrs
            for attr in attrs:
                if attr[0] == 'href':
                    self.currHref = attr[1]

    def handle_endtag(self, tag):
        # print "Encountered the end of a %s tag" % tag
        if tag == "title":
            self.inTitle = False
        if tag == "a":
            # This is to avoid us adding the parent directory to the list.
            if self.currHref != "":
                self.dirList.append(self.currHref)
            self.currAttrs = ""
            self.currHref = ""
            self.inHyperLink = False

    def handle_data(self, data):
        if self.inTitle:
            self.title = data
            logging.info("title=%s" % data)
            if "Index of" in self.title:
                # print "it is an index!!!!"
                self.isDirListing = True
        if self.inHyperLink:
            # We do not include parent directory in listing.
            if "Parent Directory" in data:
                self.currHref = ""

    def getDirListing(self):
        return self.dirList
