# Function created by Gert Mulder
# Institute TU Delft
# Date 9-11-2016
# Part of Doris 5.0

# This function creates a dem based on either a shape/kml file or a given bounding box. If a shape/kml file is given a
# minimum offset of about 0.1 degrees is used.
# All grids are based on the WGS84 projection.
# Downloaded data is based on SRTM void filled data:
# Documentation: https://lpdaac.usgs.gov/sites/default/files/public/measures/docs/NASA_SRTM_V3.pdf

# Description srtm data: https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/SRTMGL1_v003
# Description srtm q data: https://lpdaac.usgs.gov/node/505

import os
import pickle
import shutil
import zipfile
import numpy as np
import requests
from multiprocessing import Pool

from rippl.external_dems.srtm.srtm_dir_listing import ParseHTMLDirectoryListing
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.image_processing_meta import ImageProcessingMeta
from rippl.resampling.coor_new_extend import CoorNewExtend


class SrtmDownloadTile(object):
    # To enable parallel processing we create another class for the actual processing.

    def __init__(self, srtm_folder, username, password, srtm_type):

        self.srtm_folder = srtm_folder
        self.srtm_type = srtm_type
        self.username = username
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
            if not os.path.exists(file_unzip):
                command = 'wget ' + url + ' --user ' + self.username + ' --password ' \
                          + self.password + ' -O ' + file_zip
                os.system(command)
                zip_data = zipfile.ZipFile(file_zip)
                source = zip_data.open(zip_data.namelist()[0])
                target = open(file_unzip, 'wb')
                shutil.copyfileobj(source, target, length=-1)
                target.close()
                os.remove(file_zip)
        except:
            print('Failed to download ' + url)
            return False

        return True


class SrtmDownload(object):

    def __init__(self, srtm_folder, username, password, srtm_type='SRTM3', quality=False, n_processes=4):
        # srtm_folder
        self.srtm_folder = srtm_folder
        self.quality = quality

        # credentials
        self.username = username
        self.password = password

        # List of files to be downloaded
        self.filelist = self.srtm_listing(srtm_folder, username, password)

        # shapes and limits of these shapes
        self.shapes = []
        self.latlims = []
        self.lonlims = []

        # meta and polygons
        self.meta = ''
        self.polygon = ''
        self.shapefile = ''

        # Resolution of files (either SRTM1, SRTM3 or STRM30)
        self.srtm_type = srtm_type

        # EGM96
        self.egm96_interp = []

        # processes
        self.n_processes = n_processes

    def __call__(self, meta, buffer=1.0, rounding=1.0, parallel=True):

        if isinstance(meta, ImageProcessingData):
            self.meta = meta.meta
        elif isinstance(meta, ImageProcessingMeta):
            self.meta = meta
        else:
            raise TypeError('Input meta data should be an ImageProcessingData or ImageProcessingMeta object.')

        if self.srtm_type == 'SRTM3':
            step = 1.0 / 3600 * 3
        elif self.srtm_type == 'SRTM1':
            step = 1.0 / 3600
        else:
            print('Unkown SRTM type' + self.srtm_type)
            return

        # Create output coordinates.
        radar_coor = CoordinateSystem()
        radar_coor.create_radar_coordinates()
        radar_coor.load_readfile(self.meta.readfiles['original'])
        radar_coor.orbit = self.meta.find_best_orbit('original')
        self.coordinates = CoordinateSystem()
        self.coordinates.create_geographic(step, step)
        new_coor = CoorNewExtend(radar_coor, self.coordinates, buffer=buffer, rounding=rounding)
        self.coordinates = new_coor.out_coor

        tiles, download_tiles, [tile_lats, tile_lons], urls, tiles_zip = \
            self.select_tiles(self.filelist, self.coordinates, self.srtm_folder, self.srtm_type, self.quality)

        # First create a download class.
        tile_download = SrtmDownloadTile(self.srtm_folder, self.username, self.password, self.srtm_type)

        # Loop over all images
        download_dat = [[url, file_zip, file_unzip, lat, lon] for
                     url, file_zip, file_unzip, lat, lon in
                     zip(urls, tiles_zip, download_tiles, tile_lats, tile_lons)]
        if parallel:
            pool = Pool(self.n_processes)
            pool.map(tile_download, download_dat)
        else:
            for download_info in download_dat:
                tile_download(download_info)

    @staticmethod
    def select_tiles(filelist, coordinates, srtm_folder, srtm_type='SRTM3', quality=True, download=True):
        # Adds SRTM files to the list of files to be downloaded

        # Check coordinates
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')
            return
        elif coordinates.grid_type != 'geographic':
            print('only geographic coordinate systems can be used to download SRTM data')
            return

        tiles_zip = []
        tiles = []
        download_tiles = []
        tile_lats = []
        tile_lons = []
        url = []

        lat0 = coordinates.lat0 + coordinates.dlat * coordinates.first_line
        lon0 = coordinates.lon0 + coordinates.dlon * coordinates.first_pixel
        lats = np.arange(np.floor(lat0), np.ceil(lat0 + coordinates.shape[0] * coordinates.dlat)).astype(np.int32)
        lons = np.arange(np.floor(lon0), np.ceil(lon0 + coordinates.shape[1] * coordinates.dlon)).astype(np.int32)

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

                # Check if file exists in filelist
                if str(lat) not in filelist[srtm_type]:
                    continue
                elif str(lon) not in filelist[srtm_type][str(lat)]:
                    continue
                
                if os.path.join(srtm_folder, latstr + lonstr + 'SRTMGL3.hgt.zip') not in tiles_zip or not download:
                    unzip = os.path.join(srtm_folder, srtm_type + '__' + latstr + lonstr + '.hgt')
                    tiles.append(unzip)

                    if not os.path.exists(unzip) or not download:
                        tiles_zip.append(os.path.join(srtm_folder, latstr + lonstr + 'SRTMGL3.hgt.zip'))

                        download_tiles.append(unzip)
                        tile_lats.append(lat)
                        tile_lons.append(lon)
                        url.append(filelist[srtm_type][str(lat)][str(lon)])

                    if quality:
                        unzip = os.path.join(srtm_folder, srtm_type + '__' + latstr + lonstr + '.q')
                        tiles.append(unzip)

                        if not os.path.exists(unzip) or not download:
                            tiles_zip.append(os.path.join(srtm_folder, latstr + lonstr + 'SRTMGL3.q.zip'))

                            download_tiles.append(unzip)
                            tile_lats.append(lat)
                            tile_lons.append(lon)
                            url.append(filelist[srtm_type][str(lat)][str(lon)][:-7] + 'num.zip')

        return tiles, download_tiles, [tile_lats, tile_lons], url, tiles_zip

    @staticmethod
    def srtm_listing(srtm_folder, username='', password=''):
        # This script makes a list of all the available 1,3 and 30 arc second datafiles.
        # This makes it easier to detect whether files do or don't exist.

        data_file = os.path.join(srtm_folder, 'filelist')
        if os.path.exists(data_file):
            dat = open(data_file, 'rb')
            filelist = pickle.load(dat)
            dat.close()
            return filelist

        server = "http://e4ftl01.cr.usgs.gov"

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

        for folder, key_value in zip(folders, keys):

            if len(username) == 0 or len(password) == 0:
                print('username and or password is missing for downloading. If you get this message when processing an '
                      'InSAR stack, be sure you download the SRTM files first to solve this. (download_srtm function in '
                      'stack class)')

            conn = requests.get(server + '/' + folder, auth=(username, password))
            if conn.status_code == 200:
                print("status200 received ok")
            else:
                print("an error occurred during connection")

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

        file_list = open(os.path.join(srtm_folder, 'filelist'), 'wb')
        pickle.dump(filelist, file_list)
        file_list.close()

        return filelist
