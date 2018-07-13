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
from fiona import collection
from shapely.geometry import Polygon
from scipy.interpolate import RectBivariateSpline
from orbit_dem_functions.srtm_dir_listing import ParseHTMLDirectoryListing
from joblib import Parallel, delayed
from image_data import ImageData


class SrtmDownloadTile(object):
    # To enable parallel processing we create another class for the actual processing.

    def __init__(self, srtm_folder, username, password, resolution):

        self.srtm_folder = srtm_folder
        self.resolution = resolution
        self.username = username
        self.password = password

        filename = os.path.join(self.srtm_folder, 'EGM96_15min.dat')
        self.egm96_interp = self.load_egm96(filename)

    def __call__(self, url, file_zip, file_unzip, lat, lon):

        if not os.path.exists(file_unzip):
            success = self.download_dem_file(url, file_zip, file_unzip)
        else:
            return

        if success or os.path.exists(file_unzip):
            if file_unzip.endswith('.hgt') and (not os.path.exists(file_unzip[:-3] + 'x') or
                                                not os.path.exists(file_unzip[:-3] + 'y') or
                                                not os.path.exists(file_unzip[:-3] + 'z')):
                self.correct_egm96(file_unzip, lat, lon)

    @staticmethod
    def load_egm96(filename):
        # Download the EGM96 data

        # Load egm96 grid and resample to input grid using gdal.
        # (For this purpose the grid is downloaded from:
        # http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/binarygeoid.html

        if not os.path.exists(filename):
            # Download egm96 file
            command = 'wget http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/WW15MGH.DAC -O ' + filename
            os.system(command)

        # Load data
        egm96 = np.fromfile(filename, dtype='>i2').reshape((721, 1440)).astype('float32')
        egm96 = np.concatenate((egm96[:, 721:], egm96[:, :721]), axis=1)
        lats = np.linspace(-90, 90, 721)
        lons = np.linspace(-180, 179.75, 1440)
        egm96_interp = RectBivariateSpline(lats, lons, egm96)

        return egm96_interp

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

    def correct_egm96(self, tile, lat, lon):
        # This function converts srtm data to cartesian coordinates

        # Load data
        if self.resolution == 'SRTM1':
            shape = (3601, 3601)
        elif self.resolution == 'SRTM3':
            shape = (1201, 1201)
        else:
            print('quality should be either SRTM1 or SRTM3!')
            return

        image = np.fromfile(tile, dtype='>i2').astype('float32').reshape(shape)

        # Create lat/lon list
        lons = np.linspace(lon, lon+1, shape[0])
        lats = - np.linspace(lat+1, lat, shape[1])
        egm96 = self.egm96_interp(lats, lons) / 100.0

        # Correct for egm96
        im = np.memmap(tile, mode='w+', dtype='float32', shape=image.shape)
        im[:, :] = image + egm96
        im.flush()

        del egm96


class SrtmDownload(object):

    def __init__(self, srtm_folder, username, password, resolution='SRTM3', n_processes=4):
        # srtm_folder
        self.srtm_folder = srtm_folder

        # credentials
        self.username = username
        self.password = password

        # List of files to be downloaded
        self.srtm_listing()
        self.tiles_zip = []
        self.tiles = []
        self.download_tiles = []
        self.tile_lats = []
        self.tile_lons = []
        self.url = []

        # shapes and limits of these shapes
        self.shapes = []
        self.latlims = []
        self.lonlims = []

        # meta and polygons
        self.meta = ''
        self.polygon = ''
        self.shapefile = ''

        # Resolution of files (either SRTM1, SRTM3 or STRM30)
        self.resolution = resolution

        # EGM96
        self.egm96_interp = []

        # processes
        self.n_processes = n_processes

    def __call__(self, meta='', shapefile='', polygon='', border=1, rounding=1):

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.meta = meta

        if self.meta:
            self.polygon = self.meta.polygon
        elif shapefile:
            self.polygon = SrtmDownload.load_shp(shapefile)
        elif polygon:
            self.polygon = polygon
        else:
            print('No shape selected')
            return

        self.lonlim, self.latlim = SrtmDownload.load_limits(self.polygon, border, rounding)
        self.download_list(self.latlim, self.lonlim)

        # First create a download class.
        tile_download = SrtmDownloadTile(self.srtm_folder, self.username, self.password, self.resolution)

        # Loop over all images
        if len(self.tiles) > 0:
            Parallel(n_jobs=self.n_processes)(delayed(tile_download)(url, file_zip, file_unzip, lat, lon) for
                     url, file_zip, file_unzip, lat, lon in
                     zip(self.url, self.tiles_zip, self.download_tiles, self.tile_lats, self.tile_lons))

    def download_list(self, latlim, lonlim, quality=True):
        # Adds SRTM files to the list of files to be downloaded

        lats = np.arange(np.floor(latlim[0]), np.ceil(latlim[1]))
        lons = np.arange(np.floor(lonlim[0]), np.ceil(lonlim[1]))

        if self.resolution == 'SRTM1' or self.resolution == 'SRTM3':
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
                    if str(lat) not in self.filelist[self.resolution]:
                        continue
                    elif str(lon) not in self.filelist[self.resolution][str(lat)]:
                        continue
                    
                    if os.path.join(self.srtm_folder, latstr + lonstr + 'SRTMGL3.hgt.zip') not in self.tiles_zip:
                        unzip = os.path.join(self.srtm_folder, self.resolution + '__' + latstr + lonstr + '.hgt')
                        self.tiles.append(unzip)

                        if not os.path.exists(unzip):
                            self.tiles_zip.append(os.path.join(self.srtm_folder, latstr + lonstr + 'SRTMGL3.hgt.zip'))

                            self.download_tiles.append(unzip)
                            self.tile_lats.append(lat)
                            self.tile_lons.append(lon)
                            self.url.append(self.filelist[self.resolution][str(lat)][str(lon)])

                        if quality:
                            unzip = os.path.join(self.srtm_folder, self.resolution + '__' + latstr + lonstr + '.q')
                            self.tiles.append(unzip)

                            if not os.path.exists(unzip):
                                self.tiles_zip.append(os.path.join(self.srtm_folder, latstr + lonstr + 'SRTMGL3.q.zip'))

                                self.download_tiles.append(unzip)
                                self.tile_lats.append(lat)
                                self.tile_lons.append(lon)
                                self.url.append(self.filelist[self.resolution][str(lat)][str(lon)][:-7] + 'num.zip')

    def srtm_listing(self):
        # This script makes a list of all the available 1,3 and 30 arc second datafiles.
        # This makes it easier to detect whether files do or don't exist.

        data_file = os.path.join(self.srtm_folder, 'filelist')
        if os.path.exists(data_file):
            dat = open(data_file, 'r')
            self.filelist = pickle.load(dat)
            dat.close()
            return

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

        self.filelist = dict()
        self.filelist['SRTM1'] = dict()
        self.filelist['SRTM3'] = dict()
        self.filelist['SRTM30'] = dict()

        for folder, key_value in zip(folders, keys):

            conn = requests.get(server + '/' + folder, auth=(self.username, self.password))
            if conn.status_code == 200:
                print "status200 received ok"
            else:
                print "an error occurred during connection"

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
                if not str(n) in self.filelist[key_value]:
                    self.filelist[key_value][str(n)] = dict()
                self.filelist[key_value][str(n)][str(e)] = server + '/' + os.path.dirname(folder) + '/' + filename

        file_list = open(os.path.join(self.srtm_folder, 'filelist'), 'w')
        pickle.dump(self.filelist, file_list)
        file_list.close()

    @staticmethod
    def load_shp(filename):
        # from kml and shape file to a bounding box. We will always use a bounding box to create the final product.

        if filename.endswith('.shp'):
            with collection(filename, "r") as inputshape:

                shapes = [shape for shape in inputshape]
                # only first shape
                polygon = Polygon(shapes[0]['geometry']['coordinates'][0])
        else:
            print('Shape not recognized')
            return []

        return polygon

    @staticmethod
    def load_limits(polygon, buffer=1, rounding=1):

        polygon = polygon.buffer(buffer)
        coor = polygon.exterior.coords

        lon = [l[0] for l in coor]
        lat = [l[1] for l in coor]

        latlim = [min(lat), max(lat)]
        lonlim = [min(lon), max(lon)]

        # Add the rounding and borders to add the sides of our image
        # Please use rounding as a n/60 part of a degree (so 1/15 , 1/10 or 1/20 of a degree for example..)
        latlim = [np.floor((latlim[0]) / rounding) * rounding,
                       np.ceil((latlim[1]) / rounding) * rounding]
        lonlim = [np.floor((lonlim[0]) / rounding) * rounding,
                       np.ceil((lonlim[1]) / rounding) * rounding]

        return latlim, lonlim
