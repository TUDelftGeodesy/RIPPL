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
import numpy as np
import gdal

from collections import OrderedDict, defaultdict
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from orbit_dem_functions.srtm_download import SrtmDownload
from image_data import ImageData
from coordinate_system import CoordinateSystem

class CreateSrtmDem(OrbitCoordinates):
    # This class stitches the different files together. If no data is available values will be zero. Which is
    # generally true because it is above sealevel.
    # The resampling is either
    # - none
    # - regular_grid (based on vectors of lats/lons)
    # - irregular_grid (based on lats/lons)

    """
    :type meta = ImageData
    """

    def __init__(self, meta, srtm_folder='', quality=False, buf=0.2, rounding=0.2, srtm_type='SRTM3', s_lin=0, s_pix=0, lines=0):

        # Define the limits of our image
        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # Quality file also created?
        self.quality = quality
        self.srtm_type = srtm_type
        self.srtm_folder = srtm_folder

        # Buffer and rounding
        self.buf = buf
        self.rounding = rounding

        # Image coordinates
        self.coordinates = CoordinateSystem()

        if self.srtm_type == 'SRTM3':
            step = 1.0 / 3600 * 3
        elif self.srtm_type == 'SRTM1':
            step = 1.0 / 3600
        else:
            print('Unkown SRTM type' + self.srtm_type)
            return

        self.coordinates.create_geographic(dlat=step, dlon=step)
        self.coordinates.add_res_info(self.meta, buf=self.buf, round=self.rounding)

        # Initialize output files
        self.out_folder = os.path.dirname(self.meta.res_path)
        self.dem = os.path.join(self.out_folder, 'DEM' + self.coordinates.sample + '.raw')
        self.q_dem = os.path.join(self.out_folder, 'DEM_q' + self.coordinates.sample + '.raw')

    def __call__(self):
        # Create the DEM and xyz coordinates
        # First download and load needed tiles

        # Download cannot be done here, because concurrent DEM creation processes can conflict with each other.
        # self.srtm_tiles(polygon=self.polygon, border=self.border, rounding=self.rounding)

        # Find the needed tiles.
        filelist = SrtmDownload.srtm_listing(self.srtm_folder)
        tiles, n, v, t = SrtmDownload.select_tiles(filelist, self.coordinates, self.srtm_folder, self.srtm_type, self.quality, download=False)
        tiles = [tile[:-4] for tile in tiles if tile.endswith('.hgt')]

        # Finally add metadata if it is directly linked to a SAR image
        self.add_meta_data(self.meta, self.coordinates, self.quality)

        # Then create needed files
        if self.quality:
            self.create_output_files(self.meta, file_type=['DEM_q' + self.coordinates.sample])
            self.q_dat = self.create_image(dat_type='.q', tiles=tiles)
            self.meta.image_new_data_memory(self.q_dat, 'import_DEM', 0, 0, file_type='DEM_q' + self.coordinates.sample)

        self.create_output_files(self.meta, file_type=['DEM' + self.coordinates.sample])
        self.dem_dat = self.create_image(dat_type='.hgt', tiles=tiles)
        self.meta.image_new_data_memory(self.dem_dat, 'import_DEM', 0, 0, file_type='DEM' + self.coordinates.sample)

    def create_image(self, tiles, dat_type='.hgt'):
        # This function adds tiles to np.memmap file

        if dat_type == '.q':
            outputdata = np.memmap(self.q_dem, dtype=np.int8, shape=tuple(self.coordinates.shape), mode='w+')
        elif dat_type == '.hgt':
            outputdata = np.memmap(self.dem, dtype=np.float32, shape=tuple(self.coordinates.shape), mode='w+')
        else:
            print('images type should be ".q or .hgt"')
            return

        if self.srtm_type == 'SRTM1':
            shape = (3601, 3601)
            s_size = 1.0 / 3600.0
            step_lat = 1
            step_lon = 1
        elif self.srtm_type == 'SRTM3':
            shape = (1201, 1201)
            s_size = 1.0 / 1200.0
            step_lat = 1
            step_lon = 1
        else:
            print('quality should be either SRTM1 or SRTM3!')
            return

        for tile in tiles:
            tile_name = tile + dat_type
            if not os.path.exists(tile_name):
                print('Tile ' + os.path.basename(tile_name) + ' does not exist!')

            if dat_type == '.q':
                image = np.fromfile(tile + dat_type, dtype='>u1').reshape(shape)
            elif dat_type == '.hgt':
                image = np.fromfile(tile + dat_type, dtype='float32').reshape(shape)
            else:
                print('images type should be ".q or .hgt"')
                #return

            if os.path.basename(tile)[7] == 'N':
                lat = float(os.path.basename(tile)[8:10])
            else:
                lat = - float(os.path.basename(tile)[8:10])
            if os.path.basename(tile)[10] == 'E':
                lon = float(os.path.basename(tile)[11:14])
            else:
                lon = - float(os.path.basename(tile)[11:14])

            print('adding ' + tile)

            latlim = [self.coordinates.lat0, self.coordinates.lat0 + self.coordinates.dlat * (self.coordinates.shape[0] -1)]
            lonlim = [self.coordinates.lon0, self.coordinates.lon0 + self.coordinates.dlon * (self.coordinates.shape[1] -1)]

            # Find the coordinates of the part of the tile that should be written to the output data.
            t_latlim = [max(lat, latlim[0]), min(lat + step_lat, latlim[1])]
            t_lonlim = [max(lon, lonlim[0]), min(lon + step_lon, lonlim[1])]
            t_latid = [shape[0] - int(round((t_latlim[0] - lat) / s_size)), 
                       shape[0] - (int(round((t_latlim[1] - lat) / s_size)) + 1)]
            t_lonid = [int(round((t_lonlim[0] - lon) / s_size)), 
                       int(round((t_lonlim[1] - lon) / s_size)) + 1]
            
            latsize = int(round((latlim[1] - latlim[0]) / s_size)) + 1
            latid = [latsize - int(round((t_latlim[0] - latlim[0]) / s_size)),
                     latsize - (int(round((t_latlim[1] - latlim[0]) / s_size)) + 1)]
            lonid = [int(round((t_lonlim[0] - lonlim[0]) / s_size)),
                     int(round((t_lonlim[1] - lonlim[0]) / s_size)) + 1]

            print('Adding tile lat ' + str(t_latid[1] + 1) + ' to ' + str(t_latid[0]) + ' into DEM file ' +
                  str(latid[1] + 1) + ' to ' + str(latid[0]))
            print('Adding tile lon ' + str(t_lonid[0] + 1) + ' to ' + str(t_lonid[1]) + ' into DEM file ' +
                  str(lonid[0] + 1) + ' to ' + str(lonid[1]))

            # Assign values from tiles to outputdata
            outputdata[latid[1]: latid[0], lonid[0]: lonid[1]] = image[t_latid[1]: t_latid[0], t_lonid[0]: t_lonid[1]]

        outputdata.flush()

        return outputdata

    @staticmethod
    def add_meta_data(meta, coordinates, quality=False):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'import_DEM' in meta.processes.keys():
            meta_info = meta.processes['import_DEM']
        else:
            meta_info = OrderedDict()

        if quality:
            meta_info = coordinates.create_meta_data(['DEM', 'DEM_q'], ['real4', 'int8'], meta_info)
        else:
            meta_info = coordinates.create_meta_data(['DEM'], ['real4'], meta_info)

        meta.image_add_processing_step('import_DEM', meta_info)

    @staticmethod
    def processing_info(coordinates, meta_type='', quality=False):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()

        # line and pixel output files.
        if quality:
            names = ['DEM', 'DEM_q']
        else:
            names = ['DEM']

        output_dat = recursive_dict()
        for name in names:
            output_dat['slave']['import_DEM'][name]['files'] = [name + coordinates.sample + '.raw']
            output_dat['slave']['import_DEM'][name]['coordinates'] = coordinates
            output_dat['slave']['import_DEM'][name]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 2

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('import_DEM', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('import_DEM', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('import_DEM', file_type, coordinates)

# This class loads a non SRTM DEM grid. Inputs are:
# - grid with heights
# - grid with lat/lon coordinate of all points
# Grid should be given in geotiff format.


class CreateExternalDem(CreateSrtmDem):

    def use_external_tiff(self, lat_tiff, lon_tiff, h_tiff, n_processes=1, block_size=10000000):

        h = gdal.Open(h_tiff, gdal.GA_ReadOnly)
        lat = gdal.Open(lat_tiff, gdal.GA_ReadOnly)
        lon = gdal.Open(lon_tiff, gdal.GA_ReadOnly)

        if h is None or lat is None or lon is None:
            print 'Unable to open one of the DEM files'
            return

        # Find the start and end line by reading lat and lon information in chunks.
        in_size = (h.RasterYSize, h.RasterXSize)
        lat_mins = np.zeros()


        n_lines = 10


        s_line = 0
        e_line = 10**10
        s_pixel = 0
        e_pixel = 10**10

        #
        h = h.GetRasterBand(1)

        # Convert to xyz coordinates


