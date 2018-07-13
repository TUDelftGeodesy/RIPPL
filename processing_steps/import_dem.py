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

from collections import OrderedDict
from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from orbit_dem_functions.srtm_download import SrtmDownload
from image_data import ImageData


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

    def __init__(self, out_folder, dem_data_folder='', resolution='SRTM3', meta='', quality=False,
                 shapefile='', polygon='', border=0.1, rounding=0.1,
                 username='', password='',
                 s_lin=0, s_pix=0, lines=0,
                 n_processes=1, max_pix_num=10000000):

        # Define the limits of our image
        self.srtm_tiles = SrtmDownload(dem_data_folder, username, password, resolution, n_processes)
        self.max_pix_num = max_pix_num
        self.n_processes = n_processes
        self.latlim = []
        self.lonlim = []

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.meta = meta
        else:
            self.meta = []

        if self.meta:
            self.polygon = self.meta.polygon
        elif shapefile:
            self.polygon = SrtmDownload.load_shp(shapefile)
        elif polygon:
            self.polygon = polygon
        else:
            print('No shape selected')
            return

        self.rounding = rounding
        self.border = border
        self.resolution = resolution
        self.quality = quality

        # Image shapes/limits
        self.pixel_degree = int
        self.tiff_latlim = []
        self.tiff_lonlim = []
        self.lat_size = ''
        self.lon_size = ''

        # Initialize output files
        self.out_folder = out_folder
        self.dem = os.path.join(self.out_folder, 'dem_' + resolution + '.raw')
        self.q_dem = os.path.join(self.out_folder, 'dem_q_' + resolution + '.raw')

    def __call__(self):
        # Create the DEM and xyz coordinates
        # First download and load needed tiles

        self.srtm_tiles(polygon=self.polygon, border=self.border, rounding=self.rounding)
        self.lonlim = self.srtm_tiles.lonlim
        self.latlim = self.srtm_tiles.latlim
        self.define_image_coverage()

        tiles = [tile[:-4] for tile in self.srtm_tiles.tiles if tile.endswith('.hgt')]

        # Then create needed files
        if self.quality:
            self.q = self.create_image(dat_type='.q', tiles=tiles)
        self.dem = self.create_image(dat_type='.hgt', tiles=tiles)

        # Finally add metadata if it is directly linked to a SAR image
        if isinstance(self.meta, ImageData):
            self.add_meta_data()
            self.meta.image_new_data_memory(self.dem, 'import_dem', 0, 0, file_type='Dem_' + self.resolution)

    @staticmethod
    def create_output_files(meta, to_disk='', resolution='SRTM3'):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = ['Dem_' + resolution]

        for s in to_disk:
            meta.image_create_disk('import_dem', s)

    def save_to_disk(self, to_disk=''):

        if not to_disk:
            to_disk = ['Dem_' + self.resolution]

        for s in to_disk:
            self.meta.image_memory_to_disk('import_dem', s)

    def define_image_coverage(self):
        # Define image resolution
        if self.resolution == 'SRTM1':
            self.pixel_degree = 3600
        elif self.resolution == 'SRTM3':
            self.pixel_degree = 1200
        else:
            print('quality should be either SRTM1 or SRTM3!')
            return

        self.dtype = np.int16
        self.tiff_latlim = [self.latlim[0] - 0.5 / self.pixel_degree, self.latlim[1] + 0.5 / self.pixel_degree]
        self.tiff_lonlim = [self.lonlim[0] - 0.5 / self.pixel_degree, self.lonlim[1] + 0.5 / self.pixel_degree]

        # Define the final size of the grid
        self.lat_size = int(np.round((self.latlim[1] - self.latlim[0]) * self.pixel_degree)) + 1
        self.lon_size = int(np.round((self.lonlim[1] - self.lonlim[0]) * self.pixel_degree)) + 1
        print('Bounding box is:')
        print('from ' + str(self.latlim[0]) + ' latitude to ' + str(self.latlim[1]))
        print('from ' + str(self.lonlim[0]) + ' longitude to ' + str(self.lonlim[1]))

    def create_image(self, tiles, dat_type='.hgt'):
        # This function adds tiles to np.memmap file

        if dat_type == '.q':
            outputdata = np.memmap(self.q_dem, dtype=np.int8, shape=(self.lat_size, self.lon_size), mode='w+')
        elif dat_type == '.hgt':
            outputdata = np.memmap(self.dem, dtype=np.float32, shape=(self.lat_size, self.lon_size), mode='w+')

        if self.resolution == 'SRTM1':
            shape = (3601, 3601)
            s_size = 1.0 / 3600.0
            step_lat = 1
            step_lon = 1
        elif self.resolution == 'SRTM3':
            shape = (1201, 1201)
            s_size = 1.0 / 1200.0
            step_lat = 1
            step_lon = 1
        else:
            print('quality should be either SRTM1 or SRTM3!')
            return
        
        print('total file size is ' + str(self.lat_size) + ' in latitude and '
                                    + str(self.lon_size) + ' in longitude')

        for tile in tiles:
            tile_name = tile + dat_type
            if not os.path.exists(tile_name):
                print('Tile ' + os.path.basename(tile_name) + ' does not exist!')

            if dat_type == '.q':
                image = np.fromfile(tile + dat_type, dtype='>u1').reshape(shape)
            elif dat_type == '.hgt':
                image = np.fromfile(tile + dat_type, dtype='float32').reshape(shape)

            if os.path.basename(tile)[7] == 'N':
                lat = float(os.path.basename(tile)[8:10])
            else:
                lat = - float(os.path.basename(tile)[8:10])
            if os.path.basename(tile)[10] == 'E':
                lon = float(os.path.basename(tile)[11:14])
            else:
                lon = - float(os.path.basename(tile)[11:14])
            if self.resolution == 'SRTM30':
                lat = lat - 50 + (s_size / 2)
                lon += (s_size / 2)

            print('adding ' + tile)

            # Find the coordinates of the part of the tile that should be written to the output data.
            t_latlim = [max(lat, self.latlim[0]), min(lat + step_lat, self.latlim[1])]
            t_lonlim = [max(lon, self.lonlim[0]), min(lon + step_lon, self.lonlim[1])]
            t_latid = [shape[0] - int(round((t_latlim[0] - lat) / s_size)), 
                       shape[0] - (int(round((t_latlim[1] - lat) / s_size)) + 1)]
            t_lonid = [int(round((t_lonlim[0] - lon) / s_size)), 
                       int(round((t_lonlim[1] - lon) / s_size)) + 1]
            
            latsize = int(round((self.latlim[1] - self.latlim[0]) / s_size)) + 1
            latid = [latsize - int(round((t_latlim[0] - self.latlim[0]) / s_size)), 
                     latsize - (int(round((t_latlim[1] - self.latlim[0]) / s_size)) + 1)]
            lonid = [int(round((t_lonlim[0] - self.lonlim[0]) / s_size)), 
                     int(round((t_lonlim[1] - self.lonlim[0]) / s_size)) + 1]

            print('Adding tile lat ' + str(t_latid[1] + 1) + ' to ' + str(t_latid[0]) + ' into dem file ' +
                  str(latid[1] + 1) + ' to ' + str(latid[0]))
            print('Adding tile lon ' + str(t_lonid[0] + 1) + ' to ' + str(t_lonid[1]) + ' into dem file ' +
                  str(lonid[0] + 1) + ' to ' + str(lonid[1]))

            # Assign values from tiles to outputdata
            outputdata[latid[1]: latid[0], lonid[0]: lonid[1]] = image[t_latid[1]: t_latid[0], t_lonid[0]: t_lonid[1]]

        outputdata.flush()

        return outputdata

    def add_meta_data(self):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'import_dem' in self.meta.processes.keys():
            meta_info = self.meta.processes['import_dem']
        else:
            meta_info = OrderedDict()

        for dat, dat_type in zip(['Dem_' + self.resolution, 'Dem_q_' + self.resolution],
                                 ['real4', 'int8']):
            meta_info[dat + '_output_file'] = dat + '.raw'
            meta_info[dat + '_output_format'] = dat_type

            meta_info[dat + '_regular_grid'] = 'True'
            meta_info[dat + '_size_in_latitude'] = self.lat_size
            meta_info[dat + '_size_in_longitude'] = self.lon_size
            meta_info[dat + '_latitude_start'] = str(self.latlim[0])
            meta_info[dat + '_longitude_start'] = str(self.lonlim[0])
            meta_info[dat + '_latitude_step'] = str(1.0 / self.pixel_degree)
            meta_info[dat + '_longitude_step'] = str(1.0 / self.pixel_degree)

        self.meta.image_add_processing_step('import_dem', meta_info)

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
            print 'Unable to open one of the dem files'
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


