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
import fiona
from shapely.geometry import Polygon

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

    def __init__(self, meta, srtm_folder='', quality=False, buf=0.2, rounding=0.2, srtm_type='SRTM3'):

        # Define the limits of our image
        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # Quality file also created?
        self.quality = quality
        self.srtm_type = srtm_type
        self.srtm_folder = srtm_folder

        # Image coordinates
        self.coordinates = CreateSrtmDem.srtm_coordinates(self.meta.polygon, srtm_type=srtm_type,
                                                          buf=buf, rounding=rounding)

        # Initialize output files
        self.out_folder = os.path.dirname(self.meta.res_path)
        self.dem = os.path.join(self.out_folder, 'Dem_' + self.coordinates.sample + '.raw')
        self.q_dem = os.path.join(self.out_folder, 'Dem_q_' + self.coordinates.sample + '.raw')

    def __call__(self):
        # Create the DEM and xyz coordinates
        # First download and load needed tiles

        # Download cannot be done here, because concurrent DEM creation processes can conflict with each other.
        # self.srtm_tiles(polygon=self.polygon, border=self.border, rounding=self.rounding)

        # Find the needed tiles.
        filelist = SrtmDownload.srtm_listing(self.srtm_folder)
        tiles, n, v, t = SrtmDownload.select_tiles(filelist, self.srtm_folder, self.srtm_type, self.quality)
        tiles = [tile[:-4] for tile in tiles if tile.endswith('.hgt')]

        # Finally add metadata if it is directly linked to a SAR image
        self.add_meta_data(self.meta, self.coordinates, self.quality)
        self.create_output_files(self.meta)

        # Then create needed files
        if self.quality:
            self.q = self.create_image(dat_type='.q', tiles=tiles)
            self.meta.image_new_data_memory(self.q, 'import_dem', 0, 0, file_type='Dem_q_' + self.coordinates.sample)

        self.dem = self.create_image(dat_type='.hgt', tiles=tiles)
        self.meta.image_new_data_memory(self.dem, 'import_dem', 0, 0, file_type='Dem_' + self.coordinates.sample)

    def create_image(self, tiles, dat_type='.hgt'):
        # This function adds tiles to np.memmap file

        if dat_type == '.q':
            outputdata = np.memmap(self.q_dem, dtype=np.int8, shape=self.coordinates.shape, mode='w+')
        elif dat_type == '.hgt':
            outputdata = np.memmap(self.dem, dtype=np.float32, shape=self.coordinates.shape, mode='w+')
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
                return

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

            print('Adding tile lat ' + str(t_latid[1] + 1) + ' to ' + str(t_latid[0]) + ' into dem file ' +
                  str(latid[1] + 1) + ' to ' + str(latid[0]))
            print('Adding tile lon ' + str(t_lonid[0] + 1) + ' to ' + str(t_lonid[1]) + ' into dem file ' +
                  str(lonid[0] + 1) + ' to ' + str(lonid[1]))

            # Assign values from tiles to outputdata
            outputdata[latid[1]: latid[0], lonid[0]: lonid[1]] = image[t_latid[1]: t_latid[0], t_lonid[0]: t_lonid[1]]

        outputdata.flush()

        return outputdata

    @staticmethod
    def add_meta_data(meta, coordinates, quality=False):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'import_dem' in meta.processes.keys():
            meta_info = meta.processes['import_dem']
        else:
            meta_info = OrderedDict()

        if quality:
            meta_info = coordinates.create_meta_data(['Dem', 'Dem_q'], ['real4', 'int8'], meta_info)
        else:
            meta_info = coordinates.create_meta_data(['Dem'], ['real4'], meta_info)

        meta.image_add_processing_step('import_dem', meta_info)

    @staticmethod
    def processing_info(coordinates, quality=False):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()

        # line and pixel output files.
        if quality:
            names = ['Dem', 'Dem_q']
        else:
            names = ['Dem']

        output_dat = defaultdict()
        for name in names:
            output_dat['slave']['import_dem'][name]['files'] = [name + coordinates.sample + '.raw']
            output_dat['slave']['import_dem'][name]['coordinates'] = coordinates
            output_dat['slave']['import_dem'][name]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 2

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, output_file_steps=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not output_file_steps:
            meta_info = meta.processes['import_dem']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_create_disk('import_dem', s)

    @staticmethod
    def save_to_disk(meta, output_file_steps=''):

        if not output_file_steps:
            meta_info = meta.processes['import_dem']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_memory_to_disk('import_dem', s)

    @staticmethod
    def srtm_coordinates(polygon='', buf=1.0, rounding=1.0, srtm_type='SRTM3', shapefile=''):

        # If shapefile exist the polygon is not needed.
        if shapefile:
            polygon = CreateSrtmDem.load_shp(shapefile)

        polygon = polygon.buffer(buf)
        coor = polygon.exterior.coords

        lon = [l[0] for l in coor]
        lat = [l[1] for l in coor]

        latlim = [min(lat), max(lat)]
        lonlim = [min(lon), max(lon)]

        # Add the rounding and borders to add the sides of our image
        # Please use rounding as a n/60 part of a degree (so 1/15 , 1/10 or 1/20 of a degree for example..)
        latlim = np.array([np.floor((latlim[0]) / rounding) * rounding, np.ceil((latlim[1]) / rounding) * rounding])
        lonlim = np.array([np.floor((lonlim[0]) / rounding) * rounding, np.ceil((lonlim[1]) / rounding) * rounding])

        # Define image resolution
        if srtm_type == 'SRTM1':
            pixel_degree = 3600
        elif srtm_type == 'SRTM3':
            pixel_degree = 1200
        else:
            print('quality should be either SRTM1 or SRTM3!')
            return

        shape = np.array(np.round(np.diff(latlim) * pixel_degree + 1), np.round(np.diff(lonlim) * pixel_degree + 1))

        coordinates = CoordinateSystem()
        coordinates.create_geographic_coordinates(shape=shape, lat0=latlim[0], lon0=lonlim[0],
                                                  dlat=1 / float(pixel_degree), dlon=1 / float(pixel_degree))

        return coordinates

    @staticmethod
    def load_shp(filename):
        # from kml and shape file to a bounding box. We will always use a bounding box to create the final product.

        if filename.endswith('.shp'):
            with fiona.collection(filename, "r") as inputshape:

                shapes = [shape for shape in inputshape]
                # only first shape
                polygon = Polygon(shapes[0]['geometry']['coordinates'][0])
        else:
            print('Shape not recognized')
            return []

        return polygon

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


