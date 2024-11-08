# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy/scipy functions.
import numpy as np
from collections import OrderedDict
import os
import logging

# Import the parent class Process for processing steps.
from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.external_dems.geoid import GeoidInterp
from rippl.meta_data.process import Process
from rippl.external_dems.srtm.srtm_download import SrtmDownload
from rippl.external_dems.tandem_x.tandem_x_download import TandemXDownload
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.user_settings import UserSettings


class ImportDem(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor='', out_coor='', reference_slc='reference_slc', dem_folder=None, quality=False,
                 buffer=0, rounding=0, expected_min_height=0, expected_max_height=500,
                 dem_type='SRTM3', lon_resolution=3, geoid_file=None, overwrite=False):

        """
        This function creates a dem. Current options are SRTM1 and SRTM3. This could be extended in the future.
        If you only want to import a dem of which the coordinate system is already known and does not need to be
        cropped use the import dem function.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param CoordinateSystem in_coor: Coordinate system of the input radar grid for which we create a grid. An output
                    grid will be generated in this function. (Not needed, only when output grid is not defined)
        :param CoordinateSystem out_coor: Coordinate system of the input radar grid for which we create a grid. An output
                    grid will be generated in this function. (Not needed if input grid is defined)
        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampling etc.
        """

        # Dummy input info because we do not have inputs, but needed for correct processing
        self.input_info = {'process_names': [], 'image_types': [], 'polarisations': [], 'data_ids': [], 'coor_types': [],
                           'file_names': [], 'data_types': [], 'in_coor_types': [], 'aliases_processing': []}

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'dem'
        self.output_info['image_type'] = 'reference_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        if not quality:
            self.output_info['file_names'] = ['dem']
            self.output_info['data_types'] = ['real4']
        else:
            self.output_info['file_names'] = ['dem', 'dem_quality']
            self.output_info['data_types'] = ['real4', 'int8']

        # Save the settings for dem generation
        settings = UserSettings()
        settings.load_settings()
        if not dem_folder:
            if dem_type.startswith('SRTM'):
                self.dem_folder = os.path.join(settings.settings['paths']['DEM_database'], settings.settings['path_names']['DEM']['SRTM'])
            elif dem_type.startswith('TDM'):
                self.dem_folder = os.path.join(settings.settings['paths']['DEM_database'], settings.settings['path_names']['DEM']['TanDEM-X'])
            else:
                self.dem_folder = os.path.join(settings.settings['paths']['DEM_database'], dem_type)
        else:
            self.dem_folder = dem_folder
        if not geoid_file:
            self.geoid_file = os.path.join(settings.settings['paths']['DEM_database'], 'geoid', 'egm96.dat')
        else:
            self.geoid_file = geoid_file

        if not os.path.exists(self.dem_folder):
            raise FileExistsError('Path to DEM folder does not exist. Enter valid path for DEM data.')
        self.dem_type = dem_type
        self.quality = quality
        self.settings = OrderedDict({'in_coor': dict(), 'out_coor': dict()})
        self.settings['dem_folder'] = self.dem_folder
        self.settings['quality'] = self.quality
        self.settings['out_coor']['buffer'] = buffer
        self.settings['out_coor']['rounding'] = rounding
        self.settings['out_coor']['min_height'] = expected_min_height
        self.settings['out_coor']['max_height'] = expected_max_height
        self.settings['dem_type'] = dem_type
        self.settings['lon_resolution'] = lon_resolution

        self.overwrite = overwrite

        # Coordinate systems
        if isinstance(in_coor, CoordinateSystem):
            logging.info('Generating DEM input size from input grid')
            out_coor = self.create_dem_coor(dem_type, in_coor, True, buffer, rounding, lon_resolution, expected_min_height, expected_max_height)
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['reference_slc'] = reference_slc

    @staticmethod
    def create_dem_coor(dem_type, in_coor, calc_shape=True, buffer=0, rounding=0, lon_resolution=3, min_height=0, max_height=500):

        out_coor = CoordinateSystem()
        if dem_type == 'SRTM1':
            out_coor.create_geographic(1.0 / 3600, 1.0 / 3600)
        elif dem_type == 'SRTM3':
            out_coor.create_geographic(3.0 / 3600, 3.0 / 3600)
        elif dem_type == 'TDM30':
            out_coor.create_geographic(1.0 / 3600.0, lon_resolution / 3600.0)
        elif dem_type == 'TDM90':
            out_coor.create_geographic(3.0 / 3600.0, lon_resolution / 3600.0)
        else:
            logging.info('dem type not supported. Options are SRTM1, SRTM3 and TanDEM-X')

        # Calculate the extent of the output grid.
        if calc_shape:
            out_coor = CoorNewExtend(in_coor=in_coor, out_coor=out_coor, buffer=buffer, rounding=rounding,
                                     min_height=min_height, max_height=max_height).out_coor
            return out_coor
        else:
            return out_coor

    def process_calculations(self):
        """
        Find the already downloaded dem files and subtract the needed regions. Save these files in the right format
        to disk.

        :return:
        """

        # Because there is no input coordinate system we have to give the chunk coordinate system if the dem is loaded
        # in chunks.
        geoid = GeoidInterp.create_geoid(self.coordinate_systems['out_coor_chunk'], self.geoid_file)

        if self.settings['dem_type'] in ['SRTM1', 'SRTM3']:
            self['dem'] = np.flipud(self.create_dem(self.coordinate_systems['out_coor_chunk'], self.dem_folder, self.dem_type, quality=False))
            if self.quality:
                self['dem_quality_' + self.dem_type] = np.flipud(self.create_dem(self.coordinate_systems['out_coor_chunk'], self.dem_folder, self.dem_type, quality=True))

            self['dem'] += geoid
        elif self.settings['dem_type'] in ['TDM30', 'TDM90']:
            # TDM90 gives incorrect values over the ocean. Use TDM30 or SRTM data there instead.
            self['dem'] = np.flipud(self.create_dem(self.coordinate_systems['out_coor_chunk'], self.dem_folder, self.dem_type, lon_resolution=self.settings['lon_resolution']))
            self['dem'][self['dem'] == -99999] = geoid[self['dem'] == -99999]

        self['dem'] = self['dem'].astype(np.float32)

    @staticmethod
    def create_dem(coordinates, dem_folder, dem_type='SRTM3', quality=False, lon_resolution=3):
        """
        Create a grid for SRTM

        :param CoordinateSystem coordinates:
        :param str dem_folder: Folder where downloaded DEM data is stored
        :param str dem_type: Type of DEM data (SRTM1, SRTM3 or TanDEM-X)
        :param bool quality: Defines whether we create a quality or regular dem grid
        :return: dem or quality grid
        :rtype: np.ndarray
        """

        if not dem_folder:
            settings = UserSettings()
            settings.load_settings()
            if dem_type.startswith('SRTM'):
                dem_folder = os.path.join(settings.settings['paths']['DEM_database'], settings.settings['path_names']['DEM']['SRTM'])
            elif dem_type.startswith('TDM'):
                dem_folder = os.path.join(settings.settings['paths']['DEM_database'], settings.settings['path_names']['DEM']['TanDEM-X'])

        if dem_type.startswith('SRTM'):
            filelist = SrtmDownload.srtm_listing()
            tiles = SrtmDownload.select_tiles(filelist, coordinates, dem_folder, dem_type, quality, download=False)[0]
            if quality:
                tiles = [tile[:-4] for tile in tiles if tile.endswith('.hgt')]
            else:
                tiles = [tile[:-4] for tile in tiles if tile.endswith('.hgt')]
        elif dem_type.startswith('TDM'):
            filelist = TandemXDownload.tandem_x_listing()
            tiles = TandemXDownload.select_tiles(filelist, coordinates, dem_folder, tandem_x_type=dem_type,
                                                 lon_resolution=lon_resolution)[0]

        if quality:
            if dem_type.startswith('TDM'):
                raise TypeError('No quality files available for Tandem-x data.')
            outputdata = np.zeros(coordinates.shape, dtype=np.int8)
            dat_type = '.q'
        else:
            if dem_type.startswith('TDM'):
                outputdata = np.ones(coordinates.shape, dtype=np.float32) * -99999
                dat_type = ''
            else:
                outputdata = np.zeros(coordinates.shape, dtype=np.float32)
                dat_type = '.hgt'

        if dem_type == 'SRTM1':
            tile_shape = (3601, 3601)
        elif dem_type == 'SRTM3':
            tile_shape = (1201, 1201)
        elif dem_type == 'TDM30':
            tile_shape = (3601, int(np.round(3600 / lon_resolution + 1)))
        elif dem_type == 'TDM90':
            tile_shape = (1201, int(np.round(3600 / lon_resolution + 1)))
        lat_size = coordinates.dlat
        lon_size = coordinates.dlon
        step_lat = 1
        step_lon = 1

        for tile in tiles:
            tile_name = tile + dat_type
            if not os.path.exists(tile_name):
                raise FileExistsError('Tile ' + os.path.basename(tile_name) + ' does not exist!')

            if dat_type == '.q':
                image = np.fromfile(tile + dat_type, dtype='>u1').reshape(tile_shape)
            elif dat_type == '.hgt':
                image = np.fromfile(tile + dat_type, dtype='>i2').reshape(tile_shape)
            elif dat_type == '' and dem_type.startswith('TDM'):
                image = np.fromfile(tile + dat_type, dtype=np.float32).reshape(tile_shape)
            else:
                raise TypeError('images type should be ".q, .raw or .hgt"')

            if dem_type.startswith('TDM'):
                s_id = 8
            elif dem_type.startswith('SRTM'):
                s_id = 7

            if os.path.basename(tile)[s_id] == 'N':
                lat = float(os.path.basename(tile)[s_id+1:s_id+3])
            else:
                lat = - float(os.path.basename(tile)[s_id+1:s_id+3])
            if os.path.basename(tile)[s_id + 3] == 'E':
                lon = float(os.path.basename(tile)[s_id+4:s_id+7])
            else:
                lon = -float(os.path.basename(tile)[s_id+4:s_id+7])

            logging.info('adding ' + tile)
            lat0 = coordinates.lat0 + coordinates.dlat * coordinates.first_line
            lon0 = coordinates.lon0 + coordinates.dlon * coordinates.first_pixel

            latlim = [lat0, lat0 + coordinates.dlat * (coordinates.shape[0] - 1)]
            lonlim = [lon0, lon0 + coordinates.dlon * (coordinates.shape[1] - 1)]

            # Find the coordinates of the part of the tile that should be written to the output data.
            t_latlim = [max(lat, latlim[0]), min(lat + step_lat, latlim[1])]
            t_lonlim = [max(lon, lonlim[0]), min(lon + step_lon, lonlim[1])]
            t_latid = [tile_shape[0] - int(round((t_latlim[0] - lat) / lat_size)),
                       tile_shape[0] - (int(round((t_latlim[1] - lat) / lat_size)) + 1)]
            t_lonid = [int(round((t_lonlim[0] - lon) / lon_size)),
                       int(round((t_lonlim[1] - lon) / lon_size)) + 1]

            lat_total_size = int(round((latlim[1] - latlim[0]) / lat_size)) + 1
            latid = [lat_total_size - int(round((t_latlim[0] - latlim[0]) / lat_size)),
                     lat_total_size - (int(round((t_latlim[1] - latlim[0]) / lat_size)) + 1)]
            lonid = [int(round((t_lonlim[0] - lonlim[0]) / lon_size)),
                     int(round((t_lonlim[1] - lonlim[0]) / lon_size)) + 1]

            logging.info('Adding tile lat ' + str(t_latid[1] + 1) + ' to ' + str(t_latid[0]) + ' into dem file ' +
                  str(latid[1] + 1) + ' to ' + str(latid[0]))
            logging.info('Adding tile lon ' + str(t_lonid[0] + 1) + ' to ' + str(t_lonid[1]) + ' into dem file ' +
                  str(lonid[0] + 1) + ' to ' + str(lonid[1]))

            # Assign values from tiles to outputdata
            outputdata[latid[1]: latid[0], lonid[0]: lonid[1]] = image[t_latid[1]: t_latid[0], t_lonid[0]: t_lonid[1]]

        return outputdata
