# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy/scipy functions.
import numpy as np
from collections import OrderedDict
import os

# Import the parent class Process for processing steps.
from rippl.external_dems.geoid import GeoidInterp
from rippl.meta_data.process import Process
from rippl.external_dems.srtm.srtm_download import SrtmDownload
from rippl.external_dems.tandem_x.tandem_x_download import TandemXDownload
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.user_settings import UserSettings


class ImportDem(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor='', coreg_master='coreg_master', dem_folder=None, quality=False,
                 buffer=0.2, rounding=0.2, dem_type='SRTM3', lon_resolution=3, geoid_file=None, overwrite=False):

        """
        This function creates a dem. Current options are SRTM1 and SRTM3. This could be extended in the future.
        If you only want to import a dem of which the coordinate system is already known and does not need to be
        cropped use the import dem function.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param CoordinateSystem in_coor: Coordinate system of the input radar grid for which we create a grid. An output
                    grid will be generated in this function.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampling etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'dem'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        if not quality:
            self.output_info['file_types'] = ['dem']
            self.output_info['data_types'] = ['real4']
        else:
            self.output_info['file_types'] = ['dem', 'dem_quality']
            self.output_info['data_types'] = ['real4', 'int8']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master']
        self.input_info['process_types'] = ['crop']
        self.input_info['file_types'] = ['crop']
        self.input_info['polarisations'] = ['']
        self.input_info['data_ids'] = ['']
        self.input_info['coor_types'] = ['in_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['type_names'] = ['in_coor_grid']

        # Save the settings for dem generation
        settings = UserSettings()
        settings.load_settings()
        if not dem_folder:
            if dem_type.startswith('SRTM'):
                self.dem_folder = os.path.join(settings.DEM_database, settings.dem_sensor_name['srtm'])
            elif dem_type == 'TanDEM-X':
                self.dem_folder = os.path.join(settings.DEM_database, settings.dem_sensor_name['tdx'])
            else:
                self.dem_folder = os.path.join(settings.DEM_database, dem_type)
        else:
            self.dem_folder = dem_folder
        if not geoid_file:
            self.geoid_file = os.path.join(settings.DEM_database, 'geoid', 'egm96.dat')
        else:
            self.geoid_file = geoid_file

        if not os.path.exists(self.dem_folder):
            raise FileExistsError('Path to DEM folder does not exist. Enter valid path for DEM data.')
        self.dem_type = dem_type
        self.quality = quality
        self.settings = OrderedDict()
        self.settings['dem_folder'] = self.dem_folder
        self.settings['quality'] = self.quality
        self.settings['buffer'] = buffer
        self.settings['rounding'] = rounding
        self.settings['dem_type'] = dem_type
        self.settings['lon_resolution'] = lon_resolution

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        out_coor = self.create_dem_coor(dem_type, lon_resolution=lon_resolution)
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master

    @staticmethod
    def create_dem_coor(dem_type, lon_resolution=3):

        out_coor = CoordinateSystem()
        if dem_type == 'SRTM1':
            out_coor.create_geographic(1.0 / 3600, 1.0 / 3600)
        elif dem_type == 'SRTM3':
            out_coor.create_geographic(3.0 / 3600, 3.0 / 3600)
        elif dem_type == 'TanDEM-X':
            out_coor.create_geographic(3 / 3600.0, lon_resolution / 3600.0)
        else:
            print('dem type not supported. Options are SRTM1, SRTM3 and TanDEM-X')

        return out_coor

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(ImportDem, self).__init__(
                       input_info=self.input_info,
                       output_info=self.output_info,
                       coordinate_systems=self.coordinate_systems,
                       processing_images=self.processing_images,
                       overwrite=self.overwrite,
                       settings=self.settings)

    def process_calculations(self):
        """
        Find the already downloaded dem files and subtract the needed regions. Save these files in the right format
        to disk.

        :return:
        """

        # Because there is no input coordinate system we have to give the block coordinate system if the dem is loaded
        # in blocks.
        geoid = GeoidInterp.create_geoid(self.block_coor, self.geoid_file, download=False)

        if self.settings['dem_type'] in ['SRTM1', 'SRTM3']:
            self['dem'] = np.flipud(self.create_dem(self.block_coor, self.dem_folder, self.dem_type, quality=False))
            if self.quality:
                self['dem_quality_' + self.dem_type] = np.flipud(self.create_dem(self.block_coor, self.dem_folder, self.dem_type, quality=True))

            self['dem'] -= geoid
        elif self.settings['dem_type'] == 'TanDEM-X':
            self['dem'] = np.flipud(self.create_dem(self.block_coor, self.dem_folder, self.dem_type, lon_resolution=self.settings['lon_resolution']))
            self['dem'][self['dem'] == -99999] = -geoid[self['dem'] == -99999]

    def def_out_coor(self):
        """
        Based on the buffer and rounding

        :return:
        """

        new_coor = CoorNewExtend(self.coordinate_systems['in_coor'], self.coordinate_systems['out_coor'],
                                 buffer=self.settings['buffer'], rounding=self.settings['rounding'])
        self.coordinate_systems['out_coor'] = new_coor.out_coor
        self.coordinate_systems['in_coor'] = self.coordinate_systems['out_coor']

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
                dem_folder = os.path.join(settings.DEM_database, settings.dem_sensor_name['srtm'])
            elif dem_type == 'TanDEM-X':
                dem_folder = os.path.join(settings.DEM_database, settings.dem_sensor_name['tdx'])

        if dem_type.startswith('SRTM'):
            filelist = SrtmDownload.srtm_listing(dem_folder)
            tiles = SrtmDownload.select_tiles(filelist, coordinates, dem_folder, dem_type, quality, download=False)[0]
            if quality:
                tiles = [tile[:-4] for tile in tiles if tile.endswith('.hgt')]
            else:
                tiles = [tile[:-4] for tile in tiles if tile.endswith('.hgt')]
        elif dem_type == 'TanDEM-X':
            filelist = TandemXDownload.tandem_x_listing(dem_folder)
            tiles = TandemXDownload.select_tiles(filelist, coordinates, dem_folder, lon_resolution=lon_resolution)[0]

        if quality:
            if dem_type == 'TanDEM-X':
                raise TypeError('No quality files available for Tandem-x data.')
            outputdata = np.zeros(coordinates.shape, dtype=np.int8)
            dat_type = '.q'
        else:
            if dem_type == 'TanDEM-X':
                outputdata = np.ones(coordinates.shape, dtype=np.float32) * -99999
                dat_type = ''
            else:
                outputdata = np.zeros(coordinates.shape, dtype=np.float32)
                dat_type = '.hgt'

        if dem_type == 'SRTM3':
            tile_shape = (1201, 1201)
        elif dem_type == 'TanDEM-X':
            tile_shape = (1201, int(np.round(3600 / lon_resolution + 1)))
        elif dem_type == 'SRTM1':
            tile_shape = (3601, 3601)
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
            elif dat_type == '' and dem_type == 'TanDEM-X':
                image = np.fromfile(tile + dat_type, dtype=np.float32).reshape(tile_shape)
            else:
                raise TypeError('images type should be ".q, .raw or .hgt"')

            if dem_type == 'TanDEM-X':
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

            # print('adding ' + tile)

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

            #print('Adding tile lat ' + str(t_latid[1] + 1) + ' to ' + str(t_latid[0]) + ' into dem file ' +
            #      str(latid[1] + 1) + ' to ' + str(latid[0]))
            #print('Adding tile lon ' + str(t_lonid[0] + 1) + ' to ' + str(t_lonid[1]) + ' into dem file ' +
            #      str(lonid[0] + 1) + ' to ' + str(lonid[1]))

            # Assign values from tiles to outputdata
            outputdata[latid[1]: latid[0], lonid[0]: lonid[1]] = image[t_latid[1]: t_latid[0], t_lonid[0]: t_lonid[1]]

        return outputdata
