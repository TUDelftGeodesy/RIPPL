# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy/scipy functions.
import numpy as np
from scipy.interpolate import RectBivariateSpline
from collections import OrderedDict
import os

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.external_dems.srtm.srtm_download import SrtmDownload
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.coor_new_extend import CoorNewExtend


class CreateDem(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coor_in=[], in_image_types=[], coreg_master=[],
                 dem_folder='', quality=False, buffer=0.2, rounding=0.2, dem_type='SRTM3'):

        """
        This function creates a dem. Current options are SRTM1 and SRTM3. This could be extended in the future.
        If you only want to import a dem of which the coordinate system is already known and does not need to be
        cropped use the import dem function.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param CoordinateSystem coor_in: Coordinate system of the input radar grid for which we create a grid. An output
                    grid will be generated in this function.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.

        """

        # Output grid
        self.process_name = 'import_dem'
        self.dem_type = dem_type
        self.quality = quality
        if not quality:
            file_types = ['dem']
            data_types = ['real4']
        else:
            file_types = ['dem', 'dem_quality']
            data_types = ['real4', 'int8']

        # Input steps are not needed for dem creation.
        in_image_types = []
        in_coor_types = []
        in_data_ids = []
        in_polarisations = []
        in_processes = []
        in_file_types = []

        # Save the settings for dem generation
        self.dem_folder = dem_folder
        self.dem_type = dem_type
        self.settings = OrderedDict()
        self.settings['dem_folder'] = self.dem_folder
        self.settings['quality'] = self.quality
        self.settings['buffer'] = buffer
        self.settings['rounding'] = rounding
        self.settings['srtm_type'] = dem_type

        # Create the output grid coordinate system
        orbit = coreg_master.find_best_orbit('original')
        coor_in.orbit = orbit
        self.coor_out = self.create_dem_coordinates(coor_in, buffer, rounding, dem_type)

        # Call process initialization.
        super(CreateDem, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       coor_out=self.coor_out,
                       coreg_master=coreg_master,
                       out_processing_image='coreg_master')

    def process_calculations(self):
        """
        Find the already downloaded dem files and subtract the needed regions. Save these files in the right format
        to disk.

        :return:
        """

        # Because there is no input coordinate system we have to give the block coordinate system if the dem is loaded
        # in blocks.
        self['dem'] = self.create_srtm(self.block_coor, self.dem_folder, self.dem_type, quality=False)
        if self.quality:
            self['dem_quality'] = self.create_srtm(self.block_coor, self.dem_folder, self.dem_type, quality=True)

        geoid_file = os.path.join(self.dem_folder, 'egm96.dat')
        geoid = self.create_geoid(self.coor_out, geoid_file, download=False)
        self.process.data_id = self.dem_type
        self['dem'] -= geoid

    @staticmethod
    def create_dem_coordinates(coor_in, buffer, rounding, dem_type='SRTM3'):
        """
        Based on the buffer and rounding

        :param CoordinateSystem coor_in: Input coordinate system from radar grid
        :param float buffer: Buffer region around image to ensure full coverage
        :param float rounding: Rounding to create nice rounded image sizes
        :param str dem_type: dem type of input, defines the step size
        :return:
        """
        coor_out = CoordinateSystem()

        if dem_type == 'SRTM1':
            coor_out.create_geographic(1.0 / 3600, 1.0 / 3600)
        elif dem_type == 'SRTM3':
            coor_out.create_geographic(3.0 / 3600, 3.0 / 3600)
        else:
            print('dem type not supported')

        new_coor = CoorNewExtend(coor_in, coor_out, buffer=buffer, rounding=rounding)

        return new_coor.coor_out

    @staticmethod
    def create_srtm(coordinates, srtm_folder, srtm_type='SRTM3', quality=False):
        """
        Create a grid for SRTM

        :param CoordinateSystem coordinates:
        :param str srtm_folder: Folder where downloaded SRTM data is stored
        :param str srtm_type: Type of SRTM data (SRTM1 or SRTM3)
        :param bool quality: Defines whether we create a quality or regular dem grid
        :return: dem or quality grid
        :rtype: np.ndarray
        """

        filelist = SrtmDownload.srtm_listing(srtm_folder)
        tiles = SrtmDownload.select_tiles(filelist, coordinates, srtm_folder, srtm_type, quality, download=False)[0]
        if quality:
            tiles = [tile[:-4] for tile in tiles if tile.endswith('.hgt')]
        else:
            tiles = [tile[:-4] for tile in tiles if tile.endswith('.hgt')]

        if quality:
            outputdata = np.zeros(coordinates.shape, dtype=np.int8)
            dat_type = '.q'
        else:
            outputdata = np.zeros(coordinates.shape, dtype=np.float32)
            dat_type = '.hgt'

        if srtm_type == 'SRTM1':
            tile_shape = (3601, 3601)
        elif srtm_type == 'SRTM3':
            tile_shape = (1201, 1201)
        s_size = coordinates.dlat
        step_lat = 1
        step_lon = 1

        for tile in tiles:
            tile_name = tile + dat_type
            if not os.path.exists(tile_name):
                print('Tile ' + os.path.basename(tile_name) + ' does not exist!')

            if dat_type == '.q':
                image = np.fromfile(tile + dat_type, dtype='>u1').reshape(tile_shape)
            elif dat_type == '.hgt':
                image = np.fromfile(tile + dat_type, dtype='>i2').reshape(tile_shape)
            else:
                print('images type should be ".q or .hgt"')
                # return

            if os.path.basename(tile)[7] == 'N':
                lat = float(os.path.basename(tile)[8:10])
            else:
                lat = - float(os.path.basename(tile)[8:10])
            if os.path.basename(tile)[10] == 'E':
                lon = float(os.path.basename(tile)[11:14])
            else:
                lon = - float(os.path.basename(tile)[11:14])

            print('adding ' + tile)

            lat0 = coordinates.lat0 + coordinates.dlat * coordinates.first_line
            lon0 = coordinates.lon0 + coordinates.dlon * coordinates.first_pixel

            latlim = [lat0, lat0 + coordinates.dlat * (coordinates.shape[0] - 1)]
            lonlim = [lon0, lon0 + coordinates.dlon * (coordinates.shape[1] - 1)]

            # Find the coordinates of the part of the tile that should be written to the output data.
            t_latlim = [max(lat, latlim[0]), min(lat + step_lat, latlim[1])]
            t_lonlim = [max(lon, lonlim[0]), min(lon + step_lon, lonlim[1])]
            t_latid = [tile_shape[0] - int(round((t_latlim[0] - lat) / s_size)),
                       tile_shape[0] - (int(round((t_latlim[1] - lat) / s_size)) + 1)]
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

        return outputdata

    @staticmethod
    def create_geoid(coordinates, egm_96_file, download=True):
        """
        Creates a egm96 geoid to correct input dem data to ellipsoid value.

        :param CoordinateSystem coordinates: Coordinate system of grid
        :param str egm_96_file: Filename of egm96 geoid grid. If not available it will be downloaded.
        :return: geoid grid for coordinate system provided
        :rtype: np.ndarray
        """

        # (For this purpose the grid is downloaded from:
        # http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/binarygeoid.html

        if not os.path.exists(egm_96_file):
            # Download egm96 file
            if download:
                command = 'wget http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/WW15MGH.DAC -O ' + egm_96_file
                os.system(command)
            else:
                print('Cannot download http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/WW15MGH.DAC to '
                      + egm_96_file + ' ,please try to do it manually and try again.')

        # Load data
        egm96 = np.fromfile(egm_96_file, dtype='>i2').reshape((721, 1440)).astype('float32')
        egm96 = np.concatenate((egm96[:, 721:], egm96[:, :721], egm96[:, 721][:, None]), axis=1)
        lats = np.linspace(-90, 90, 721)
        lons = np.linspace(-180, 180, 1441)
        egm96_interp = RectBivariateSpline(lats, lons, egm96)

        if coordinates.grid_type == 'geographic':
            lats = coordinates.lat0 + (np.arange(coordinates.shape[0]) + coordinates.first_line) * coordinates.dlat
            lons = coordinates.lon0 + (np.arange(coordinates.shape[1]) + coordinates.first_pixel) * coordinates.dlon

            lons[lons < -180] = lons[lons < -180] + 360
            lons[lons > 180] = lons[lons > 180] - 360

            egm96_grid = np.transpose(egm96_interp(lons, lats))

        elif coordinates.grid_type == 'projection':

            ys = coordinates.y0 + (np.arange(coordinates.shape[0]) + coordinates.first_line) * coordinates.dy
            xs = coordinates.x0 + (np.arange(coordinates.shape[1]) + coordinates.first_pixel) * coordinates.dx
            x, y = np.meshgrid(xs, ys)
            lats, lons = coordinates.proj2ell(np.ravel(x), np.ravel(y))
            del x, y

            lons[lons < -180] = lons[lons < -180] + 360
            lons[lons > 180] = lons[lons > 180] - 360

            egm96_grid = np.reshape(egm96_interp(lons, lats, grid=False), coordinates.shape)
        else:
            print('Radar grid inputs cannot be used to interpolate geoid')

        return egm96_grid / 100
