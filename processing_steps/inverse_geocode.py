"""
This script is mainly used to fit a DEM to the line and pixel coordinates of a specific orbit.
To do so we follow the next 3 steps:
1. Based on image metadata we load the dem.
2. For irregular lat/lon grids, we load lat/lon files. / For regular lat/lon grids we calculate it from metadata
3. If needed, we correct the height grid for the geoid based on EGM96
4. Calculate the xyz cartesian coordinates for every pixel from latitude, longitude and height using the WGS84 ellipsoid
5. Finally the xyz coordinates are used to find line and pixel coordinates of the grid.

This steps follows the create SRTM dem function or load the external DEM function.

The main function is called inverse geocoding, because the performed steps are also done during geocoding but in the
opposite order, where the line and pixel coordinates are converted to lat/lon/height coordinates via cartesian coordi-
nates.

The results of this step is a grid of line and pixel coordinates

"""

from orbit_dem_functions.orbit_coordinates import OrbitCoordinates
from image_data import ImageData
from collections import OrderedDict
import numpy as np
import logging
import os


class InverseGeocode(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, s_lin=0, s_pix=0, lines=0, resolution='SRTM3'):
        # There are three options for processing:

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.meta = meta

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.resolution = resolution
        dem_key = 'Dem_' + self.resolution

        self.shape = np.array(self.meta.data_sizes['import_dem'][dem_key])
        if lines != 0:
            self.shape = [np.minimum(lines, self.shape[0] - s_lin), self.shape[1] - s_pix]

        self.s_lin = s_lin
        self.s_pix = s_pix
        self.e_lin = self.s_lin + self.shape[0]
        self.e_pix = self.s_pix + self.shape[1]

        # And the output data
        self.dem = []
        self.lat = []
        self.lon = []
        self.x = []
        self.y = []
        self.z = []
        self.line = []
        self.pixel = []
        self.orbits = OrbitCoordinates(self.meta)

        # Load height and create lat/lon or load lat/lon
        self.dem = self.meta.image_load_data_memory('import_dem', self.s_lin, self.s_pix, self.shape, dem_key)

        if self.meta.processes['import_dem'][dem_key + '_regular_grid'] == 'True':
            start_lat = float(self.meta.processes['import_dem'][dem_key + '_latitude_start'])
            step_lat = float(self.meta.processes['import_dem'][dem_key + '_latitude_step'])
            start_lon = float(self.meta.processes['import_dem'][dem_key + '_longitude_start'])
            step_lon = float(self.meta.processes['import_dem'][dem_key + '_longitude_step'])

            self.lat = np.flip(np.linspace(start_lat, start_lat + step_lat *
                                           (self.meta.data_sizes['import_dem'][dem_key][0] - 1),
                                            self.meta.data_sizes['import_dem'][dem_key][0]),
                               axis=0)[self.s_lin:self.s_lin + self.shape[0], None] * np.ones((1, self.shape[1]))
            self.lon = np.linspace(start_lon + step_lon * self.s_pix,
                                   start_lon + step_lon * (self.s_pix + self.shape[1] - 1),
                                   self.shape[1])[None, :] * np.ones((self.shape[0], 1))

        elif self.meta.processes['import_dem'][dem_key + '_regular_grid'] == 'False':
            lat_key = 'Dem_lat_' + self.resolution
            lon_key = 'Dem_lon_' + self.resolution

            self.lat = self.meta.image_load_data_memory('import_dem', self.s_lin, self.s_pix, self.shape, lat_key)
            self.lon = self.meta.image_load_data_memory('import_dem', self.s_lin, self.s_pix, self.shape, lon_key)

    def __call__(self):

        if len(self.lat) == 0 or len(self.lon) == 0 or len(self.dem) == 0:
            print('Missing input data for inverse geocoding for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Here the actual geocoding is done.

            # Then calculate the x,y,z coordinates
            self.x, self.y, self.z = self.orbits.ell2xyz(np.ravel(self.lat), np.ravel(self.lon), np.ravel(self.dem))

            # Finally calculate the lat/lon coordinates
            self.line, self.pixel = self.orbits.xyz2lp(self.x, self.y, self.z)
            del self.x, self.y, self.z
            self.line = self.line.reshape(self.shape).astype(np.float32)
            self.pixel = self.pixel.reshape(self.shape).astype(np.float32)

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.resolution)
            self.meta.image_new_data_memory(self.line, 'inverse_geocode', self.s_lin, self.s_pix, file_type='Dem_line_' + self.resolution)
            self.meta.image_new_data_memory(self.pixel, 'inverse_geocode', self.s_lin, self.s_pix, file_type='Dem_pixel_' + self.resolution)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed geocoding for ' +
                              self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed geocoding for ' +
                  self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_output_files(meta, to_disk='', resolution='SRTM3'):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = ['Dem_line_' + resolution, 'Dem_pixel_' + resolution]

        for s in to_disk:
            meta.image_create_disk('inverse_geocode', s)

    def save_to_disk(self, to_disk=''):

        if not to_disk:
            to_disk = ['Dem_line_' + self.resolution, 'Dem_pixel_' + self.resolution]

        for s in to_disk:
            self.meta.image_memory_to_disk('inverse_geocode', s)

    @staticmethod
    def add_meta_data(meta, resolution='SRTM3'):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'inverse_geocode' in meta.processes.keys():
            meta_info = meta.processes['inverse_geocode']
        else:
            meta_info = OrderedDict()

        dem_dat = 'Dem_' + resolution

        for dat in ['Dem_line_' + resolution, 'Dem_pixel_' + resolution]:
            meta_info[dat + '_output_file'] = dat + '.raw'
            meta_info[dat + '_output_format'] = 'real4'

            meta_info[dat + '_size_in_latitude'] = meta.processes['import_dem'][dem_dat + '_size_in_latitude']
            meta_info[dat + '_size_in_longitude'] = meta.processes['import_dem'][dem_dat + '_size_in_longitude']
            meta_info[dat + '_latitude_start'] = meta.processes['import_dem'][dem_dat + '_latitude_start']
            meta_info[dat + '_longitude_start'] = meta.processes['import_dem'][dem_dat + '_longitude_start']
            meta_info[dat + '_latitude_step'] = meta.processes['import_dem'][dem_dat + '_latitude_step']
            meta_info[dat + '_longitude_step'] = meta.processes['import_dem'][dem_dat + '_longitude_step']

        meta.image_add_processing_step('inverse_geocode', meta_info)

    @staticmethod
    def processing_info(resolution='SRTM3'):

        # Information on this processing step
        input_dat = dict()
        input_dat['slave']['import_dem'] = ['Dem_' + resolution]

        output_dat = dict()
        output_dat['slave']['inverse_geocode'] = ['Dem_line_' + resolution, 'Dem_pixel_' + resolution]

        # Number of times input data is used in ram. Bit difficult here but 15 times is ok guess.
        mem_use = 10

        return input_dat, output_dat, mem_use
