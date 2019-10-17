# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class Geocode(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coordinates=[], coreg_master=[], overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param CoordinateSystem coordinates: Coordinate system of the input grids.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'geocode'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['X', 'Y', 'Z', 'lat', 'lon']
        self.output_info['data_types'] = ['real8', 'real8', 'real8', 'real4', 'real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master']
        self.input_info['process_types'] = ['dem']
        self.input_info['file_types'] = ['dem']
        self.input_info['data_types'] = ['real4']
        self.input_info['polarisations'] = ['']
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['out_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['type_names'] = ['dem']

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = coordinates
        self.coordinate_systems['out_coor'] = coordinates

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master
        self.settings = dict()

    def init_super(self):

        super(Geocode, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        With geocoding we find the XYZ values using the DEM height and line/pixel spacing. From the XYZ values the
        latitude and longitude can directly be derived too.

        :return:
        """

        if self.out_coor.grid_type == 'radar_coordinates':
            # Get orbit
            orbit = self.processing_images['coreg_master'].find_best_orbit('original')
            self.block_coor.create_radar_lines()
            orbit_interp = OrbitCoordinates(coordinates=self.block_coor, orbit=orbit)

            # Add DEM values
            orbit_interp.height = self['dem']
            orbit_interp.lp_time()

            # Calculate cartesian coordinates
            orbit_interp.lph2xyz()
            self['X'] = np.reshape(orbit_interp.xyz[0, :], self.block_coor.shape)
            self['Y'] = np.reshape(orbit_interp.xyz[1, :], self.block_coor.shape)
            self['Z'] = np.reshape(orbit_interp.xyz[2, :], self.block_coor.shape)

            # Calculate lat/lon values
            orbit_interp.xyz2ell()
            self['lat'] = np.reshape(orbit_interp.lat, self.block_coor.shape)
            self['lon'] = np.reshape(orbit_interp.lon, self.block_coor.shape)

        else:

            if self.out_coor.grid_type == 'projection':

                X, Y = self.block_coor.create_xy_grid()
                lat, lon = self.block_coor.proj2ell(np.ravel(X), np.ravel(Y))

                self['lat'] = np.reshape(lat, self.block_coor.shape)
                self['lon'] = np.reshape(lon, self.block_coor.shape)
            elif self.out_coor.grid_type == 'geographic':

                self['lat'], self['lon'] = self.block_coor.create_latlon_grid()

            xyz = OrbitCoordinates.ell2xyz(self['lat'], self['lon'])

            self['X'] = np.reshape(xyz[0, :], self.block_coor.shape)
            self['Y'] = np.reshape(xyz[1, :], self.block_coor.shape)
            self['Z'] = np.reshape(xyz[2, :], self.block_coor.shape)
