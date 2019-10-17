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


class InverseGeocode(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coordinates=[], coor_ref=[], dem_type='SRTM1',
                 in_processes=[], in_file_types=[], in_data_ids=[], coreg_master=[], overwrite=False):

        """
        This function is used to find the line/pixel coordinates of the dem grid. These can later on be used to
        calculate the radar dem grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem coordinates: Coordinate system of the input grids. If dem_type is given this parameter can
                be left empty!
        :param CoordinateSystem coor_ref: Coordinate system of the radar grid we should convert to
        :param str dem_type: In the case we want to use an imported DEM a dem_type should be defined. At the moment
                automatic generation of SRTM1 and SRTM3 are implemented, but also other DEMs can be imported manually.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'inverse_geocode'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['lines', 'pixels']
        self.output_info['data_types'] = ['real4', 'real4']

        # Input data information
        if dem_type:
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
        else:
            self.input_info = dict()
            self.input_info['image_types'] = ['coreg_master']
            self.input_info['process_types'] = ['dem', 'geocode', 'geocode', 'geocode']
            self.input_info['file_types'] = ['dem', 'X', 'Y', 'Z']
            self.input_info['data_types'] = ['real4', 'real4', 'real4', 'real4']
            self.input_info['polarisations'] = ['', '', '', '']
            self.input_info['data_ids'] = [data_id, data_id, data_id, data_id]
            self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', '', '', '']
            self.input_info['type_names'] = ['dem', 'X', 'Y', 'Z']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = coordinates
        self.coordinate_systems['in_coor'] = coordinates
        self.coor_ref = coor_ref

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.dem_type = dem_type

    def init_super(self):

        super(InverseGeocode, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        We calculate the line and pixel coordinates of the DEM grid as a preparation for the conversion to a radar grid

        :return:
        """

        processing_data = self.processing_images['coreg_master']

        # Evaluate the orbit and create orbit coordinates object.
        orbit_interp = OrbitCoordinates(self.coor_ref)

        if self.dem_type:
            if self.in_coor.grid_type == 'geographic':
                lats, lons = self.block_coor.create_latlon_grid()
            elif self.in_coor.grid_type == 'projection':
                x, y = self.block_coor.create_xy_grid()
                lats, lons = self.block_coor.proj2ell(x, y)

            xyz = OrbitCoordinates.ell2xyz(np.ravel(lats), np.ravel(lons), np.ravel(self['dem']))
        else:
            xyz = np.vstack((np.ravel(self['X'])[None, :], np.ravel(self['Y'])[None, :], np.ravel(self['Z'])[None, :]))

        self.ml_step = np.array(self.coor_ref.multilook) / np.array(self.coor_ref.oversample)

        lines, pixels = orbit_interp.xyz2lp(xyz)
        self['lines'] = np.reshape((lines - self.coor_ref.first_line) / self.ml_step[0], self.block_coor.shape)
        self['pixels'] = np.reshape((pixels  - self.coor_ref.first_pixel) / self.ml_step[1], self.block_coor.shape)
        