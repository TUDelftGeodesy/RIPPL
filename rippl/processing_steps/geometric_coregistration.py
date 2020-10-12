# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class GeometricCoregistration(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], dem_type='SRTM1',
                 in_image_types=[], in_processes=[], in_file_types=[], in_data_ids=[],
                 slave='slave', coreg_master='coreg_master', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'geometric_coregistration'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['coreg_lines', 'coreg_pixels']
        self.output_info['data_types'] = ['real8', 'real8']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master']
        self.input_info['process_types'] = ['geocode', 'geocode', 'geocode']
        self.input_info['file_types'] = ['X', 'Y', 'Z']
        self.input_info['polarisations'] = ['', '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['type_names'] = ['X_coreg', 'Y_coreg', 'Z_coreg']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = in_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(GeometricCoregistration, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Calculate the lines and pixels using the orbit of the slave and coreg_master image.

        :return:
        """

        # Get the orbits
        orbit_slave = self.processing_images['slave'].find_best_orbit('original')
        readfile_slave = self.processing_images['slave'].readfiles['original']

        # Now initialize the orbit estimation.
        coordinates = CoordinateSystem()
        coordinates.create_radar_coordinates()
        coordinates.load_readfile(readfile=readfile_slave)
        orbit_interp = OrbitCoordinates(coordinates=coordinates, orbit=orbit_slave)
        xyz = np.vstack((np.ravel(self['X_coreg'])[None, :], np.ravel(self['Y_coreg'])[None, :], np.ravel(self['Z_coreg'])[None, :]))
        lines, pixels = orbit_interp.xyz2lp(xyz)

        self['coreg_lines'] = np.reshape(lines, self.block_coor.shape)
        self['coreg_pixels'] = np.reshape(pixels, self.block_coor.shape)
