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


class HeightToPhase(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', out_coor=[], slave='slave', coreg_master='coreg_master', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'height_to_phase'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['height_to_phase']
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master', 'slave']
        self.input_info['process_types'] = ['radar_ray_angles', 'baseline']
        self.input_info['file_types'] = ['incidence_angle', 'perpendicular_baseline']
        self.input_info['polarisations'] = ['', '']
        self.input_info['data_ids'] = [data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['type_names'] = ['incidence', 'baseline']

        if out_coor.grid_type != 'radar_coordinates':
            self.input_info['image_types'].extend(['coreg_master', 'coreg_master', 'coreg_master'])
            self.input_info['process_types'].extend(['geocode', 'geocode', 'geocode'])
            self.input_info['file_types'].extend(['X', 'Y', 'Z'])
            self.input_info['polarisations'].extend(['', '', ''])
            self.input_info['data_ids'].extend([data_id, data_id, data_id])
            self.input_info['coor_types'].extend(['out_coor', 'out_coor', 'out_coor'])
            self.input_info['in_coor_types'].extend(['', '', ''])
            self.input_info['type_names'].extend(['X', 'Y', 'Z'])

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(HeightToPhase, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        readfile_slave = self.processing_images['slave'].readfiles['original']
        wavelength = readfile_slave.wavelength
        sol = 299792458

        if self.coordinate_systems['in_block_coor'].grid_type == 'radar_coordinates':

            coordinates = self.coordinate_systems['in_block_coor']
            coordinates.create_radar_lines()
            R = (coordinates.ra_time + coordinates.interval_pixels * coordinates.ra_step) * sol

            self['height_to_phase'] = self['baseline'] / (R[None, :] * np.sin(np.deg2rad(self['incidence'])))

        else:
            orbit_slave = self.processing_images['slave'].find_best_orbit('original')

            # Now initialize the orbit estimation.
            coordinates = CoordinateSystem()
            coordinates.create_radar_coordinates()
            coordinates.load_readfile(readfile=readfile_slave)

            xyz = np.vstack((np.ravel(self['X'])[None, :], np.ravel(self['Y'])[None, :], np.ravel(self['Z'])[None, :]))
            orbit_coordinates = OrbitCoordinates(coordinates=coordinates, orbit=orbit_slave)
            lines, pixels = orbit_coordinates.xyz2lp(xyz)

            R = (coordinates.ra_time + np.reshape(pixels, self.block_coor.shape) * coordinates.ra_step) * sol

            self['height_to_phase'] = self['baseline'] / (R * np.sin(np.deg2rad(self['incidence'])))
