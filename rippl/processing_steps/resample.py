# Try to do all calculations using numpy functions.
import numpy as np
from collections import OrderedDict

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_regular2irregular import Regural2irregular
from rippl.resampling.multilook_regular import MultilookRegular


class Resample(Process):  # Change this name to the one of your processing step.

    def __init__(self, in_coor=[], out_coor=[], resample_type='linear',
                 in_image_type='', in_process='', in_file_type='', in_polarisation='', in_data_id='',
                 slave='slave', coreg_master='coreg_master', overwrite=False):

        """
        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param str in_image_type: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.) for
                    the multilooked image
        :param str in_process: Which process outputs are we using as an input for the multilooked image
        :param str in_file_type: What are the exact outputs we use from these processes
        :param str in_polarisation: For which polarisation is it done. Leave empty if not relevant
        :param str in_data_id: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        # Check whether we need to do regular or irregular multilooking
        self.regular = False # MultilookRegular.check_same_coordinate_system(in_coor, out_coor)

        # Output data information
        self.output_info = dict()
        if not in_image_type:
            if isinstance(slave, ImageProcessingData):
                self.output_info['image_type'] = 'slave'
            else:
                self.output_info['image_type'] = 'coreg_master'

        # Output data information
        self.output_info['process_name'] = in_process
        self.output_info['polarisation'] = in_polarisation
        self.output_info['data_id'] = in_data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = [in_file_type]
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        if self.regular:
            self.input_info['image_types'] = [self.output_info['image_type']]
            self.input_info['process_types'] = [in_process]
            self.input_info['file_types'] = [in_file_type]
            self.input_info['polarisations'] = [in_polarisation]
            self.input_info['data_ids'] = [in_data_id]
            self.input_info['coor_types'] = ['in_coor']
            self.input_info['in_coor_types'] = ['']
            self.input_info['type_names'] = ['input_data']
        else:
            self.input_info['image_types'] = [self.output_info['image_type'], 'coreg_master', 'coreg_master']
            self.input_info['process_types'] = [in_process, 'reproject', 'reproject']
            self.input_info['file_types'] = [in_file_type, 'out_coor_lines', 'out_coor_pixels']
            self.input_info['polarisations'] = [in_polarisation, '', '']
            self.input_info['data_ids'] = [in_data_id, '', '']
            self.input_info['coor_types'] = ['in_coor', 'out_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', 'in_coor', 'in_coor']
            self.input_info['type_names'] = ['input_data', 'lines', 'pixels']

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master
        self.processing_images['slave'] = slave
        self.settings = dict()
        self.settings['resample_type'] = resample_type
        self.settings['buf'] = 3
        self.settings['out_irregular_grids'] = ['lines', 'pixels']

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(Resample, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Resampling of radar grid. For non radar grid this function is not advised as it uses

        :return:
        """

        in_block_coor = self.coordinate_systems['in_block_coor']

        # Change line/pixel coordinates to right value
        if in_block_coor.grid_type == 'radar_coordinates':
            lines = (self['lines'] - in_block_coor.first_line) / \
                                   (self.block_coor.multilook[0] / self.block_coor.oversample[0])
            pixels = (self['pixels'] - in_block_coor.first_pixel) / \
                                   (self.block_coor.multilook[1] / self.block_coor.oversample[1])
        else:
            lines = (self['lines'] - in_block_coor.first_line)
            pixels = (self['pixels'] - in_block_coor.first_pixel)

        # Init resampling
        resample = Regural2irregular(self.settings['resample_type'])
        self[self.output_info['file_types'][0]] = resample(self['input_data'], lines, pixels)
