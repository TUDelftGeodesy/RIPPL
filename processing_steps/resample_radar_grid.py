# Try to do all calculations using numpy functions.
import numpy as np
from collections import OrderedDict

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_regular2irregular import Resample


class ResampleRadarGrid(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', resample_type='4p_cubic',
                 in_coor=[], out_coor=[],
                 slave='slave', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'resample'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['resampled']
        self.output_info['data_types'] = ['complex_real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave', 'slave', 'slave']
        self.input_info['process_types'] = ['deramp', 'geometric_coregistration', 'geometric_coregistration']
        self.input_info['file_types'] = ['deramped', 'coreg_lines', 'coreg_pixels']
        self.input_info['polarisations'] = [polarisation, '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id]
        self.input_info['coor_types'] = ['in_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['type_names'] = ['input_data', 'coreg_lines', 'coreg_pixels']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = in_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['resample_type'] = resample_type

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(ResampleRadarGrid, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def load_irregular_grids(self):
        """
        Load the coordinates the grid has to be resampled to.

        :return:
        """

        # Define the irregular grid for output. This is needed if you want to calculate in blocks.
        if len(self.in_images['coreg_lines'].disk['data']) > 0:
            s_lin = self.block_coor.first_line - self.coordinate_systems['out_coor'].first_line
            s_pix = self.block_coor.first_pixel - self.block_coor.first_pixel
            e_lin = s_lin + self.block_coor.shape[0]
            e_pix = s_pix + self.block_coor.shape[1]

            self.out_irregular_grids = [self.in_images['coreg_lines'].disk['data'][s_lin:e_lin, s_pix:e_pix],
                                        self.in_images['coreg_pixels'].disk['data'][s_lin:e_lin, s_pix:e_pix]]
        elif len(self.in_images['coreg_lines'].memory['data']) > 0:
            self.out_irregular_grids = [self.in_images['coreg_lines'].memory['data'],
                                        self.in_images['coreg_pixels'].memory['data']]
        else:
            raise FileNotFoundError('Data for irregular grids not available.')
        self.in_irregular_grids = [None]

    def process_calculations(self):
        """
        Resampling of radar grid. For non radar grid this function is not advised as it uses

        :return:
        """

        # Init resampling
        resample = Resample(self.settings['resample_type'])
        in_block_coor = self.coordinate_systems['in_block_coor']

        # Change line/pixel coordinates to right value
        lines = (self['coreg_lines'] - in_block_coor.first_line) / \
                               (self.block_coor.multilook[0] / self.block_coor.oversample[0])
        pixels = (self['coreg_pixels'] - in_block_coor.first_pixel) / \
                               (self.block_coor.multilook[1] / self.block_coor.oversample[1])

        self['resampled'] = resample(self['input_data'], lines, pixels)
