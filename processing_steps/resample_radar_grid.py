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
                 coor_in=[], coor_out=[],
                 in_coor_types=[], in_processes=[], in_file_types=[], in_polarisations=[], in_data_ids=[],
                 slave=[], overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem coor_in: Coordinate system of the input grids.
        :param CoordinateSystem coor_out: Coordinate system of output grids, if not defined the same as coor_in
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined coor_in. Options are 'coor_in', 'coor_out' or anything else defined in the
                coordinate system input.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        self.process_name = 'resample'
        file_types = ['resampled']
        data_types = ['complex_short']
        self.settings = OrderedDict()
        self.settings['resample_type'] = resample_type

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """
        in_image_types = ['slave', 'slave', 'slave']
        if len(in_coor_types) == 0:
            in_coor_types = ['coor_in', 'coor_out', 'coor_out']
        if len(in_data_ids) == 0:
            in_data_ids = ['none', '', '']
        if len(in_polarisations) == 0:
            in_polarisations = [polarisation, 'none', 'none']
        if len(in_processes) == 0:
            in_processes = ['deramp', 'geometric_coregistration', 'geometric_coregistration']
        if len(in_file_types) == 0:
            in_file_types = ['deramped', 'lines', 'pixels']

        in_type_names = ['input_data', 'lines', 'pixels']

        # Initialize process
        super(ResampleRadarGrid, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id,
                       settings=self.settings,
                       polarisation=polarisation,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       coor_out=coor_out,
                       in_type_names=in_type_names,
                       in_coor_types=in_coor_types,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       slave=slave,
                       overwrite=overwrite)
        if self.process_finished:
            return

        # Define the irregular grid for output. This is needed if you want to calculate in blocks.
        self.out_irregular_grids = [self.in_images['lines'].disk['data'],
                                   self.in_images['pixels'].disk['data']]


    def process_calculations(self):
        """
        Resampling of radar grid. For non radar grid this function is not advised as it uses

        :return:
        """

        # Init resampling
        resample = Resample(self.settings['resample_type'])

        # Change line/pixel coordinates to right value
        lines = (self['lines'] + (self.coor_out.first_line - self.coor_in.first_line) - self.block_coor.first_line) / \
                               (self.block_coor.multilook[0] / self.block_coor.oversample[0])
        pixels = (self['pixels'] + (self.coor_out.first_pixel - self.coor_in.first_pixel) - self.block_coor.first_pixel) / \
                               (self.block_coor.multilook[1] / self.block_coor.oversample[1])

        self['resampled'] = resample(self['input_data'], lines, pixels)
