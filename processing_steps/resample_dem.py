# Try to do all calculations using numpy functions.
import numpy as np
from scipy.interpolate import RectBivariateSpline

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_irregular2regular import Irregular2Regular


class ResampleDem(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], resample_type='delaunay',
                 in_coor_types=[], in_processes=[], in_file_types=[], in_data_ids=[],
                 coreg_master=[], overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined in_coor. Options are 'in_coor', 'out_coor' or anything else defined in the
                coordinate system input.

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
        self.output_info['process_name'] = 'dem'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['dem']
        self.output_info['data_types'] = ['real4']

        # Input data information
        if resample_type == 'delaunay':
            self.input_info = dict()
            self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master']
            self.input_info['process_types'] = ['dem', 'inverse_geocode', 'inverse_geocode']
            self.input_info['file_types'] = ['dem', 'lines', 'pixels']
            self.input_info['data_types'] = ['real4', 'real4', 'real4']
            self.input_info['polarisations'] = ['', '', '']
            self.input_info['data_ids'] = [data_id, data_id, data_id]
            self.input_info['coor_types'] = ['in_coor', 'in_coor', 'in_coor']
            self.input_info['in_coor_types'] = ['', '', '']
            self.input_info['type_names'] = ['input_dem', 'lines', 'pixels']
        else:
            self.input_info = dict()
            self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master']
            self.input_info['process_types'] = ['dem', 'reproject', 'reproject']
            self.input_info['file_types'] = ['dem', 'in_coor_lines', 'in_coor_pixels']
            self.input_info['data_types'] = ['real4', 'real4', 'real4']
            self.input_info['polarisations'] = ['', '', '']
            self.input_info['data_ids'] = [data_id, data_id, data_id]
            self.input_info['coor_types'] = ['in_coor', 'out_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', 'in_coor', 'in_coor']
            self.input_info['type_names'] = ['input_dem', 'lines', 'pixels']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        super(ResampleDem, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

        if self.process_finished:
            return

        # Define the irregular grid for input. This is needed if you want to calculate in blocks.
        self.in_irregular_grids = [self.in_images['lines'].disk['data'],
                                   self.in_images['pixels'].disk['data']]

    def process_calculations(self):
        """
        Calculate radar DEM using the

        :return:
        """

        if self.out_coor.grid_type == 'radar_coordinates':
            resample = Irregular2Regular(self['lines'], self['pixels'], self['input_dem'], coordinates=self.block_coor)
            self['dem'] = resample()
        else:
            x_coor = np.arange(self.block_coor.shape[1]) - (self.block_coor.first_pixel - self.out_coor.first_pixel)
            y_coor = np.arange(self.block_coor.shape[0]) - (self.block_coor.first_line - self.out_coor.first_line)

            bilinear_interp = RectBivariateSpline(x_coor, y_coor, self['input_dem'])
            self['dem'] = bilinear_interp.ev(self['pixels'], self['lines'])
