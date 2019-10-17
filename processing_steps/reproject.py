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

from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.resampling.grid_transforms import GridTransforms


class Reproject(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], coreg_master=[], conversion_type='multilook', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData coreg_master: Coreg master image, used as the default for input and output for processing.
        """

        if conversion_type not in ['multilook', 'resample']:
            raise ValueError('Choose either multilook or resample as convesion type')

        self.conversion_type = conversion_type

        # If the boundaries of the output grid are not jet defined, define them.
        if out_coor.shape == [0, 0] or out_coor.shape == '':
            new_extend = CoorNewExtend(in_coor, out_coor)
            out_coor = new_extend.out_coor

        # In case of multilooking we want to know the coordinates in the output grid of all the pixels of the input
        # grid, while for resampling we want to know the coordinates in the input grid of the output pixels.
        if conversion_type == 'multilook':
            output_grid = in_coor
            output_grid_coordinates = out_coor
        elif conversion_type == 'resample':
            output_grid = out_coor
            output_grid_coordinates = in_coor

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'reproject'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['in_coor_lines', 'in_coor_pixels']
        self.output_info['data_types'] = ['real4', 'real4']

        # Input data information
        self.input_info = dict()
        if output_grid_coordinates.grid_type == 'radar_coordinates':
            self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master']
            self.input_info['process_types'] = ['geocode', 'geocode', 'geocode']
            self.input_info['file_types'] = ['X', 'Y', 'Z']
            self.input_info['data_types'] = ['real4', 'real4', 'real4']
            self.input_info['polarisations'] = ['', '', '']
            self.input_info['data_ids'] = [data_id, data_id, data_id]
            self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', '', '']
            self.input_info['type_names'] = ['X', 'Y', 'Z']
        else:
            self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master']
            self.input_info['process_types'] = ['dem', 'geocode', 'geocode']
            self.input_info['file_types'] = ['dem', 'lat', 'lon']
            self.input_info['data_types'] = ['real4', 'real4', 'real4']
            self.input_info['polarisations'] = ['', '', '']
            self.input_info['data_ids'] = [data_id, data_id, data_id]
            self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', '', '']
            self.input_info['type_names'] = ['dem', 'lat', 'lon']

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = output_grid_coordinates
        self.coordinate_systems['out_coor'] = output_grid

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master
        self.settings = dict()

    def init_super(self):

        super(Reproject, self).__init__(
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

        transform = GridTransforms(self.in_coor, self.out_coor)

        if self.out_coor.grid_type == 'radar_coordinates':
            if self.in_coor.grid_type == 'radar_coordinates':
                transform.add_xyz(self['X'], self['Y'], self['Z'])
            elif self.in_coor.grid_type in ['projection', 'geographic']:
                transform.add_dem(self['dem'])
                transform.add_lat_lon(self['lat'], self['lon'])

        self['in_coor_lines'], self['in_coor_pixels'] = transform()
