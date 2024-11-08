
# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_regular2irregular import Regural2irregular


class Resample(Process):  # Change this name to the one of your processing step.

    def __init__(self, in_coor=[], out_coor=[], resample_type='linear',
                 image_type='', process='', file_type='', polarisation='', data_id='',
                 secondary_slc='secondary_slc', reference_slc='reference_slc', overwrite=False,
                 buffer=3, expected_min_height=0, expected_max_height=500, same_coordinate_system=False):

        """
        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        if not image_type:
            if isinstance(secondary_slc, ImageProcessingData):
                self.output_info['image_type'] = 'secondary_slc'
            else:
                self.output_info['image_type'] = 'reference_slc'

        # Output data information
        self.output_info['process_name'] = process
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = [file_type]
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        if same_coordinate_system:
            self.input_info['image_types'] = [self.output_info['image_type']]
            self.input_info['process_names'] = [process]
            self.input_info['file_names'] = [file_type]
            self.input_info['polarisations'] = [polarisation]
            self.input_info['data_ids'] = [data_id]
            self.input_info['coor_types'] = ['in_coor']
            self.input_info['in_coor_types'] = ['']
            self.input_info['aliases_processing'] = ['input_data']
        else:
            self.input_info['image_types'] = [self.output_info['image_type'], 'reference_slc', 'reference_slc']
            self.input_info['process_names'] = [process, 'grid_transform', 'grid_transform']
            self.input_info['file_names'] = [file_type, 'resample_lines', 'resample_pixels']
            self.input_info['polarisations'] = [polarisation, '', '']
            self.input_info['data_ids'] = [data_id, '', '']
            self.input_info['coor_types'] = ['in_coor', 'out_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', 'in_coor', 'in_coor']
            self.input_info['aliases_processing'] = ['input_data', 'lines', 'pixels']

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['reference_slc'] = reference_slc
        self.processing_images['secondary_slc'] = secondary_slc
        self.settings = dict()
        self.settings['resample_type'] = resample_type
        self.settings['regular'] = same_coordinate_system
        self.settings['in_coor'] = dict()
        self.settings['in_coor']['buffer'] = buffer
        self.settings['in_coor']['min_height'] = expected_min_height
        self.settings['in_coor']['max_height'] = expected_max_height

    def process_calculations(self):
        """
        Resampling of radar grid. For non radar grid this function is not advised as it uses

        :return:
        """

        in_coor_chunk = self.coordinate_systems['in_coor_chunk']

        # Change line/pixel coordinates to right value
        if self.settings['regular']:
            if in_coor_chunk.grid_type == 'radar_coordinates':
                lines = (self['lines'] - in_coor_chunk.first_line) / \
                                       (self.coordinate_systems['out_coor_chunk'].multilook[0] / self.coordinate_systems['out_coor_chunk'].oversample[0])
                pixels = (self['pixels'] - in_coor_chunk.first_pixel) / \
                                       (self.coordinate_systems['out_coor_chunk'].multilook[1] / self.coordinate_systems['out_coor_chunk'].oversample[1])
            else:
                lines = (self['lines'] - in_coor_chunk.first_line)
                pixels = (self['pixels'] - in_coor_chunk.first_pixel)

            # Init resampling
            resample = Regural2irregular(self.settings['resample_type'])
            self[self.output_info['file_names'][0]] = resample(self['input_data'], lines, pixels)
        else:
            # Do a bilinear spline interpolation #TODO implement bilinear regular to regular interpolation
            lines = (self['lines'] - in_coor_chunk.first_line)
            pixels = (self['pixels'] - in_coor_chunk.first_pixel)

            resample = Regural2irregular(self.settings['resample_type'])
            self[self.output_info['file_names'][0]] = resample(self['input_data'], lines, pixels)
