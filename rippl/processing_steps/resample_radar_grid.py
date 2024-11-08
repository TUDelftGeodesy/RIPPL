
# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_regular2irregular import Regural2irregular


class ResampleRadarGrid(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', resample_type='4p_cubic',
                 in_coor=[], out_coor=[],
                 secondary_slc='secondary_slc', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'resample'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['resampled']
        self.output_info['data_types'] = ['complex32']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['secondary_slc', 'secondary_slc', 'secondary_slc']
        self.input_info['process_names'] = ['deramp', 'geometric_coregistration', 'geometric_coregistration']
        self.input_info['file_names'] = ['deramped', 'coreg_lines', 'coreg_pixels']
        self.input_info['polarisations'] = [polarisation, '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id]
        self.input_info['coor_types'] = ['in_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['aliases_processing'] = ['input_data', 'coreg_lines', 'coreg_pixels']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = in_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['resample_type'] = resample_type
        self.settings['in_coor'] = dict()
        self.settings['in_coor']['buffer'] = 10

    def process_calculations(self):
        """
        Resampling of radar grid. For non radar grid this function is not advised as it uses

        :return:
        """

        # Init resampling
        resample = Regural2irregular(self.settings['resample_type'])
        in_coor_chunk = self.coordinate_systems['in_coor_chunk']

        # Change line/pixel coordinates to right value
        lines = (self['coreg_lines'] - in_coor_chunk.first_line) / \
                               (self.coordinate_systems['out_coor_chunk'].multilook[0] / self.coordinate_systems['out_coor_chunk'].oversample[0])
        pixels = (self['coreg_pixels'] - in_coor_chunk.first_pixel) / \
                               (self.coordinate_systems['out_coor_chunk'].multilook[1] / self.coordinate_systems['out_coor_chunk'].oversample[1])

        self['resampled'] = resample(self['input_data'], lines, pixels)
