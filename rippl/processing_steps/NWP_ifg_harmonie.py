# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class InterferogramHarmonie(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='',
                 out_coor=[],
                 slave='slave', master='master', coreg_master='coreg_master', ifg='ifg', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'harmonie_interferogram'
        self.output_info['image_type'] = 'ifg'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['harmonie_interferogram']
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave', 'master']
        self.input_info['process_types'] = ['harmonie_aps', 'harmonie_aps']
        self.input_info['file_types'] = ['harmonie_aps', 'harmonie_aps']

        self.input_info['polarisations'] = [polarisation, polarisation]
        self.input_info['data_ids'] = [data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['type_names'] = ['slave', 'master']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['master'] = master
        self.processing_images['coreg_master'] = coreg_master
        self.processing_images['ifg'] = ifg

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(InterferogramHarmonie, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        self['harmonie_interferogram'] = self['slave'] - self['master']
