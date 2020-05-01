# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.processing_steps.deramp import Deramp


class CorrectPhases(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', out_coor=[], slave='slave', overwrite=False):
        """
        This function deramps the ramped data from TOPS mode to a deramped data. Input data of this function should
        be a radar coordinates grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'correct_phases'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['phase_corrected']
        self.output_info['data_types'] = ['complex_short']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave', 'slave', 'slave']
        self.input_info['process_types'] = ['resample', 'calc_reramp', 'calc_earth_topo_phase']
        self.input_info['file_types'] = ['resampled', 'ramp', 'earth_topo_phase']
        self.input_info['polarisations'] = [polarisation, '', '']
        self.input_info['data_ids'] = [data_id, '', '']
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['type_names'] = ['input_data', 'ramp', 'earth_topo_phase']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(CorrectPhases, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Because of the change in baselines, every pixel is shifted in azimuth time. This also influences the delay time
        to the same point on the ground for different orbits. Therefore we correct here for this effect using the
        geometrical shift in range.

        :return:
        """

        ramp = np.exp(-1j * (-self['ramp'] + self['earth_topo_phase'])).astype(np.complex64)

        # Finally calced the deramped image.
        self['phase_corrected'] = self['input_data'] * np.conjugate(ramp)
