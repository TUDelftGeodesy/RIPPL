# Try to do all calculations using numpy functions.
import numpy as np
import copy

# Import the parent class Process for processing steps.
from rippl.meta_data.multilook_process import MultilookProcess
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.resampling.multilook_irregular import MultilookIrregular
from rippl.resampling.multilook_regular import MultilookRegular

class AASRAmplitudeMultilook(MultilookProcess):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='',
                 in_coor=[], out_coor=[],
                 coreg_master='coreg_master', slave='slave', overwrite=False, batch_size=1000000):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        """

        # Check whether we need to do regular or irregular multilooking
        self.regular = MultilookRegular.check_same_coordinate_system(in_coor, out_coor)
        # If the grid size of the output grid are not defined yet, they are calculated here.

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'AASR_amplitude_multilook'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['ambiguities_calibrated_amplitude', 'ambiguities_calibrated_amplitude_db', 'AASR_db']
        self.output_info['data_types'] = ['real4', 'real4', 'real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave', 'slave', 'coreg_master']
        self.input_info['process_types'] = ['combined_ambiguities', 'calibrated_amplitude', 'radar_ray_angles']
        self.input_info['file_types'] = ['combined_ambiguities', 'calibrated_amplitude_db', 'incidence_angle']

        self.input_info['polarisations'] = [polarisation, polarisation, '']
        self.input_info['data_ids'] = [data_id, data_id, '']
        self.input_info['coor_types'] = ['in_coor', 'out_coor', 'in_coor']
        self.input_info['in_coor_types'] = ['', 'in_coor', '']
        self.input_info['type_names'] = ['complex_data', 'calibrated_amplitude_db', 'incidence_angle']

        if not self.regular:
            self.input_info['image_types'].extend(['coreg_master', 'coreg_master'])
            self.input_info['process_types'].extend(['reproject', 'reproject'])
            self.input_info['file_types'].extend(['in_coor_lines', 'in_coor_pixels'])
            self.input_info['polarisations'].extend(['', ''])
            self.input_info['data_ids'].extend(['', ''])
            self.input_info['coor_types'].extend(['in_coor', 'in_coor'])
            self.input_info['in_coor_types'].extend(['out_coor', 'out_coor'])
            self.input_info['type_names'].extend(['lines', 'pixels'])

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # Image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['out_irregular_grids'] = ['lines', 'pixels']
        self.settings['memory_data'] = False
        self.settings['buf'] = 0
        self.settings['multilooked_grids'] = ['ambiguities_calibrated_amplitude']
        self.batch_size = batch_size

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(AASRAmplitudeMultilook, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def __call__(self):

        super(AASRAmplitudeMultilook, self).__call__(memory_in=False)

    def before_multilook_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        beta_0 = 237.0  # TODO Read in from meta data files.

        self['ambiguities_calibrated_amplitude'] = np.abs(self['complex_data'])**2 * np.sin(self['incidence_angle'] / 180 * np.pi) \
                                                   / (beta_0**2)

    def after_multilook_calculations(self):

        valid_pixels = (self['ambiguities_calibrated_amplitude'] != 0) * ~np.isnan(self['ambiguities_calibrated_amplitude'])

        # Calculate the db values.
        self['ambiguities_calibrated_amplitude_db'][valid_pixels] = 10 * np.log10(self['ambiguities_calibrated_amplitude'][valid_pixels])
        self['AASR_db'][valid_pixels] = self['ambiguities_calibrated_amplitude_db'][valid_pixels] - self['calibrated_amplitude_db'][valid_pixels]
