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

class CalibratedAmplitudeMultilook(MultilookProcess):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', no_of_looks=False,
                 in_coor=[], out_coor=[], master_image=False, db_only=True, resampled=True, no_line_pixel_input=False,
                 coreg_master='coreg_master', slave='slave', overwrite=False, batch_size=1000000, oceans=True):

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
        self.output_info['process_name'] = 'calibrated_amplitude'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        if db_only:
            self.output_info['file_types'] = ['calibrated_amplitude_db']
            self.output_info['data_types'] = ['real4']
        else:
            self.output_info['file_types'] = ['calibrated_amplitude', 'calibrated_amplitude_db']
            self.output_info['data_types'] = ['real4', 'real4']
        self.db_only = db_only

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave']
        if master_image or not resampled:
            self.input_info['process_types'] = ['crop']
            self.input_info['file_types'] = ['crop']
        else:
            self.input_info['process_types'] = ['correct_phases']
            self.input_info['file_types'] = ['phase_corrected']

        self.input_info['polarisations'] = [polarisation]
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['in_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['type_names'] = ['complex_data']

        if not self.regular and not no_line_pixel_input:
            self.input_info['image_types'].extend(['coreg_master', 'coreg_master', 'coreg_master'])
            self.input_info['process_types'].extend(['reproject', 'reproject', 'radar_ray_angles'])
            self.input_info['file_types'].extend(['in_coor_lines', 'in_coor_pixels', 'incidence_angle'])
            self.input_info['polarisations'].extend(['', '', ''])
            self.input_info['data_ids'].extend(['', '', ''])
            self.input_info['coor_types'].extend(['in_coor', 'in_coor', 'in_coor'])
            self.input_info['in_coor_types'].extend(['out_coor', 'out_coor', ''])
            self.input_info['type_names'].extend(['lines', 'pixels', 'incidence_angle'])
        elif not self.regular and no_line_pixel_input and not oceans:
            self.input_info['image_types'].extend(['coreg_master'])
            self.input_info['process_types'].extend(['dem'])
            self.input_info['file_types'].extend(['dem'])
            self.input_info['polarisations'].extend(['',])
            self.input_info['data_ids'].extend([''])
            self.input_info['coor_types'].extend(['in_coor',])
            self.input_info['in_coor_types'].extend([''])
            self.input_info['type_names'].extend(['dem'])
        else:
            no_line_pixel_input = True

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
        self.settings['multilooked_grids'] = ['calibrated_amplitude']
        self.settings['memory_data'] = False
        self.settings['buf'] = 0
        self.settings['no_line_pixel_input'] = no_line_pixel_input
        self.settings['regular'] = self.regular
        if not oceans:
            self.settings['dem'] = True
        self.batch_size = batch_size
        self.add_no_of_looks(no_of_looks)

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(CalibratedAmplitudeMultilook, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def before_multilook_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        beta_0 = 237.0  # TODO Read in from meta data files.

        self['calibrated_amplitude'] = np.abs(self['complex_data'])**2 * np.sin(self['incidence_angle'] / 180 * np.pi) \
                                                   / (beta_0 ** 2)

    def after_multilook_calculations(self):

        valid_pixels = self['calibrated_amplitude'] != 0

        # Calculate the db values.
        self['calibrated_amplitude_db'][valid_pixels] = 10 * np.log10(self['calibrated_amplitude'][valid_pixels])
