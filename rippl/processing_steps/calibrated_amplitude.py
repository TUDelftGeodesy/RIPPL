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


class CalibratedAmplitude(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', out_coor='', master_image=False,
                 slave='slave', coreg_master='coreg_master', overwrite=False, resampled=True, db_only=True):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Coreg master where the coordinates are given.
        """

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
        self.input_info['image_types'] = ['slave', 'coreg_master']
        if master_image or not resampled:
            self.input_info['process_types'] = ['crop', 'radar_ray_angles']
            self.input_info['file_types'] = ['crop', 'incidence_angle']
        else:
            self.input_info['process_types'] = ['earth_topo_phase', 'radar_ray_angles']
            self.input_info['file_types'] = ['earth_topo_phase_corrected', 'incidence_angle']

        self.input_info['polarisations'] = [polarisation, '']
        self.input_info['data_ids'] = [data_id, '']
        self.input_info['coor_types'] = ['out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['type_names'] = ['complex_data', 'incidence_angle']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(CalibratedAmplitude, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Create a square amplitude image. (Very basic operation)

        :return:
        """

        beta_0 = 237.0    # TODO Read in from meta data files.
        if self.db_only:
            cal_amplitude = np.abs(self['complex_data'])**2 * np.sin(self['incidence_angle'] / 180 * np.pi) / beta_0**2
            valid_pixels = (cal_amplitude != 0)
            self['calibrated_amplitude_db'][valid_pixels] = 10 * np.log10(cal_amplitude[valid_pixels])
        else:
            self['calibrated_amplitude'] = np.abs(self['complex_data'])**2 * np.sin(self['incidence_angle'] / 180 * np.pi) / beta_0**2
            valid_pixels = (self['calibrated_amplitude'] != 0)
            self['calibrated_amplitude_db'][valid_pixels] = 10 * np.log10(self['calibrated_amplitude'][valid_pixels])
