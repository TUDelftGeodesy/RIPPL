# Try to do all calculations using numpy functions.
import numpy as np
from collections import OrderedDict
from scipy import signal
from scipy.interpolate import RectBivariateSpline

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_regular2irregular import Regural2irregular


class CalcAzimuthAmbiguities(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', gaussian_spread=1, kernel_size=5, amb_num=1, amb_loc='left',
                 out_coor=[], coreg_master='coreg_master', slave='slave', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData, str slave: Slave image, used as the default for input and output for processing.
        """

        # Output data information
        amb_str = '_ambiguity_' + amb_loc + '_no_' + str(amb_num)

        self.output_info = dict()
        self.output_info['process_name'] = 'azimuth_ambiguities'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = amb_loc + '_no_' + str(amb_num)
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['ambiguities']
        self.output_info['data_types'] = ['complex_real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave', 'slave', 'coreg_master', 'coreg_master', 'coreg_master']

        if coreg_master == slave:
            self.input_info['process_types'] = ['crop', 'calc_reramp']
            self.input_info['file_types'] = ['crop', 'ramp']
        else:
            self.input_info['process_types'] = ['correct_phases', 'calc_reramp']
            self.input_info['file_types'] = ['phase_corrected', 'ramp']
        self.input_info['file_types'].extend(['gain' + amb_str, 'range' + amb_str, 'azimuth' + amb_str])
        self.input_info['process_types'].extend(['azimuth_ambiguities_locations', 'azimuth_ambiguities_locations',
                                                 'azimuth_ambiguities_locations'])
        self.input_info['polarisations'] = [polarisation, '', '', '', '']
        self.input_info['data_ids'] = [data_id, '', '', '', '']
        self.input_info['coor_types'] = ['in_coor', 'in_coor', 'out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '', '']
        self.input_info['type_names'] = ['in_data', 'ramp', 'gain', 'range', 'azimuth']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['gaussian_spread'] = [gaussian_spread, gaussian_spread]
        self.settings['kernel_size'] = [kernel_size, kernel_size]
        self.settings['amb_name'] = amb_str
        self.settings['out_irregular_grids'] = ['azimuth', 'range']
        self.settings['buf'] = int(np.ceil(kernel_size / 2) + 1)

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(CalcAzimuthAmbiguities, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Resampling of radar grid. For non radar grid this function is not advised as it uses

        :return:
        """

        # Init resampling
        resample_filter = self.create_ambiguity_filters_gaussian(self.settings['gaussian_spread'], self.settings['kernel_size'])
        resample = Regural2irregular(w_type='custom', custom_kernel=resample_filter, kernel_size=self.settings['kernel_size'])
        in_block_coor = self.coordinate_systems['in_block_coor']

        # Change line/pixel coordinates to right value
        lines = (self['azimuth'] - in_block_coor.first_line) / \
                               (self.block_coor.multilook[0] / self.block_coor.oversample[0])
        pixels = (self['range'] - in_block_coor.first_pixel) / \
                               (self.block_coor.multilook[1] / self.block_coor.oversample[1])

        valid = (self['gain'] != 0) * (self['azimuth'] != 0) * (self['range'] != 0)
        unramped = self['in_data'] * np.conjugate(np.exp(-1j * self['ramp']).astype(np.complex64))
        self['ambiguities'][valid] = resample(unramped, lines[valid], pixels[valid]) * np.sqrt(self['gain'][valid])

    def create_ambiguity_filters_gaussian(self, spread=[1, 1], kernel_size=[7, 7], no_samples=[50, 50]):
        """
        Create a filter based on a gaussian in range and azimuth.

        :param spread: How many pixels the gaussian is spread.

        :return:
        """

        kernel_size = np.array(kernel_size) + np.array([2, 2])
        az_values = np.linspace(-kernel_size[0] / 2, kernel_size[0] / 2, no_samples[0] + 1)
        ra_values = np.linspace(-kernel_size[1] / 2, kernel_size[1] / 2, no_samples[1] + 1)

        gaussian_az = signal.gaussian(no_samples[0] + 1, spread[0] / kernel_size[0] * no_samples[0])
        gaussian_ra = signal.gaussian(no_samples[1] + 1, spread[1] / kernel_size[1] * no_samples[1])

        gaussian_filter = RectBivariateSpline(az_values, ra_values, gaussian_az[:, None] * gaussian_ra[None, :])

        return gaussian_filter
