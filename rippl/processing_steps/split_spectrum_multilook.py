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

class SplitSpectrumMultilook(MultilookProcess):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='',
                 in_coor=[], out_coor=[],
                 slave='slave', master='master', coreg_master='coreg_master', ifg='ifg', overwrite=False, batch_size=1000000):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        """

        # Check whether we need to do regular or irregular multilooking
        self.regular = MultilookRegular.check_same_coordinate_system(in_coor, out_coor)
        # If the grid size of the output grid are not defined yet, they are calculated here.

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'split_spectrum'
        self.output_info['image_type'] = 'ifg'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['low_band_interferogram', 'high_band_interferogram']
        self.output_info['data_types'] = ['complex_real4', 'complex_real4']

        # Input data information
        self.input_info = dict()

        if coreg_master == slave:
            self.input_info['process_types'] = ['crop', 'correct_phases', 'calc_reramp', 'calc_reramp']
            self.input_info['file_types'] = ['crop', 'phase_corrected', 'ramp', 'ramp']
        elif coreg_master == master:
            self.input_info['process_types'] = ['correct_phases', 'crop', 'calc_reramp', 'calc_reramp']
            self.input_info['file_types'] = ['phase_corrected', 'crop', 'ramp', 'ramp']
        else:
            self.input_info['process_types'] = ['correct_phases', 'correct_phases', 'calc_reramp', 'calc_reramp']
            self.input_info['file_types'] = ['phase_corrected', 'phase_corrected', 'ramp', 'ramp']

        self.input_info['image_types'] = ['slave', 'master', 'slave', 'master']
        self.input_info['polarisations'] = [polarisation, polarisation, '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id, data_id]
        self.input_info['coor_types'] = ['in_coor', 'in_coor', 'in_coor', 'in_coor']
        self.input_info['in_coor_types'] = ['', '', '', '']
        self.input_info['type_names'] = ['slave', 'master', 'slave_ramp', 'master_ramp']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # Image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['master'] = master
        self.processing_images['coreg_master'] = coreg_master
        self.processing_images['ifg'] = ifg

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['out_irregular_grids'] = ['lines', 'pixels']
        self.settings['multilooked_grids'] = ['low_band_interferogram', 'high_band_interferogram']
        self.settings['memory_data'] = False
        self.settings['buf'] = 0
        self.batch_size = batch_size

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(SplitSpectrumMultilook, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def __call__(self):

        super(SplitSpectrumMultilook, self).__call__(memory_in=False)

    def before_multilook_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        # Deramp data
        ramp_slave = np.cos(self['slave_ramp']) + 1j * np.sin(self['slave_ramp'])
        ramp_master = np.cos(self['master_ramp']) + 1j * np.sin(self['master_ramp'])

        # Split in two bands
        fft_master = np.fft.fft(self['master'] * ramp_master, axis=1)
        fft_slave = np.fft.fft(self['slave'] * ramp_slave, axis=1)

        # Create high and low band
        split_line = int(fft_slave.shape[1] / 2)
        low_slave = np.zeros(shape=fft_slave)
        low_slave[:, :split_line] = fft_slave[:, :split_line]
        low_master = np.zeros(shape=fft_master)
        low_master[:, :split_line] = fft_master[:, :split_line]
        self['low_band_interferogram'] = np.fft.ifft(low_master, aixs=1) * np.conjugate(np.fft.ifft2(low_slave, axis=1))
        del low_slave, low_master

        high_slave = np.zeros(shape=fft_slave)
        high_slave[:, split_line:] = fft_slave[:, split_line:]
        high_master = np.zeros(shape=fft_master)
        high_master[:, split_line:] = fft_master[:, split_line:]
        self['high_band_interferogram'] = np.fft.ifft(high_master, axis=1) * np.conjugate(np.fft.ifft2(high_slave))
        del high_slave, high_master
