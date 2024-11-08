# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

class CalibratedAmplitudeC2(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', no_of_samples=False,
                 in_coor=[], out_coor=[], primary_slc_image=False, resampled=True,
                 reference_slc='reference_slc', secondary_slc='secondary_slc', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData primary_slc: primary image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a primary_slc/secondary_slc combination
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'calibrated_amplitude_c2'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['calibrated_amplitude_pol1', 'calibrated_amplitude_pol2', 'cal_C12', 'cal_C21']
        self.output_info['data_types'] = ['real4', 'real4', 'complex_real4', 'complex_real4']

        # Input data information
        self.input_info = dict()
        if primary_slc_image or not resampled:
            self.input_info['image_types'] = ['secondary_slc', 'secondary_slc', 'secondary_slc']
            self.input_info['process_names'] = ['crop', 'crop', 'radar_geometry']
            self.input_info['file_names'] = ['crop', 'crop', 'incidence_angle']
        else:
            self.input_info['image_types'] = ['secondary_slc', 'secondary_slc', 'reference_slc']
            self.input_info['process_names'] = ['earth_topo_phase', 'earth_topo_phase', 'radar_geometry']
            self.input_info['file_names'] = ['earth_topo_phase_corrected', 'earth_topo_phase_corrected', 'incidence_angle']

        self.input_info['polarisations'] = [polarisation[0], polarisation[1], '']
        self.input_info['data_ids'] = [data_id, data_id, '']
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['aliases_processing'] = ['complex_data_pol1', 'complex_data_pol2', 'incidence_angle'] #here how we should input VV and VH Pol

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # Image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc
        self.processing_images['reference_slc'] = reference_slc

        # Finally define whether we overwrite or not
        self.settings = dict()
        self.overwrite = overwrite

    def process_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        beta_0 = 237.0    # TODO Read in from meta data files.
        complex_1 = self['complex_data_pol1']**2 * np.sin(self['incidence_angle'] / 180 * np.pi) / beta_0**2
        complex_2 = self['complex_data_pol2']**2 * np.sin(self['incidence_angle'] / 180 * np.pi) / beta_0**2
        self['calibrated_amplitude_pol1'] = np.abs(complex_1)
        self['calibrated_amplitude_pol2'] = np.abs(complex_2)

        self['cal_C12'] = complex_1 * np.conjugate(complex_2)
        self['cal_C21'] = -1 * complex_1 * np.conjugate(complex_2)
