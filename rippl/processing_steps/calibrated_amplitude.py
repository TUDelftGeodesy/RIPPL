# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class CalibratedAmplitude(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', out_coor='', multilooked=False, resampled=True,
                 secondary_slc='secondary_slc', reference_slc='reference_slc', overwrite=False, db_only=True):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData reference_slc: Reference where the coordinates are given.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'calibrated_amplitude'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        if db_only:
            self.output_info['file_names'] = ['calibrated_amplitude_db']
            self.output_info['data_types'] = ['real4']
        else:
            self.output_info['file_names'] = ['calibrated_amplitude', 'calibrated_amplitude_db']
            self.output_info['data_types'] = ['real4', 'real4']
        self.db_only = db_only

        # Input data information
        self.input_info = dict()
        if resampled:
            self.input_info['image_types'] = ['secondary_slc', 'reference_slc']
        else:
            self.input_info['image_types'] = ['secondary_slc', 'secondary_slc']
        self.input_info['process_names'] = ['intensity', 'radar_geometry']
        self.input_info['file_names'] = ['intensity', 'incidence_angle']
        self.input_info['polarisations'] = [polarisation, '']
        self.input_info['data_ids'] = [data_id, '']
        self.input_info['coor_types'] = ['out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['aliases_processing'] = ['intensity', 'incidence_angle']

        if multilooked:
            self.input_info['image_types'].append('secondary_slc')
            self.input_info['process_names'].append('intensity')
            self.input_info['file_names'].append('number_of_samples')
            self.input_info['polarisations'].append('')
            self.input_info['data_ids'].append(data_id)
            self.input_info['coor_types'].append('out_coor')
            self.input_info['in_coor_types'].append('')
            self.input_info['aliases_processing'].append('number_of_samples')

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc
        self.processing_images['reference_slc'] = reference_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['resampled'] = resampled
        self.settings['multilooked'] = multilooked

    def process_calculations(self):
        """
        Create a square amplitude image. (Very basic operation)

        Processing is done based on description in https://sentinel.esa.int/documents/247904/685163/s1-radiometric-calibration-v1.0.pdf

        :return:
        """

        beta_0 = 237.0    # TODO Read in from meta data files.
        if self.settings['multilooked']:
            cal_amplitude = self['intensity'] / self['number_of_samples'] / beta_0 ** 2 * np.sin(np.deg2rad(self['incidence_angle']))
        else:
            cal_amplitude = self['intensity'] / beta_0**2 * np.sin(np.deg2rad(self['incidence_angle']))

        if self.db_only:
            valid_pixels = (cal_amplitude != 0)
            self['calibrated_amplitude_db'] = np.zeros(cal_amplitude.shape).astype(np.float32)
            self['calibrated_amplitude_db'][valid_pixels] = 10 * np.log10(cal_amplitude[valid_pixels])
        else:
            self['calibrated_amplitude'] = cal_amplitude.astype(np.float32)
            valid_pixels = (self['calibrated_amplitude'] != 0)
            self['calibrated_amplitude_db'][valid_pixels] = 10 * np.log10(self['calibrated_amplitude'][valid_pixels])
