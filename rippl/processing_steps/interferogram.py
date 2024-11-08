# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class Interferogram(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='',
                 out_coor=[], in_image='earth_topo_phase',
                 secondary_slc='secondary_slc', primary_slc='primary_slc', reference_slc='reference_slc', ifg='ifg', overwrite=False):

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
        self.output_info['process_name'] = 'interferogram'
        self.output_info['image_type'] = 'ifg'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['interferogram']
        self.output_info['data_types'] = ['complex_real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['secondary_slc', 'primary_slc']
        if in_image == 'earth_topo_phase':
            self.input_info['process_names'] = ['earth_topo_phase', 'earth_topo_phase']
            self.input_info['file_names'] = ['earth_topo_phase_corrected', 'earth_topo_phase_corrected']
        elif in_image == 'reramped':
            self.input_info['process_names'] = ['reramp', 'reramp']
            self.input_info['file_names'] = ['reramped', 'reramped']
        else:
            self.input_info['process_names'] = ['resample', 'resample']
            self.input_info['file_names'] = ['resampled', 'resampled']

        self.input_info['polarisations'] = [polarisation, polarisation]
        self.input_info['data_ids'] = [data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['aliases_processing'] = ['secondary_slc', 'primary_slc']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc
        self.processing_images['primary_slc'] = primary_slc
        self.processing_images['ifg'] = ifg

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def process_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        self['interferogram'] = self['primary_slc'] * np.conjugate(self['secondary_slc'])
