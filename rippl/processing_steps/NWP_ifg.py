# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class NWPInterferogram(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], split_signal=False, geometry_correction=False,
                 primary_slc='primary_slc', secondary_slc='secondary_slc', reference_slc='reference_slc', ifg='ifg',
                 nwp_model_database_folder='', model_name='era5',
                 overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData primary_slc: primary image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a primary_slc/secondary_slc combination
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = model_name + '_nwp_interferogram'
        self.output_info['image_type'] = 'ifg'
        self.output_info['data_id'] = data_id
        self.output_info['polarisation'] = ''
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = []
        self.output_info['data_types'] = []

        # Define all the outputs
        geometry_types = ['', '_geometry_corrected'] if geometry_correction else ['']
        signal_types = ['_aps', '_hydrostatic_delay', '_wet_delay', '_liquid_delay'] if split_signal else ['_aps']
        for geom in geometry_types:
            for sig in signal_types:
                self.output_info['file_names'].append(model_name + '_ifg' + sig + geom)
                self.output_info['data_types'].append('real4')

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = []
        self.input_info['process_names'] = []
        self.input_info['file_names'] = []
        self.input_info['data_ids'] = []
        self.input_info['coor_types'] = []
        self.input_info['polarisations'] = []
        self.input_info['in_coor_types'] = []
        self.input_info['aliases_processing'] = []

        for slc_type in ['secondary_slc', 'primary_slc']:
            for geom in geometry_types:
                for sig in signal_types:
                    self.input_info['file_names'].append(model_name + sig + geom)
                    self.input_info['image_types'].append(slc_type)
                    self.input_info['process_names'].append(model_name + '_nwp_delay')
                    self.input_info['data_ids'].append(data_id)
                    self.input_info['coor_types'].append('out_coor')
                    self.input_info['polarisations'].append('')
                    self.input_info['in_coor_types'].append('')
                    self.input_info['aliases_processing'].append(slc_type + sig + geom)

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc
        self.processing_images['primary_slc'] = primary_slc
        self.processing_images['reference_slc'] = reference_slc
        self.processing_images['ifg'] = ifg

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['model_name'] = model_name
        self.settings['signal_types'] = signal_types
        self.settings['geometry_types'] = geometry_types

    def process_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        model_name = self.settings['model_name']

        for geom in self.settings['geometry_types']:
            for sig in self.settings['signal_types']:
                self[model_name + '_ifg' + sig + geom] = (self['secondary_slc' + sig + geom] -
                                                          self['primary_slc' + sig + geom])
