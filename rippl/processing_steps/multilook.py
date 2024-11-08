# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np
import copy

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.multilook_regular import MultilookRegular
from rippl.resampling.multilook_irregular import MultilookIrregular


class Multilook(Process):  # Change this name to the one of your processing step.

    def __init__(self, in_coor=[], out_coor=[], regular=False,
                 image_type='', process='', file_type='', polarisation='', data_id='', data_type='real4',
                 secondary_slc='secondary_slc', reference_slc='reference_slc', overwrite=False,
                 calculation_type='sum', number_of_samples=False,
                 min_height=0, max_height=0, buffer=0, rounding=0):

        """
        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param str image_type: The type of the input ImageProcessingData objects (e.g. secondary_slc/primary_slc/ifg etc.) for
                    the multilooked image
        :param str process: Which process outputs are we using as an input for the multilooked image
        :param str file_type: What are the exact outputs we use from these processes
        :param str polarisation: For which polarisation is it done. Leave empty if not relevant
        :param str data_id: If processes are used multiple times in different parts of the processing they can be
                distinguished using a data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampline etc.
        """

        self.output_info = dict()
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['process_name'] = process
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        if isinstance(file_type, str):
            self.output_info['file_names'] = [file_type]
        elif isinstance(file_type, list):
            self.output_info['file_names'] = file_type
        if isinstance(data_type, str):
            self.output_info['data_types'] = [data_type]
        elif isinstance(data_type, list):
            self.output_info['data_types'] = data_type

        # Input data information
        n_file_types = len(self.output_info['file_names'])
        self.input_info = dict()
        self.input_info['image_types'] = [self.output_info['image_type'] for n in range(n_file_types)]
        self.input_info['process_names'] = [process for n in range(n_file_types)]
        self.input_info['file_names'] = copy.copy(self.output_info['file_names'])
        self.input_info['polarisations'] = [polarisation for n in range(n_file_types)]
        self.input_info['data_ids'] = [data_id for n in range(n_file_types)]
        self.input_info['coor_types'] = ['in_coor' for n in range(n_file_types)]
        self.input_info['in_coor_types'] = ['' for n in range(n_file_types)]
        self.input_info['aliases_processing'] = [file_type + '_input_data' for file_type in self.output_info['file_names']]

        if number_of_samples:
            self.output_info['file_names'].append('number_of_samples')
            self.output_info['data_types'].append('int32')

        if not regular:
            self.input_info['image_types'].extend(['reference_slc', 'reference_slc'])
            self.input_info['process_names'].extend(['grid_transform', 'grid_transform'])
            self.input_info['file_names'].extend(['multilook_lines', 'multilook_pixels'])
            self.input_info['polarisations'].extend(['', ''])
            self.input_info['data_ids'].extend(['', ''])
            self.input_info['coor_types'].extend(['in_coor', 'in_coor'])
            self.input_info['in_coor_types'].extend(['out_coor', 'out_coor'])
            self.input_info['aliases_processing'].extend(['lines', 'pixels'])

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['reference_slc'] = reference_slc
        self.processing_images['secondary_slc'] = secondary_slc

        self.settings = dict()
        self.settings['regular'] = regular
        if calculation_type not in ['sum', 'mean']:     # Would be interesting to allow min/max/median values though
            raise TypeError('calculation_type can only be sum or mean!')
        self.settings['calculation_type'] = calculation_type
        self.settings['number_of_samples'] = number_of_samples
        self.settings['in_coor'] = dict()
        self.settings['in_coor']['buffer'] = buffer
        self.settings['in_coor']['rounding'] = rounding
        self.settings['in_coor']['min_height'] = min_height
        self.settings['in_coor']['max_height'] = max_height

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        for file_type in self.output_info['file_names']:
            if file_type != 'number_of_samples':
                if self.settings['regular']:
                    multilook = MultilookRegular(self.coordinate_systems['in_coor_chunk'], self.coordinate_systems['out_coor_chunk'])
                    multilook(self[file_type + '_input_data'])
                else:
                    multilook = MultilookIrregular(self.coordinate_systems['in_coor_chunk'], self.coordinate_systems['out_coor_chunk'])
                    multilook.create_conversion_grid(self['lines'], self['pixels'])
                    if self.settings['number_of_samples']:
                        multilook.apply_multilooking(self[file_type + '_input_data'], remove_unvalid=True)
                    else:
                        multilook.apply_multilooking(self[file_type + '_input_data'], remove_unvalid=False)

                if self.settings['calculation_type'] == 'sum':
                    self[file_type] = multilook.multilooked
                elif self.settings['calculation_type'] == 'mean':
                    multilooked = np.zeros(multilook.multilooked.shape).astype(multilook.multilooked.dtype)
                    valid = multilook.samples > 0
                    multilooked[valid] = multilook.multilooked[valid] / multilook.samples[valid]
                    self[file_type] = multilooked

                if self.settings['number_of_samples']:
                    self['number_of_samples'] = multilook.samples
