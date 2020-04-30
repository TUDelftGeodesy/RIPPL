# Try to do all calculations using numpy functions.
import numpy as np
from collections import OrderedDict

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_regular2irregular import Regural2irregular


class combined_ambiguities(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', amb_no=2, out_coor=[], slave='slave', master_image=False, overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'combined_ambiguities'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['combined_ambiguities']
        self.output_info['data_types'] = ['complex_real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave']
        if master_image:
            self.input_info['process_types'] = ['crop']
            self.input_info['file_types'] = ['crop']
        else:
            self.input_info['process_types'] = ['earth_topo_phase']
            self.input_info['file_types'] = ['earth_topo_phase_corrected']

        self.input_info['polarisations'] = [polarisation]
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['in_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['type_names'] = ['orig_data']

        for amb_loc in ['left', 'right']:
            for amb_num in range(amb_no):
                self.input_info['image_types'].append('slave')
                self.input_info['file_types'].append('ambiguity_' + amb_loc + '_no_' + str(amb_num))
                self.input_info['process_types'].extend('azimuth_ambiguities')
                self.input_info['polarisations'].append(polarisation)
                self.input_info['data_ids'].append(data_id)
                self.input_info['coor_types'].append('in_coor')
                self.input_info['in_coor_types'].append('')
                self.input_info['type_names'].append('ambiguity_' + amb_loc + '_no_' + str(amb_num))

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['ambiguity_no'] = amb_no

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(AASR, self).__init__(
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

        # Add all ambiguities
        orig_power = np.abs(self['in_data'])**2
        ambiguities_power = np.zeros(orig_power.shape, np.comlex_float32)

        for amb_loc in ['left', 'right']:
            for amb_num in range(self.settings['ambiguity_no']):
                amb_str = 'ambiguity_' + amb_loc + '_no_' + str(amb_num)
                ambiguities_power += self[amb_str]

        self['AASR'] = ambiguities_power / orig_power
