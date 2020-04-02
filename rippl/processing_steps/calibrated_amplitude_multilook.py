# Try to do all calculations using numpy functions.
import numpy as np
import copy

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.resampling.multilook_irregular import MultilookIrregular
from rippl.resampling.multilook_regular import MultilookRegular

class CalibratedAmplitudeMultilook(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='',
                 in_coor=[], out_coor=[], master_image=False, db_only=True, resampled=True,
                 coreg_master='coreg_master', slave='slave', overwrite=False, batch_size=1000000):

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
        self.input_info['image_types'] = ['slave', 'coreg_master']
        if master_image or not resampled:
            self.input_info['process_types'] = ['crop', 'radar_ray_angles']
            self.input_info['file_types'] = ['crop', 'incidence_angle']
        else:
            self.input_info['process_types'] = ['earth_topo_phase', 'radar_ray_angles']
            self.input_info['file_types'] = ['earth_topo_phase_corrected', 'incidence_angle']

        self.input_info['polarisations'] = [polarisation, '']
        self.input_info['data_ids'] = [data_id, '']
        self.input_info['coor_types'] = ['in_coor', 'in_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['type_names'] = ['complex_data', 'incidence_angle']

        if not self.regular:
            self.input_info['image_types'].extend(['coreg_master', 'coreg_master'])
            self.input_info['process_types'].extend(['reproject', 'reproject'])
            self.input_info['file_types'].extend(['in_coor_lines', 'in_coor_pixels'])
            self.input_info['polarisations'].extend(['', ''])
            self.input_info['data_ids'].extend(['', ''])
            self.input_info['coor_types'].extend(['in_coor', 'in_coor'])
            self.input_info['in_coor_types'].extend(['out_coor', 'out_coor'])
            self.input_info['type_names'].extend(['lines', 'pixels'])

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
        self.batch_size = batch_size

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(CalibratedAmplitudeMultilook, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def __call__(self):

        super(CalibratedAmplitudeMultilook, self).__call__(memory_in=False)

    def process_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        shape = self.in_images['complex_data'].coordinates.shape
        no_lines = int(np.ceil(self.batch_size / shape[1]))
        no_blocks = int(np.ceil(shape[0] / no_lines))
        pixels = self.in_images['pixels'].disk['data']
        lines = self.in_images['lines'].disk['data']
        data = self.in_images['complex_data'].disk['data']
        data_type = self.in_images['complex_data'].disk['meta']['dtype']
        incidence = self.in_images['incidence_angle'].disk['data']
        incidence_type = self.in_images['incidence_angle'].disk['meta']['dtype']

        self.coordinate_systems['out_coor'].create_radar_lines()
        self.coordinate_systems['in_coor'].create_radar_lines()
        looks = np.zeros(self[self.output_info['file_types'][0]].shape)

        beta_0 = 237.0    # TODO Read in from meta data files.

        for block in range(no_blocks):
            coordinates = copy.deepcopy(self.coordinate_systems['in_coor'])
            coordinates.first_line += block * no_lines
            coordinates.shape[0] = np.minimum(shape[0] - block * no_lines, no_lines)

            amplitude_data = np.abs(ImageData.disk2memory(data[block * no_lines: (block + 1) * no_lines, :], data_type))**2 * \
                             np.sin(ImageData.disk2memory(incidence[block * no_lines: (block + 1) * no_lines, :], incidence_type) / 180 * np.pi) / (beta_0 ** 2)

            print('Processing ' + str(block) + ' out of ' + str(no_blocks))

            if self.regular:
                multilook = MultilookRegular(coordinates, self.coordinate_systems['out_coor'])
                self['calibrated_amplitude_db'] += multilook(amplitude_data)
            else:
                multilook = MultilookIrregular(coordinates, self.coordinate_systems['out_coor'])
                multilook.create_conversion_grid(lines[block * no_lines: (block + 1) * no_lines, :],
                                                 pixels[block * no_lines: (block + 1) * no_lines, :])
                multilook.apply_multilooking(amplitude_data)
                self['calibrated_amplitude_db'] += ImageData.memory2disk(multilook.multilooked, self.output_info['data_types'][0])
                looks += multilook.looks

        valid_pixels = self[self.output_info['file_types'][0]] != 0
        self['calibrated_amplitude_db'][valid_pixels] /= looks[valid_pixels]

        if not self.db_only:
            self['calibrated_amplitude'][valid_pixels] = self['calibrated_amplitude_db'][valid_pixels]

        # Calculate the db values.
        self['calibrated_amplitude_db'][valid_pixels] = 10 * np.log10(self['calibrated_amplitude_db'][valid_pixels])

    def def_out_coor(self):
        """
        Calculate extend of output coordinates.

        :return:
        """

        new_coor = CoorNewExtend(self.coordinate_systems['in_coor'], self.coordinate_systems['out_coor'])
        self.coordinate_systems['out_coor'] = new_coor.out_coor
