# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np
import copy

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

from rippl.resampling.multilook_irregular import MultilookIrregular
from rippl.resampling.multilook_regular import MultilookRegular
from rippl.resampling.coor_new_extend import CoorNewExtend


class Multilook(Process):  # Change this name to the one of your processing step.

    def __init__(self, in_coor=[], out_coor=[],
                 in_image_type='', in_process='', in_file_type='', in_polarisation='', in_data_id='', in_data_type='real4',
                 slave='slave', coreg_master='coreg_master', overwrite=False, batch_size=1000000):

        """
        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param str in_image_type: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.) for
                    the multilooked image
        :param str in_process: Which process outputs are we using as an input for the multilooked image
        :param str in_file_type: What are the exact outputs we use from these processes
        :param str in_polarisation: For which polarisation is it done. Leave empty if not relevant
        :param str in_data_id: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Check whether we need to do regular or irregular multilooking
        self.regular = MultilookRegular.check_same_coordinate_system(in_coor, out_coor)
        # If the grid size of the output grid are not defined yet, they are calculated here.

        self.output_info = dict()
        if not in_image_type:
            if isinstance(slave, ImageProcessingData):
                self.output_info['image_type'] = 'slave'
            else:
                self.output_info['image_type'] = 'coreg_master'

        # Output data information
        self.output_info['process_name'] = in_process
        self.output_info['polarisation'] = in_polarisation
        self.output_info['data_id'] = in_data_id
        self.output_info['coor_type'] = 'out_coor'
        if isinstance(in_file_type, str):
            self.output_info['file_types'] = [in_file_type]
            num_types = 1
        elif isinstance(in_file_type, list):
            self.output_info['file_types'] = in_file_type
            num_types = len(in_file_type)
        if isinstance(in_data_type, str):
            self.output_info['data_types'] = [in_data_type for n in range(num_types)]
        elif isinstance(in_data_type, list):
            self.output_info['data_types'] = in_data_type

        self.file_types = self.output_info['file_types']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = [self.output_info['image_type'] for n in range(num_types)]
        self.input_info['process_types'] = [in_process for n in range(num_types)]
        self.input_info['file_types'] = copy.copy(self.output_info['file_types'])
        self.input_info['polarisations'] = [in_polarisation for n in range(num_types)]
        self.input_info['data_ids'] = [in_data_id for n in range(num_types)]
        self.input_info['coor_types'] = ['in_coor' for n in range(num_types)]
        self.input_info['in_coor_types'] = ['' for n in range(num_types)]
        self.input_info['type_names'] = [file_type + '_input_data' for file_type in self.output_info['file_types']]

        if not self.regular:
            self.input_info['image_types'].extend(['coreg_master', 'coreg_master'])
            self.input_info['process_types'].extend(['reproject', 'reproject'])
            self.input_info['file_types'].extend(['in_coor_lines', 'in_coor_pixels'])
            self.input_info['polarisations'].extend(['', ''])
            self.input_info['data_ids'].extend(['', ''])
            self.input_info['coor_types'].extend(['in_coor', 'in_coor'])
            self.input_info['in_coor_types'].extend(['out_coor', 'out_coor'])
            self.input_info['type_names'].extend(['lines', 'pixels'])

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master
        self.processing_images['slave'] = slave
        self.batch_size = batch_size
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(Multilook, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def __call__(self):

        super(Multilook, self).__call__(memory_in=False)

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        for file_type in self.file_types:
            shape = self.in_images[file_type + '_input_data'].coordinates.shape
            no_lines = int(np.ceil(self.batch_size / shape[1]))
            no_blocks = int(np.ceil(shape[0] / no_lines))
            pixels = self.in_images['pixels'].disk['data']
            lines = self.in_images['lines'].disk['data']
            input_data = self.in_images[file_type + '_input_data'].disk['data']
            input_dtype = self.in_images[file_type + '_input_data'].disk['meta']['dtype']

            self.coordinate_systems['out_coor'].create_radar_lines()
            self.coordinate_systems['in_coor'].create_radar_lines()
            looks = np.zeros(self[file_type].shape)

            for block in range(no_blocks):
                coordinates = copy.deepcopy(self.coordinate_systems['in_coor'])
                coordinates.first_line += block * no_lines
                coordinates.shape[0] = np.minimum(shape[0] - block * no_lines, no_lines)

                print('Processing ' + str(block) + ' out of ' + str(no_blocks))

                if self.regular:
                    multilook = MultilookRegular(coordinates, self.coordinate_systems['out_coor'])
                    self[file_type] += multilook(input_data[block * no_lines: (block + 1) * no_lines, :])
                else:
                    multilook = MultilookIrregular(coordinates, self.coordinate_systems['out_coor'])
                    multilook.create_conversion_grid(lines[block * no_lines: (block + 1) * no_lines, :],
                                                     pixels[block * no_lines: (block + 1) * no_lines, :])
                    in_data = ImageData.disk2memory(input_data[block * no_lines: (block + 1) * no_lines, :], input_dtype)
                    multilook.apply_multilooking(in_data)
                    self[file_type] += ImageData.memory2disk(multilook.multilooked, input_dtype)
                    looks += multilook.looks
            valid_pixels = self[file_type] != 0
            self[file_type][valid_pixels] /= looks[valid_pixels]

    def def_out_coor(self):
        """
        Calculate extend of output coordinates.

        :return:
        """

        new_coor = CoorNewExtend(self.coordinate_systems['in_coor'], self.coordinate_systems['out_coor'])
        self.coordinate_systems['out_coor'] = new_coor.out_coor
