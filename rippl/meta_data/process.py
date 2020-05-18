'''

This is a super class for all processing step. This class manages:
- Importing data from a processing stack (either SLC or interferogram)
- Initialize a processing data/metadata object.
- Writing results to disk if needed
- Removing results/ins from memory

'''

import numpy as np
import copy

from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.process_data import ProcessData
from rippl.meta_data.image_data import ImageData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.select_input_window import SelectInputWindow


class Process():

    def __init__(self, input_info, output_info, coordinate_systems, processing_images, settings, overwrite):
        """
        Input steps Process. These input steps are a summary of the inputs and outputs of every processing step.
        The reading and writing of data and creation of metadata information is all summarized in this step.

        :param dict[list[str]] input_info: Input data information of your processing step which consists of
                - process_types > the names of the processes of the input data
                - file_types > the names of the file types of the inputs
                - image_types > the names of the images used (images should be defined in images parameter)
                - type_names > self assigned names to call the variables during the processing. If missing it will be
                                replaced by the file_types
                - data_ids > possible data_ids of inputs. (empty by default)
                - polarisation > polarisation of inputs. (only applicable for radar data, otherwises it should be none)
                - coor_types > type of coordinate systems of the inputs. The coordinate systems themselves should be
                                given in the coordinates input variable.
        :param dict[str or list[str]] output_info: Information on the outputs of the processing step which consists of
                - process_name > name of the process
                - image_type > name of image used for output. Should be defined in the images parameter
                - polarisation > polarisation of radar data used in process
                - data_id > data id of process (can be anything but empty by default.)
                - coor_type > the coordinate type of the output. Coordinate systems should be part of the coordinates
                                list.
                - file_types > list of file types of the output
                - data_types > data types of outputs
        :param dict[CoordinateSystem] coordinate_systems: Dictionary of coordinate systems. All names of coordinate systems
                in output and input info should be defined here.
        :param dict[ImageProcessingData] processing_images: Dictionary with image data object. The datasets defined in the input
                and output info should be defined here.
        :param dict[str] settings: Additional settings on the processing that should be saved to metadata file
        :param bool overwrite: Do we process this step if it already exists or not
        """

        # First check the inputs
        self.input_info = self.check_input_info(input_info)
        self.output_info = self.check_output_info(output_info)
        self.processing_images = self.check_processing_images(processing_images, self.input_info, self.output_info)
        self.coordinate_systems = self.check_coordinate_systems(coordinate_systems, self.input_info, self.output_info)

        # Predefine all image definitions
        self.process_name = output_info['process_name']     # type: str
        self.process = []                                   # type: ProcessData

        # Block coordinate system is the coordinate system create for processing in blocks.
        self.block_coor = self.coordinate_systems['block_coor']

        # Input and output image (slave/master/ifg etc.)
        self.out_processing_image = self.processing_images[self.output_info['image_type']]

        # Input and output image data. To load from disk to memory and save results
        self.in_images = dict()                              # type: dict(ImageData)
        self.out_images = dict()                             # type: dict(ImageData)

        # Create the process metadata and images. Load the input images too.
        if not overwrite:
            process_exist = self.check_output_exists(self.out_processing_image, self.process_name,
                                                      self.coordinate_systems['out_coor'], self.coordinate_systems['in_coor'],
                                                      self.output_info['polarisation'],
                                                      self.output_info['data_id'],
                                                      self.output_info['file_types'])
            self.process_finished = process_exist[0]
            self.process_on_disk = process_exist[1]
            if self.process_finished and self.process_on_disk:
                return

        self.create_process_metadata(self.output_info['polarisation'], self.output_info['data_id'], self.settings)
        self.create_output_images(self.output_info['file_types'], self.output_info['data_types'], [])

        # Information for processing of dataset in blocks. If you want to do so, run the define_block method.
        self.settings = settings
        if 'memory_data' in self.settings.keys():
            self.memory_data = self.settings['memory_data']
        else:
            self.memory_data = True
        if not 'multilook_grids' in self.settings.keys():
            self.settings['multilook_grids'] = []
        if 'buf' in self.settings.keys():
            self.buf = self.settings['buf']
        else:
            self.buf = 5
        self.s_lin = 0
        self.s_pix = 0
        self.lines = self.coordinate_systems['out_coor'].shape[0]
        self.pixels = self.coordinate_systems['out_coor'].shape[1]
        self.out_irregular_grids = [None]
        self.in_irregular_grids = [None]
        self.blocks = False

        # Finally set a few variables for the multilooking case.
        self.no_block = 0
        self.no_blocks = 0
        self.no_lines = 1
        self.block_shape = (0, 0)
        self.ml_in_data = dict()
        self.ml_out_data = dict()

    def __call__(self, memory_in=True):
        """
        This function does basically the same as the __call__ function, but assumes that we apply no multiprocessing
        and or pipeline processing. Therefore it includes an extended number of steps:
        1. Reading in of input data
        2. Creation of output data on disk
        3. Create output memory files
        4. Perform the actual processing
        5. Save data to disk
        6. Clean memory steps.

        :return:
        """

        self.init_super()
        if self.process_finished and self.process_on_disk:
            print('Process already finished')
            return

        # Create the input and output info
        self.load_input_info()

        for image_key in self.processing_images.keys():
            self.processing_images[image_key].load_memmap_files()

        if memory_in:  # For the multilooking we do not work with parts of files, but the whole file at once.
            self.load_input_data()
        self.create_output_data_files()
        self.create_memory()
        self.process_calculations()
        self.save_to_disk()

        self.clean_memory()
        self.out_processing_image.save_json()

        self.process_finished = True

    def __getitem__(self, key):
        # Check if there is a memory file with this name and give the output.
        if key in list(self.in_images.keys()):
            data = self.in_images[key].memory['data']
            return data
        elif key in list(self.out_images.keys()):
            data = self.out_images[key].memory['data']
            return data
        else:
            raise LookupError('The input or output dataset ' + key + ' does not exist.')

    def __setitem__(self, key, data):
        # Set the data of one variable in memory.
        if key in list(self.in_images.keys()):
            self.in_images[key].memory['data'] = data
        elif key in list(self.out_images.keys()):
            self.out_images[key].memory['data'] = data
        else:
            raise LookupError('The input or output dataset ' + key + ' does not exist.')

    def load_input_info(self):
        """
        This method initializes the input and output.

        :return:
        """

        self.load_input_images(self.input_info['image_types'],
                               self.input_info['process_types'],
                               self.input_info['file_types'],
                               self.input_info['polarisations'],
                               self.input_info['data_ids'],
                               self.input_info['coor_types'],
                               self.input_info['in_coor_types'],
                               self.input_info['type_names']) # We load the input images only. The data is not loaded to memory yet.

    @staticmethod
    def check_output_exists(out_processing_image, process_name, out_coor, in_coor, polarisation, data_id, file_types):
        """
        Check if processing of this step is already done.

        :param str polarisation: Polarisation of output process
        :param str data_id: Data ID of output process
        :param list(str) file_types: List of file types output process
        :return:
        """

        if out_coor == in_coor:
            in_coor = ''

        for file_type in file_types:
            image = out_processing_image.processing_image_data_exists(process_name, out_coor, in_coor, data_id, polarisation,
                                                                      file_type, data=True, message=False)
            if not isinstance(image, ImageData):
                return (False, False)
            else:
                if not image.check_data_disk_valid() == (True, True):
                    return (True, False)

        # If al data sources exist and are available on disk cancel processing.
        return (True, True)

    def process_calculations(self):
        """
        This is the function in every processing step where the actual calculations are done. Because this is the
        master function this method is empty, as there should be an override of this function in the child functions.

        :return:
        """

        pass

    def def_out_coor(self):
        """
        This function is used to define the shape of the output coordinates based on the input coordinates. Function
        specific information is given in

        :return:
        """

    def init_super(self):
        """
        This function is used to initialize the super class, so these are called from the functions individually.

        :return:
        """

        print('Only call this function with a child object of Process')

    def load_coordinate_system_sizes(self, find_out_coor=True):
        """
        To do the final processing it is important that we know the exact shape of both the input and output coordinate
        systems. These are either imported or calculated here.

        :return:
        """

        if 'in_coor' not in self.coordinate_systems.keys():
            self.coordinate_systems['in_coor'] = copy.deepcopy(self.coordinate_systems['out_coor'])
        no_in_shape = len(self.coordinate_systems['in_coor'].shape) == 0 or self.coordinate_systems['in_coor'].shape == [0, 0]
        no_out_shape = len(self.coordinate_systems['out_coor'].shape) == 0 or self.coordinate_systems['out_coor'].shape == [0, 0]

        # If the input is given load one or the other.
        if 'in_coor' in list(self.coordinate_systems.keys()):
            if 'in_coor' in list(self.input_info['coor_types']) and no_in_shape:
                n_in_coor = self.input_info['coor_types'].index('in_coor')
                if self.input_info['in_coor_types'][n_in_coor]:
                    input_coor = self.coordinate_systems[self.input_info['in_coor_types'][n_in_coor]]
                else:
                    input_coor = ''

                image_in = self.processing_images[self.input_info['image_types'][n_in_coor]].\
                    processing_image_data_exists(self.input_info['process_types'][n_in_coor],
                                                 self.coordinate_systems['in_coor'],
                                                 in_coordinates=input_coor,
                                                 data_id=self.input_info['data_ids'][n_in_coor],
                                                 polarisation=self.input_info['polarisations'][n_in_coor],
                                                 file_type=self.input_info['file_types'][n_in_coor])

                if image_in == False:
                    raise LookupError('Could not find processing step ' + self.input_info['process_types'][n_in_coor] +
                                      ' file type ' + self.input_info['file_types'][n_in_coor] + ' for coordinate system')
                self.coordinate_systems['in_coor'] = image_in.coordinates

        if 'out_coor' in list(self.coordinate_systems.keys()) and find_out_coor:
            if 'out_coor' in list(self.input_info['coor_types']) and no_out_shape:
                n_out_coor = self.input_info['coor_types'].index('out_coor')
                if self.input_info['in_coor_types'][n_out_coor]:
                    input_coor = self.coordinate_systems[self.input_info['in_coor_types'][n_out_coor]]
                else:
                    input_coor = ''

                image_out = self.processing_images[self.input_info['image_types'][n_out_coor]]. \
                    processing_image_data_exists(self.input_info['process_types'][n_out_coor],
                                                 self.coordinate_systems['out_coor'],
                                                 in_coordinates=input_coor,
                                                 data_id=self.input_info['data_ids'][n_out_coor],
                                                 polarisation=self.input_info['polarisations'][n_out_coor],
                                                 file_type=self.input_info['file_types'][n_out_coor])
                if image_out == False:
                    raise LookupError('Could not find processing step ' + self.input_info['process_types'][n_out_coor] +
                                      ' file type ' + self.input_info['file_types'][n_out_coor] + ' for coordinate system')
                self.coordinate_systems['out_coor'] = image_out.coordinates

        # Add the orbits when using radar grids.
        if self.coordinate_systems['in_coor'].grid_type == 'radar_coordinates':
            if 'in_coor' in self.input_info['coor_types']:
                n_in_coor = self.input_info['coor_types'].index('in_coor')
                orbit_in = self.processing_images[self.input_info['image_types'][n_in_coor]].find_best_orbit()
                self.coordinate_systems['in_coor'].orbit = orbit_in

        if self.coordinate_systems['out_coor'].grid_type == 'radar_coordinates':
            orbit_out = self.processing_images[self.output_info['image_type']].find_best_orbit()
            self.coordinate_systems['out_coor'].orbit = orbit_out

        no_in_shape = len(self.coordinate_systems['in_coor'].shape) == 0 or self.coordinate_systems['in_coor'].shape == [0, 0]
        no_out_shape = len(self.coordinate_systems['out_coor'].shape) == 0 or self.coordinate_systems['out_coor'].shape == [0, 0]

        # Otherwise calculate one using the other.
        if 'out_coor' in list(self.coordinate_systems.keys()) and 'in_coor' in list(self.coordinate_systems.keys()):
            if no_out_shape or no_in_shape:
                # If this is the case it is relevant to extract the orbit for the input and output coordinate system.
                self.def_out_coor()

    def create_process_metadata(self, polarisation, data_id, settings):
        """
        Create the process data object. This creates the meta data of the image and creates a link to the output data
        on disk and in memory.

        :param str polarisation: Polarisation of data
        :param str data_id: data_id of dataset, if not relevant leave blank
        :param str settings: Specific settings of this processing step
        :return:
        """

        if 'in_coor' in list(self.coordinate_systems.keys()):
            if self.coordinate_systems['in_coor'] == self.coordinate_systems['out_coor']:
                in_coordinates = []
            else:
                in_coordinates = self.coordinate_systems['in_coor']
        else:
            in_coordinates = []

        coordinates = self.coordinate_systems['out_coor']
        self.process_data = ProcessData(self.out_processing_image.folder, self.output_info['process_name'],
                                        coordinates=coordinates, in_coordinates=in_coordinates,
                                        settings=settings, polarisation=polarisation, data_id=data_id)
        self.out_processing_image.add_process(self.process_data)
        self.process = self.process_data.meta

    def create_output_images(self, file_types, data_types, shapes=[]):
        """
        This step is used to generate the output files of this function. These have to be defined in the initialization
        of a processing step. Best practice is to use a default set of inputs, but leave it to the user to define
        other input steps.

        :param list(str) file_types: Names of the file types to be generated.
        :param list(str) data_types: Data types of these files. Check the ImageData file for possible data formats
        :param list(tuple) shapes: Shapes of outputs. Defaults to shape of output coordinate system if not defined.
        :return:
        """

        # Add image data.
        self.process_data.add_process_images(file_types, data_types, shapes)
        # Define the output images.
        self.out_images = self.process_data.images

    @staticmethod
    def check_input_info(input_info):
        '''
        Check whether there is no information missing in the output data.

        :param dict() input_info:
        :return:
        '''

        if not isinstance(input_info, dict):
            raise TypeError('Output info should be a dictionary')

        input_keys = set(input_info.keys())
        needed_keys = set({'process_types', 'coor_types', 'file_types'})
        if len(needed_keys - input_keys) > 0:
            raise LookupError(
                'One of the needed processes, coor_types, file_types or data_types is missing in the input_info')
        n = len(input_info['file_types'])
        if n != len(input_info['image_types']) or n != len(input_info['process_types']):
            raise LookupError('Number of image types or process types is not correct.')
        if not 'polarisations' in input_keys:
            input_info['polarisation'] = ['' for i in range(n)]
        elif n != len(input_info['polarisations']):
            raise LookupError('Number of polarisations is not the same as other parameters')
        if not 'data_ids' in input_keys:
            input_info['data_id'] = ['' for i in range(n)]
        elif n != len(input_info['data_ids']):
            raise LookupError('Number of data_ids is not the same as other parameters')
        if not 'in_coor_types' in input_keys:
            input_info['in_coor_types'] = ['' for i in range(n)]
        elif n != len(input_info['in_coor_types']):
            raise LookupError('Number of in_coor_types is not the same as other parameters')
        if not 'name_types' in input_keys:
            input_info['name_types'] = input_info['file_types']
        elif n != len(input_info['file_types']):
            raise LookupError('Number of file_types is not the same as other parameters')

        return input_info

    @staticmethod
    def check_output_info(output_info):
        '''
        Check whether there is no information missing in the output data.

        :param dict() output_info:
        :return:
        '''

        if not isinstance(output_info, dict):
            raise TypeError('Output info should be a dictionary')

        output_keys = set(output_info.keys())
        needed_keys = set({'process_name', 'coor_type', 'file_types', 'data_types'})
        if len(needed_keys - output_keys) > 0:
            raise LookupError('One of the needed process, coor_type, file_types or data_types is missing in the output_info')
        if len(output_info['file_types']) != len(output_info['data_types']):
            raise LookupError('data_types and file_types are not the same length. Both should be lists')
        if not 'polarisation' in output_keys:
            output_info['polarisation'] = ''
        if not 'data_id' in output_keys:
            output_info['data_id'] = ''
        if not 'name_type' in output_keys:
            output_info['type_names'] = output_info['file_types']

        return output_info

    @staticmethod
    def check_processing_images(images, input_info, output_info):
        '''
        This function is to check whether the input images and the input_info and output_info are aligned. It checks
        for two things:
        1. Whether there are images missing which are mentioned in the input_info or output_info
        2. Whether there are images that are superfluous, these will be removed.

        :param dict[ImageProcessingData] images:
        :param list[list[str] or str] input_info:
        :param list[list[str]] output_info:
        :return:
        '''

        # Check which image types are needed.
        needed_image_types = set()
        for image_type in input_info['image_types']:
            needed_image_types.add(image_type)
        needed_image_types.add(output_info['image_type'])

        # Cleanup not used images.
        image_types = set()
        for key in images.keys():
            if not isinstance(images[key], ImageProcessingData):
                images.pop(key)
            else:
                image_types.add(key)

        # Check if images are missing and throw an error if so
        if len(needed_image_types - image_types) > 0:
            raise LookupError('Needed image type for input or output does not exist')

        # Remove not needed images
        for image_type in (image_types - needed_image_types):
            images.pop(image_type)

        return images

    @staticmethod
    def check_coordinate_systems(coordinates, input_info, output_info):
        '''
        This function is to check whether the input images and the input_info and output_info are aligned. It checks
        for two things:
        1. Whether there are coordinate sytems missing which are mentioned in the input_info or output_info
        2. Whether there are coordinate systems that are superfluous, these will be removed.
        Finally the block coor coordinate system is added.

        :param dict[CoordinateSystem] coordinates:
        :param list[list[str] or str] input_info:
        :param list[list[str]] output_info:
        :return:
        '''

        needed_coordinate_systems = set()
        for coordinate_system in input_info['coor_types']:
            needed_coordinate_systems.add(coordinate_system)
        needed_coordinate_systems.add(output_info['coor_type'])

        # Check if images are missing and throw an error if so
        for coordinates_key in needed_coordinate_systems:
            if coordinates_key not in list(coordinates.keys()):
                raise LookupError('Needed image type for input or output does not exist')

        coordinates['block_coor'] = copy.copy(coordinates['out_coor'])
        if 'in_coor' not in coordinates.keys():
            coordinates['in_coor'] = coordinates['out_coor']
        coordinates['in_block_coor'] = copy.copy(coordinates['in_coor'])

        return coordinates

    def define_processing_block(self, s_lin=0, s_pix=0, lines=0, pixels=0):
        """
        Here we check whether the processing blocks can be processed based on the given sizes. Also we check whether
        it is possible to do block processing
        If the in coordinates and output coordinates are not the same either an irregular in or output grid is
        needed to do the reading in.
        Exception for this are the 'regular' multilooking, which does not need any precreated grid.

        :param int s_lin: Start line with respect too total image size
        :param int s_pix: Start pixel with respect too total image size
        :param int lines: Number of lines to process
        :param int pixels: Number of pixels to process
        :return:
        """

        if not self.memory_data:
            raise AssertionError('Creating blocks for data that is not loaded in memory is not possible! Aborting.')

        self.s_lin = s_lin
        self.s_pix = s_pix

        if self.coordinate_systems['out_coor'].grid_type == 'radar_coordinates':
            ml_step = np.array(self.coordinate_systems['out_coor'].multilook) / \
                      np.array(self.coordinate_systems['out_coor'].oversample)
            s_lin = s_lin * ml_step[0]
            s_pix = s_pix * ml_step[1]

        self.lines = lines
        self.pixels = pixels
        self.blocks = True

        # Check the overlap and limit the number of lines if needed.
        if self.s_lin >= self.coordinate_systems['out_coor'].shape[0] or self.s_pix >= self.coordinate_systems['out_coor'].shape[1]:
            print('Start line and pixel are too high')
            return False
        if self.lines > (self.coordinate_systems['out_coor'].shape[0] - self.s_lin) or self.lines == 0:
            self.lines = self.coordinate_systems['out_coor'].shape[0] - self.s_lin
        if self.pixels > (self.coordinate_systems['out_coor'].shape[1] - self.s_pix) or self.pixels == 0:
            self.pixels = self.coordinate_systems['out_coor'].shape[1] - self.s_pix

        self.coordinate_systems['block_coor'] = copy.deepcopy(self.coordinate_systems['out_coor'])
        self.coordinate_systems['block_coor'].first_line += int(s_lin)
        self.coordinate_systems['block_coor'].first_pixel += int(s_pix)
        self.coordinate_systems['block_coor'].shape = [int(self.lines), int(self.pixels)]
        self.block_coor = self.coordinate_systems['block_coor']

        # Check if the input irregular/regular grid is given to select the right inputs.
        if not self.coordinate_systems['in_coor'].same_coordinates(self.coordinate_systems['out_coor'], strict=False):
            if len(list(self.input_info['image_types'])) == 0:
                # If there are no input images it is fine too.
                return True
            if isinstance(self.in_irregular_grids[0], ImageData):
                if self.in_irregular_grids[0].shape == self.coordinate_systems['in_coor'].shape:
                    return True
            if isinstance(self.out_irregular_grids[0], ImageData):
                if self.out_irregular_grids[0].shape == self.coordinate_systems['out_coor'].shape:
                    return True
            # in the case we have to change coordinate system the coordinates of the input/output grid have to be cal-
            # culated beforehand to apply the calculation.
            return False
        return True

    # Handling of in and output data. This is the main
    def load_input_images(self, image_types, processes, file_types, polarisations, data_ids, coor_types, in_coor_types,
                          type_names):
        """
        This function loads the input data needed to do the processing. To do so, the exact source of the input data 
        should be defined using the given variables.
        
        :param list[str] image_types: List of the image types we need (slave/master/ifg). If empty they are assumed to
                    be the same as the output image.
        :param list[str] processes: List of names of the processes
        :param list[str] file_types: List of file types, which are part of these processes.
        :param list[str] polarisations: List of polarisation of inputs. If not defined we select wat is available.
        :param list[str] data_ids: List of data ids. Only used when the same processing step is used more than once in
                    a processing chain. Generally empty.
        :param list[str] coor_types: The type of coordinate systems used. If not specified we assume that all inputs
                    are of the input coordinate system type.
        :param list[str] in_coor_types: The input coordinate system of the the needed input data. For example if you
                    want a specific conversion grid from one projection to another.
        :return:
        """

        n_inputs = len(processes)
        if n_inputs == 0:
            return True

        # First check whether the image_types/coor_types/polarisations/data_ids are empty.
        if image_types == []:
            image_types = [self.out_processing_image for i in range(n_inputs)]
        else:
            check_image_types = [image_type in list(self.processing_images.keys()) for image_type in image_types]
            if False in check_image_types:
                TypeError('Specified image type does not exist in loaded input image types.')

        if coor_types == []:
            coor_types = ['in_coor' for i in range(n_inputs)]
        else:
            check_coor_types = [coor_type in list(self.coordinate_systems.keys()) for coor_type in coor_types]
            if False in check_coor_types:
                TypeError('Specified coordinate type does not exist in loaded coordinate systems.')

        if polarisations == []:
            polarisations = ['' for i in range(n_inputs)]
        if data_ids == []:
            data_ids = ['' for i in range(n_inputs)]

        # Check of they exist and get the images.
        for i, [image_type, process, file_type, polarisation, data_id, coor, in_coor, name] in enumerate(zip(image_types,
                                                                                            processes,
                                                                                            file_types,
                                                                                            polarisations,
                                                                                            data_ids,
                                                                                            coor_types,
                                                                                            in_coor_types,
                                                                                            type_names)):
            # If the grid is only defined to extract the coordinate system for the coordinates input.
            if name in ['in_coor_grid', 'out_coor_grid']:
                continue
            if in_coor != '':
                in_coor = self.coordinate_systems[in_coor]

            image_data = self.processing_images[image_type].\
                processing_image_data_exists(process, self.coordinate_systems[coor], in_coor, data_id, polarisation, file_type)
            if image_data == False:
                raise TypeError('No processing information for ' + process + ' file type ' + file_type + ' found.')
            else:
                self.process_data.add_input_image(image_data, name)
                self.in_images[name] = image_data

        return True

    def load_input_data(self):
        """
        Here we actually load the needed input data. The images should already be loaded

        :return:
        """

        if not self.memory_data:
            return

        # We first load the output data grids. Because they can contain information for loading the input grids.
        self.load_out_coor_input_data()
        self.load_in_coor_input_data()

    def load_in_coor_input_data(self):
        """
        This function is used to load the input data of a function with the input coordinate system.

        """

        if not self.blocks:
            s_lin = 0
            s_pix = 0
            shape = self.coordinate_systems['in_coor'].shape

        elif 'out_irregular_grids' not in self.settings.keys() and 'in_irregular_grids' not in self.settings.keys():
            # Assuming that the input grid and output grid have the same coordinate system.
            s_lin = self.s_lin
            s_pix = self.s_pix
            shape = [self.lines, self.pixels]

        elif 'out_irregular_grids' in self.settings.keys():
            out_grids = self.settings['out_irregular_grids']
            first_line, first_pixel, shape = \
                SelectInputWindow.output_irregular_rectangle(np.copy(self[out_grids[0]]),
                                                             np.copy(self[out_grids[1]]),
                                                             max_shape=self.coordinate_systems['in_coor'].shape,
                                                             min_line=self.coordinate_systems['in_coor'].first_line,
                                                             min_pixel=self.coordinate_systems['in_coor'].first_pixel,
                                                             buf=self.buf)

            self.coordinate_systems['in_block_coor'].shape = shape
            self.coordinate_systems['in_block_coor'].first_line = first_line
            self.coordinate_systems['in_block_coor'].first_pixel = first_pixel
            s_lin = first_line - self.coordinate_systems['in_coor'].first_line
            s_pix = first_pixel - self.coordinate_systems['in_coor'].first_pixel

            # print('Shape of used inputs is ' + str(shape[0]) + ' lines and ' + str(shape[1]) + ' pixels')

        elif 'in_irregular_grids' in self.settings.keys():
            in_grids = self.settings['in_irregular_grids']
            first_line, first_pixel, shape = \
                SelectInputWindow.input_irregular_rectangle(self.in_images[in_grids[0]].disk['data'],
                                                             self.in_images[in_grids[1]].disk['data'],
                                                             shape=self.coordinate_systems['block_coor'].shape,
                                                             s_lin=self.coordinate_systems['block_coor'].first_line,
                                                             s_pix=self.coordinate_systems['block_coor'].first_pixel,
                                                             buf=self.buf)

            self.coordinate_systems['in_block_coor'].shape = shape
            self.coordinate_systems['in_block_coor'].first_line = first_line
            self.coordinate_systems['in_block_coor'].first_pixel = first_pixel
            s_lin = first_line
            s_pix = first_pixel

        else:
            raise AttributeError('Not possible to load needed input range for in coordinate system. Aborting.')

        keys = [key for key, input_type in zip(self.in_images.keys(), self.input_info['coor_types']) if input_type == 'in_coor']
        for key in keys:
            success = self.in_images[key].load_memory_data(shape, s_lin, s_pix)
            if not success:
                raise ValueError('Input data ' + key + ' for ' + self.process_name + ' from image ' +
                                 self.in_images[key].folder + ' cannot be loaded.')

    def load_out_coor_input_data(self):
        """
        This function is used to load the input data of a function with the output coordinate system.

        """

        if not self.blocks:
            s_lin = 0
            s_pix = 0
            shape = self.coordinate_systems['out_coor'].shape

        else:
            s_lin = self.s_lin
            s_pix = self.s_pix
            shape = [self.lines, self.pixels]

        keys = [key for key, input_type in zip(self.in_images.keys(), self.input_info['coor_types']) if
                input_type == 'out_coor']
        for key in keys:
            success = self.in_images[key].load_memory_data(shape, s_lin, s_pix)
            if not success:
                raise ValueError('Input data ' + key + ' for ' + self.process_name + ' from image ' +
                                 self.in_images[key].folder + ' cannot be loaded.')

    def create_memory(self, file_types=[]):
        """
        This function is used to create the outputs of the process to memory.

        :param list[str] file_types: File type of output grid
        :return:
        """

        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            succes = self.out_images[file_type].new_memory_data([self.lines, self.pixels], self.s_lin, self.s_pix)

    # Next steps are not performed during the processing. These are executed before (create_output_data_file) or after
    # (save_to_disk, clean_memory) processing. However, they only need to be seperated when we work with data blocks.
    # Therefore, there is an added function beside __call__, called call_full, which does so.
    def create_output_data_files(self, file_types=[]):
        """
        This function preallocates the output files on disk. So it creates only files with zeros which can be filled
        later on in the process.

        :param list[str] file_types: File types to be preallocated on disk. If not specified, all output files will be
                        preallocated on disk.
        :return:
        """

        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            create_new = self.out_images[file_type].create_disk_data()
            if not create_new:
                load_new = self.out_images[file_type].load_disk_data()
                if not load_new:
                    raise FileNotFoundError('Data on disk not found and not able to create')

        # If all files are succesfully loaded.
        return True

    def save_to_disk(self, file_types=[]):
        """
        This function saves all data that is stored in memory to disk. So during a processing step the data is first
        stored in memory and then saved to disk. This makes it possible to keep data in memory and use it for a proces
        sing pipeline. This function will therefore only be used as the last step of a pipeline.

        :param list(str) file_types: File types which are saved to disk. If not specified all file types are used.
        :return:
        """

        # Save data from memory to disk.
        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            save_data = self.out_images[file_type].save_memory_data_to_disk()
            if not save_data:
                return False

        # If all files are succesfully saved.
        return True

    def clean_memory(self):
        """
        Clean all data from memory related to this processing step (in and output data). This is generally done after
        the processing of a single step or a processing pipeline is finished. This will prevent the use of to memory
        space for processing.

        :return:
        """

        for image in self.in_images.keys():
            self.in_images[image].remove_memory_data()
        for image in self.out_images.keys():
            self.out_images[image].remove_memory_data()
