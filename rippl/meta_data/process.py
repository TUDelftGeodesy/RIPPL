'''

This is a super class for all processing step. This class manages:
- Importing data from a processing stack (either SLC or interferogram)
- Initialize a processing data/metadata object.
- Writing results to disk if needed
- Removing results/ins from memory

'''

import numpy as np
import copy
import json
import logging

from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.process_data import ProcessData
from rippl.meta_data.image_data import ImageData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.coor_new_extend import CoorNewExtend

class Process():

    def initialize(self, input_info, output_info, coordinate_systems, processing_images, settings, overwrite):
        """
        Input steps Process. These input steps are a summary of the inputs and outputs of every processing step.
        The reading and writing of data and creation of metadata information is all summarized in this step.

        :param dict[list[str]] input_info: Input data information of your processing step which consists of
                - process_types > the names of the processes of the input data
                - file_types > the names of the file types of the inputs
                - image_types > the names of the images used (images should be defined in images parameter)
                - type_names > self-assigned names to call the variables during the processing. If missing it will be
                                replaced by the file_types
                - data_ids > possible data_ids of inputs. (empty by default)
                - polarisation > polarisation of inputs. (only applicable for radar data, otherwises it should be none)
                - coor_types > type of coordinate systems of the inputs. The coordinate systems themselves should be
                                given in the coordinates input variable.
        :param dict[str or list[str]] output_info: Information on the outputs of the processing step which consists of
                - process_name > name of the process
                - image_type > name of image used for output. Should be defined in the images' parameter
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
        self.settings = self.check_settings(settings)
        self.processing_images = self.check_processing_images(processing_images, self.input_info, self.output_info)
        self.coordinate_systems = self.check_coordinate_systems(coordinate_systems, self.input_info, self.output_info)

        # Predefine all image definitions
        self.process_name = output_info['process_name']     # type: str
        self.process = []                                   # type: ProcessData

        # Input and output image (secondary/primary/ifg etc.)
        self.out_processing_image = self.processing_images[self.output_info['image_type']]

        # Input and output image data. To load from disk to memory and save results
        self.in_images = dict()                              # type: dict(ImageData)
        self.in_images_types = dict()
        self.out_images = dict()                             # type: dict(ImageData)
        self.out_images_types = dict()

        # Create the process metadata and images. Load the input images too.
        self.overwrite = overwrite
        if not overwrite:
            if 'in_coor' in self.coordinate_systems.keys():
                in_coor = self.coordinate_systems['in_coor']
            else:
                in_coor = ''
            process_exist = self.check_output_exists(self.out_processing_image, self.process_name,
                                                      self.coordinate_systems['out_coor'], in_coor,
                                                      self.output_info['polarisation'],
                                                      self.output_info['data_id'],
                                                      self.output_info['file_names'])
            self.process_finished = process_exist[0]
            self.process_on_disk = process_exist[1]
            if self.process_finished and self.process_on_disk:
                return
        else:
            self.process_finished = False
            self.process_on_disk = False

        self.create_process_metadata(self.output_info['polarisation'], self.output_info['data_id'], settings)
        self.create_output_images(self.output_info['file_names'], self.output_info['data_types'], [])

        # Information for processing of dataset in chunks. If you want to do so, run the define_chunk method.
        self.chunks = False

        # Finally set a few variables for the multilooking case.
        self.no_chunk = 0
        self.no_chunks = 0
        self.no_lines = 1
        self.chunk_shape = (0, 0)
        self.ml_in_data = dict()
        self.ml_out_data = dict()

    def __call__(self, memory_in=True, scratch_disk_dir='', internal_memory_dir=''):
        """
        This function does basically the same as the __call__ function, but assumes that we apply no multiprocessing
        and or pipeline processing. Therefore, it includes an extended number of steps:
        1. Reading in of input data
        2. Creation of output data on disk
        3. Create output memory files
        4. Perform the actual processing
        5. Save data to disk
        6. Clean memory steps.

        :return:
        """

        self.initialize(input_info=self.input_info,
                        output_info=self.output_info,
                        coordinate_systems=self.coordinate_systems,
                        processing_images=self.processing_images,
                        overwrite=self.overwrite,
                        settings=self.settings)
        if self.process_finished and self.process_on_disk:
            logging.info('Process already finished')
            return

        # Create the input and output info
        self.load_input_info()
        self.load_input_data_files(scratch_disk_dir=scratch_disk_dir, internal_memory_dir=internal_memory_dir)
        self.load_input_data(scratch_disk_dir=scratch_disk_dir, internal_memory_dir=internal_memory_dir)
        self.create_output_data_files()
        self.create_memory()
        self.process_calculations()

        self.save_to_disk()

        self.clean_memory()
        self.out_processing_image.save_json()

        self.process_finished = True

    def fake_processing(self, output_file_name='', output_file_type=''):
        """
        This function creates the same processing step in metadata without actually doing the processing. This is
        useful in some cases.

        """

        self.initialize(input_info=self.input_info,
                        output_info=self.output_info,
                        coordinate_systems=self.coordinate_systems,
                        processing_images=self.processing_images,
                        overwrite=self.overwrite,
                        settings=self.settings)
        if self.process_finished and self.process_on_disk:
            logging.info('Process already finished')
            return

        self.out_processing_image.update_json()
        # Change the filenames
        if output_file_name:
            for output_file_key in self.process.json_dict['output_files'].keys():
                new_name = self.process.json_dict['output_files'][output_file_key]['file_name']
                self.process.json_dict['output_files'][output_file_key]['file_name'] = new_name.replace(output_file_key, output_file_name)
        if output_file_type:
            for output_file_key in self.process.json_dict['output_files'].keys():
                self.process.json_dict['output_files'][output_file_key]['dtype'] = output_file_type

        self.out_processing_image.save_json(update=False)
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
            raise LookupError('The input or output dataset ' + key + ' in processing calculations for'
                              + str(self.__class__) + ' does not exist.')

    def __setitem__(self, key, data):
        # Set the data of one variable in memory.
        if key in list(self.in_images.keys()):
            self.in_images[key].memory['data'] = data
        elif key in list(self.out_images.keys()):
            self.out_images[key].memory['data'] = data
        else:
            raise LookupError('The input or output dataset ' + key + ' in processing calculations for'
                              + str(self.__class__) + ' does not exist.')

    def load_input_info(self):
        """
        This method initializes the input and output.

        :return:
        """

        self.load_input_images(self.input_info['image_types'],
                               self.input_info['process_names'],
                               self.input_info['file_names'],
                               self.input_info['polarisations'],
                               self.input_info['data_ids'],
                               self.input_info['coor_types'],
                               self.input_info['in_coor_types'],
                               self.input_info['aliases_processing']) # We load the input images only. The data is not loaded to memory yet.

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
        primary function this method is empty, as there should be an override of this function in the child functions.

        :return:
        """

        pass

    def def_out_coor(self):
        """
        This function is used to define the shape of the output coordinates based on the input coordinates. Function
        specific information is given in

        :return:
        """

    def create_process_metadata(self, polarisation, data_id, settings):
        """
        Create the process data object. This creates the metadata of the image and creates a link to the output data
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
        new_process = ProcessData(self.out_processing_image.folder, self.output_info['process_name'],
                                        coordinates=coordinates, in_coordinates=in_coordinates,
                                        settings=settings, polarisation=polarisation, data_id=data_id)
        process = self.out_processing_image.add_process(new_process)
        self.process = process

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
        self.process.add_process_images(file_types, data_types, shapes)
        # Define the output images.
        self.out_images = self.process.images

    def check_input_info(self, input_info):
        '''
        Check whether there is no information missing in the output data.

        :param dict() input_info:
        :return:
        '''

        if not isinstance(input_info, dict):
            raise TypeError('Output info should be a dictionary for ' + str(self.__class__) + '.')

        input_keys = set(input_info.keys())
        needed_keys = set({'process_names', 'coor_types', 'file_names'})
        if len(needed_keys - input_keys) > 0:
            raise LookupError(
                'One of the needed process_names, coor_types or file_names is missing in the input_info for ' + str(self.__class__) + '.')
        n = len(input_info['file_names'])
        if n != len(input_info['image_types']) or n != len(input_info['process_names']):
            raise LookupError('Number of image types or process types is not correct for ' + str(self.__class__) + '.')
        if not 'polarisations' in input_keys:
            input_info['polarisation'] = ['' for i in range(n)]
        elif n != len(input_info['polarisations']):
            raise LookupError('Number of polarisations is not the same as other parameters for ' + str(self.__class__) + '.')
        if not 'data_ids' in input_keys:
            input_info['data_id'] = ['' for i in range(n)]
        elif n != len(input_info['data_ids']):
            raise LookupError('Number of data_ids is not the same as other parameters for ' + str(self.__class__) + '.')
        if not 'in_coor_types' in input_keys:
            input_info['in_coor_types'] = ['' for i in range(n)]
        elif n != len(input_info['in_coor_types']):
            raise LookupError('Number of in_coor_types is not the same as other parameters for ' + str(self.__class__) + '.')
        if not 'aliases_processing' in input_keys:
            input_info['aliases_processing'] = input_info['file_names']
        elif n != len(input_info['file_names']):
            raise LookupError('Number of file_types is not the same as other parameters for ' + str(self.__class__) + '.')

        try:
            json.dumps(input_info)
        except Exception as e:
            raise TypeError('Input info should only hold str values to be JSON Serializable. Check the input settings '
                            'for your processing step ' + str(self.__class__) + '. ' + str(e))

        return input_info

    def check_output_info(self, output_info):
        '''
        Check whether there is no information missing in the output data.

        :param dict() output_info:
        :return:
        '''

        if not isinstance(output_info, dict):
            raise TypeError('Output info should be a dictionary for ' + str(self.__class__) + '.')

        output_keys = set(output_info.keys())
        needed_keys = set({'process_name', 'coor_type', 'file_names', 'data_types'})
        if len(needed_keys - output_keys) > 0:
            raise LookupError('One of the needed process, coor_type, file_types or data_types is missing in the output_info of ' + str(self.__class__))
        if len(output_info['file_names']) != len(output_info['data_types']):
            raise LookupError('data_types and file_types are not the same length for ' + str(self.__class__) + '. Both should be lists')
        if not 'polarisation' in output_keys:
            output_info['polarisation'] = ''
        if not 'data_id' in output_keys:
            output_info['data_id'] = ''
        if not 'name_type' in output_keys:
            output_info['aliases_processing'] = output_info['file_names']

        try:
            json.dumps(output_info)
        except Exception as e:
            raise TypeError('Output info should only hold str values to be JSON Serializable. Check the output settings '
                            'for your processing step ' + str(self.__class__) + '. ' + str(e))

        return output_info

    def check_settings(self, settings):
        """
        This function checks the validity of the input settings. By first replacing any numpy values to float/int values
        and

        """

        # We check for two levels deep. We do not allow further nesting of settings.
        if not isinstance(settings, dict):
            raise TypeError('Settings should be a dictionary for ' + str(self.__class__) + '.')
        for key in settings.keys():
            # Check for numpy values in first layer.
            val = settings[key]
            if isinstance(val, np.integer):
                settings[key] = int(val)
            elif isinstance(val, np.floating):
                settings[key] = float(val)
            elif isinstance(val, dict):
                for nested_key in settings[key].keys():
                    # Check for numpy values in first layer.
                    nested_val = settings[key][nested_key]
                    if isinstance(nested_val, np.integer):
                        settings[key][nested_key] = int(nested_val)
                    elif isinstance(nested_val, np.floating):
                        settings[key][nested_key] = float(nested_val)
                    elif isinstance(nested_val, dict):
                        raise TypeError('Only create nested dictionaries of max 2 layers in settings of your processing'
                                        'step. Check the input you provided to ' + str(self.__class__) + '.')

        try:
            json.dumps(settings)
        except Exception as e:
            raise TypeError('Output info should only hold str values to be JSON Serializable. Check the output settings '
                            'for your processing step ' + str(self.__class__) + '. ' + str(e))

        return settings

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
        for key in list(images.keys()):
            if not isinstance(images[key], ImageProcessingData):
                images.pop(key)
                if isinstance(images[key], list):
                    logging.info('Please provide a single image as input for ' + key + ' and not a list. Data not used in process')
                else:
                    logging.info('Image data for ' + key + ' is not a ImageProcessingData (or child class) object, data not used in process.')
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
        Finally, the chunk coor coordinate system is added.

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

        coordinates['out_coor_chunk'] = copy.copy(coordinates['out_coor'])
        if 'in_coor' in coordinates.keys():
            coordinates['in_coor_chunk'] = copy.copy(coordinates['in_coor'])

        return coordinates

    def define_coordinate_system_size(self, reference_coor):
        """
        Here we define the size of the full output image for this process, if it is not defined yet.

        """

        if 'out_coor' in self.coordinate_systems.keys():
            if not self.coordinate_systems['out_coor'].same_coordinates(reference_coor, strict=False):
                if 'out_coor' not in self.settings.keys():
                    self.settings['out_coor'] = dict()

                self.coordinate_systems['out_coor'] = CoorNewExtend(reference_coor,
                                                                    self.coordinate_systems['out_coor'],
                                                                    min_height=self.settings['out_coor'].get('min_height', 0),
                                                                    max_height=self.settings['out_coor'].get('max_height', 0),
                                                                    corners_midpoints=True,
                                                                    buffer=self.settings['out_coor'].get('buffer', 0),
                                                                    rounding=self.settings['out_coor'].get('rounding', 0),
                                                                    out_coor_limits = True
                                                                    ).out_coor
            else:
                self.coordinate_systems['out_coor'] = copy.deepcopy(reference_coor)

        if 'in_coor' in self.coordinate_systems.keys():
            if not self.coordinate_systems['in_coor'].same_coordinates(reference_coor, strict=False):
                if 'in_coor' not in self.settings.keys():
                    self.settings['in_coor'] = dict()

                self.coordinate_systems['in_coor'] = CoorNewExtend(reference_coor,
                                                                   self.coordinate_systems['in_coor'],
                                                                   min_height=self.settings['in_coor'].get('min_height', 0),
                                                                   max_height=self.settings['in_coor'].get('max_height', 0),
                                                                   corners_midpoints=True,
                                                                   buffer=self.settings['in_coor'].get('buffer', 0),
                                                                   rounding=self.settings['in_coor'].get('rounding', 0),
                                                                   out_coor_limits=True
                                                                   ).out_coor
            else:
                self.coordinate_systems['in_coor'] = copy.deepcopy(reference_coor)

    def define_processing_chunk(self, reference_coor_chunk, s_lin=0, s_pix=0):
        """
        Here we check whether the processing chunks can be processed based on the given sizes. Also, we check whether
        it is possible to do chunk processing
        If the in coordinates and output coordinates are not the same either an irregular in or output grid is
        needed to do the reading in.
        Exception for this are the 'regular' multilooking, which does not need any precreated grid.

        :param CoordinateSystem reference_coor_chunk: Coordinate system of input chunk
        :param int s_lin: Start line
        :param int s_pix: Start pixel
        :return:
        """

        self.s_lin = s_lin
        self.s_pix = s_pix

        if self.coordinate_systems['out_coor'].grid_type == 'radar_coordinates':
            ml_step = np.array(self.coordinate_systems['out_coor'].multilook) / \
                      np.array(self.coordinate_systems['out_coor'].oversample)
            self.s_lin = (self.s_lin * ml_step[0]).astype(np.int64)
            self.s_pix = (self.s_pix * ml_step[1]).astype(np.int64)

        self.chunks = True

        # Check if the input and output coordinates are the same and if not calculate the coverage for these coordinate
        # systems.
        if 'out_coor' in self.coordinate_systems.keys():
            if not self.coordinate_systems['out_coor'].same_coordinates(reference_coor_chunk, strict=False):
                self.coordinate_systems['out_coor_chunk'] = CoorNewExtend(reference_coor_chunk,
                                                                   self.coordinate_systems['out_coor'],
                                                                   min_height=self.settings['out_coor'].get('min_height', 0),
                                                                   max_height=self.settings['out_coor'].get('max_height', 0),
                                                                   corners_midpoints=True,
                                                                   buffer=self.settings['out_coor'].get('buffer', 0),
                                                                   rounding=self.settings['out_coor'].get('rounding', 0),
                                                                   out_coor_limits=True
                                                                   ).out_coor
            else:
                self.coordinate_systems['out_coor_chunk'] = copy.deepcopy(reference_coor_chunk)

        if 'in_coor' in self.coordinate_systems.keys():
            if not self.coordinate_systems['in_coor'].same_coordinates(reference_coor_chunk, strict=False):
                self.coordinate_systems['in_coor_chunk'] = CoorNewExtend(reference_coor_chunk,
                                                                   self.coordinate_systems['in_coor'],
                                                                   min_height=self.settings['in_coor'].get('min_height', 0),
                                                                   max_height=self.settings['in_coor'].get('max_height', 0),
                                                                   corners_midpoints=True,
                                                                   buffer=self.settings['in_coor'].get('buffer', 0),
                                                                   rounding=self.settings['in_coor'].get('rounding', 0),
                                                                   out_coor_limits=True
                                                                   ).out_coor
            else:
                self.coordinate_systems['in_coor_chunk'] = copy.deepcopy(reference_coor_chunk)

    # Handling of in and output data. This is the main
    def load_input_images(self, image_types, processes, file_types, polarisations, data_ids, coor_types, in_coor_types,
                          type_names):
        """
        This function loads the input data needed to do the processing. To do so, the exact source of the input data 
        should be defined using the given variables.
        
        :param list[str] image_types: List of the image types we need (secondary/primary/ifg). If empty they are assumed to
                    be the same as the output image.
        :param list[str] processes: List of names of the processes
        :param list[str] file_types: List of file types, which are part of these processes.
        :param list[str] polarisations: List of polarisation of inputs. If not defined we select wat is available.
        :param list[str] data_ids: List of data ids. Only used when the same processing step is used more than once in
                    a processing chain. Generally empty.
        :param list[str] coor_types: The type of coordinate systems used. If not specified we assume that all inputs
                    are of the input coordinate system type.
        :param list[str] in_coor_types: The input coordinate system of the needed input data. For example if you
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
            if in_coor != '':
                in_coor = self.coordinate_systems[in_coor]

            image_data = self.processing_images[image_type].\
                processing_image_data_exists(process, self.coordinate_systems[coor], in_coor, data_id, polarisation, file_type)
            if image_data == False:
                main_error = 'Error while loading input data for process ' + str(self.__class__) + '. '
                error_message = ''
                # First check whether this data exists but for another coordinate system
                coors = self.processing_images[image_type].processing_image_data_iterator(
                    processes=[process], coordinates=[], in_coordinates=[in_coor], data_ids=[data_id],
                    polarisations=[polarisation], file_types=[file_type])[2]
                if len(coors) > 0:
                    self.coordinate_systems[coor].create_short_coor_id()
                    error_message += 'Data for coordinate system ' + self.coordinate_systems[coor].short_id_str + \
                                     ' with shape ' + str(self.coordinate_systems[coor].shape) + ' ' ' not available ' \
                                     'but alternative coordinate systems with coordinates '
                    for coordinates in coors:
                        coordinates.create_short_coor_id()
                        error_message += coordinates.short_id_str + ' with shape ' + str(coordinates.shape) + ' '
                    error_message += 'available.'

                # For the input coordinates
                coors = self.processing_images[image_type].processing_image_data_iterator(
                    processes=[process], coordinates=[self.coordinate_systems[coor]], in_coordinates=[], data_ids=[data_id],
                    polarisations=[polarisation], file_types=[file_type])[3]
                if len(coors) > 0:
                    in_coor.create_short_coor_id()
                    error_message += 'Data with input coordinate system ' + in_coor.short_id_str + \
                                     ' with shape ' + str(in_coor.shape) + ' ' ' not available ' \
                                     'but alternative input coordinate systems with coordinates '
                    for coordinates in coors:
                        coordinates.create_short_coor_id()
                        error_message += coordinates.short_id_str + ' with shape ' + str(coordinates.shape) + ' '
                    error_message += 'available.'

                # Then check wether a dataset could be found for a different polarisation
                diff_pol_available = self.processing_images[image_type].processing_image_data_iterator(
                    processes=[process], coordinates=[self.coordinate_systems[coor]], in_coordinates=[in_coor], data_ids=[data_id],
                    polarisations=[], file_types=[file_type])[-1]
                if len(diff_pol_available):
                    error_message += ' Image with polarisation ' + polarisation + ' not available, but alternative ' \
                                     ' polarisation are available.'
                # And data id
                diff_data_id_available = self.processing_images[image_type].processing_image_data_iterator(
                    processes=[process], coordinates=[self.coordinate_systems[coor]], in_coordinates=[in_coor], data_ids=[],
                    polarisations=[polarisation], file_types=[file_type])[-1]
                if len(diff_data_id_available):
                    error_message += ' Image with data id ' + data_id + ' not available, but alternative ' \
                                                                                  ' data ids are available.'

                # Raise the error
                raise TypeError(main_error + 'No processing information for ' + process + ' file type ' + file_type + ' with '
                                'coordinate system ' + self.coordinate_systems[coor].short_id_str + ' found. Data '
                                ' should be loaded from ' + image_type + ' but does not exist. ' + error_message)
            else:
                self.process.add_input_image(image_data, name)
                self.in_images[name] = image_data
                self.in_images_types[name] = image_type

        return True

    def load_input_data_files(self, scratch_disk_dir='', internal_memory_dir=''):
        """
        Load needed input data files as memmap
        """

        if not internal_memory_dir:
            internal_memory_dir = scratch_disk_dir

        file_types = list(self.in_images.keys())

        for file_type in file_types:
            load_input = self.in_images[file_type].load_disk_data([internal_memory_dir, scratch_disk_dir])

            if not load_input:
                raise FileNotFoundError('Data on disk not found.')

        # If all files are succesfully loaded.
        return True

    def load_input_data(self, scratch_disk_dir='', internal_memory_dir=''):
        """
        This function is used to load the input data of a function with the output coordinate system.

        """

        if not internal_memory_dir:
            internal_memory_dir = scratch_disk_dir

        for in_coor_type in list(set(self.input_info['coor_types'])):
            in_coor_chunk_type = in_coor_type + '_chunk'

            if in_coor_chunk_type in self.coordinate_systems.keys():
                coor = self.coordinate_systems[in_coor_chunk_type]
            elif in_coor_type in self.coordinate_systems.keys():
                coor = self.coordinate_systems[in_coor_type]
            else:
                raise TypeError('No coordinate system for ' + in_coor_type + ' or ' + in_coor_chunk_type + ' available. '
                                'Make sure that you added an out coordinate system to the processing step.')

            keys = [key for key, input_type in zip(self.in_images.keys(), self.input_info['coor_types']) if
                    input_type == in_coor_type]
            new_coor = copy.deepcopy(coor)
            for key in keys:
                # And check the limits of the input data and needed input chunk.
                source_coor = self.in_images[key].coordinates
                # First check the source location. Normally these should be the same for the whole stack.
                offset_lines, offset_pixels = source_coor.get_offset(new_coor)

                # Select only the overlapping area for lines
                if offset_lines < 0:
                    new_coor.shape = [new_coor.shape[0] - np.abs(offset_lines), new_coor.shape[1]]
                    new_coor.first_line += np.abs(offset_lines)
                else:
                    s_lin = offset_lines
                if offset_lines + new_coor.shape[0] > source_coor.shape[0]:
                    new_coor.shape = [source_coor.shape[0] - offset_lines, new_coor.shape[1]]
                # And for pixels
                if offset_pixels < 0:
                    new_coor.shape = [new_coor.shape[0], new_coor.shape[1] - np.abs(offset_pixels)]
                    new_coor.first_pixel += np.abs(offset_pixels)
                else:
                    s_pix = offset_pixels
                if offset_pixels + new_coor.shape[1] > source_coor.shape[1]:
                    new_coor.shape = [new_coor.shape[0], source_coor.shape[1] - offset_pixels]

            # Check the new shape for negative or zero values
            if new_coor.shape[0] <= 0 or new_coor.shape[1] <= 0:
                logging.info('Input data for ' + str(self.__class__) + ' from image ' +  self.in_images[key].folder
                        + ' cannot be loaded. No overlap between needed input and actual input data. Using zero values'
                          ' instead')
                new_coor.shape = [0, 0]
            # Notify is shape adjusted
            elif new_coor.shape != coor.shape:
                logging.info('Input data for ' + str(self.__class__) + ' from image ' +  self.in_images[key].folder
                        + ' has only partial overlap. Only overlapping data is loaded for processing.')

            if in_coor_chunk_type in self.coordinate_systems.keys():
                self.coordinate_systems[in_coor_chunk_type] = new_coor
            elif in_coor_type in self.coordinate_systems.keys():
                self.coordinate_systems[in_coor_type] = new_coor

            # Then load the data.
            for key in keys:
                source_coor = self.in_images[key].coordinates
                s_lin, s_pix = source_coor.get_offset(new_coor)
                # In case no overlapping data is available
                if new_coor.shape == [0,0]:
                    # Create fake memory dataset.
                    dtype = self.in_images[key].dtype_memory[self.in_images[key].dtype]
                    self.in_images[key].memory['data'] = np.zeros((0, 0)).astype(dtype)

                    # Updata meta data
                    self.in_images[key].memory['meta']['s_lin'] = s_lin
                    self.in_images[key].memory['meta']['s_pix'] = s_pix
                    self.in_images[key].memory['meta']['shape'] = new_coor.shape
                else:

                    success = self.in_images[key].load_memory_data(new_coor.shape, s_lin, s_pix, tmp_directories=[internal_memory_dir, scratch_disk_dir])
                    if not success:
                        raise ValueError('Input data ' + key + ' for ' + self.process_name + ' for processing ' +
                                         str(self.__class__) + ' from image ' + self.in_images[key].folder + ' cannot be loaded.')

    def create_memory(self, file_types=[]):
        """
        This function is used to create the outputs of the process to memory. This is an essential step in the parallel
        processing. Before any data is created in one of the processing step, first the outputs should be generated in
        memory.

        :param list[str] file_types: File type of output grid
        :return:
        """

        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        if self.chunks:
            out_coor = self.coordinate_systems['out_coor_chunk']        # type: CoordinateSystem
            out_coor_orig = self.coordinate_systems['out_coor']         # type: CoordinateSystem
            s_lin, s_pix = out_coor_orig.get_offset(out_coor)
        else:
            out_coor = self.coordinate_systems['out_coor']
            s_lin = 0
            s_pix = 0

        for file_type in file_types:
            succes = self.out_images[file_type].new_memory_data(out_coor.shape, s_lin, s_pix)

    # Next steps are not performed during the processing. These are executed before (create_output_data_file) or after
    # (save_to_disk, clean_memory) processing. However, they only need to be seperated when we work with data chunks.
    # Therefore, there is an added function beside __call__, called call_full, which does so.
    def create_output_data_files(self, file_types=[], tmp_directories=[]):
        """
        This function preallocates the output files on disk. So it creates only files with zeros which can be filled
        later on in the process. This can only be done if we are not in a parallel processing run. Otherwise, this could
        cause a system crash!

        :param list[str] file_types: File types to be preallocated on disk. If not specified, all output files will be
                        preallocated on disk.
        :return:
        """

        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            create_new = self.out_images[file_type].create_disk_data(tmp_directories=tmp_directories)
            if not create_new:
                load_new = self.out_images[file_type].load_disk_data()
                if not load_new:
                    raise FileNotFoundError('Data on disk not found and not able to create')

        # If all files are succesfully loaded.
        return True

    def load_output_data_files(self, file_types=[]):
        """
        Load output data files during processing. Return an error if files do not exist.

        In a processing pipeline the output datasets are first created on disk to create a space for the parallel
        processes to write their data to disk. This should be done before the parallel processing is started because
        individual parallel processes cannot do so because if they could the different parallel processes could create
        the same file at the same moment, which would cause the system to crash.

        """

        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            load_output = self.out_images[file_type].load_disk_data()
            if not load_output:
                file_path = self.out_images[file_type].file_path
                raise FileNotFoundError('Data on disk not found for ' + file_path)

        # If all files are succesfully loaded.
        return True

    def save_to_disk(self, file_types=[]):
        """
        This function saves all data that is stored in memory to disk. So during a processing step the data is first
        stored in memory and then saved to disk. This makes it possible to keep data in memory and use it for a process
        in the processing pipeline. This function will therefore only be used as the last step of a pipeline.

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

    def flush_to_disk(self, file_types=[]):
        """
        This function flushes all memmap data to disk. This is the last step for multilooking of other steps where
        data is directly written to disk.

        :param list(str) file_types: File types which are saved to disk. If not specified all file types are used.
        :return:

        TODO This function can be removed after multilooking is implemented in parallel way
        """

        # Save data from memory to disk.
        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            save_data = self.out_images[file_type].disk['data'].flush()
            if not save_data:
                return False

        # If all files are succesfully saved.
        return True

    def remove_memmap_files(self):
        """
        Remove all loaded memmap files for this function. This can be done after processing is finished. Mainly used to
        clear up memory space after the processing is finished.

        """

        for image in self.out_images.keys():
            self.out_images[image].remove_disk_data_memmap()            # type: ImageData
        for image in self.in_images.keys():
            self.in_images[image].remove_disk_data_memmap()             # type: ImageData

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
