import multiprocessing
import os
import numpy as np
import copy
import inspect

from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.process import Process
from rippl.run_parallel import run_parallel


class Pipeline():


    def __init__(self, pixel_no=1000000, processes=1):
        """
        The pipeline function is used to create pipelines of functions. This serves several purposes:
        1. Results from different functions can be saved to memory, reducing disk input/output
        2. Saves disk space as only the final output has to be saved to disk
        3. Makes the splitting of the images in parts possible, which enables multiprocessing.

        :param int pixel_no: Number of pixels processed per individual block
        :param int processes: Number of processes used for multiprocessing. For debugging use 1 core.
        """

        # The steps that will be performed in this pipeline
        self.processing_steps = []
        self.save_processsing_steps = []

        # The processing data that will be processed.
        # This should be defined beforehand otherwise it is not possible to divide it in blocks. The placeholders in
        # processing steps will therefore be replaced later on.
        self.processing_data = dict()
        self.coordinate_systems = dict()

        self.pixel_no = pixel_no
        self.processes = processes

        self.pipelines = []
        self.block_pipelines = []

    def add_processing_data(self, processing_data, typename='slave'):
        """
        With this function the ImageProcessingData steps are added.

        :param list[ImageProcessingData] or ImageProcessingData processing_data: The dataset itself
        :param str typename: The type of the dataset or datasets
        :return:
        """

        if not isinstance(processing_data, list):
            processing_data = [processing_data]
        if typename not in self.processing_data.keys():
            self.processing_data[typename] = []

        for dat in processing_data:

            if not isinstance(dat, ImageProcessingData):
                raise TypeError('Input data should be an ImageProcessingData object')

            self.processing_data[typename].append(dat)

    def add_coordinate_systems(self, coordinate_systems, typename='out_coor'):
        """
        With this function the ImageProcessingData steps are added.

        :param list[CoordinateSystem] or CoordinateSystem coordinate_systems: The used coordinate systems.
        :param str typename: The type of the dataset or datasets
        :return:
        """

        if not isinstance(coordinate_systems, list):
            coordinate_systems = [coordinate_systems]
        if typename not in self.processing_data.keys():
            self.processing_data[typename] = []

        for dat in coordinate_systems:

            if not isinstance(dat, CoordinateSystem):
                raise TypeError('Input data should be an CoordinateSystem object')

            self.processing_data[typename].append(dat)

    def add_processing_step(self, processing_step, save_to_disk):
        """
        Here we build up the pipeline of consecutive steps.

        :param Process processing_step: Add the used steps for the processing. Check if the datasets are not yet loaded
                but are defined in the input. Otherwise remove them.
        :param bool save_to_disk: Should the output of this step be saved to disk. The main purpose of the pipeline
                is to reduce the number of outputs. Therefore, for every step should be defined whether the outputs of
                this step should be saved to disk or not. Generally only the last step of the pipeline is interesting,
                but in some cases
        :return:
        """

        if not isinstance(processing_step, Process):
            raise TypeError('Input should be a Process object')

        for key in processing_step.processing_images.keys():
            if processing_step.processing_images[key] != []:
                processing_step.processing_images[key] = []

        self.save_processsing_steps.append(save_to_disk)


    def __call__(self):

        # First check all processing data and coordinate systems.
        self.check_processing_data()
        self.check_coordinate_systems()

        # Then assign the processing data and coordinate systems.
        self.assign_coordinate_systems_processing_data()

        # Finally create the block sizes.
        self.divide_processing_blocks()

        # Create the output files. This is not done using multiprocessing, as it only assigns disk space which is
        # generally fast.

        # First check whether there are no duplicates of the same ImageProcessingData objects which are used to write
        # outputs. This could cause problems in two ways:
        # 1. It is not possible to write to the same file at the same moment, so it could cause an IO conflict
        # 2. This will result in two non-unique metadata outputs, so it will miss part of the outputs in the final
        #           resulting datastack.
        unique_images = []
        for pipeline in self.pipelines:
            for process, save_process in zip(pipeline['processes'], self.save_processsing_steps):
                if save_process:
                    unique_images.append(process.out_processing_image)
        if not len(unique_images) == len(set(unique_images)):
            raise BlockingIOError('Not possible to run parallel processes that have the same output data files.')

        # Create the actual outputs.
        for pipeline in self.pipelines:
            for process, save_process in zip(pipeline['processes'], self.save_processsing_steps):
                process.create_output_images()

        # Now run the individual blocks using multiprocessing of cores > 1
        processing_data_lists = []
        if self.processes == 1:

            for pipeline in self.block_pipelines:
                processing_data_lists.append(run_parallel(pipeline))
        else:
            pool = multiprocessing.Pool(processes=self.processes)

            for result in pool.imap_unordered(run_parallel, self.block_pipelines):
                processing_data_lists.append(result)

        # Finally write the output resfile to disk.
        processing_data = []
        json_files = []

        # First find the unique images. (These will be there multiple times because data is processed in blocks)
        for processing_data_list in processing_data_lists:
            for processing in processing_data_list:
                processing_data.append(processing)
                json_files.append(processing.folder)

        # Write away the .json files.
        unique_json_files, unique_ids = np.unique(json_files, return_index=True)
        for processing in processing_data[unique_ids]:
            processing.update_json()

        # Finally replace the already existing images that were assigned to the pipelines in the first place with the
        # new ones after processing.
        for type_names in self.processing_data.keys():
            for id, processing_data in enumerate(self.processing_data[type_names]):
                self.processing_data[type_names][id] = next(processing for processing in processing_data[unique_ids]
                                                            if processing_data.folder == processing.folder)

    def check_processing_data(self):
        """
        Here we check whether the provided processing data is correct for processing. This includes the following steps.
        - Remove all loaded memmap or loaded memory data to be sure that we can savely copy these variables.
        - Check if the number of images is the same for all types. If there is only one of one type we assume that that
                one will be the same for all other images. This one will be copied for all other cases.

        :return:
        """

        type_names = list(self.processing_data.keys())
        type_len = [len(self.processing_data[type_name]) for type_name in type_names]
        max_len = np.max(type_len)

        for lenght in type_len:
            if type_len != 1 and type_len != max_len:
                raise AssertionError('The number of one of the processing data types is too low.')

        # Now be sure that all memory and memmap data is removed.
        for type_name in type_names:
            for processing_data in self.processing_data[type_name]:
                if not isinstance(processing_data, ImageProcessingData):
                    raise TypeError('All objects in the processing data should be ImageProcessingData objects.')

                processing_data.remove_memmap_files()
                processing_data.remove_memory_files()

        # Finally make sure that all the different types of processing data are aligned to create pipelines.
        if max_len != 1:
            for type_name, length in zip(type_names, type_len):
                if length == 1:
                    self.processing_data[type_name] = [self.processing_data[type_name][0] for n in range(max_len)]

    def check_coordinate_systems(self):
        """
        Always call the check_processing_data before the check_coordinate_systems to prevent false negatives!

        Here we check whether we have the right coordinate systems loaded. This consists of the following steps.
        - Check which coordinate types are needed from the processing steps.
        - Check whether we have all needed coordinate systems as input
        - Check whether the number of datasets and number of coordinate systems are the same. Or if only one coordinate
                system is given create a list of coordinate systems.
        - If the output coordinate system is not defined check if the first function provides a way to calculate the
                exact extend. If so do this for all provided coordinate systems before running the final script.

        :return:
        """

        type_names = list(self.coordinate_systems.keys())
        type_len = [len(self.coordinate_systems[type_name]) for type_name in type_names]
        max_len = np.max(type_len)

        for lenght in type_len:
            if type_len != 1 and type_len != max_len:
                raise AssertionError('The number of one of the coordinate systems types is too low.')

        # Make sure that all the different types of coordinate systems are aligned to create pipelines.
        if max_len != 1:
            for type_name, length in zip(type_names, type_len):
                if length == 1:
                    self.coordinate_systems[type_name] = [self.coordinate_systems[type_name][0] for n in range(max_len)]

        # Finally make sure that the sizes of the output images are already defined.
        out_type = self.processing_steps[0].output_info['coor_type']
        in_types = self.processing_steps[0].input_info['coor_types']

        for i, coor_system in enumerate(self.coordinate_systems[out_type]):
            if coor_system.shape == [0, 0]:
                # If the new coordinates are not defined yet.
                for process in self.processing_steps:
                    methods = inspect.getmembers(process, predicate=inspect.ismethod)
                    if 'define_coordinates_shape' in methods:
                        in_coor_type = inspect.getfullargspec(process.define_coordinates_shape)[0][1]
                        in_coor = self.coordinate_systems[in_coor_type][i]
                        if in_coor.shape == [0, 0]:
                            raise ValueError('The shape of the input coordinate system should be predefined if the '
                                             'shape of the output coordinate system is not known')

                        coor_system = process.define_coordinates_shape(in_coor)


    def assign_coordinate_systems_processing_data(self):
        """
        Be sure you run the check_processing_data and check_coordinate_systems before running this step.

        This method assigns the coordinate systems and processing data to the processing steps. To do so it copies the
        the different processing steps, coordinate_systems and processing_data objects to different packages for
        processing.

        :return:
        """

        self.pipelines = []
        image_types = list(self.processing_data.keys())
        coor_types = list(self.coordinate_systems.keys())

        n_processes = self.processing_data[image_types[0]]
        for i in range(n_processes):

            pipeline = dict()
            pipeline['processes'] = []
            pipeline['save_processes'] = []

            for process, save in zip(self.processing_steps, self.save_processsing_steps):

                process = copy.deepcopy(process)
                for image_type in process.processing_images.keys():
                    if image_type not in image_types:
                        raise TypeError('Image type ' + image_type + ' does not exist.')
                    else:
                        process.processing_images[image_type] = copy.deepcopy(self.processing_data[image_type][i])

                for coor_type in process.coordinate_systems.keys():
                    if image_type not in image_types:
                        raise TypeError('Coordinates type ' + image_type + ' does not exist.')
                    else:
                        process.coordinate_systems[image_type] = copy.deepcopy(self.coordinate_systems[image_type][i])

                pipeline['processes'].append(process)
                pipeline['save_processes'].append(save)

    def divide_processing_blocks(self):
        """
        Be sure you run the assign_coordinate_systems_processing_data before this step.

        This method takes the individual processing packages provided by the assign_coordinate_systems_processing_data
        to create packages for

        :return:
        """

        if self.pixel_no == 0:
            self.block_pipelines = self.pipelines
            return

        self.block_pipelines = []
        for pipeline in self.pipelines:
            # First find the output coordinate system
            coor_type = pipeline['processes'][0].output_info['coor_type']
            coor_system = pipeline['processes'][0].coordinate_systems[coor_type]

            pixel_no = coor_system.shape[0] * coor_system.shape[1]
            if pixel_no < self.pixel_no:
                # In the case the defined block size is larger than the size of the output image, we can run the full
                # image.
                self.block_pipelines.append(pipeline)
            else:
                lines = np.ceil(float(self.pixel_no) / coor_system.shape[1]).astype(np.int32)
                blocks = np.ceil(float(coor_system.shape[0]) / lines).astype(np.int32)

                for n in range(blocks):
                    new_pipeline = copy.deepcopy(pipeline)
                    for process in new_pipeline['processes']:
                        process.define_processing_block(s_lin=n * lines, lines=lines)

                    self.block_pipelines.append(new_pipeline)
