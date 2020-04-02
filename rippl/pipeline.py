import multiprocessing
import os
import numpy as np
import copy
import random

from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.process import Process
from rippl.run_parallel import run_parallel


class Pipeline():


    def __init__(self, pixel_no=1000000, processes=1, run_no_datasets=25):
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
        self.memory_in_processing_steps = []

        # The processing data that will be processed.
        # This should be defined beforehand otherwise it is not possible to divide it in blocks. The placeholders in
        # processing steps will therefore be replaced later on.
        self.processing_data = dict()

        self.pixel_no = pixel_no
        self.processes = processes
        self.run_no_datasets = run_no_datasets
        self.total_blocks = 0

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

            dat.remove_memmap_files()
            dat.remove_memory_files()
            self.processing_data[typename].append(copy.deepcopy(dat))

    def add_processing_step(self, processing_step, save_to_disk, memory_in=False):
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
            if not isinstance(processing_step.processing_images[key], str):
                processing_step.processing_images[key] = key

        self.processing_steps.append(processing_step)
        self.save_processsing_steps.append(save_to_disk)
        self.memory_in_processing_steps.append(memory_in)
        if memory_in == True and self.pixel_no != 0:
            raise BlockingIOError('Not possible to load only from memory and running the code in blocks. Either do not '
                                  'use memory in or set pixel_no to zero.')

    def __call__(self):

        # First check all processing data and coordinate systems.
        self.check_processing_data()
        self.check_coordinate_systems()

        # Number of blocks to run datasets.
        image_types = list(self.processing_data.keys())
        n_datasets = len(self.processing_data[image_types[0]])

        blocks = int(np.ceil(n_datasets / self.run_no_datasets))

        for block in range(blocks):
            # Print which block we are processing.
            print('Processing pipeline block ' + str(block) + ' out of ' + str(blocks))

            # Then assign the processing data and coordinate systems.
            self.assign_coordinate_systems_processing_data(block)

            # Finally create the block sizes.
            self.divide_processing_blocks(block)

            # The output resfile to disk.
            self.processing_datasets = []
            self.json_files = []

            # Create the output files. This is not done using multiprocessing, as it only assigns disk space which is
            # generally fast.

            # First check whether there are no duplicates of the same ImageProcessingData objects which are used to write
            # outputs. This could cause problems in two ways:
            # 1. It is not possible to write to the same file at the same moment, so it could cause an IO conflict
            # 2. This will result in two non-unique metadata outputs, so it will miss part of the outputs in the final
            #           resulting datastack.

            """
            This check is not really necessary...
            
            unique_images = []
            for pipeline in self.pipelines:
                for process, save_process in zip(pipeline['processes'], self.save_processsing_steps):
                    if save_process:
                        unique_images.append(process.out_processing_image)
            if not len(unique_images) == len(set(unique_images)):
                raise BlockingIOError('Not possible to run parallel processes that have the same output data files.')
            """

            # Create the actual outputs.
            for pipeline in self.pipelines:
                for process, save_process in zip(pipeline['processes'], self.save_processsing_steps):
                    if save_process:
                        process.create_output_data_files()

            # Now run the individual blocks using multiprocessing of cores > 1
            processing_data_lists = []
            if self.processes == 1:

                for pipeline in self.block_pipelines:
                    processing_data_lists.append(run_parallel(pipeline))
            else:
                pool = multiprocessing.Pool(processes=self.processes)

                for result in pool.imap_unordered(run_parallel, self.block_pipelines):
                    processing_data_lists.append(result)
                pool.close()


            # First find the unique images. (These will be there multiple times because data is processed in blocks)
            for processing_data_list in processing_data_lists:
                for processing in processing_data_list:
                    self.processing_datasets.append(processing)
                    self.json_files.append(processing.folder)

            self.save_processing_results()

    def save_processing_results(self):
        # Write away the .json files.
        if len(self.processing_datasets) == 0:
            return

        unique_json_files, unique_ids = np.unique(self.json_files, return_index=True)
        for id in unique_ids:
            self.processing_datasets[id].update_json()

        # Finally replace the already existing images that were assigned to the pipelines in the first place with the
        # new ones after processing.
        for type_names in self.processing_data.keys():
            for id, in_processing_data in enumerate(self.processing_data[type_names]):
                in_processing_data = [processing for processing in self.processing_datasets if processing.folder == processing.folder][0]

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

        for length in type_len:
            if length != 1 and length != max_len:
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
        - Load the output coordinate system of the first step
        - Check whether the out coordinate system of the following steps is the same.
        :return:
        """

        coor_out = self.processing_steps[0].coordinate_systems['out_coor']      # type: CoordinateSystem
        coor_out.create_short_coor_id()

        for processing_step in self.processing_steps:

            new_coor_out = processing_step.coordinate_systems['out_coor']
            new_coor_out.create_short_coor_id()

            if coor_out.short_id_str != new_coor_out.short_id_str:
                raise TypeError('All out coordinates systems in a pipeline should be the same!')

    def assign_coordinate_systems_processing_data(self, run_block=0):
        """
        Be sure you run the check_processing_data and check_coordinate_systems before running this step.

        This method assigns the coordinate systems and processing data to the processing steps. To do so it copies the
        the different processing steps, coordinate_systems and processing_data objects to different packages for
        processing.

        :return:
        """

        self.pipelines = []
        image_types = list(self.processing_data.keys())

        # Loop over all the datasets
        n_datasets = len(self.processing_data[image_types[0]])
        max_blocks = int(np.ceil(n_datasets / self.run_no_datasets))

        for i in range(run_block * self.run_no_datasets, np.minimum((run_block + 1) * self.run_no_datasets, n_datasets)):

            pipeline = dict()
            pipeline['processes'] = []
            pipeline['save_processes'] = []
            pipeline['memory_in'] = []

            save_anything = False

            # Loop over the processing steps.
            for n, (process, save, memory_in) in enumerate(zip(self.processing_steps,
                                                               self.save_processsing_steps,
                                                               self.memory_in_processing_steps)):

                process = copy.deepcopy(process)
                image_type_names = list(process.processing_images.keys())
                for image_type_name in image_type_names:
                    image_type = copy.copy(process.processing_images[image_type_name])
                    if image_type not in image_types:
                        raise TypeError('Image type ' + image_type + ' does not exist.')
                    else:
                        process.processing_images[image_type_name] = self.processing_data[image_type][i]

                # If we are looking at the first processing step, we have to define the out coordinate system.
                # For the input coordinate systems this is not really necessary as it is done during the processing.
                if n == 0:
                    process.load_coordinate_system_sizes()
                    out_coor = process.coordinate_systems['out_coor']

                process.coordinate_systems['out_coor'] = copy.deepcopy(out_coor)
                Process.__init__(process, input_info=process.input_info,
                                 output_info=process.output_info,
                                 coordinate_systems=process.coordinate_systems,
                                 processing_images=process.processing_images,
                                 overwrite=process.overwrite,
                                 settings=process.settings)

                if not process.process_finished or not process.process_on_disk:
                    pipeline['processes'].append(process)
                    pipeline['save_processes'].append(save)
                    pipeline['memory_in'].append(memory_in)

                    if save:
                        save_anything = True

            if save_anything:
                self.pipelines.append(pipeline)
            else:
                print('Skipping processing. Process already finished')

    def divide_processing_blocks(self, run_block=0):
        """
        Be sure you run the assign_coordinate_systems_processing_data before this step.

        This method takes the individual processing packages provided by the assign_coordinate_systems_processing_data
        to create packages for

        :return:
        """

        image_types = list(self.processing_data.keys())
        n_datasets = len(self.processing_data[image_types[0]])
        max_blocks = int(np.ceil(n_datasets / self.run_no_datasets))
        self.block_pipelines = []

        if self.pixel_no == 0:
            for pipeline in self.pipelines:
                pipeline['block'] = 0
                pipeline['total_blocks'] = 1
                self.block_pipelines.append(pipeline)

        else:
            for pipeline in self.pipelines:
                # First find the output coordinate system
                coor_type = pipeline['processes'][0].output_info['coor_type']
                coor_system = pipeline['processes'][0].coordinate_systems[coor_type]

                pixel_no = coor_system.shape[0] * coor_system.shape[1]
                if pixel_no < self.pixel_no:
                    # In the case the defined block size is larger than the size of the output image, we can run the full
                    # image.
                    pipeline['block'] = 0
                    pipeline['total_blocks'] = 1
                    self.block_pipelines.append(pipeline)
                else:
                    lines = np.ceil(float(self.pixel_no) / coor_system.shape[1]).astype(np.int32)
                    blocks = np.ceil(float(coor_system.shape[0]) / lines).astype(np.int32)

                    for n in range(blocks):
                        new_pipeline = copy.deepcopy(pipeline)
                        for process in new_pipeline['processes']:
                            process.define_processing_block(s_lin=n * lines, lines=lines)

                        new_pipeline['block'] = n
                        new_pipeline['total_blocks'] = blocks
                        self.block_pipelines.append(new_pipeline)

        random.shuffle(self.block_pipelines)

        self.total_blocks += len(self.block_pipelines)
        for block_no, pipeline in enumerate(self.block_pipelines):
                pipeline['process_block_no'] = block_no + self.total_blocks - len(self.block_pipelines)
                # This is an estimate....
                pipeline['total_process_block_no'] = self.total_blocks + len(self.block_pipelines) * (max_blocks - run_block - 1)
