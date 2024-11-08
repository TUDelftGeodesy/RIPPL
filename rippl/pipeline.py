from multiprocessing import get_context
import numpy as np
import copy
import random
import json
import logging
import datetime

from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.process import Process
from rippl.run_parallel import run_parallel
from rippl.meta_data.stack import Stack


class Pipeline(object):

    def __init__(self, pixel_no=1000000, processes=1, run_no_datasets=32, chunk_orientation='lines', include_reference=False,
                 scratch_disk_dir='', internal_memory_dir='', processing_type='', image_type='', stack=''):
        """
        The pipeline function is used to create pipelines of functions. This serves several purposes:
        1. Results from different functions can be saved to memory, reducing disk input/output
        2. Saves disk space as only the final output has to be saved to disk
        3. Makes the splitting of the images in parts possible, which enables multiprocessing.

        :param int pixel_no: Number of pixels processed per individual chunk
        :param int processes: Number of processes used for multiprocessing. For debugging use 1 core.
        """

        # The steps that will be performed in this pipeline
        self.processing_steps = []
        self.save_processing_steps = []
        self.new_radar_shapes = []
        if processing_type not in ['slc', 'reference_slc', 'secondary_slc', 'ifg']:
            raise TypeError('processing_type should be slc, reference_slc, secondary_slc or ifg! This defines what '
                            'SLCs or ifgs will be processed. '
                            'slc > all available SLCs will be processed (for calibrated amplitude for example)'
                            'reference_slc > only the reference SLC will be processed (common for geocoding)'
                            'secondary_slc > only the secondary SLCs are processed (common for resampling)'
                            'ifg > All interferograms in the stack are processed (for interferogram, coherence etc.)')
        else:
            self.processing_type = processing_type
            self.include_reference = include_reference

        if image_type not in ['slice', 'full']:
            raise TypeError('image_type should either be slice of full! '
                            'slice > processing is done over individual slices (bursts in case of Sentinel-1 data)'
                            'full > processing is done over the full image (concatenated slices in most cases)')
        else:
            self.image_type = image_type

        if not isinstance(stack, Stack):
            raise TypeError('Variable stack should be a rippl Stack object.')
        else:
            self.stack = stack
            # Reload stack meta data
            self.stack.reload_stack()

        # If disk I/O is an issue we can work with a tmp directory
        self.scratch_disk_dir = scratch_disk_dir
        self.internal_memory_dir = internal_memory_dir

        # The processing data that will be processed.
        # This should be defined beforehand otherwise it is not possible to divide it in chunks. The placeholders in
        # processing steps will therefore be replaced later on.
        self.processing_data = dict()
        self.coordinate_systems = dict()

        self.pixel_no = pixel_no
        self.processing_cores = processes
        self.run_no_datasets = run_no_datasets
        self.total_chunks = 0
        self.chunk_orientation = chunk_orientation
        self.json_dicts = []
        self.json_files = []

        self.pipelines = []
        self.chunk_pipelines = []

    def __call__(self):

        # First check all processing data and coordinate systems.
        self.load_processing_data_coordinates()
        self.check_processing_data()

        # Number of chunks to run datasets.
        image_types = list(self.processing_data.keys())
        n_datasets = len(self.processing_data[image_types[0]])

        blocks = int(np.ceil(n_datasets / self.run_no_datasets))

        for block in range(blocks):
            # Print which chunk we are processing.
            logging.info('Processing pipeline block ' + str(block + 1) + ' out of ' + str(blocks))
            start_time = datetime.datetime.now()
            logging.info('Start time block ' + str(block + 1) + ' is ' + str(start_time))

            # Then assign the processing data and coordinate systems.
            self.assign_coordinate_systems_processing_data(block)

            # Finally create the chunk sizes.
            self.divide_processing_chunks(block, self.chunk_orientation)
            # Create the output files. This is not done using multiprocessing, as it only assigns disk space which is
            # generally fast.

            # First check whether there are no duplicates of the same ImageProcessingData objects which are used to write
            # outputs. This could cause problems in two ways:
            # 1. It is not possible to write to the same file at the same moment, so it could cause an IO conflict
            # 2. This will result in two non-unique metadata outputs, so it will miss part of the outputs in the final
            #           resulting data_stack.

            # Create the actual outputs.
            for pipeline in self.pipelines:
                for process, save_process in zip(pipeline['processes'], pipeline['save_processes']):
                    if save_process:
                        process.create_output_data_files()
                        process.remove_memmap_files()

            # Now run the individual chunks using multiprocessing of cores > 1
            self.json_dicts = []
            self.json_files = []
            if self.processing_cores == 1:

                for pipeline in self.chunk_pipelines:
                    json_out = run_parallel(copy.deepcopy(pipeline))
                    self.json_dicts.extend(json_out[0])
                    self.json_files.extend(json_out[1])
            else:
                with get_context("spawn").Pool(processes=self.processing_cores, maxtasksperchild=5) as pool:

                    json_outs = pool.map_async(run_parallel, self.chunk_pipelines, chunksize=1).get()
                    for json_out in json_outs:
                        self.json_dicts.extend(json_out[0])
                        self.json_files.extend(json_out[1])

            logging.info('Finished processing pipeline block ' + str(block + 1) + ' out of ' + str(blocks))
            logging.info('Finished block ' + str(block + 1) + ' in ' + str(datetime.datetime.now() - start_time))
            self.save_processing_results()

    def add_processing_data(self, processing_data, typename='secondary', overwrite=False):
        """
        With this function the ImageProcessingData steps are added.

        :param list[ImageProcessingData] or ImageProcessingData processing_data: The dataset itself
        :param str typename: The type of the dataset or datasets
        :return:
        """

        # Check for validity
        if isinstance(processing_data, dict):
            for key in processing_data.keys():
                if not isinstance(processing_data[key], ImageProcessingData):
                    raise TypeError('All coordinate systems should be a CoordinateSystem object')
                else:
                    # Remove data loaded in memory
                    processing_data[key].remove_memmap_files()
                    processing_data[key].remove_memory_files()

        if typename not in self.processing_data.keys():
            self.processing_data[typename] = copy.deepcopy(processing_data)
        elif overwrite:
            logging.info(typename + ' already exists and is overwritten with new processing data')
        else:
            raise LookupError(typename + ' already exists but overwrite is False')

    def add_coordinate_systems(self, coordinate_system, typename='coor_out', overwrite=False):
        """
        Here we add the relevant coordinate systems for the pipelines. If we work with different bursts / images these
        can differ between datasets.

        This can be of different types:
        1. Just a single coordinatesystem
        2. A dictionary with one 'full' image dataset
        3. A dictionary with a nested dictionary with slice names so ['slices'][slice_name]
        4. A combination of 2 and 3

        """

        # Check for validity
        if isinstance(coordinate_system, dict):
            for key in coordinate_system.keys():
                if not isinstance(coordinate_system[key], CoordinateSystem):
                    raise TypeError('All coordinate systems should be a CoordinateSystem object')
        elif isinstance(coordinate_system, CoordinateSystem):
            coordinate_system = {'general': coordinate_system}

        if typename not in self.coordinate_systems.keys():
            self.coordinate_systems[typename] = copy.deepcopy(coordinate_system)
        elif overwrite:
            logging.info(typename + ' already exists and is overwritten with new coordinate system')
        else:
            logging.info(typename + ' already exists but overwrite is False. Keeping the old settings')

    def load_processing_data_coordinates(self):
        """
        This step assigns the processing data and coordinates to the stack

        """

        if self.image_type == 'slice':
            slice = True
        else:
            slice = False

        images, coordinates = self.stack.get_processing_data(self.processing_type, slice=slice,
                                                             include_reference_slc=self.include_reference)

        # Initialize data
        image_types = ['reference_slc', 'primary_slc', 'secondary_slc', 'ifg']
        for image_type in image_types:
            if len(images[image_type]) > 0:
                self.add_processing_data(images[image_type], image_type)
                self.add_coordinate_systems(coordinates[image_type], image_type)

        if len(list(self.processing_data.keys())):
            logging.info('No images found for processing. Check whether you stack is properly initialized or whether '
                         'you created the interferograms.')

        # For all other coordinate systems try to load from stack or throw error if missing.
        coor_dict = {'in_coor': [], 'out_coor': []}
        for process in self.processing_steps:
            for coor_type in coor_dict.keys():
                if coor_type in process.coordinate_systems.keys():
                    if isinstance(process.coordinate_systems[coor_type], str):
                        coor_dict[coor_type].append(process.coordinate_systems[coor_type])
        coor_strs = list(set(coor_dict['in_coor'] + coor_dict['out_coor']))

        for coor_str in coor_strs:
            if not coor_str in self.coordinate_systems.keys():
                if coor_str not in self.stack.coordinates.keys():
                    raise TypeError('Coordinates of type ' + coor_str + ' do not exist in the processing stack. Make sure '
                                    'that you choose the right processing type that matches the coordinate system types. '
                                    'Otherwise, check if there is no typo in the coordinate type or create a '
                                    'new coordinate system using the create_ml_coordinates function from the stack object.')
                else:
                    coor_keys = list(self.stack.coordinates[coor_str].keys())
                    coor_systems = dict()
                    if slice:
                        slice_names = self.stack.slcs[self.stack.reference_date].slice_names
                        data_key = list(self.processing_data.keys())[0]
                        slice_names_list = [dat[-17:] for dat in self.processing_data[data_key].keys()]
                        for slice_name in slice_names_list:
                            for coor_name in [coor_name for coor_name in coor_keys if slice_name in coor_name]:
                                coor_systems[coor_name] = self.stack.coordinates[coor_str][coor_name]
                        self.add_coordinate_systems(coor_systems, typename=coor_str)
                    else:
                        for coor_name in [coor_name for coor_name in coor_keys if 'full' in coor_name]:
                            coor_systems[coor_name] = self.stack.coordinates[coor_str][coor_name]
                    if 'general' in coor_keys:
                        coor_systems['general'] = self.stack.coordinates[coor_str]['general']

                    self.add_coordinate_systems(coor_systems, typename=coor_str)

    def add_processing_step(self, processing_step, save_to_disk=False, new_radar_shape=False):
        """
        Here we build up the pipeline of consecutive steps.

        :param Process processing_step: Add the used steps for the processing. Check if the datasets are not yet loaded
                but are defined in the input. Otherwise, remove them.
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

        self.new_radar_shapes.append(new_radar_shape)
        self.processing_steps.append(processing_step)
        self.save_processing_steps.append(save_to_disk)

    def save_processing_results(self):
        # Write away the .json files.
        if len(self.json_dicts) == 0:
            return

        unique_json_files, unique_ids, reverse_ids = np.unique(self.json_files, return_index=True, return_inverse=True)
        # Load the existing .json dictionaries.
        unique_json_dicts = []
        for unique_json_file in unique_json_files:
            with open(unique_json_file, 'r+') as old_json_file:
                old_json_dict = json.load(old_json_file)
                unique_json_dicts.append(old_json_dict)

            [copy.deepcopy(self.json_dicts[json_id]) for json_id in unique_ids]

        # Merge the new json files with the old json files.
        for reverse_id, json_dict in zip(reverse_ids, self.json_dicts):
            # First check if it is the selected unique id.
            unique_json_dicts[reverse_id] = self.merge(unique_json_dicts[reverse_id], json_dict)

        # Finally write the relevant .json files to disk.
        for unique_json_dict, unique_json_file in zip(unique_json_dicts, unique_json_files):
            with open(unique_json_file, 'w+') as file:
                try:
                    json.dumps(unique_json_dict)
                except Exception as e:
                    raise ValueError('Part of the .json file is not json serializable. Make sure that the processing'
                                     'step settings only accept dictionaries with regular int, float or string values.' + str(e))
                json.dump(unique_json_dict, file, indent=3)

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
        if len(type_len) == 0:
            raise ValueError('No input data detected in pipeline. Do not forget to add data with the '
                             'add_processing_data function!')
        max_len = np.max(type_len)

        for length in type_len:
            if length != 1 and length != max_len:
                raise ValueError('The number of different processing data types is not consistent')

        # Now be sure that all memory and memmap data is removed.
        for type_name in type_names:
            for processing_data in self.processing_data[type_name]:
                if isinstance(self.processing_data[type_name], dict):
                    processing_data = self.processing_data[type_name][processing_data]
                if not isinstance(processing_data, ImageProcessingData):
                    raise TypeError('All objects in the processing data should be ImageProcessingData objects.')

                processing_data.remove_memmap_files()
                processing_data.remove_memory_files()

        # Finally make sure that all the different types of processing data are aligned to create pipelines.
        if max_len != 1:
            for type_name, length in zip(type_names, type_len):
                if length == 1:
                    self.processing_data[type_name] = [self.processing_data[type_name][0] for n in range(max_len)]

    def check_coordinate_systems(self, pipeline_processing_steps):
        """
        Here we check whether we have the right coordinate systems loaded. This consists of the following steps.
        - Load the output coordinate system of the first step
        - Check whether the out coordinate system of the following steps is the same.

        :return:
        """

        # Gather all the input and output coordinate systems
        short_ids = []

        # Settings for the different coordinate systems, to find extends of the input data. These should be synchronized
        # between images...
        pipeline_coor_settings = dict()
        for processing_step in reversed(pipeline_processing_steps):     # type: Process
            for coor_type in ['out_coor', 'in_coor']:
                if coor_type in processing_step.coordinate_systems.keys():
                    processing_step.coordinate_systems[coor_type].create_short_coor_id()
                    coor = processing_step.coordinate_systems[coor_type]      # type: CoordinateSystem
                    pol = processing_step.output_info['polarisation']

                    short_id = coor.short_id_str
                    if not short_id in pipeline_coor_settings.keys():
                        pipeline_coor_settings[short_id] = {'buffer': 0, 'rounding': 0, 'min_height': 0, 'max_height':0}
                    if coor_type not in processing_step.settings.keys():
                        processing_step.settings[coor_type] = dict()

                    coor_settings = dict()
                    # Skip the rounding, as it is not necessary and could cause problems.
                    for var_str in ['buffer', 'min_height', 'max_height']:
                        if var_str in processing_step.settings.keys():
                            raise TypeError('Variable buffer, rounding, min_height, max_height given in the settings '
                                            'of processing step ' + str(processing_step.__class__) + ' but is not defined '
                                            'for coor_in of coor_out. Please nest it like ["coor_in"][' + var_str + '] = .')
                        else:
                            coor_settings[var_str] = processing_step.settings[coor_type].get(var_str, 0)

                        # Add value to make sure that all earlier steps crop a large enough area.
                        if var_str == 'buffer':
                            pipeline_coor_settings[short_id][var_str] += coor_settings[var_str]
                        elif var_str == 'min_height':
                            pipeline_coor_settings[short_id][var_str] = np.minimum(
                                pipeline_coor_settings[short_id][var_str], coor_settings[var_str])
                        elif var_str == 'max_height':
                            pipeline_coor_settings[short_id][var_str] = np.maximum(
                                pipeline_coor_settings[short_id][var_str], coor_settings[var_str])
                        # And assign new value to coordinate system
                        processing_step.settings[coor_type][var_str] = pipeline_coor_settings[short_id][var_str]

        for process, new_radar_shape, save in zip(pipeline_processing_steps,
                                                  self.new_radar_shapes,
                                                  self.save_processing_steps):
            # Assign the correct coordinate system size to the output grids that are saved to disk.
            # Start by checking whether it is defined already. If the shape is defined, this is fine already
            for coor_type in ['in_coor', 'out_coor']:
                if coor_type in process.coordinate_systems.keys():
                    coor = process.coordinate_systems[coor_type]  # type: CoordinateSystem

                    if coor.grid_type == 'radar_coordinates':
                        if not coor.shape or not coor.readfile or not coor.orbit:
                            # Check if used image data contains the needed radar coordinate system
                            if coor_type == 'out_coor':
                                coor.radar_grid_type = process.output_info['image_type']
                            elif coor_type == 'in_coor' and len(process.input_info['image_types']) > 0:
                                coor.radar_grid_type = process.input_info['image_types'][process.input_info['coor_type'].index('in_coor')]
                            elif coor.radar_grid_type not in ['primary_slc', 'secondary_slc', 'reference_slc', 'ifg']:
                                raise LookupError('Define your radar grid type of your input/output CoordinateSystem. '
                                                  'Choose from slc, primary_slc, secondary_slc, reference_slc and ifg'
                                                  ' for ' + str(processing_step.__class__))
                            elif coor.radar_grid_type not in process.processing_images.keys():
                                raise TypeError('Make sure that one of the input or output images used in the this '
                                                'processing step is the same as the input or output radar grid for ' +
                                                str(processing_step.__class__))
                            new_coor = process.processing_images[coor.radar_grid_type].radar_coordinates['original']
                            if coor.shape and not new_radar_shape:
                                if not coor.shape == new_coor.shape:
                                    logging.info('The input radar shape has a different shape than the original radar'
                                          'shape. Defaulting to original radar shape. If you want to force the new'
                                          'radar shape set new_radar_shape variable to True')
                                coor = new_coor
                            # If the shape is not defined, most likely first line / last line are missing too.
                            elif not coor.shape:
                                coor = new_coor
                    else:
                        if not coor.shape and save and coor_type == 'out_coor':
                            raise LookupError('Could not find a shape for output coordinate system from existing data. '
                                              'Please provide the exact shape of the data creating a pipeline using the '
                                              'CoorNewExtend() function')

                    # If we are looking at the first processing step, we have to define the out coordinate system.
                    # For the input coordinate systems this is not really necessary as it is done during the processing
                    process.coordinate_systems[coor_type] = copy.deepcopy(coor)
                    process.coordinate_systems[coor_type].create_short_coor_id()

        # Finally check if the output data all have the same coordinate system. These should be the same, because
        # otherwise overlapping regions from different parallel processes will be saved at the same time, which
        # can cause the whole processing chain to crash.
        out_coors = []
        saved_out_coors = []
        for process, save in zip(pipeline_processing_steps, self.save_processing_steps):
            out_coors.append(process.coordinate_systems['out_coor'])
            if save:
                saved_out_coors.append(process.coordinate_systems['out_coor'])
                for var_str in ['buffer', 'min_height', 'max_height']:
                    if process.settings['out_coor'][var_str] != 0 and self.pixel_no != 0:
                        raise ValueError('The buffer, rounding, min_height, max_height for an output grid should be '
                                         'zero! Otherwise, processing in chunks is not possible. If you have to run '
                                         'it this way, use pixel_no=0 (means unlimited) to run every image with a '
                                         'single process. Note that this could cause a strong increase in memory use.')

        # Check if there are no more than two different output coordinate systems.
        if len(set([coor.short_id_str for coor in out_coors])) > 2:
            raise TypeError('Only processing pipelines with a maximum of two different output coordinate systems for '
                            'the different processing steps is possible. Split up the current pipeline in different '
                            'parts to allow for multiprocessing.')

        if len(set([coor.short_id_str for coor in saved_out_coors])) > 1:
            raise TypeError('All the processing steps that write an output dataset to disk in one pipeline, should'
                            'have te same coordinate system. Otherwise, the parallel processing can cause '
                            'concurrent writing to the same part of a file on disk, breaking the code.')

    def assign_coordinate_systems_processing_data(self, run_chunk=0):
        """
        Be sure you run the check_processing_data and check_coordinate_systems before running this step.

        This method assigns the coordinate systems and processing data to the processing steps. To do so it copies the
        different processing steps, coordinate_systems and processing_data objects to different packages for
        processing.

        :return:
        """

        self.pipelines = []
        image_types = list(self.processing_data.keys())
        coor_types = list(self.coordinate_systems.keys())

        # Loop over all the datasets
        max_datasets = image_types[np.argmax([len(self.processing_data[image_types[n]].keys()) for n in range(len(image_types))])]
        n_datasets = len(self.processing_data[max_datasets])
        max_chunks = int(np.ceil(n_datasets / self.run_no_datasets))

        start_n = run_chunk * self.run_no_datasets
        end_n = np.minimum((run_chunk + 1) * self.run_no_datasets, n_datasets)
        dataset_keys = list(self.processing_data[max_datasets].keys())[start_n:end_n]

        for pipeline_no, dataset_key in enumerate(dataset_keys):

            pipeline = dict()
            pipeline['processes'] = []
            pipeline['save_processes'] = []
            pipeline['scratch_disk_dir'] = self.scratch_disk_dir            # Path temporary data loaded on scratch
            pipeline['internal_memory_dir'] = self.internal_memory_dir      # Path temporary data loaded in RAM

            # To be changed if there is an output to disk for one of the processing steps. Otherwise, processing makes
            # no sense
            if np.sum(np.array(self.save_processing_steps)) == 0:
                logging.info('Skipping processing. Process already finished or no data that should be saved to disk after '
                      'processing.')
                continue

            # Loop over the processing steps and assign processing data and coordinate systems.
            processing_steps = []
            for process in self.processing_steps:
                process = copy.deepcopy(process)
                image_type_names = list(process.processing_images.keys())
                for image_type_name in image_type_names:
                    # Skip if the processing data is already assigned.
                    if isinstance(process.processing_images[image_type_name], ImageProcessingData):
                        continue
                    image_type = process.processing_images[image_type_name]
                    if image_type not in image_types:
                        raise TypeError('Image type ' + image_type + ' does not exist.')
                    else:
                        if dataset_key in self.processing_data[image_type].keys():
                            process.processing_images[image_type_name] = self.processing_data[image_type][dataset_key]
                        else:
                            if 'full' in dataset_key and 'full' in self.processing_data[image_type].keys():
                                process.processing_images[image_type_name] = self.processing_data[image_type]['full']
                            elif 'slice' in dataset_key:
                                slice_name = 'slice' + dataset_key.split('slice')[-1]
                                if slice_name in self.processing_data[image_type].keys():
                                    process.processing_images[image_type_name] = self.processing_data[image_type][slice_name]
                                else:
                                    raise TypeError('Could not find the slice ' + slice_name + ' for image type ' + image_type)
                            else:
                                raise TypeError('Could not find data for ' + dataset_key + ' for image type ' + image_type)

                coor_type_names = list(process.coordinate_systems.keys())
                for coor_type_name in coor_type_names:
                    # Skip if the coordinate system is already assigned
                    if isinstance(process.coordinate_systems[coor_type_name], CoordinateSystem):
                        continue
                    coor_type = process.coordinate_systems[coor_type_name]
                    if coor_type not in coor_types:
                        raise TypeError('Coordinate type ' + coor_type + ' does not exist.')
                    else:
                        if dataset_key in self.coordinate_systems[coor_type].keys():
                            process.coordinate_systems[coor_type_name] = self.coordinate_systems[coor_type][dataset_key]
                        else:
                            if 'full' in dataset_key and 'full' in self.coordinate_systems[coor_type].keys():
                                process.coordinate_systems[coor_type_name] = self.coordinate_systems[coor_type]['full']
                            elif 'slice' in dataset_key:
                                slice_name = 'slice' + dataset_key.split('slice')[-1]
                                if slice_name in self.coordinate_systems[coor_type].keys():
                                    process.coordinate_systems[coor_type_name] = self.coordinate_systems[coor_type][slice_name]
                                else:
                                    raise TypeError('Could not find the slice ' + slice_name + ' for coordinate type ' + coor_type)
                            elif 'general' in self.coordinate_systems[coor_type].keys():
                                logging.info('Could not find a specific coordinate system for ' + dataset_key + ' using the '
                                      ' general coordinate system instead. This could cause problems with the shape of'
                                      ' different datasets with the same coordinates though.')
                                process.coordinate_systems[coor_type_name] = self.coordinate_systems[coor_type]['general']
                            else:
                                raise TypeError('Could not find data for ' + dataset_key + ' for coordinate type ' + coor_type)
                processing_steps.append(process)

            self.check_coordinate_systems(processing_steps)

            # Finally assign the image shape sizes to all but the reference image.
            saved_out_coors = [process.coordinate_systems['out_coor'] for process, save in zip(processing_steps,
                                                               self.save_processing_steps) if save == True]
            pipeline['reference_coor'] = copy.deepcopy(saved_out_coors[0])
            for process, save in zip(processing_steps, self.save_processing_steps):

                process.define_coordinate_system_size(pipeline['reference_coor'])
                Process.initialize(process, input_info=process.input_info,
                                 output_info=process.output_info,
                                 coordinate_systems=process.coordinate_systems,
                                 processing_images=process.processing_images,
                                 overwrite=process.overwrite,
                                 settings=process.settings)

                # In case the process is not finished or not already on disk, we should add it to the list to be
                # processed
                if not process.process_finished or not process.process_on_disk:
                    pipeline['processes'].append(process)
                    pipeline['save_processes'].append(save)

            # Now all the input and output image sizes of the pipeline are set and can be split in blocks
            # Save pipeline
            if np.sum(np.array(pipeline['save_processes'])) == 0:
                logging.info('Skipping processing. Process already finished or no data that should be saved to disk after '
                      'processing.')
            else:
                logging.info('Prepared pipeline for image ' + str(pipeline_no + 1 - start_n) + ' of total ' + str(end_n - start_n))
                self.pipelines.append(pipeline)


    def divide_processing_chunks(self, run_chunk=0, chunk_orientation='lines'):
        """
        Be sure you run the assign_coordinate_systems_processing_data before this step.

        This method takes the individual processing packages provided by the assign_coordinate_systems_processing_data
        to create packages for

        :return:
        """

        image_types = list(self.processing_data.keys())
        n_datasets = len(self.processing_data[image_types[0]])
        max_chunks = int(np.ceil(n_datasets / self.run_no_datasets))
        self.chunk_pipelines = []

        if self.pixel_no == 0:
            for pipeline in self.pipelines:
                pipeline['chunk'] = 0
                pipeline['total_chunks'] = 1
                pipeline['s_lin'] = 0
                pipeline['s_pix'] = 0
                coor_type = pipeline['processes'][-1].output_info['coor_type']
                coor_system = pipeline['processes'][-1].coordinate_systems[coor_type]
                pipeline['lines'] = coor_system.shape[0]
                pipeline['pixels'] = coor_system.shape[1]
                pipeline['total_lines'] = coor_system.shape[0]
                pipeline['total_pixels'] = coor_system.shape[1]
                self.chunk_pipelines.append(pipeline)
        else:
            for im_no, pipeline in enumerate(self.pipelines):
                # First find the output coordinate system
                coor_type = pipeline['processes'][-1].output_info['coor_type']
                coor_system = pipeline['processes'][-1].coordinate_systems[coor_type]        # type: CoordinateSystem

                pixel_no = coor_system.shape[0] * coor_system.shape[1]
                if pixel_no < self.pixel_no:
                    # In the case the defined chunk size is larger than the size of the output image, we can run the full
                    # image.
                    pipeline['chunk'] = 0
                    pipeline['total_chunks'] = 1
                    pipeline['s_lin'] = 0
                    pipeline['s_pix'] = 0
                    pipeline['lines'] = coor_system.shape[0]
                    pipeline['pixels'] = coor_system.shape[1]
                    pipeline['total_lines'] = coor_system.shape[0]
                    pipeline['total_pixels'] = coor_system.shape[1]
                    self.chunk_pipelines.append(pipeline)
                else:
                    if chunk_orientation == 'lines':
                        lines = np.ceil(float(self.pixel_no) / coor_system.shape[1]).astype(np.int32)
                        pixels = coor_system.shape[1]
                        l_chunks = np.ceil(float(coor_system.shape[0]) / lines).astype(np.int32)
                        p_chunks = 1
                    elif chunk_orientation == 'pixels':
                        lines = coor_system.shape[0]
                        pixels = np.ceil(float(self.pixel_no) / coor_system.shape[0]).astype(np.int32)
                        l_chunks = 1
                        p_chunks = np.ceil(float(coor_system.shape[1]) / pixels).astype(np.int32)
                    elif chunk_orientation == 'chunks':
                        # If this option is chosen the image is divided in chunks in equal lines/pixel chunks.
                        lines = np.ceil(np.sqrt(float(self.pixel_no))).astype(np.int32)
                        pixels = lines
                        l_chunks = np.ceil(float(coor_system.shape[0]) / lines).astype(np.int32)
                        p_chunks = np.ceil(float(coor_system.shape[1]) / pixels).astype(np.int32)
                    else:
                        raise TypeError('chunk orientation can only be lines, pixels or chunks')

                    for n in range(l_chunks):
                        for m in range(p_chunks):
                            new_pipeline = dict()
                            for key in ['save_processes', 'scratch_disk_dir', 'internal_memory_dir', 'reference_coor']:
                                new_pipeline[key] = pipeline[key]
                            # Create the processing coordinate system based
                            pipeline_coor_chunk = copy.deepcopy(coor_system)
                            pipeline_coor_chunk.first_line += n * lines
                            pipeline_coor_chunk.first_pixel += m * pixels
                            shape_lines = np.minimum(lines, coor_system.shape[0] - n * lines)
                            shape_pixels = np.minimum(pixels, coor_system.shape[1] - m * pixels)
                            pipeline_coor_chunk.shape = (shape_lines, shape_pixels)

                            # Copy all the processes, to maintain the shape settings for the individual chunks when the
                            # new pipeline is created. However, do it for the whole pipeline processes at once to make
                            # sure that the image data still refers to the same objects for different processes.
                            new_pipeline['processes'] = []
                            for process in pipeline['processes']:
                                new_process = copy.copy(process)
                                # Deep copy for chunk coordinate systems
                                new_process.coordinate_systems = copy.copy(process.coordinate_systems)
                                if 'in_coor_chunk' in process.coordinate_systems.keys():
                                    new_process.coordinate_systems['in_coor_chunk'] = copy.deepcopy(process.coordinate_systems['in_coor_chunk'])
                                if 'out_coor_chunk' in process.coordinate_systems.keys():
                                    new_process.coordinate_systems['out_coor_chunk'] = copy.deepcopy(process.coordinate_systems['out_coor_chunk'])
                                new_process.define_processing_chunk(reference_coor_chunk=pipeline_coor_chunk, s_lin=n*lines, s_pix=m*pixels)
                                new_pipeline['processes'].append(new_process)

                            new_pipeline['chunk'] = n * p_chunks + (m + 1)
                            new_pipeline['total_chunks'] = l_chunks * p_chunks
                            new_pipeline['s_lin'] = n * lines
                            new_pipeline['s_pix'] = m * pixels
                            new_pipeline['lines'] = lines
                            new_pipeline['pixels'] = pixels
                            new_pipeline['total_lines'] = coor_system.shape[0]
                            new_pipeline['total_pixels'] = coor_system.shape[1]
                            self.chunk_pipelines.append(new_pipeline)
                            logging.info('Prepared pipeline for processing chunk ' + str(new_pipeline['chunk']) +
                                  ' out of ' + str(new_pipeline['total_chunks']) + ' for image ' + str(im_no) + ' out of '
                                  + str(len(self.pipelines)))

        random.shuffle(self.chunk_pipelines)

        self.total_chunks += len(self.chunk_pipelines)
        for chunk_no, pipeline in enumerate(self.chunk_pipelines):
                pipeline['process_chunk_no'] = chunk_no + self.total_chunks - len(self.chunk_pipelines)
                # This is an estimate....
                pipeline['total_process_chunk_no'] = self.total_chunks + len(self.chunk_pipelines) * (max_chunks - run_chunk - 1)

    @staticmethod
    def merge(a, b, path=None):
        "merges b into a"
        if path is None: path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    Pipeline.merge(a[key], b[key], path + [str(key)])
                elif a[key] == b[key]:
                    pass # same leaf value
                elif key == 'last_date_changed':
                    a[key] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a
