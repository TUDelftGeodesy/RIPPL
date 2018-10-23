# The main goal of this function is to connect different functions to create a pipeline.
# The pipeline code will follow the following steps:
#   - List all needed input and output of the requested processing_steps
#   - Order the different processing_steps in such a way that we will never miss one of the outputs
#   - Create a list of inputs (data on disk), intermediate results (in memory) and outputs (data on disk)
#   - From these list of inputs and outputs we define when they should be loaded in memory
#   Final result of the former will be a list with loading data, processing data, unloading data (either removal or saving to disk)
#   - Results can be altered to create output of certain intermediate steps if needed
#
# If the processing_steps do not fit together the pipeline creator will throw an error
# If the whole process itself is ok the input image is checked whether the needed input files are there. If not the
# function will throw an error.
#
# The final created function

# image meta data
import copy
from collections import OrderedDict
from processing_list import ProcessingList
from image_data import ImageData
import numpy as np
from image import Image
from interferogram import Interferogram
from processing_steps.concatenate import Concatenate
from coordinate_system import CoordinateSystem
from multiprocessing import Pool


class Pipeline():

    def __init__(self, memory=500, cores=6, cmaster=[], master=[], slave=[], ifg=[], parallel=True):
        # Individual images are always defined as slaves.
        # Be sure you input the right image types. Otherwise the function will not run!
        # master/slave/ifg are full images, not just the slices. The lists are just multiples of these. Only functions
        # where lists are needed, we run with lists.

        # First load the available functions
        processes = ProcessingList()
        self.processes = processes.processing
        self.processing_inputs = processes.processing_inputs

        # Maximum memory per process in MB.
        self.memory = memory

        # For now we just assume that every pixel can be run with 10000 pixels met MB (Likely to be higher...)
        self.pixels = self.memory * 10000

        # Cores. The number of cores we use to run our processing.
        self.cores = cores
        # The parallel parameter should be True (otherwise there is no use for the parallel option. 
        # However, this option exists to simplify debugging of the program)
        self.parallel = parallel

        # Load input meta data.
        self.cmaster = cmaster
        self.master = master
        self.slave = slave
        self.ifg = ifg

        self.slice_ids = []

        # Find slice ids
        for im in [self.cmaster, self.master, self.slave, self.ifg]:
            if isinstance(im, Image) or isinstance(im, Interferogram):
                self.slice_ids.extend(im.slices.keys())

        self.slice_ids = sorted(list(set(self.slice_ids)))

        # Re-organize the different full images and slices
        self.res_dat = OrderedDict()

        for slice_id in self.slice_ids:
            for im, im_str in zip([self.cmaster, self.master, self.slave, self.ifg], ['cmaster', 'slave', 'master', 'ifg']):
                slice = dict()

                if isinstance(im, Image) or isinstance(im, Interferogram):
                    if slice_id in im.slices.keys():
                        slice[im_str] = im.slices[slice_id]
                self.res_dat[slice_id] = slice

        for im, im_str in zip([self.cmaster, self.master, self.slave, self.ifg], ['cmaster', 'slave', 'master', 'ifg']):
            image = dict()

            if isinstance(im, Image) or isinstance(im, Interferogram):
                if im.res_file:
                    image[im_str] = im.res_dat

            self.res_dat['full'] = image

        # Init function running variables
        self.function = ''
        self.coor = []
        self.settings = []
        self.meta_type = ''

        # Init the pipeline information.
        self.pipelines = []
        self.processing_packages = []

        # The order of processes and there accompanying functions.
        self.run_full = []
        self.run_slice = [[] for slice_id in self.slice_ids]

        # Init pool of workers for multprocessing
        self.pool = Pool(self.processes)

    def __call__(self, function, settings, coor, meta_type, slice=True, file_type=''):
        # Here we prepare and run the parallel processing for this image at once.

        # The specific settings for some of the functions. Sometimes we want to run with specific settings for one
        # of the intermediate or the final function.
        # This variable should be a dictionary, containing dictionaries for individual functions.
        self.settings = settings
        if not isinstance(coor, CoordinateSystem):
            print('coor should be an CoordinateSystem object')
            return

        self.coor = coor
        self.function = function
        self.slice = slice
        self.meta_type = meta_type

        if not file_type:
            self.file_type = function
        else:
            self.file_type = file_type

        self.define_function_order()
        self.run_parallel_processing()

    def define_function_order(self):

        # First find all dependencies for this step.
        in_dep = dict()
        in_dep['slave'] = dict()
        in_dep['cmaster'] = dict()
        in_dep['master'] = dict()
        in_dep['ifg'] = dict()

        out_dep = dict()
        out_dep['slave'] = dict()
        out_dep['cmaster'] = dict()
        out_dep['master'] = dict()
        out_dep['ifg'] = dict()

        # First process for the full image
        # Check if step already exist and output is generated.
        if not self.function in self.processes.keys():
            print(self.function + ' does not exist!')
            return

        # Initialize the function steps before any concatenation or multilooking.

        # Initialize the different 'pipelines'. Or combinations of functions that can be run using memory and data
        # slicing. The most important thing here is that the input resolution of the data does not change.
        # e.g. this changes when multilooking/concatenation or resampling
        self.pipelines = []
        self.concatenations = []

        # Create a list of steps that still have to be performed but did not fit in the current pipeline.
        concat = dict()
        concat['step'] = []
        concat['type'] = []
        concat['meta'] = []
        concat['coor'] = []

        # Shift of coordinates by multilooking
        multilook = dict()
        multilook['step'] = []
        multilook['type'] = []
        multilook['meta'] = []
        multilook['coor_in'] = []
        multilook['coor_out'] = []
        multilook['name'] = []

        # Start of a new pipeline. Before multilooking, concatenation or resampling.
        dummy_pipeline = dict()
        dummy_pipeline['step'] = []
        dummy_pipeline['type'] = []
        dummy_pipeline['meta'] = []
        dummy_pipeline['coor'] = []
        dummy_pipeline['name'] = []
        new_pipeline = copy.deepcopy(dummy_pipeline)

        proc_depth = 1

        ############################################################################################################
        # Full image processing

        # First perform all needed steps if we process the full image.
        # Check if step exists
        if not self.slice:
            if self.res_dat['full'][self.meta_type].check_datafile(self.function, file_type=self.file_type, warn=False):
                print('Processing not needed, file already available')
                return
            else:
                start_funcs = [self.function]
                start_meta = [self.meta_type]
                start_meta_type = 'full'
                start_coor = self.coor

            concat, multilook, new_pipeline = self.create_pipeline(start_funcs, start_meta, start_meta_type, start_coor,
                                                                   concat, multilook, new_pipeline, proc_depth)
            proc_depth += 1

            # First iterate over the full image till there are no resolution shifts existing anymore.
            while len(new_pipeline['step']) > 0 or len(multilook['step']) > 0 :

                old_meta = ''

                # Create new pipelines if needed.
                while len(new_pipeline['step']) > 0:
                    step_ids = np.where(np.array(new_pipeline['coor'].sample) == new_pipeline['coor'][0].sample *
                                        np.array(new_pipeline['meta']) == new_pipeline['meta'][0])
                    start_funcs = np.array(new_pipeline['step'])[step_ids]
                    start_meta = np.array(new_pipeline['meta'])[step_ids]
                    start_meta_type = np.array(new_pipeline['meta_type'])[step_ids]
                    start_coor = new_pipeline['coor'][-1]

                    # If we start working on the same slice, independent calculations are not possible anymore
                    if old_meta == start_meta[0]:
                        proc_depth += 1

                    # Remove the processed types from the list
                    for id in np.sort(step_ids)[::-1]:
                        for key in new_pipeline.keys():
                            new_pipeline[key].pop(id)

                    # Create the pipeline
                    concat, multilook, new_pipeline = self.create_pipeline(start_funcs, start_meta, start_meta_type, start_coor,
                                                                       concat, multilook, new_pipeline, proc_depth)
                    old_meta = start_meta[0]


                # Then do the multilooking. These can be done all at once as they are all independent.
                if len(multilook['step']) > 0:
                    proc_depth += 1
                    func_set, new_pipeline = self.create_ml_processing(multilook, new_pipeline, proc_depth)
                    self.pipelines.append(func_set)
                    proc_depth += 1

                    # Remove the existing multilook steps
                    for key in multilook.keys():
                        multilook[key] = []

        ##########################################################################################################
        # Concatenation

        # Now find the concatenation steps needed. These are seperate steps in the whole process
        if not self.slice:
            new_pipeline, proc_depth = self.create_concatenate_processing(concat, new_pipeline, proc_depth)

        #############################################################################################################
        # Individual slices

        # If we are only processing slices, this is the moment the processing starts.
        if self.slice:
            for slice_name in self.slice_ids:
                if self.res_dat[slice_name][self.meta_type].check_datafile(self.function, file_type=self.file_type, warn=False):
                    print('Processing not needed, file already available')
                    return
                else:
                    start_funcs = [self.function]
                    start_meta = [self.meta_type]
                    start_meta_type = slice_name
                    start_coor = self.coor

                concat, multilook, new_pipeline = self.create_pipeline(start_funcs, start_meta, start_meta_type, start_coor,
                                                                       concat, multilook, new_pipeline, proc_depth)
            proc_depth += 1

        # Now we can perform a similar process as for the full image, but now for the slices.
        # First iterate over the full image till there are no resolution shifts existing anymore.
        while len(new_pipeline['step']) > 0 or len(multilook['step']) > 0:

            old_meta = ''

            # Create new pipelines if needed.
            while len(new_pipeline['step']) > 0:
                step_ids = np.where(np.array(new_pipeline['coor'].sample) == new_pipeline['coor'][0].sample *
                                    np.array(new_pipeline['meta']) == new_pipeline['meta'][0])
                start_funcs = np.array(new_pipeline['step'])[step_ids]
                start_meta = np.array(new_pipeline['meta'])[step_ids]
                start_meta_type = np.array(new_pipeline['meta_type'])[step_ids]
                start_coor = new_pipeline['coor'][0]

                # If we start working on the same slice, independent calculations are not possible anymore
                if old_meta == start_meta[0]:
                    proc_depth += 1

                # Remove the processed types from the list
                for id in np.sort(step_ids)[::-1]:
                    for key in new_pipeline.keys():
                        new_pipeline[key].pop(id)

                # Create the pipeline
                concat, multilook, new_pipeline = self.create_pipeline(start_funcs, start_meta, start_meta_type,
                                                                       start_coor,
                                                                       concat, multilook, new_pipeline, proc_depth)
                old_meta = start_meta[0]

            # Then do the multilooking. These can be done all at once as they are all independent.
            if len(multilook['step']) > 0:
                proc_depth += 1
                func_set, new_pipeline = self.create_ml_processing(multilook, new_pipeline, proc_depth)
                self.pipelines.append(func_set)
                proc_depth += 1

                # Remove the existing multilook steps
                for key in multilook.keys():
                    multilook[key] = []

    @staticmethod
    def order_input_data(input_data):

        meta_types = []
        step_types = []
        file_types = []

        for meta_type in input_data.keys():
            for step_type in input_data[meta_type].keys():
                for file_type in input_data[meta_type][step_type].keys():

                    meta_types.append(meta_type)
                    step_types.append(step_type)
                    file_types.append(file_type)

        return meta_types, step_types, file_types

    def create_concatenate_processing(self, concat, new_pipeline, depth):
        # Create a set of functions that can be run for a concatenation step.
        # If the concatenation also includes an multilooking of the bursts, these two steps are combined.

        ml = dict()

        for con_step, con_type, con_meta, con_coor in zip(
                concat['step'], concat['type'], concat['meta'], concat['coor']):

            input_data, output_data, mem_use = self.processes[con_step].processing_info(con_coor)
            meta_types, step_types, file_types = Pipeline.order_input_data(input_data)

            for meta_type, step_type, file_type in zip(meta_types, step_types, file_types):

                concat_names = [slice_str for slice_str in self.res_dat.keys() if slice_str != 'full']
                concat_slices = [self.res_dat[slice_str][meta_type] for slice_str in concat_names]
                concat_res, concat_coors = Concatenate.find_slice_coordinates(concat_slices, con_coor)

                # Additional settings for concatenation.
                settings = dict()
                settings['step'] = step_type
                settings['file_type'] = file_type

                for slice_name, slice_coor in zip(concat_names, concat_coors):

                    input_dat, output_data, mem_use = self.processes[con_step].processing_info(slice_coor)
                    meta_tps, step_tps, file_tps = Pipeline.order_input_data(input_data)
                    # We can use one input as reference as the coordinates of all inputs are the same
                    input_1 = input_dat[meta_tps[0]][step_tps[0]][file_tps[0]]

                    if slice_coor.sample != input_1['coordinates'].sample and \
                            input_1['coor_change'] == 'multilook':
                            ml['step'].append(step_type)
                            ml['type'].append(file_type)
                            ml['meta_type'].append(meta_type)
                            ml['meta'].append(slice_name)
                            ml['coor_in'].append(input_1['coordinates'])
                            ml['coor_out'].append(slice_coor)

                    elif input_1['coor_change'] == 'resample' or slice_coor.sample == input_1['coordinates'].sample:
                        new_pipeline['step'].append(step_type)
                        new_pipeline['type'].append(file_type)
                        new_pipeline['meta_type'].append(meta_type)
                        new_pipeline['meta'].append(slice_name)
                        new_pipeline['coor'].append(slice_coor.sample)

                func_set, new_pipeline = self.create_ml_processing(ml, new_pipeline, depth)

                func_set['main_proc_depth'] = depth
                depth += 1
                func_set['type'] = 'concatenate'
                func_set['step'].append('concatenate')
                func_set['file_type'].append('')
                func_set['meta'].append(concat_names)
                func_set['meta_type'].append(meta_type)
                func_set['coor'].append(con_coor)
                func_set['settings'].append(settings)
                func_set['proc_depth'] = [2 for a in func_set['proc_depth']]
                func_set['proc_depth'].append(1)
                self.pipelines.append(func_set)

        return new_pipeline, depth

    def create_ml_processing(self, multilook, new_pipeline, depth):
        # This function extracts all processing steps for multilooking of the full or slave images.
        # These functions are merely a set of independent functions and do not belong to a pipeline as parallel processing
        # of parts of images is not possible (and almost never needed)

        # Get the individual lists
        ml_func = multilook['step']
        ml_file_type = multilook['type']
        ml_meta_type = multilook['meta_type']
        ml_meta_names = multilook['meta']
        ml_in_coor = multilook['coor_in']
        ml_out_coor = multilook['coor_out']

        # Init the func_set variable
        func_set = dict()
        func_set['type'] = 'multilook'
        func_set['step'] = []
        func_set['file_type'] = []
        func_set['meta'] = []
        func_set['meta_type'] = []
        func_set['coor'] = []
        func_set['save_disk'] = []
        func_set['settings'] = []

        # If the direct step before concatenation is a multilooking step, these two can be combined.
        # An exception is the interferogram step, which is in most cases a multilooking step too.
        for step, file_type, meta_name, meta_type, in_coor, out_coor in zip(ml_func, ml_file_type, ml_meta_type, ml_meta_names, ml_in_coor, ml_out_coor):

            f_type = file_type + out_coor.sample
            # Check if the needed step is already processed. Otherwise further processing is not needed.
            if not self.res_dat[meta_name][meta_type].check_datafile(self.function, file_type=f_type, warn=False):
                # If it is not already there. We check whether there is a difference in input and
                # output coordinate system.
                # If that is the case we add these to these steps to the concatenation procedure.

                # This means that we need to do a multilooking operation. The used function is
                # either the multilooking function or
                settings = dict()

                if step == 'interferogram':
                    m_step = step
                    m_file_type = file_type

                    input_data, output_data, mem_use = self.processes[step].processing_info(out_coor)
                    meta_tps, step_tps, file_tps = Pipeline.order_input_data(input_data)
                else:
                    m_step = 'multilook'
                    m_file_type = ''

                    meta_tps = [meta_type]
                    step_tps = [step]
                    file_tps = [file_type]

                    settings['step'] = step
                    settings['file_type'] = file_type
                    settings['coor_out'] = out_coor

                func_set['main_proc_depth'] = depth
                func_set['step'].append(m_step)
                func_set['file_type'].append(m_file_type)
                func_set['meta'].append(meta_name)
                func_set['meta_type'].append(meta_type)
                func_set['coor'].append(in_coor)
                func_set['settings'].append(settings)
                func_set['proc_depth'].append(1)

                # Now add the original requested files to our to be processed list.
                for meta_tp, step_tp, file_tp in zip(meta_tps, step_tps, file_tps):

                    f_type = file_tp + in_coor.sample
                    if not self.res_dat[meta_name][meta_type].check_datafile(self.function, file_type=f_type, warn=False):
                        new_pipeline['step'].append(step_tp)
                        new_pipeline['type'].append(file_tp)
                        new_pipeline['meta_type'].append(meta_tp)
                        new_pipeline['meta'].append(meta_name)
                        new_pipeline['coor'].append(in_coor)

        return func_set, new_pipeline

    def create_pipeline(self, start_func, start_meta_type, meta_name, start_coor, concat, multilook, new_pipeline, pipeline_depth):

        # These variables are used to store the steps that cannot be processed in the current pipeline. There are therefore first stored and then
        meta_type_n = []    # What meta type are we dealing with (slave, master, cmaster, ifg?)
        step_n = []         # What step should be processed?
        file_type_n = []         # What is the file type we are working with?
        meta_n = []         # What is the meta data file of this step? (normally one ImageData and multiple for Concatenate steps
        coor_n = []         # What is the coordinate system of this step?
        save_disk_n = []    # Should the data be saved to disk?
        proc_depth = []     # This variable indicates the processing depth (e.g. 1,2,3 etc in line. It increases with
                            # one every iteration. This can later be used to remove unused data in memory

        depend = start_func
        depend_meta_type = start_meta_type
        depth = 0

        while len(depend) > 0:

            depth += 1

            functions = depend
            functions_type = depend_meta_type

            depend = []
            depend_meta_type = []

            for function, meta, meta_type in zip(functions, meta_name, functions_type):

                input_dat, output_dat, mem_use = self.processes[function].processing_info(start_coor)
                meta_types, step_types, file_types = Pipeline.order_input_data(input_dat)

                for meta_type, step, file_type in zip(meta_types, step_types, file_types):
                    # If data already exist we do not need to process it
                    if self.res_dat['full'][meta_type].check_datafile(function, file_type=function, warn=False):
                        pass
                    elif self.slice != input_dat[meta_type][step][file_type]['slice']:
                        concat['step'].append(step)
                        concat['type'].append(file_type)
                        concat['meta'].append(meta_type)
                        concat['coor'].append(input_dat[meta_type][step][file_type]['coordinates'])

                    elif start_coor.sample != input_dat[meta_type][step][file_type]['coordinates'].sample:

                        if input_dat[meta_type][step][file_type]['coor_change'] == 'multilook':
                            multilook['step'].append(step)
                            multilook['type'].append(file_type)
                            multilook['meta'].append(meta_type)
                            multilook['coor_in'].append(input_dat[meta_type][step][file_type]['coordinates'])
                            multilook['coor_out'].append(start_coor)
                        elif input_dat[meta_type][step][file_type]['coor_change'] == 'resample':
                            new_pipeline['step'].append(step)
                            new_pipeline['type'].append(file_type)
                            new_pipeline['meta'].append(meta_type)
                            new_pipeline['coor'].append(input_dat[meta_type][step][file_type]['coordinates'])
                    else:
                        step_n.append(step)
                        file_type_n.append(file_type)
                        meta_n.append(self.res_dat[meta_name][meta_type])
                        meta_type_n.append(meta_type)
                        coor_n.append(input_dat[meta_type][step][file_type]['coordinates'])
                        proc_depth.append(depth)

                        if step in start_func:
                            save_disk_n.append(True)
                        else:
                            save_disk_n.append(False)

                        # To iterate further.
                        depend.append(step)
                        depend_meta_type.append(meta_type)

        # Add list of processing steps and remove steps occurring in the list.
        # Find unique ids starting from the end of the list (prefer last occurrences)
        func_set = dict()
        func_set['main_proc_depth'] = pipeline_depth
        func_set['type'] = 'pipeline'
        func_set['step'], ids = np.unique(np.array(step_n)[::-1], return_index=True)

        # Now gather the file types and bundle them per processing step.
        func_set['file_type'] = []
        for step in func_set['step']:
            func_set['file_type'].append(list(set(file_type_n[np.where(np.array(step_n) == step)[0]])))
        func_set['meta'] = np.array(meta_n)[::-1][ids]
        func_set['meta_type'] = np.array(meta_type_n)[::-1][ids]
        func_set['coor'] = np.array(coor_n)[::-1][ids]
        func_set['save_disk'] = np.array(save_disk_n)[::-1][ids]

        # proc depth should preserve the lowest number.
        a, ids = np.unique(np.array(step_n), return_index=True)
        func_set['proc_depth'] = np.array(save_disk_n)[::-1][ids]
        func_set['settings'] = [dict() for i in ids]

        # Save pipeline
        self.pipelines.append(func_set)

        return concat, multilook, new_pipeline

    def run_parallel_processing(self):
        # This function generates a list of functions and inputs to do the processing.

        # There are four types of inputs for individual packages
        # 1. Processing step > this processes and runs the function for either a part or a full slice
        # 2. Metadata step > this only generates the metadata needed to initialize the processing step to write to disk
        # 3. Create step > this step creates a new file on disk
        # 4. Save step > this step saves the data from memory to disk

        # Apart from this we have an independent step that is always run:
        dummy_processing = dict()
        dummy_processing['functions'] = []

        # Store the used res information to restore after processing.
        dummy_processing['res_dat'] = dict()
        for key in self.res_dat.keys():
            dummy_processing['res_dat'][key] = dict()

        dummy_processing['proc'] = False
        dummy_processing['proc_var_name'] = []
        dummy_processing['proc_var'] = []
        dummy_processing['meta'] = False
        dummy_processing['meta_var_name'] = []
        dummy_processing['meta_var'] = []
        dummy_processing['create'] = False
        dummy_processing['create_var_name'] = []
        dummy_processing['create_var'] = []
        dummy_processing['save'] = False
        dummy_processing['save_var_name'] = []
        dummy_processing['save_var'] = []
        dummy_processing['clear_mem'] = False
        dummy_processing['clear_mem_var_name'] = []
        dummy_processing['clear_mem_var'] = []

        proc_depths = [pipeline['proc_depth'] for pipeline in self.pipelines]

        for proc_depth in range(1, np.max(proc_depths)):
            for pipeline in [pipeline for pipeline in self.pipelines if self.pipelines['proc_depth'] == proc_depth]:
                # Init intialization and processing package
                processing_package = []

                # create pipeline parallel processing packages.
                if pipeline['type'] == 'pipeline':

                    pipeline_init = copy.deepcopy(dummy_processing)
                    pipeline_init['meta'] = True
                    pipeline_init['create'] = True
                    res_dat = pipeline_init['res_dat']

                    # First define the processing order
                    save_disk = []

                    pds = pipeline['proc_depth']
                    for pd, i in pds, range(len(pds)):
                        if pd == 1:
                            save_disk.append(pipeline['function'][i])

                    #  Then add the variables which are independent from the block size
                    for func, file_type, meta, meta_type, coordinates, coor_out, clear_mem in zip(pipeline['steps'], pipeline['file_type'],
                            pipeline['meta'], pipeline['meta_type'], pipeline['coordinate'], pipeline['coor_out']):

                        if func in save_disk:
                            meta_var, meta_var_names, res_dat = self.create_var_package(
                                func, meta, meta_type, coordinates, coor_out, file_type=file_type, package_type='metadata', res_dat=res_dat)
                            create_var, create_var_names, res_dat = self.create_var_package(
                                func, meta, meta_type, coordinates, coor_out, file_type=file_type, package_type='disk_data', res_dat=res_dat)

                            pipeline_init['function'].append(self.processes[func])
                            pipeline_init['meta_var_name'].append(meta_var_names)
                            pipeline_init['meta_var'].append(meta_var)
                            pipeline_init['create_var_name'].append(create_var_names)
                            pipeline_init['create_var'].append(create_var)

                    processing_package.append(pipeline_init)

                elif pipeline['type'] == 'multilook' or pipeline['type'] == 'concatenate':

                    # Filter all multilook steps from both multilook and concatenate sets
                    for func, file_type, settings, meta, meta_type, coordinates, coor_out in zip(pipeline['steps'], pipeline['file_type'], pipeline['settings'],
                            pipeline['meta'], pipeline['meta_type'], pipeline['coordinate'], pipeline['coor_out']):

                        # All multilook steps are independent so we give them independent steps here too.
                        pipeline_ml = copy.deepcopy(dummy_processing)
                        pipeline_ml['processing'] = True
                        res_dat = pipeline_ml['res_dat']

                        if pipeline['type'] == 'multilook':
                            pipeline_ml['create'] = True
                            pipeline_ml['save'] = True
                            pipeline_ml['clear_mem'] = True

                        if func in ['multilook', 'interferogram']:
                            pipeline_ml['function'] = [self.processes[func]]

                            if pipeline['type'] == 'multilook':

                                if func == 'multilook':

                                    meta_var, meta_var_names, res_dat = self.create_var_package(
                                        func, meta, meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                        res_dat=res_dat, step=settings['step'], package_type='disk_data')
                                    disk_var, disk_var_names, res_dat = self.create_var_package(
                                        func, meta, meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                        res_dat=res_dat, step=settings['step'], package_type='disk_data')
                                else:  # If it is interferogram

                                    meta_var, meta_var_names, res_dat = self.create_var_package(
                                        func, meta, meta_type, coordinates, coor_out, file_type=file_type,
                                        res_dat=res_dat, package_type='disk_data')
                                    disk_var, disk_var_names, res_dat = self.create_var_package(
                                        func, meta, meta_type, coordinates, coor_out, file_type=file_type,
                                        res_dat=res_dat, package_type='disk_data')

                                if pipeline['type'] == 'multilook':
                                    pipeline_ml['clear_mem_var_name'] = [[disk_var_names]]
                                    pipeline_ml['clear_mem_var'] = [[disk_var]]
                                    pipeline_ml['create_var_name'] = [disk_var_names]
                                    pipeline_ml['create_var'] = [disk_var]
                                    pipeline_ml['save_var_name'] = [disk_var_names]
                                    pipeline_ml['save_var'] = [disk_var]

                                pipeline_ml['proc_var_name'] = [meta_var_names]
                                pipeline_ml['proc_var'] = [meta_var]

                            else:
                                # When it is part of the concatenate step we do not remove data from memory!
                                pipeline_ml['remove_mem'] = [[]]

                        processing_package.append(pipeline_ml)

                # Now run this package in parallel.
                self.run_parallel_package(processing_package)
                self.write_res()

            # Run again for the concatenation scripts only. (Others can be done in 1 parallel package)
            for pipeline in [pipeline for pipeline in self.pipelines if self.pipelines['proc_depth'] == proc_depth ]:
                # Init intialization and processing package
                processing_package = []

                # create pipeline parallel processing packages.
                if pipeline['type'] == 'pipeline':

                    pipeline_processing = copy.deepcopy(dummy_processing)
                    pipeline_processing['proc'] = True
                    pipeline_processing['save'] = True
                    pipeline_processing['clear_mem'] = True

                    clear_mem_var = dict()
                    clear_mem_var_names = dict()

                    # First define the processing order
                    remove_mem = []
                    save_disk = []

                    pds = pipeline['proc_depth']
                    for pd, i in pds, range(len(pds)):

                        if pd == 1:
                            save_disk.append(pipeline['function'][i])

                        # If there is not a step with the same processing depth remove the set of earlier steps
                        if pd not in np.array(pds)[i + 1:]:
                            remove_mem.append(np.array(pipeline['step'])[np.array(pds) == pd + 1])

                    #  Then add the variables which are independent from the block size
                    for func, file_type, meta, meta_type, coordinates, coor_out, clear_mem in zip(pipeline['steps'], pipeline['file_type'],
                            pipeline['meta'], pipeline['meta_type'], pipeline['coordinate'], pipeline['coor_out'], remove_mem):

                        pipeline_processing['function'].append(self.processes[func])

                        proc_var, proc_var_names = self.create_var_package(
                            func, meta, meta_type, coordinates, coor_out, file_type=file_type, package_type='processing')
                        pipeline_processing['proc_var_name'].append(proc_var_names)
                        pipeline_processing['proc_var'].append(proc_var)

                        save_var, save_var_names = self.create_var_package(
                            func, meta, meta_type, coordinates, coor_out, file_type=file_type, package_type='disk_data')
                        clear_mem_var[func] = save_var
                        clear_mem_var_names[func] = save_var_names

                        if func in save_disk:
                            pipeline_processing['save_var_name'].append(save_var_names)
                            pipeline_processing['save_var'].append(save_var)
                        else:
                            pipeline_processing['save_var_name'].append([])
                            pipeline_processing['save_var'].append([])

                        clear_vars = []
                        clear_names = []

                        for clear in remove_mem:
                            clear_vars.append(clear_mem_var[clear])
                            clear_names.append(clear_mem_var_names[clear])

                        pipeline_processing['clear_mem_var_name'].append(clear_names)
                        pipeline_processing['clear_mem_var'].append(clear_vars)

                    # Then the parallel processing in blocks
                    blocks = (pipeline['coordinate'][0].shape[0] * pipeline['coordinate'][0].shape[0]) / self.pixels + 1
                    lines = pipeline['coordinate'][0].shape[0] / blocks + 1

                    for block_no in range(blocks):

                        start_line = block_no * lines
                        block_pipeline = copy.deepcopy(pipeline_processing)

                        for i in len(block_pipeline['function']):
                            block_pipeline['proc_var_name'][i].append('s_lin')
                            block_pipeline['proc_var'][i].append(start_line)
                            block_pipeline['proc_var_name'][i].append('lines')
                            block_pipeline['proc_var'][i].append(lines)

                        self.processing_packages.append(block_pipeline)
                        self.write_res()

                elif pipeline['type'] == 'concatenate':

                    # Filter all multilook steps from both multilook and concatenate sets
                    for func, file_type, settings, meta, meta_type, coordinates, coor_out in zip(pipeline['steps'], pipeline['file_type'], pipeline['settings'],
                            pipeline['meta'], pipeline['meta_type'], pipeline['coordinate'], pipeline['coor_out']):

                        # All multilook steps are independent so we give them independent steps here too.
                        pipeline_ml = copy.deepcopy(dummy_processing)
                        pipeline_ml['processing'] = True
                        res_dat = pipeline_ml['res_dat']

                        if func == 'concatenate':
                            pipeline_ml['function'] = [self.processes[func]]

                            meta_var, meta_var_names, res_dat = self.create_var_package(
                                func, meta, meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                step=settings['step'], package_type='processing', res_dat=res_dat)
                            disk_var, disk_var_names, res_dat = self.create_var_package(
                                func, meta, meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                step=settings['step'], package_type='disk_data', res_dat=res_dat)

                            pipeline_ml['proc_var_name'] = [meta_var_names]
                            pipeline_ml['proc_var'] = [meta_var]
                            pipeline_ml['create_var_name'] = [disk_var_names]
                            pipeline_ml['create_var'] = [disk_var]
                            pipeline_ml['save_var_name'] = [disk_var_names]
                            pipeline_ml['save_var'] = [disk_var]
                            pipeline_ml['clear_mem_var_name'] = [[disk_var_names]]
                            pipeline_ml['clear_mem_var'] = [[disk_var]]

                        # After concatenation the memory stored data should be removed.
                        elif func in ['multilook', 'interferogram']:

                            pipeline_ml['function'] = [self.processes[func]]
                            pipeline_ml['clear_mem'] = True

                            if func == 'multilook':
                                disk_var, disk_var_names = self.create_var_package(
                                    func, meta, meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                    step=settings['step'], package_type='disk_data')
                            else:  # If it is interferogram
                                disk_var, disk_var_names = self.create_var_package(
                                    func, meta, meta_type, coordinates, coor_out, file_type=file_type,
                                    package_type='disk_data')

                            pipeline_ml['clear_mem_var_name'] = [[disk_var_names]]
                            pipeline_ml['clear_mem_var'] = [[disk_var]]

                        processing_package.append(pipeline_ml)

                # Now run this package in parallel.
                self.run_parallel_package(processing_package)
                self.write_res()

    def create_var_package(self, function, meta, meta_type, coordinates, coor_out, step='', file_type='', package_type='processing', res_dat=''):

        vars = []
        var_names = []

        if not isinstance(res_dat, dict):
            print('res_dat should be a nested dict containing res files')

        if package_type == 'processing':
            func_var_names = self.processing_inputs[function]['__init__']
        elif package_type == 'metadata':
            func_var_names = self.processing_inputs[function]['add_meta_data']
        elif package_type == 'disk_data':
            func_var_names = self.processing_inputs[function]['save_to_disk']
        else:
            print('package type should processing, metadata, disk_data')
            return

        # First the variables that can be used for every case here.
        if 'meta' in func_var_names:
            var_names.append('meta')
            vars.append(self.res_dat[meta][meta_type])
            if res_dat:
                res_dat[meta][meta_type] = self.res_dat[meta][meta_type]

        # Coordinate of input and output
        if 'coordinates' in func_var_names:
            var_names.append('coordinates')
            vars.append(coordinates)
        if 'coor_out' in func_var_names:
            var_names.append('coor_out')
            vars.append(coor_out)

        # Step and file types
        if 'step' in func_var_names:
            var_names.append('step')
            vars.append(step)
        if 'file_type' in func_var_names:
            var_names.append('file_type')
            vars.append(file_type)

        if package_type in ['metadata', 'processing']:
            if 'master_meta' in func_var_names:
                var_names.append('master_meta')
                vars.append(self.res_dat[meta]['master'])
                if res_dat:
                    res_dat[meta]['master'] = self.res_dat[meta]['master']
            if 'cmaster_meta' in func_var_names:
                var_names.append('cmaster_meta')
                vars.append(self.res_dat[meta]['cmaster'])
                if res_dat:
                    res_dat[meta]['cmaster'] = self.res_dat[meta]['cmaster']
            if 'meta_slices' in func_var_names:
                var_names.append('meta_slices')
                meta_keys = [m for m in self.res_dat.keys() if meta != 'full']
                vars.append([self.res_dat[m][meta_type] for m in meta_keys])
                if res_dat:
                    for m in meta_keys:
                        res_dat[m][meta_type] = self.res_dat[m][meta_type]

            # Todo create ifg on the fly if needed.
            if 'ifg_meta' in func_var_names:
                var_names.append('ifg_meta')
                vars.append(self.res_dat[meta]['ifg'])
                if res_dat:
                    res_dat[meta]['ifg'] = self.res_dat[meta]['ifg']

            return vars, var_names, res_dat
        else:
            return vars, var_names

    def write_res(self):
        # This function writes all the available .res files to disk. Applied after processing/multilooking/concatenating
        # This method ensures us that in case processing fails somewhere and the program is exited prematurely
        # the datastack will not be corrupted.

        for key in self.res_dat.keys():
            for meta_type in self.res_dat[key].keys():
                self.res_dat[key][meta_type].write()

    def run_parallel_package(self, parallel_package):
        # This function runs a parallel package.

        for key in parallel_package.res_dat.keys():
            for dat_type in parallel_package.res_dat[key].keys():
                if not isinstance(parallel_package.res_dat[key][dat_type], ImageData):
                    print(dat_type + ' is missing for processing.')
                    return dat_type

        res_dat = self.pool.map(parallel_package)

        # Update the .res files of this image
        for key in res_dat.keys():
            for meta_type in res_dat[key].keys():
                self.res_dat[key][meta_type] = res_dat[key][meta_type]

    @staticmethod
    def run_parallel(dat):

        # First split the functions and variables
        functions = dat['function']

        for func, n in zip(functions, range(len(functions))):

            if dat['meta']:

                var = dat['meta_var'][n]
                var_names = dat['meta_var_names'][n]

                if len(var_names) > 0:
                    # Because the number of variables can vary we use the eval functions.
                    func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                    eval_str = 'proc_func = func.add_meta_data(' + ','.join(func_str) + ')'
                    eval(eval_str)

            if dat['create']:

                var = dat['create_var'][n]
                var_names = dat['create_var_names'][n]

                if len(var_names) > 0:
                    # Because the number of variables can vary we use the eval functions.
                    func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                    eval_str = 'func.create_output_files(' + ','.join(func_str) + ')'
                    eval(eval_str)

            if dat['proc']:

                var = dat['proc_var'][n]
                var_names = dat['proc_var_names'][n]

                if len(var_names) > 0:
                    # Because the number of variables can vary we use the eval functions.
                    func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                    eval_str = 'proc_func = func(' + ','.join(func_str) + ')'
                    eval(eval_str)

                # Run the function created by the eval() string
                proc_func()

            if dat['save']:
                var = dat['save_var'][n]
                var_names = dat['save_var_names'][n]

                if len(var_names) > 0:
                    # Because the number of variables can vary we use the eval functions.
                    func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                    eval_str = 'func.save_to_disk(' + ','.join(func_str) + ')'
                    eval(eval_str)

            if dat['clear_mem']:

                for var, var_names in zip(dat['clear_mem_var'][n], dat['clear_mem_var_names'][n]):

                    if len(var_names) > 0:
                        # Because the number of variables can vary we use the eval functions.
                        func_str = [var_names[i] + '=var[' + str(i) + ']' for i in range(len(var))]
                        eval_str = 'func.clear_memory(' + ','.join(func_str) + ')'
                        eval(eval_str)

        return dat['res_dat']
