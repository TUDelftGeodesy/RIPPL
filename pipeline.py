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
from rippl.run_parallel import run_parallel
from collections import defaultdict
from rippl.processing_list import ProcessingList
from rippl.image_data import ImageData
import numpy as np
from rippl.processing_steps.concatenate import Concatenate
from rippl.coordinate_system import CoordinateSystem
from multiprocessing import Pool
import sys


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
            if type(im).__name__ == 'Image' or type(im).__name__ == 'Interferogram':

                if len(self.slice_ids) == 0:
                    self.slice_ids.extend(im.slices.keys())
                else:
                    self.slice_ids = list(set(self.slice_ids) & set(im.slices.keys()))

        self.slice_ids = sorted(list(set(self.slice_ids)))

        # Re-organize the different full images and slices
        self.res_dat = dict()

        for slice_id in self.slice_ids:
            slice = dict()
            for im, im_str in zip([self.cmaster, self.master, self.slave, self.ifg], ['cmaster', 'master', 'slave', 'ifg']):

                if type(im).__name__ == 'Image' or type(im).__name__ == 'Interferogram':
                    if slice_id in im.slices.keys():
                        slice[im_str] = im.slices[slice_id]

            self.res_dat[slice_id] = copy.deepcopy(slice)

        image = dict()
        for im, im_str in zip([self.cmaster, self.master, self.slave, self.ifg], ['cmaster', 'master', 'slave', 'ifg']):

            if type(im).__name__ == 'Image' or type(im).__name__ == 'Interferogram':
                if im.res_file:
                    image[im_str] = copy.deepcopy(im.res_data)

            self.res_dat['full'] = image

        if self.cmaster == self.slave:
            for slice_id in self.slice_ids:
                self.res_dat[slice_id]['cmaster'] = self.res_dat[slice_id]['slave']
            self.res_dat['full']['cmaster'] = self.res_dat['full']['slave']
        elif self.cmaster == self.master:
            for slice_id in self.slice_ids:
                self.res_dat[slice_id]['cmaster'] = self.res_dat[slice_id]['master']
            self.res_dat['full']['cmaster'] = self.res_dat['full']['master']

        self.clean_memmaps()
        self.clean_memory()

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
        self.pool = []

    def __call__(self, function, settings, coor, meta_type, file_type=''):
        # Here we prepare and run the parallel processing for this image at once.

        # The specific settings for some of the functions. Sometimes we want to run with specific settings for one
        # of the intermediate or the final function.
        # This variable should be a dictionary, containing dictionaries for individual functions.
        self.settings = settings
        if not isinstance(coor, CoordinateSystem):
            print('coor should be an CoordinateSystem object')
            return

        self.coor = coor
        self.slice = coor.slice
        self.meta_type = meta_type

        self.function = []
        if isinstance(function, str):
            self.function = [function]
        elif isinstance(function, list):
            self.function = function

        self.file_type = []
        if not file_type:
            for func in self.function:
                self.file_type.append([func + coor.sample])
        elif isinstance(file_type, str):
            self.file_type = [[file_type + coor.sample]]
        elif isinstance(file_type, list) and len(self.function) == 1:
            self.file_type = [[f_type + coor.sample for f_type in file_type]]
        elif isinstance(file_type, list):
            for f_type in file_type:
                if isinstance(f_type, list):
                    self.file_type.append([f_t + coor.sample for f_t in f_type])
                elif isinstance(f_type, str):
                    self.file_type.append([f_type + coor.sample])

        if len(self.function) != len(self.file_type):
            print('Number of input functions and file types is not the same!')

        self.define_function_order()
        self.run_parallel_processing()

        self.clean_memmaps()
        self.clean_memory()

        # Finally save the resultfile information
        for slice_id in self.slice_ids:
            for im, im_str in zip([self.cmaster, self.master, self.slave, self.ifg], ['cmaster', 'master', 'slave', 'ifg']):

                if type(im).__name__ == 'Image' or type(im).__name__ == 'Interferogram':
                    im.slices[slice_id] = self.res_dat[slice_id][im_str]

        for im, im_str in zip([self.cmaster, self.master, self.slave, self.ifg], ['cmaster', 'master', 'slave', 'ifg']):

            if type(im).__name__ == 'Image' or type(im).__name__ == 'Interferogram':
                im.res_data = self.res_dat['full'][im_str]

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
        for func in self.function:
            if not func in self.processes.keys():
                print('processing function ' + func + ' does not exist!')
                return

        # Initialize the function steps before any concatenation or multilooking.

        # Initialize the different 'pipelines'. Or combinations of functions that can be run using memory and data
        # slicing. The most important thing here is that the input resolution of the data does not change.
        # e.g. this changes when multilooking/concatenation or resampling
        self.pipelines = []
        self.concatenations = []

        # Create a list of steps that still have to be performed but did not fit in the current pipeline.
        concat = dict([('step', []), ('file_type', []), ('meta', []), ('meta_type', []), ('coor_in', []), 
                       ('coor_out', [])])

        # Shift of coordinates by multilooking
        multilook = dict([('step', []), ('file_type', []), ('meta', []), ('meta_type', []), ('coor_in', []), 
                          ('coor_out', []), ('meta_name', [])])

        # Start of a new pipeline. Before multilooking, concatenation or resampling.
        dummy_pipeline = dict([('step', []), ('file_type', []), ('meta', []), ('meta_type', []), ('coor_in', []), 
                          ('coor_out', []), ('meta_name', [])])
        new_pipeline = copy.deepcopy(dummy_pipeline)

        proc_depth = 1

        ############################################################################################################
        # Full image processing

        # First perform all needed steps if we process the full image.
        # Check if step exists
        if not self.slice:
            for step_file_type, fn in zip(self.file_type, np.arange(len(self.file_type))):
                for file_type in step_file_type:
                    if self.res_dat['full'][self.meta_type].check_datafile(self.function,
                                                                file_type=file_type + self.coor.sample, warn=False):
                        self.file_type[fn].remove(file_type)
            if len(self.file_type) == 0:
                print('Processing not needed, file already available')
                return
            else:
                start_funcs = self.function
                start_meta_name = 'full'
                start_file_type = self.file_type
                start_meta_type = [self.meta_type for n in np.arange(len(self.function))]
                start_coor = copy.deepcopy(self.coor)

                if start_coor.meta_name != start_meta_name:
                    if 'cmaster' in self.res_dat['full'].keys():
                        start_coor.add_res_info(self.res_dat['full']['cmaster'])
                    else:
                        start_coor.add_res_info(self.res_dat['full'][self.meta_type])

            concat, multilook, new_pipeline = self.create_pipeline(start_funcs, start_file_type, start_meta_type, start_meta_name, start_coor,
                                                                       concat, multilook, new_pipeline, proc_depth)
            proc_depth += 1

            # First iterate over the full image till there are no resolution shifts existing anymore.
            while len(new_pipeline['step']) > 0 or len(multilook['step']) > 0:

                old_meta = ''

                # Create new pipelines if needed.
                while len(new_pipeline['step']) > 0:
                    out_samples = [coor.sample for coor in new_pipeline['coor_out']]

                    step_ids = np.where((np.array(out_samples) == new_pipeline['coor_out'][-1].sample) *
                                        (np.array(new_pipeline['meta']) == new_pipeline['meta'][-1]))[0]
                    start_funcs = np.array(new_pipeline['step'])[step_ids]
                    start_meta_type = np.array(new_pipeline['meta_type'])[step_ids]
                    start_file_type = np.array(new_pipeline['file_type'])[step_ids]
                    start_coor = new_pipeline['coor_out'][-1]

                    if start_coor.meta_name != start_meta_name:
                        if 'cmaster' in self.res_dat['full'].keys():
                            start_coor.add_res_info(self.res_dat['full']['cmaster'])
                        else:
                            start_coor.add_res_info(self.res_dat['full'][self.meta_type])

                    # If we start working on the same slice, independent calculations are not possible anymore
                    if old_meta == start_meta_name:
                        proc_depth += 1

                    # Remove the processed types from the list
                    for id in np.sort(step_ids)[::-1]:
                        for key in ['step', 'file_type', 'coor_in', 'coor_out', 'meta', 'meta_type']:
                            new_pipeline[key].pop(id)

                    # Create the pipeline
                    concat, multilook, new_pipeline = self.create_pipeline(start_funcs, start_file_type, start_meta_type, start_meta_name, start_coor,
                                                                       concat, multilook, new_pipeline, proc_depth)
                    old_meta = start_meta_name


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
            proc_depth += 1

        #############################################################################################################
        # Individual slices

        # If we are only processing slices, this is the moment the processing starts.
        if self.slice:
            for slice_name in self.slice_ids:
                for step_file_type, fn in zip(self.file_type, np.arange(len(self.file_type))):
                    for file_type in step_file_type:
                        if self.res_dat[slice_name][self.meta_type].check_datafile(self.function,
                                                                                   file_type=file_type + self.coor.sample,
                                                                                   warn=False):
                            self.file_type[fn].remove(file_type)
                if len(self.file_type) == 0:
                    print('Processing not needed, file already available')
                    return
                else:
                    start_funcs = self.function
                    start_meta = slice_name
                    start_file_type = self.file_type
                    start_meta_type = [self.meta_type for n in np.arange(len(self.function))]
                    start_coor = copy.deepcopy(self.coor)

                    if start_coor.meta_name != start_meta:
                        if 'cmaster' in self.res_dat[start_meta].keys():
                            start_coor.add_res_info(self.res_dat[start_meta]['cmaster'])
                        else:
                            start_coor.add_res_info(self.res_dat[start_meta][self.meta_type])

                concat, multilook, new_pipeline = self.create_pipeline(start_funcs, start_file_type, start_meta_type, start_meta, start_coor,
                                                                       concat, multilook, new_pipeline, proc_depth)
            proc_depth += 1

        # Now we can perform a similar process as for the full image, but now for the slices.
        # First iterate over the full image till there are no resolution shifts existing anymore.
        while len(new_pipeline['step']) > 0 or len(multilook['step']) > 0:

            old_meta = ''

            # Create new pipelines if needed.
            while len(new_pipeline['step']) > 0:
                out_samples = [coor.sample for coor in new_pipeline['coor_out']]

                step_ids = np.where((np.array(out_samples) == new_pipeline['coor_out'][-1].sample) *
                                    (np.array(new_pipeline['meta']) == new_pipeline['meta'][-1]))[0]
                start_funcs = np.array(new_pipeline['step'])[step_ids]
                start_meta_type = np.array(new_pipeline['meta_type'])[step_ids]
                start_file_type = np.array(new_pipeline['file_type'])[step_ids]
                start_meta_name = new_pipeline['meta'][-1]
                start_coor = new_pipeline['coor_out'][-1]

                if start_coor.meta_name != start_meta_name:
                    if 'cmaster' in self.res_dat[start_meta_name].keys():
                        start_coor.add_res_info(self.res_dat[start_meta_name]['cmaster'])
                    else:
                        start_coor.add_res_info(self.res_dat[start_meta_name][self.meta_type])

                # If we start working on the same slice, independent calculations are not possible anymore
                if old_meta == start_meta_name:
                    proc_depth += 1

                # Remove the processed types from the list
                for id in np.sort(step_ids)[::-1]:
                    for key in ['step', 'file_type', 'coor_in', 'coor_out', 'meta', 'meta_type']:
                        new_pipeline[key].pop(id)

                # Create the pipeline
                concat, multilook, new_pipeline = self.create_pipeline(start_funcs, start_file_type, start_meta_type, start_meta_name,
                                                                       start_coor, concat, multilook, new_pipeline, proc_depth)
                old_meta = start_meta_name

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

    def processing_info(self, step, coor, meta, meta_name='full'):
        # This function checks the input needed for the

        inputs = self.processing_inputs[step]['processing_info']

        if 'coor_in' in inputs:
            coor_in = ''
            if step in self.settings.keys():
                    if meta_name in self.settings[step].keys():
                        if 'coor_in' in self.settings[step][meta_name].keys():
                            coor_in = self.settings[step][meta_name]['coor_in']

        if 'meta_type' in inputs and 'coor_in' in inputs:
            input_dat, output_data, mem_use = self.processes[step].processing_info(coor, meta_type=meta, coor_in=coor_in)
        elif not 'meta_type' in inputs and 'coor_in' in inputs:
            input_dat, output_data, mem_use = self.processes[step].processing_info(coor, coor_in=coor_in)
        elif 'meta_type' in inputs and not 'coor_in' in inputs:
            input_dat, output_data, mem_use = self.processes[step].processing_info(coor, meta_type=meta)
        else:
            input_dat, output_data, mem_use = self.processes[step].processing_info(coor)

        return input_dat, output_data, mem_use

    def create_concatenate_processing(self, concat, new_pipeline, depth):
        # Create a set of functions that can be run for a concatenation step.
        # If the concatenation also includes an multilooking of the bursts, these two steps are combined.

        for con_step, con_file_type, con_meta_type, con_coor_in, con_coor_out in zip(
                concat['step'], concat['file_type'], concat['meta_type'], concat['coor_in'], concat['coor_out']):

            depth += 1

            ml = dict([('step', []), ('file_type', []), ('meta', []), ('meta_type', []), ('coor_in', []),
                       ('coor_out', []), ('meta_name', [])])

            input_data, output_data, mem_use = self.processing_info(con_step, con_coor_in, con_meta_type)
            meta_types, step_types, file_types = Pipeline.order_input_data(input_data)

            concat_names = [slice_str for slice_str in self.res_dat.keys() if slice_str != 'full']
            concat_slices = [self.res_dat[slice_str][con_meta_type] for slice_str in concat_names]
            concat_res, concat_in_coors, concat_coors = Concatenate.find_slice_coordinates(concat_slices, con_coor_in, con_coor_out)

            # Additional settings for concatenation.
            settings = dict()
            settings['step'] = con_step
            if con_file_type.endswith(con_coor_out.sample) and len(con_coor_out.sample) > 0:
                con_file_type = con_file_type[:-len(con_coor_out.sample)]

            settings['file_type'] = con_file_type

            for slice_name, slice_coor, slice_in_coor in zip(concat_names, concat_coors, concat_in_coors):

                slice_coor.meta_name = ''
                resample = False
                multilook = False

                # First check whether additional multilooking is needed.
                for meta_type, step_type, file_type, n in zip(meta_types, step_types, file_types, np.arange(len(file_types))):
                    if 'coor_change' in input_data[meta_type][step_type][file_type].keys():
                        if input_data[meta_type][step_type][file_type]['coor_change'] == 'resample':
                            resample = True

                    in_sample = input_data[meta_type][step_type][file_type]['coordinates'].sample
                    if (slice_coor.sample != in_sample and not resample) or con_step == 'interferogram':
                        in_coor = input_data[meta_type][step_type][file_type]['coordinates']
                        multilook = True
                    elif resample:
                        in_coor = input_data[meta_type][step_type][file_type]['coordinates']
                    else:
                        in_coor = []

                # If it already exists.
                if self.res_dat[slice_name][con_meta_type].check_datafile(con_step, file_type=con_file_type + slice_coor.sample, warn=False):
                    pass

                elif multilook:
                    ml['step'].append(con_step)
                    ml['file_type'].append(con_file_type)
                    ml['meta_type'].append(con_meta_type)
                    ml['meta'].append(slice_name)
                    ml['coor_in'].append(slice_in_coor)
                    ml['coor_out'].append(slice_coor)

                else:
                    id = Pipeline.check_step_unique(new_pipeline, con_step, slice_name, con_meta_type, slice_coor,
                                                    [], con_file_type)
                    if id is False:
                        id = Pipeline.check_step_unique(new_pipeline, con_step, slice_name, con_meta_type,
                                                        slice_coor, [])
                        if id is False:
                            new_pipeline['step'].append(con_step)
                            new_pipeline['file_type'].append([con_file_type])
                            new_pipeline['meta'].append(slice_name)
                            new_pipeline['meta_type'].append(con_meta_type)
                            new_pipeline['coor_in'].append(slice_in_coor)
                            new_pipeline['coor_out'].append(slice_coor)
                        else:
                            new_pipeline['file_type'][id[0]].append(con_file_type)

            func_set, new_pipeline = self.create_ml_processing(ml, new_pipeline, depth)

            func_set['main_proc_depth'] = copy.copy(depth)

            func_set['proc_type'] = 'concatenate'
            func_set['step'].append('concatenate')
            func_set['file_type'].append([''])
            func_set['meta'].append(concat_names)
            func_set['meta_type'].append(con_meta_type)
            func_set['coor_in'].append([])
            func_set['coor_out'].append(con_coor_out)
            func_set['settings'].append(settings)

            self.pipelines.append(func_set)

        return new_pipeline, depth

    def create_ml_processing(self, multilook, new_pipeline, depth):
        # This function extracts all processing steps for multilooking of the full or slave images.
        # These functions are merely a set of independent functions and do not belong to a pipeline as parallel processing
        # of parts of images is not possible (and almost never needed)

        # Init the func_set variable
        func_set = dict([('proc_type', 'multilook'), ('step', []), ('file_type', []), ('meta', []), ('meta_type', []),
                         ('coor_in', []), ('coor_out', []), ('save_disk', []), ('settings', []), ('rem_mem', [])])
        recursive_dict = lambda: defaultdict(recursive_dict)

        # If the direct step before concatenation is a multilooking step, these two can be combined.
        # An exception is the interferogram step, which is in most cases a multilooking step too.
        for step, file_type, meta_type, meta_name, in_coor, out_coor in zip(multilook['step'], multilook['file_type'],
                                                                            multilook['meta_type'], multilook['meta'],
                                                                            multilook['coor_in'], multilook['coor_out']):

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
                    m_file_type = [file_type]

                    input_data, output_data, mem_use = self.processing_info(step, out_coor, meta_type)
                    meta_tps, step_tps, file_tps = Pipeline.order_input_data(input_data)
                else:
                    m_step = 'multilook'
                    m_file_type = ['']

                    meta_tps = [meta_type]
                    step_tps = [step]
                    file_tps = [file_type]

                settings['step'] = step
                settings['file_type'] = file_type

                func_set['main_proc_depth'] = copy.copy(depth)
                func_set['step'].append(m_step)
                func_set['file_type'].append(m_file_type)
                func_set['meta'].append(meta_name)
                func_set['meta_type'].append(meta_type)
                func_set['coor_in'].append(in_coor)
                func_set['coor_out'].append(out_coor)
                func_set['settings'].append(settings)
                func_set['rem_mem'].append(recursive_dict())

                if step == 'interferogram':
                    func_set['rem_mem'][-1]['master'][step_tps[0]] = [file_tps[0]]
                    func_set['rem_mem'][-1]['slave'][step_tps[0]] = [file_tps[0]]
                else:
                    func_set['rem_mem'][-1][meta_type][step_tps[0]] = [file_tps[0]]

                # Now add the original requested files to our to be processed list.
                for meta_tp, step_tp, file_tp in zip(meta_tps, step_tps, file_tps):

                    f_type = file_tp + in_coor.sample
                    if not self.res_dat[meta_name][meta_type].check_datafile(self.function, file_type=f_type, warn=False):

                        id = Pipeline.check_step_unique(new_pipeline, step_tp, meta_name, meta_tp, out_coor,
                                                        in_coor, file_tp)
                        if id is False:
                            id = Pipeline.check_step_unique(new_pipeline, step_tp, meta_name, meta_tp, out_coor,
                                                            in_coor)
                            if id is False:
                                new_pipeline['step'].append(step_tp)
                                new_pipeline['file_type'].append([file_tp])
                                new_pipeline['meta_type'].append(meta_tp)
                                new_pipeline['meta'].append(meta_name)
                                new_pipeline['coor_in'].append([])
                                new_pipeline['coor_out'].append(in_coor)
                            else:
                                new_pipeline['file_type'][id[0]].append(file_tp)

        return func_set, new_pipeline

    def create_pipeline(self, start_func, start_file_type, start_meta_type, meta_name, start_coor, concat, multilook, new_pipeline, pipeline_depth):

        # These variables are used to store the steps that cannot be processed in the current pipeline. There are therefore first stored and then
        pipeline = dict()
        pipeline['meta_type'] = []    # What meta type are we dealing with (slave, master, cmaster, ifg?)
        pipeline['step'] = []         # What step should be processed?
        pipeline['file_type'] = []    # What is the file type we are working with?
        pipeline['meta'] = []         # What is the meta data file of this step? (normally one ImageData and multiple for Concatenate steps
        pipeline['coor_in'] = []      # What is the coordinate system of this step?
        pipeline['coor_out'] = []     # What is the coordinate system of this step?
        pipeline['save_disk'] = []    # Should the data be saved to disk?
        pipeline['rem_mem'] = []   # This variable indicates the processing depth (e.g. 1,2,3 etc in line. It increases with
                            # one every iteration. This can later be used to remove unused data in memory

        # Start with adding the first steps to the processing
        for step, meta_type, file_types in zip(start_func, start_meta_type, start_file_type):

            new_file_types = []
            for file_type in file_types:
                if not self.res_dat[meta_name][meta_type].check_datafile(step, file_type=file_type, warn=False):
                    new_file_types.append(file_type)

            if len(new_file_types) > 0:
                pipeline['step'].append(step)
                pipeline['file_type'].append(new_file_types)
                pipeline['meta'].append(meta_name)
                pipeline['meta_type'].append(meta_type)
                pipeline['coor_in'].append([])
                pipeline['coor_out'].append(start_coor)

                # When to remove data from memory. Everything will be removed from memory at the end for a pipeline
                input_dat, output_dat, mem_use = self.processing_info(step, start_coor, meta_type, meta_name)
                m_types, s_types, f_types = Pipeline.order_input_data(output_dat)
                out_types = [f_type for f_type in f_types]
                recursive_dict = lambda: defaultdict(recursive_dict)
                pipeline['rem_mem'].append(recursive_dict())
                pipeline['rem_mem'][-1][m_types[0]][s_types[0]] = out_types

                if step in start_func:
                    pipeline['save_disk'].append(True)
                else:
                    pipeline['save_disk'].append(False)

        depend = pipeline['step']
        depend_meta_type = pipeline['meta_type']
        depend_file_type = pipeline['file_type']
        depth = 0

        while len(depend) > 0:

            depth += 1

            functions = depend
            functions_meta_type = depend_meta_type
            functions_file_type = depend_file_type

            depend = []
            depend_meta_type = []
            depend_file_type = []

            for p_step, p_meta_type, p_file_type in zip(functions, functions_meta_type, functions_file_type):

                input_dat, output_dat, mem_use = self.processing_info(p_step, start_coor, p_meta_type, meta_name)
                meta_types, step_types, file_types = Pipeline.order_input_data(input_dat)
                parent_id = np.where(np.array(pipeline['step']) == p_step)[0][0]

                if len(meta_types) == 0:
                    continue

                # Check whether all slices already exist for concatenation or whether the processing of this step
                # should be done in slices.
                if meta_name == 'full':

                    use_slices = False
                    for p_type in p_file_type:
                        use_slices = self.check_slices_exist(p_meta_type, p_step, p_type)
                        if not use_slices:
                            continue

                    # If one of the inputs requires slices while the main image is a full this should be done first.
                    for meta_type, step, file_type in zip(meta_types, step_types, file_types):
                        if input_dat[meta_type][step][file_type]['slice'] == True:
                            use_slices = True

                    if p_step in ['inverse_geocode', 'geometrical_coreg']:
                        use_slices = False

                    coor_in = input_dat[meta_types[0]][step_types[0]][file_types[0]]['coordinates']
                    if use_slices:

                        for p_type in p_file_type:
                            id = Pipeline.check_step_unique(concat, p_step, meta_name, p_meta_type,
                                                            start_coor, coor_in, p_type)
                            if id is False:
                                concat['step'].append(p_step)
                                concat['file_type'].append(p_type)
                                concat['meta'].append(meta_name)
                                concat['meta_type'].append(p_meta_type)
                                coor = copy.deepcopy(coor_in)
                                if len(coor.shape) == 0:
                                    coor.add_res_info(self.res_dat[meta_name][p_meta_type])
                                concat['coor_in'].append(coor)
                                concat['coor_out'].append(start_coor)

                        # Remove the original processing step because it has to be created with different coordinates.
                        for dat_type in ['step', 'file_type', 'meta', 'meta_type', 'coor_in', 'coor_out', 'rem_mem']:
                            pipeline[dat_type].pop(parent_id)

                        continue

                # Now check whether the input and output coordinates match. If they do no match, a multilooking step
                # should be performed, which discontinues the pipeline.
                for meta_type, step, file_type, i in zip(meta_types, step_types, file_types, np.arange(len(meta_types))):
                    # If data already exist we do not need to process it

                    ml_count = 0
                    coor_in = input_dat[meta_type][step][file_type]['coordinates']

                    # In case of resampling or multilooking information of the master process should be added
                    if start_coor.sample != coor_in.sample:
                        # Find info of original step
                        if 'coor_change' in input_dat[meta_type][step][file_type].keys():
                            if input_dat[meta_type][step][file_type]['coor_change'] == 'resample':
                                # Add coordinate information to original step
                                pipeline['coor_in'][parent_id] = coor_in
                        else:
                            ml_count += 1

                            for p_type in p_file_type:

                                id = Pipeline.check_step_unique(multilook, p_step, meta_name, p_meta_type, start_coor,
                                                                coor_in, p_type)

                                if id is False:
                                    multilook['step'].append(p_step)
                                    multilook['file_type'].append(p_type)
                                    multilook['meta'].append(meta_name)
                                    multilook['meta_type'].append(p_meta_type)
                                    coor = copy.deepcopy(coor_in)
                                    if len(coor.shape) == 0:
                                        coor.add_res_info(self.res_dat[meta_name][p_meta_type])
                                    multilook['coor_in'].append(coor)
                                    multilook['coor_out'].append(start_coor)

                            meta_types.pop(i)
                            step_types.pop(i)
                            file_types.pop(i)

                        # Remove the original processing step because it has to be created with different coordinates.
                        if ml_count == len(meta_type):
                            for dat_type in ['step', 'file_type', 'meta', 'meta_type', 'coor_in', 'coor_out', 'rem_mem']:
                                pipeline[dat_type].pop(parent_id)

                # Now check for the slave processes, which do not need concatenation or multilooking.
                for meta_type, step, file_type in zip(meta_types, step_types, file_types):

                    parent_id = np.where(np.array(pipeline['step']) == p_step)[0][0]
                    coor_in = input_dat[meta_type][step][file_type]['coordinates']

                    if self.res_dat[meta_name][meta_type].check_datafile(step, file_type=file_type, warn=False):

                        # If the input files already exist they can be removed from memory directly after processing.
                        if meta_type in pipeline['rem_mem'][parent_id].keys():
                            if step in pipeline['rem_mem'][parent_id][meta_type].keys():
                                if not file_type in pipeline['rem_mem'][parent_id][meta_type][step]:
                                    pipeline['rem_mem'][parent_id][meta_type][step].append(file_type)
                            else:
                                pipeline['rem_mem'][parent_id][meta_type][step] = [file_type]
                        else:
                            pipeline['rem_mem'][parent_id][meta_type][step] = [file_type]

                    elif start_coor.sample != coor_in.sample:
                        # Find info of original step
                        if 'coor_change' in input_dat[meta_type][step][file_type].keys():
                            if input_dat[meta_type][step][file_type]['coor_change'] == 'resample':
                                # Add coordinate information to original step
                                pipeline['coor_in'][parent_id] = coor_in

                                id = Pipeline.check_step_unique(new_pipeline, step, meta_name, meta_type, start_coor,
                                                                coor_in, file_type)
                                if id is False:
                                    id = Pipeline.check_step_unique(new_pipeline, step, meta_name, meta_type,
                                                                    start_coor, coor_in)
                                    if id is False:
                                        new_pipeline['step'].append(step)
                                        new_pipeline['file_type'].append([file_type])
                                        new_pipeline['meta'].append(meta_name)
                                        new_pipeline['meta_type'].append(meta_type)
                                        new_pipeline['coor_in'].append([])
                                        coor = copy.deepcopy(coor_in)
                                        if len(coor.shape) == 0:
                                            coor.add_res_info(self.res_dat[meta_name][p_meta_type])
                                        new_pipeline['coor_out'].append(coor)
                                    else:
                                        new_pipeline['file_type'][id[0]].append(file_type)

                    else:
                        id = Pipeline.check_step_unique(pipeline, step, meta_name, meta_type,
                                                        start_coor, coor_in, file_type)
                        if id is False:
                            id = Pipeline.check_step_unique(pipeline, step, meta_name, meta_type,
                                                            start_coor, coor_in)

                            # In case the step also does not exist
                            if id is False:
                                id = [len(pipeline['step'])]
                                pipeline['step'].append(step)
                                pipeline['file_type'].append([file_type])
                                pipeline['meta'].append(meta_name)
                                pipeline['meta_type'].append(meta_type)
                                pipeline['coor_in'].append([])
                                pipeline['coor_out'].append(start_coor)

                                # In first instance we assume that all files are removed from memory because they are
                                # not needed or already save to disk. If they are needed later on they will be removed
                                # from here.
                                in_dat, out_dat, mem_use = self.processing_info(step, start_coor, meta_type, meta_name)
                                m_types, s_types, f_types = Pipeline.order_input_data(out_dat)
                                out_types = [f_type for f_type, s_type, m_type in zip(f_types, s_types, m_types)
                                             if s_type == step and m_type == meta_type]
                                recursive_dict = lambda: defaultdict(recursive_dict)
                                pipeline['rem_mem'].append(recursive_dict())
                                pipeline['rem_mem'][-1][m_types[0]][s_types[0]] = out_types

                                if step in start_func:
                                    pipeline['save_disk'].append(True)
                                else:
                                    pipeline['save_disk'].append(False)

                            else:
                                # In case the step exists
                                pipeline['file_type'][id[0]].append(file_type)

                            # In both cases add the remove memory step
                            parent_id = np.where(np.array(pipeline['step']) == p_step)[0][0]
                            if isinstance(pipeline['rem_mem'][id[0]][meta_type][step], list):
                                if file_type in pipeline['rem_mem'][id[0]][meta_type][step]:
                                    pipeline['rem_mem'][id[0]][meta_type][step].remove(file_type)
                            if meta_type in pipeline['rem_mem'][parent_id].keys():
                                if step in pipeline['rem_mem'][parent_id][meta_type].keys():
                                    pipeline['rem_mem'][parent_id][meta_type][step].append(file_type)
                                else:
                                    pipeline['rem_mem'][parent_id][meta_type][step] = [file_type]
                            else:
                                pipeline['rem_mem'][parent_id][meta_type][step] = [file_type]

                        if len(depend) > 0:
                            nd = np.where(np.array(depend) == step)[0]
                        else:
                            nd = []
                        # To iterate further.
                        if len(nd) > 0:
                            depend_file_type[nd[0]].append(file_type)
                        else:
                            depend.append(step)
                            depend_meta_type.append(meta_type)
                            depend_file_type.append([file_type])

                        if len(id) > 0:
                            if id[0] != len(pipeline['step']):
                                # When this is not the last excecuted step, move it to the end of the list.
                                for key in ['step', 'file_type', 'meta', 'meta_type', 'coor_in', 'coor_out', 'rem_mem', 'save_disk']:
                                    pipeline[key].append(pipeline[key].pop(id[0]))

        # Add list of processing steps and remove steps occurring in the list.
        # Find unique ids starting from the end of the list (prefer last occurrences)
        if len(pipeline['step']) > 0:
            func_set = dict()
            func_set['main_proc_depth'] = copy.copy(pipeline_depth)
            func_set['proc_type'] = 'pipeline'

            func_set['step'] = np.array(pipeline['step'])[::-1]

            # Now gather the file types and bundle them per processing step.
            func_set['file_type'] = np.array(pipeline['file_type'])[::-1]
            func_set['meta'] = np.array(pipeline['meta'])[::-1]
            func_set['meta_type'] = np.array(pipeline['meta_type'])[::-1]
            func_set['coor_in'] = np.array(pipeline['coor_in'])[::-1]
            func_set['coor_out'] = np.array(pipeline['coor_out'])[::-1]
            func_set['save_disk'] = np.array(pipeline['save_disk'])[::-1]
            func_set['rem_mem'] = np.array(pipeline['rem_mem'])[::-1]

            func_set['settings'] = []
            for step in func_set['step']:
                if step in self.settings.keys():
                    func_set['settings'].append(self.settings[step])
                else:
                    func_set['settings'].append(dict())

            # Save pipeline
            self.pipelines.insert(0, func_set)

        return concat, multilook, new_pipeline

    def check_slices_exist(self, meta_type, step, file_type):
        # This function checks whether the needed file for the full image already exists in all the slices. In that case
        # no processing is needed, only concatenation.

        slices = [self.res_dat[meta_name][meta_type] for meta_name in self.res_dat.keys() if
                  meta_type in self.res_dat[meta_name].keys() and meta_name is not 'full']

        for slice in slices:
            if not slice.check_datafile(step, file_type=file_type, warn=False):
                return False

        return True

    @staticmethod
    def check_step_unique(step_dict, step, meta, meta_type, coor_out, coor_in, file_type=''):
        # Checks whether there is a step with the same step/meta_name/coor_in/coor_out

        if len(step_dict['step']) == 0:
            return False

        ids = np.where(np.array(step_dict['step']) == step)[0]
        ids = ids[np.where(np.array(step_dict['meta_type'])[ids] == meta_type)[0]]
        ids = ids[np.where(np.array(step_dict['meta'])[ids] == meta)[0]]

        if len(ids) == 0:
            return False

        coor_in_samples = []
        coor_out_samples = []
        for coor_in, coor_out in zip(np.array(step_dict['coor_in'])[ids], np.array(step_dict['coor_out'])[ids]):
            coor_out_samples.append(coor_out.sample)
            if isinstance(coor_in, CoordinateSystem):
                coor_in_samples.append(coor_in.sample)
            else:
                coor_in_samples.append(coor_out.sample)
        if not isinstance(coor_in, CoordinateSystem):
            coor_in_sample = coor_out.sample
        else:
            coor_in_sample = coor_in.sample

        ids = ids[np.where((np.array(coor_in_samples) == coor_in_sample) * (np.array(coor_out_samples) == coor_out.sample))[0]]

        if file_type:
            if not file_type in step_dict['file_type'][ids[0]]:
                ids = []

        if len(ids) > 0:
            return ids
        else:
            return False

    def run_parallel_processing(self):
        # This function generates a list of functions and inputs to do the processing.

        # There are four types of inputs for individual packages
        # 1. Processing step > this processes and runs the function for either a part or a full slice
        # 2. Metadata step > this only generates the metadata needed to initialize the processing step to write to disk
        # 3. Create step > this step creates a new file on disk
        # 4. Save step > this step saves the data from memory to disk

        # Apart from this we have an independent step that is always run:
        dummy_processing = dict()
        dummy_processing['function'] = []

        # Store the used res information to restore after processing.
        dummy_processing['res_dat'] = dict()
        for key in self.res_dat.keys():
            dummy_processing['res_dat'][key] = dict()

        dummy_processing['proc_type'] = ''
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

        proc_depths = [pipeline['main_proc_depth'] for pipeline in self.pipelines]
        if len(proc_depths) == 0:
            return

        for proc_depth in np.arange(1, np.max(proc_depths) + 1)[::-1]:
            # Init intialization and processing package
            processing_package = []

            for pipeline in [pipeline for pipeline in self.pipelines if pipeline['main_proc_depth'] == proc_depth]:

                # create pipeline parallel processing packages.
                if pipeline['proc_type'] == 'pipeline':

                    pipeline_init = copy.deepcopy(dummy_processing)
                    pipeline_init['proc_type'] = 'pipeline_init'
                    pipeline_init['meta'] = True
                    pipeline_init['create'] = True
                    res_dat = pipeline_init['res_dat']

                    # First define the processing order
                    save_disk = [pipeline['step'][n] for n in np.arange(len(pipeline['step'])) if pipeline['save_disk'][n] == True]

                    #  Then add the variables which are independent from the block size
                    for func, f_type, meta, meta_type, coordinates, coor_out in zip(pipeline['step'], pipeline['file_type'],
                            pipeline['meta'], pipeline['meta_type'], pipeline['coor_in'], pipeline['coor_out']):

                        file_type = []
                        for f in f_type:
                            if f.endswith(coor_out.sample) and len(coor_out.sample) > 0:
                                file_type.append(f[:-len(coor_out.sample)])
                            elif len(coor_out.sample) == 0:
                                file_type.append(f)
                            else:
                                print('Out coordinate system does not fit with file type name. Review your processing_info of step ' + func)
                                return

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

                    processing_package.insert(0, pipeline_init)

                elif pipeline['proc_type'] == 'multilook' or pipeline['proc_type'] == 'concatenate':

                    # Filter all multilook steps from both multilook and concatenate sets
                    for func, file_type, settings, meta, meta_type, coordinates, coor_out, rem_mem in zip(pipeline['step'], pipeline['file_type'], pipeline['settings'],
                            pipeline['meta'], pipeline['meta_type'], pipeline['coor_in'], pipeline['coor_out'], pipeline['rem_mem']):

                        # All multilook steps are independent so we give them independent steps here too.
                        pipeline_ml = copy.deepcopy(dummy_processing)
                        pipeline_ml['proc_type'] = 'multilook'
                        pipeline_ml['proc'] = True
                        res_dat = pipeline_ml['res_dat']
                        pipeline_ml['clear_mem'] = True

                        if pipeline['proc_type'] == 'multilook' or coordinates.sample == coor_out.sample:
                            pipeline_ml['create'] = True
                            pipeline_ml['meta'] = True
                            pipeline_ml['save'] = True

                        clear_mem_var = []
                        clear_mem_var_name = []

                        for m_type in rem_mem.keys():
                            for fun in rem_mem[m_type].keys():
                                for f_type in rem_mem[m_type][fun]:

                                    clear_var = [self.res_dat[meta][m_type], fun, f_type + coordinates.sample]
                                    clear_var_names = ['meta', 'step', 'file_type']
                                    clear_mem_var.append(clear_var)
                                    clear_mem_var_name.append(clear_var_names)

                        if func in ['multilook', 'interferogram']:
                            pipeline_ml['function'] = [self.processes[func]]

                            if func == 'multilook':
                                proc_var, proc_var_names, res_dat = self.create_var_package(
                                    func, meta, meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                    res_dat=res_dat, step=settings['step'], package_type='processing')
                                disk_var, disk_var_names, res_dat = self.create_var_package(
                                    func, meta, meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                    res_dat=res_dat, step=settings['step'], package_type='disk_data')
                                meta_var, meta_var_names, res_dat = self.create_var_package(
                                    func, meta, meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                    res_dat=res_dat, step=settings['step'], package_type='metadata')
                            else:  # If it is interferogram
                                proc_var, proc_var_names, res_dat = self.create_var_package(
                                    func, meta, meta_type, coordinates, coor_out, file_type=file_type,
                                    res_dat=res_dat, package_type='processing')
                                disk_var, disk_var_names, res_dat = self.create_var_package(
                                    func, meta, meta_type, coordinates, coor_out, file_type=file_type,
                                    res_dat=res_dat, package_type='disk_data')
                                meta_var, meta_var_names, res_dat = self.create_var_package(
                                    func, meta, meta_type, coordinates, coor_out, file_type=file_type,
                                    res_dat=res_dat, package_type='metadata')

                            if pipeline['proc_type'] == 'multilook' or coordinates.sample == coor_out.sample:
                                pipeline_ml['create_var_name'] = [disk_var_names]
                                pipeline_ml['create_var'] = [copy.copy(disk_var)]
                                pipeline_ml['save_var_name'] = [disk_var_names]
                                pipeline_ml['save_var'] = [copy.copy(disk_var)]
                                pipeline_ml['meta_var_name'].append(meta_var_names)
                                pipeline_ml['meta_var'].append(meta_var)

                                if pipeline['proc_type'] == 'multilook':
                                    clear_var = [self.res_dat[meta][meta_type], settings['step'], settings['file_type'] + coor_out.sample]
                                    clear_var_names = ['meta', 'step', 'file_type']
                                else:
                                    clear_var = [self.res_dat[meta][meta_type], 'interferogram', 'interferogram' + coor_out.sample]
                                    clear_var_names = ['meta', 'step', 'file_type']

                                clear_mem_var.append(clear_var)
                                clear_mem_var_name.append(clear_var_names)

                                pipeline_ml['clear_mem_var_name'].append(clear_mem_var_name)
                                pipeline_ml['clear_mem_var'].append(clear_mem_var)

                            pipeline_ml['clear_mem_var'] = [clear_mem_var]
                            pipeline_ml['clear_mem_var_name'] = [clear_mem_var_name]
                            pipeline_ml['proc_var_name'] = [proc_var_names]
                            pipeline_ml['proc_var'] = [proc_var]

                        processing_package.insert(0, pipeline_ml)

            # Now run this package in parallel.
            self.run_parallel_package(processing_package)
            processing_package = []

            # Run again for the concatenation scripts only. (Others can be done in 1 parallel package)
            for pipeline in [pipeline for pipeline in self.pipelines if pipeline['main_proc_depth'] == proc_depth]:
                # Init intialization and processing package

                # create pipeline parallel processing packages.
                if pipeline['proc_type'] == 'pipeline':

                    pipeline_processing = copy.deepcopy(dummy_processing)
                    pipeline_processing['proc_type'] = 'pipeline_processing'
                    pipeline_processing['proc'] = True
                    pipeline_processing['save'] = True
                    pipeline_processing['clear_mem'] = True

                    # First define the processing order
                    remove_mem = []
                    save_disk = [pipeline['step'][n] for n in np.arange(len(pipeline['step'])) if pipeline['save_disk'][n] == True]
                    res_dat = pipeline_processing['res_dat']

                    #  Then add the variables which are independent from the block size
                    for func, f_type, meta, meta_type, coordinates, coor_out, rem_mem in zip(pipeline['step'], pipeline['file_type'],
                            pipeline['meta'], pipeline['meta_type'], pipeline['coor_in'], pipeline['coor_out'], pipeline['rem_mem']):

                        file_type = []
                        for f in f_type:
                            if f.endswith(coor_out.sample) and len(coor_out.sample) > 0:
                                file_type.append(f[:-len(coor_out.sample)])
                            elif len(coor_out.sample) == 0:
                                file_type.append(f)
                            else:
                                print('Out coordinate system does not fit with file type name. Review your processing_info of step ' + func)
                                return

                        pipeline_processing['function'].append(self.processes[func])

                        proc_var, proc_var_names, res_dat = self.create_var_package(func, meta, meta_type, coordinates,
                                                                                    coor_out, package_type='processing', res_dat=res_dat)
                        pipeline_processing['proc_var_name'].append(proc_var_names)
                        pipeline_processing['proc_var'].append(proc_var)

                        save_var, save_var_names, res_dat = self.create_var_package(func, meta, meta_type, coordinates, coor_out,
                                                                           file_type=file_type, package_type='disk_data', res_dat=res_dat)

                        if func in save_disk:
                            pipeline_processing['save_var_name'].append(save_var_names)
                            pipeline_processing['save_var'].append(save_var)
                        else:
                            pipeline_processing['save_var_name'].append([])
                            pipeline_processing['save_var'].append([])

                        # Find where the step is removed from memory (or not)
                        clear_mem_var = []
                        clear_mem_var_name = []

                        for m_type in rem_mem.keys():
                            for fun in rem_mem[m_type].keys():
                                for f_type in rem_mem[m_type][fun]:

                                    clear_var = [self.res_dat[meta][m_type], fun, f_type + coor_out.sample]
                                    clear_var_names = ['meta', 'step', 'file_type']
                                    clear_mem_var.append(clear_var)
                                    clear_mem_var_name.append(clear_var_names)

                        pipeline_processing['clear_mem_var_name'].append(clear_mem_var_name)
                        pipeline_processing['clear_mem_var'].append(clear_mem_var)

                    # Then the parallel processing in blocks
                    if 'unwrap' in pipeline['step']:
                        blocks = 1
                    else:
                        blocks = (pipeline['coor_out'][0].shape[0] * pipeline['coor_out'][0].shape[1]) // self.pixels + 1

                    lines = pipeline['coor_out'][0].shape[0] // blocks + 1

                    for block_no in np.arange(int(blocks)):

                        start_line = block_no * lines
                        if start_line < pipeline['coor_out'][0].shape[0]:
                            block_pipeline = copy.deepcopy(pipeline_processing)

                            for i in np.arange(len(block_pipeline['function'])):
                                block_pipeline['proc_var_name'][i].append('s_lin')
                                block_pipeline['proc_var'][i].append(start_line)
                                block_pipeline['proc_var_name'][i].append('lines')
                                block_pipeline['proc_var'][i].append(lines)

                            processing_package.insert(0, block_pipeline)

                elif pipeline['proc_type'] == 'concatenate':

                    # All multilook steps are independent so we give them independent steps here too.
                    pipeline_ml = copy.deepcopy(dummy_processing)
                    pipeline_ml['proc_type'] = 'concatenate'
                    pipeline_ml['proc'] = True
                    pipeline_ml['clear_mem'] = True
                    pipeline_ml['save_disk'] = True
                    res_dat = pipeline_ml['res_dat']

                    clear_mem_var = []
                    clear_mem_var_name = []

                    # Filter all multilook steps from both multilook and concatenate sets
                    for func, file_type, settings, meta, meta_type, coordinates, coor_out in zip(pipeline['step'], pipeline['file_type'], pipeline['settings'],
                            pipeline['meta'], pipeline['meta_type'], pipeline['coor_in'], pipeline['coor_out']):

                        if func == 'concatenate':
                            pipeline_ml['function'].append(self.processes[func])

                            proc_var, proc_var_names, res_dat = self.create_var_package(
                                func, 'full', meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                meta_list=meta, step=settings['step'], package_type='processing', res_dat=res_dat)
                            meta_var, meta_var_names, res_dat = self.create_var_package(
                                func, 'full', meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                meta_list=meta, step=settings['step'], package_type='metadata', res_dat=res_dat)
                            disk_var, disk_var_names, res_dat = self.create_var_package(
                                func, 'full', meta_type, coordinates, coor_out, file_type=settings['file_type'],
                                meta_list=meta, step=settings['step'], package_type='disk_data', res_dat=res_dat)

                            pipeline_ml['proc_var_name'].append(proc_var_names)
                            pipeline_ml['proc_var'].append(proc_var)
                            pipeline_ml['save_var_name'].append(disk_var_names)
                            pipeline_ml['save_var'].append(disk_var)

                            clear_var = [self.res_dat['full'][meta_type], settings['step'], settings['file_type'] + coor_out.sample]
                            clear_var_name = ['meta', 'step', 'file_type']
                            clear_mem_var.append(clear_var)
                            clear_mem_var_name.append(clear_var_name)

                            pipeline_ml['clear_mem_var'].append(clear_mem_var)
                            pipeline_ml['clear_mem_var_name'].append(clear_mem_var_name)

                        # After concatenation the memory stored data should be removed.
                        elif func in ['multilook', 'interferogram']:

                            if func == 'multilook':
                                clear_var = [self.res_dat[meta][meta_type], settings['step'], settings['file_type'] + coor_out.sample]
                                clear_var_name = ['meta', 'step', 'file_type']
                            else:  # If it is interferogram
                                clear_var = [self.res_dat[meta][meta_type], func, file_type[0] + coor_out.sample]
                                clear_var_name = ['meta', 'step', 'file_type']

                            clear_mem_var.append(clear_var)
                            clear_mem_var_name.append(clear_var_name)

                    processing_package.insert(0, pipeline_ml)

            # Now run this package in parallel.
            self.run_parallel_package(processing_package)
            self.write_res()

    def clean_memmaps(self):
        # This function cleans all res_dat variables from memmaps. (these cause problems when copied for parallel processing)
        for slice_key in self.res_dat.keys():
            for type_key in self.res_dat[slice_key].keys():
                self.res_dat[slice_key][type_key].clean_memmap_files()

    def clean_memory(self):
        # This function cleans all res_dat variables from memmaps. (these cause problems when copied for parallel processing)
        for slice_key in self.res_dat.keys():
            for type_key in self.res_dat[slice_key].keys():
                self.res_dat[slice_key][type_key].clean_memory()

    def create_var_package(self, function, meta, meta_type, coor_in, coor_out, meta_list='', step='', file_type='', package_type='processing', res_dat=''):

        vars = []
        var_names = []

        if not res_dat == '' and not isinstance(res_dat, dict):
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

            # interferogram and coherence calculations are special cases where slave and ifg are needed...
            if meta_type == 'ifg' and 'ifg_meta' in func_var_names:
                vars.append(self.res_dat[meta]['slave'])
                if res_dat:
                    res_dat[meta]['slave'] = self.res_dat[meta]['slave']
            else:
                vars.append(self.res_dat[meta][meta_type])
                if res_dat:
                    res_dat[meta][meta_type] = self.res_dat[meta][meta_type]

        # Coordinate of input and output
        if 'coordinates' in func_var_names:
            var_names.append('coordinates')
            vars.append(coor_out)
        if 'coor_in' in func_var_names:
            var_names.append('coor_in')
            vars.append(coor_in)

        # Step and file types
        if 'step' in func_var_names and len(step) > 0:
            var_names.append('step')
            vars.append(step)
        if 'file_type' in func_var_names and len(file_type) > 0:
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
            # For the concatenation step
            if 'meta_slices' in func_var_names and isinstance(meta_list, list):
                var_names.append('meta_slices')

                vars.append([self.res_dat[m][meta_type] for m in meta_list])
                if res_dat:
                    for m in meta_list:
                        res_dat[m][meta_type] = self.res_dat[m][meta_type]
            # Todo create ifg on the fly if needed.
            if 'ifg_meta' in func_var_names:
                var_names.append('ifg_meta')
                vars.append(self.res_dat[meta]['ifg'])
                if res_dat:
                    res_dat[meta]['ifg'] = self.res_dat[meta]['ifg']

        if isinstance(res_dat, dict):
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

        for package in parallel_package:
            for key in package['res_dat'].keys():
                for dat_type in package['res_dat'][key].keys():
                    if not isinstance(package['res_dat'][key][dat_type], ImageData):
                        print(dat_type + ' is missing for processing.')
                        return dat_type

        package_types = [package['proc_type'] for package in parallel_package]

        if not self.parallel or 'concatenate' in package_types:
            # If not parallel (for debugging purposes)
            res_dats = []
            for package in parallel_package:
                res_dat = run_parallel(package)

                # Update the .res files of this image
                if res_dat == False:
                    print('Encountered error in processing. Skipping this .res file.')
                    # sys.exit()
                else:
                    for key in res_dat.keys():
                        for meta_type in res_dat[key].keys():
                            self.res_dat[key][meta_type] = res_dat[key][meta_type]

        elif self.parallel:
            self.pool = Pool(self.cores, maxtasksperchild=1)

            for res_dat in self.pool.imap_unordered(run_parallel, parallel_package):
                # Update the .res files of this image
                if res_dat == False:
                    print('Encountered error in processing. Skipping this .res file.')
                    # sys.exit()
                else:
                    for key in res_dat.keys():
                        for meta_type in res_dat[key].keys():
                            self.res_dat[key][meta_type] = res_dat[key][meta_type]

            self.pool.close()
            self.pool = []

        self.clean_memmaps()

        #print('Output Image Data sizes:')
        #for key in self.res_dat.keys():
        #    for meta_type in self.res_dat[key].keys():
        #        print('Size of ' + key + ' of type ' + meta_type + ' is ' + str(
        #            self.res_dat[key][meta_type].get_size(self.res_dat[key][meta_type])) + ' bytes.')
