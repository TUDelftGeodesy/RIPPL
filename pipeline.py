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
from processing_steps.processing_list import import_processing_list
import numpy as np


class RunPipeline():

    def __init__(self, function, use_functions=[], settings=[], memory=1000, cores=4,
                 cmaster=[], master=[], slave=[], ifg=[],
                 ml=[[5, 20]], off=[[0, 0]], ovr=[[1, 1]]):
        # Individual images are always defined as slaves.
        # Be sure you input the right image types. Otherwise the function will not run!
        # master/slave/ifg are full images, not just the slices. The lists are just multiples of these. Only functions
        # where lists are needed, we run with lists.

        # First load the available functions
        self.processes = import_processing_list()

        # The functions that have to be run
        self.function = function
        self.use_functions = use_functions

        # The specific settings for some of the functions. Sometimes we want to run with specific settings for one
        # of the intermediate or the final function.
        # This variable should be a dictionary, containing dictionaries for individual functions.
        self.settings = settings

        # Maximum memory per process in MB.
        self.memory = memory

        # Cores. The number of cores we use to run our processing.
        self.cores = cores

        # Load input meta data.
        self.cmaster = cmaster
        self.master = master
        self.slave = slave
        self.ifg = ifg

        # The order of processes and there accompanying functions.
        self.ordered_functions = []
        self.ordered_processes = []
        self.ordered_filename = []
        self.ordered_im_type = []
        self.ordered_ml = []
        self.ordered_ovr = []
        self.ordered_off = []

        # The multilooking, offset and oversampling for ifg/coherence/geocoding
        # Extra check whether we have sets of multilooking factors.
        if type(multilook[0]) == int:
            multilook = [multilook]
        if type(oversample[0]) == int:
            oversample = [oversample]
        if type(offset[0]) == int:
            offset = [offset]

        # Assign to class variable.
        self.multilooking = multilook
        self.oversample = oversample
        self.offset = offset

    def __call__(self):
        # Here we prepare and run the parallel processing for this image at once.

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

        for function in self.functions:

            if function in self.processes.keys():
                input_dat, output_dat, mem_use = self.processes[function].processing_info()
            else:
                print(function + ' does not exist!')
                return

            depend = []
            type_depend = input_dat.keys()
            for t in type_depend:
                for s in input_dat[t].keys():
                    in_dep[t][input_dat[t].keys()], out_dep[t][input_dat[t].keys()], mem_use = self.processes[s]
                    depend.append(t + ' ' + s)

            while depend:

                depend = []

                for d in depend:

                    if function in self.processes.keys():
                        input_dat, output_dat, mem_use = self.processes[function].processing_info()
                    else:
                        print(function + ' does not exist!')
                        return

                    type_depend = input_dat.keys()
                    for t in type_depend:
                        for s in input_dat[t].keys():
                            in_dep[t][input_dat[t].keys()], out_dep[t][input_dat[t].keys()], mem_use = self.processes[s]
                            depend.append(t + ' ' + s)

        # Now define the order of all these steps.

        # Create a list of all in/out dependencies.
        functions = []
        f_in_deps = []
        file_type_in_deps = []
        in_deps = []

        for m_t in in_dep.keys():
            for m_p in in_dep[m_t].keys():
                functions.append(m_t + '_' + m_p)
                for s_t in in_dep[m_t][m_p].keys():
                    for s_p in in_dep[m_t][m_p][s_t].keys():
                        for s_f in in_dep[m_t][m_p][s_t].keys():
                            file_type_in_deps.append(m_t + '_' + m_p)
                            f_in_deps.append(m_t + '_' + m_p)
                            in_deps.append(s_t + '_' + s_p)

        f_in_deps = np.array(f_in_deps)
        in_deps = np.array(in_deps)

        self.ordered_functions = []
        self.ordered_processes = []
        self.ordered_im_type = []

        while len(functions) > 0:

            for f in functions:
                if f not in set(f_in_deps):
                    break

            if f not in functions:
                print('Error while defining order of functions at function ' + f)

            f_in_deps = f_in_deps[in_deps != f]
            in_deps = in_deps[in_deps != f]

            functions.remove(f)
            self.ordered_functions.append(f)
            self.ordered_processes.append(f.split('_', 1)[1])
            self.ordered_im_type.append(f.split('_', 1)[0])

        self.ordered_functions = np.array(self.ordered_functions)
        self.ordered_processes = np.array(self.ordered_processes)
        self.ordered_im_type = np.array(self.ordered_im_type)

    def check_existing_input(self):

        # Loop over the different slices of master/slave/master_coreg/ifg
        images = [self.master, self.slave, self.cmaster, self.ifg]
        image_strs = ['master', 'slave', 'cmaster', 'ifg']

        for im, im_str in zip(images, image_strs):

            proc_steps = self.ordered_functions[self.ordered_im_type == im_str]

            for process, filename in zip(self.ordered_processes, self.ordered_filenames):
                # Check if functions is already performed

                # Find all the existing datasets that are generated for these functions.

                # Find the coordinates for the different images:
                # - slices within the full image
                # - coordinates of the multilooked full image for different multilooking factors.
                # - coordinates of the coarse height map for interpolation.

                # Construct the file we need for that image based on the input requirements.

                # Check which files already exist and how that can satisfy the input demand of certain functions.

                # Now remove the steps that are already done for this slice.
                # 1 Remove the slice output files for this one step.
                # 2 Iteratively remove the other steps that are only needed for the respective step.
                #   This can be done using more or less the same method as in the define_function_order.

        # Resulting are all the steps that still have to be done for individual slices.

    def prepare_processing(self):
        # This function generates a list of functions and inputs to do the processing.

        # 1. Check which steps need image wide before they can be run.
        # 2. Based on these, create pipelines for individual bursts which are as long as possible.
        # 3. Provide the function and the function inputs for every individual slice and step.

        # This results in a list of sequences for this image.

        # For every slice or full image seperate functions are created.

            # Loop over all slices
            # - add to init (if needed different outputs)
            # - add to meta_info

            # Find the number of parts the image has to be divided given the memory settings.
            # - get memory use for all steps (based on 32 bit variables)
            # - add the output of the different steps before them (make distinction between 32 and 64 bit)
            # - divide total memory use by the max data use per line to get the number of lines by step.
            #   (the maximum number of lines is the total line no divided by number of cores)

            # Now create functions and variables for functions.
                # Loop over functions in pipeline
                    # Loop over different line starts
                    # - assign function
                    # - read possible extra settings and add variables
                    # - add meta data information (should be hard copied! Make sure there is minimal memory use for this
                    #   meta data object, which is normally prevented by cleaning memory data storage.)
                    # - add the start line and number of lines.

        # This should result in a prepared set of functions and variables for parallel processing.


    @staticmethod
    def init_parallel(function_vars):

        # First split the functions and variables
        functions = function_vars[0]
        output_names = function_vars[1]

        # Now init these step
        for function, output_files in zip(functions, output_names):
            # Create the initial output files. If specified the output_files are used as output, but if left empty
            # the functions will also work. (Make sure any new function does so!)
            function.create_output_files(output_files)

    @staticmethod
    def run_parallel(function_vars):

        # First split the functions and variables
        functions = function_vars[0]
        variable_names = function_vars[1]
        variables = function_vars[2]

        if len(variable_names) != len(variables):
            print('The length of the variables and variable names should be the same!')

        # Now init these step
        for function, var, var_names, n in zip(functions, variables, variable_names, range(len(variables))):
            # Because the number of variables can vary we use the eval functions.
            func_str = [variable_names[n][i] + '=variables[' + str(n) + '][' + str(i) + ']' for i in range(len(variables[n]))]
            eval_str = 'proc_func = function(' + ','.join(func_str) + ')'
            eval(eval_str)

            # Run the function
            proc_func()

    @staticmethod
    def meta_info_parallel(meta_data):

        # Now init these step
        for meta in meta_data:
            # The only function of this step is to write meta data information to file and clean the meta data function
            # from data stored in memory.

            # Write meta data to .res file. (This is the moment that
            meta.write()
            meta.clean_memory()
