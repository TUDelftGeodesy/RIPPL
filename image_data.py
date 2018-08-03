import os
import numpy as np
import warnings
from collections import defaultdict

from image_metadata import ImageMetadata


class ImageData(ImageMetadata):

    def __init__(self, filename, res_type, warn=False):

        ImageMetadata.__init__(self, filename, res_type, warn=False)

        self.folder = os.path.dirname(filename)

        # Define the storage variable
        self.file_paths = dict()
        # Every step can have multiple output files.
        # For every file on disk there is also a in memory file. These are only loaded when needed for processing.
        # These should be given as ...._output_file with ...._output_format
        self.data_disk = defaultdict()
        self.data_memory = defaultdict()

        self.data_files = defaultdict()
        self.data_types = defaultdict()

        # [lines, pixels]
        self.data_sizes = defaultdict()
        self.data_intervals = defaultdict()
        self.data_offset = defaultdict()
        # Limits are [first_line, first_pix]
        self.data_limits = defaultdict()

        # If this class is used to load only a part of the dataset (mainly for parallel processing)
        # The buffer is used
        self.data_memory_sizes = defaultdict()
        self.data_memory_intervals = defaultdict()
        self.data_memory_offset = defaultdict()
        self.data_memory_limits = defaultdict()

        # Possible data types
        self.dtype_disk = {'complex_int': np.dtype([('re', np.int16), ('im', np.int16)]),
                           'complex_short': np.dtype([('re', np.float16), ('im', np.float16)]),
                           'complex_real4': np.complex64,
                           'real8': np.float64,
                           'real4': np.float32,
                           'real2': np.float16,
                           'int8': np.int8,
                           'int16': np.int16,
                           'int32': np.int32,
                           'int64': np.int64,
                           'tiff': np.dtype([('re', np.int16), ('im', np.int16)])}  # source file format (sentinel)
        self.dtype_numpy = {'complex_int': np.complex64,
                            'complex_short': np.complex64,
                            'complex_real4': np.complex64,
                            'real8': np.float64,
                            'real4': np.float32,
                            'real2': np.float16,
                            'int8': np.int8,
                            'int16': np.int16,
                            'int32': np.int32,
                            'int64': np.int64,
                            'tiff': np.complex64}
        self.read_data()

    # Next processing_steps are used to subtract, add, read or write from this slice
    def image_read_disk(self, step, file_type='Data'):
        # This function initializes a dataset using a memmap or a simple numpy matrix. This can be used to store data
        # from a processing step that is performed in parallel blocks.

        if not self.check_datafile(step, file_type, exist=False):
            return

        self.add_data_step(step, file_type=file_type)
        self.data_disk[step][file_type] = np.memmap(self.data_files[step][file_type], mode='r+',
                                               dtype=self.dtype_disk[self.data_types[step][file_type]],
                                               shape=self.data_sizes[step][file_type])

    def image_create_disk(self, step, file_type='Data', overwrite=True):
        # With this function we add a certain data file to the image data structure. This is only used to make it
        # easily accessible, but the file is not saved to disk!
        # Metadata should be given to in this case before adding the file (to keep track of the location and size)

        path = os.path.join(os.path.dirname(self.res_path), self.data_files[step][file_type])

        if not self.check_datafile(step, file_type, exist=False, warn=False):
            if not overwrite or not self.check_datafile(step, file_type, exist=True):
                return
            else:
                print(path + ' already exists. It will be overwritten')

        self.add_data_step(step, file_type=file_type)
        self.data_disk[step][file_type] = np.memmap(path, mode='w+',
                                               dtype=self.dtype_disk[self.data_types[step][file_type]],
                                               shape=self.data_sizes[step][file_type])

    def image_memory_to_disk(self, step, file_type='Data'):
        # This function is used to save a data crop to the image file. Normally this can be done directly but in case
        # the format is not a native numpy dtype we have to adjust.

        if not self.check_datafile(step, file_type, exist=True, warn=False):
            return
        else:
            # If the file exists but is not read as a memmap, read it in as a memmap.
            if len(self.data_disk[step][file_type]) == 0:
                self.read_data_memmap(step, file_type)
            else:
                return

        if not self.check_loaded(step, file_type=file_type, warn=False):
            self.read_data()

        s_pix = self.data_memory_limits[step][file_type][1]
        e_pix = s_pix + self.data_memory_sizes[step][file_type][1]
        s_lin = self.data_memory_limits[step][file_type][0]
        e_lin = s_lin + self.data_memory_sizes[step][file_type][0]

        cpx_int = self.dtype_disk['complex_int']
        cpx_flt = self.dtype_disk['complex_short']
        dat_type = self.data_types[step][file_type]

        if dat_type == 'complex_int':
            self.data_disk[step][file_type][s_lin:e_lin, s_pix:e_pix] = \
                self.data_memory[step][file_type].view(np.float32).astype(np.int16).view(cpx_int)
        elif dat_type == 'complex_short':
            self.data_disk[step][file_type][s_lin:e_lin, s_pix:e_pix] = \
                self.data_memory[step][file_type].view(np.float32).astype(np.int16).view(cpx_flt)
        else:
            self.data_disk[step][file_type][s_lin:e_lin, s_pix:e_pix] = \
                self.data_memory[step][file_type]

        # Flush to disk
        self.data_disk[step][file_type].flush()

    def image_load_data_memory(self, step, s_lin, s_pix, shape, file_type='Data', warn=True):
        # This function loads data in memory using the most efficient way.
        #
        # First we check whether this step exists in the datafile
        # - Yes > goto 1
        # - No > Failure, return false
        #
        # - 1. Then it will check whether the data is already loaded in memory
        #   - 1.1 Yes, we will check the coverage.
        #       - 1.1.1 The same, we are done. > Succes, return data
        #       - 1.1.2 If the original data includes the requested data, the dataset is subsetted > Succes, return data
        #       - 1.1.3 If the original data does not include the requested data. > goto 2
        #   - 1.2 No, > goto 2
        #
        #   2. Loading from disk
        #   - 2. Check if step and datafile exists on disk
        #       - 2.1. Yes, we check if this file covers the requested region
        #           - 2.1.1 Yes, load data from disk to memory > Succes, return data
        #           - 2.2.2 No > Failure, return False
        #       - 2.2. No > Failure, return False

        if not self.check_step_exist(step):
            return []

        if self.check_loaded(step, loc='memory', file_type=file_type, warn=False):

            if self.check_coverage(step, s_lin, s_pix, shape, loc='memory', file_type=file_type, warn=False):
                if not list(shape) == list(self.data_memory_sizes[step][file_type]):
                    self.image_subset_memory(step, s_lin, s_pix, shape, file_type=file_type)

                return self.data_memory[step][file_type]

        if not self.check_loaded(step, loc='disk', file_type=file_type, warn=False):
            if not self.read_data_memmap(step, file_type):
                return []

        if self.check_loaded(step, loc='disk', file_type=file_type):
            if self.check_coverage(step, s_lin, s_pix, shape, loc='disk', file_type=file_type):
                self.image_disk_to_memory(step, s_lin, s_pix, shape, file_type=file_type)

                return self.data_memory[step][file_type]

        # If we did not manage to load the data either from memory or disk return False
        if warn:
            print('Failed to load data for ' + step + ' file ' + file_type + ' for image ' + self.folder)
        return False

    def image_new_data_memory(self, in_data, step, s_lin, s_pix, file_type='Data'):
        # This function replaces the in memory data for the image. To do so we need the file together with the
        # first line and pixels of the data. If not defined we assume it starts at the first line and pixel.

        if not self.check_step_exist(step):
            return

        if not self.check_coverage(step, s_lin, s_pix, in_data.shape, loc='disk', file_type=file_type):
            return

        if not in_data.dtype == self.dtype_numpy[self.data_types[step][file_type]]:
            warnings.warn('Data types of input and output dataset are not the same')
            return

        self.data_memory[step][file_type] = in_data
        self.data_memory_limits[step][file_type] = (s_lin, s_pix)
        self.data_memory_sizes[step][file_type] = in_data.shape

    def image_subset_memory(self, step, s_lin, s_pix, shape, file_type='Data'):
        # This function subsets the data stored in memory.

        if not self.check_coverage(step, s_lin, s_pix, shape, 'memory', file_type):
            return False

        # Now check whether it is covered by the memory file.
        old_s_lin = self.data_memory_limits[step][file_type][0]
        old_s_pix = self.data_memory_limits[step][file_type][1]

        self.data_memory[step][file_type] = \
            self.data_memory[step][file_type][s_lin - old_s_lin: s_lin - old_s_lin + shape[0],
                                              s_pix - old_s_pix: s_pix - old_s_pix + shape[1]]
        self.data_memory_limits[step][file_type] = [s_lin, s_pix]
        self.data_memory_sizes[step][file_type] = shape
        return True

    def image_disk_to_memory(self, step, s_lin, s_pix, shape, file_type='Data'):
        # This function loads a part of the information either from a memmap or standard numpy file. If needed the data
        # is also converted to a usable format (in case of complex int16 or complex float 16 datasets)

        if not self.check_datafile(step, file_type, exist=True):
            return False

        if not self.check_loaded(step, loc='disk', file_type=file_type):
            if not self.read_data_memmap(step, file_type):
                return []

        if not self.check_coverage(step, s_lin, s_pix, shape, 'disk', file_type):
            return False

        # Update shape sizes
        self.data_memory_limits[step][file_type] = [s_lin, s_pix]
        self.data_memory_sizes[step][file_type] = shape
        e_lin = s_lin + shape[0]
        e_pix = s_pix + shape[1]

        dat_type = self.data_types[step][file_type]

        if dat_type == 'complex_int':
            self.data_memory[step][file_type] = self.data_disk[step][file_type]\
                .view(np.int16).astype('float32', subok=False).view(np.complex64)[s_lin:e_lin, s_pix:e_pix]
        elif dat_type == 'complex_short':
            self.data_memory[step][file_type] = self.data_disk[step][file_type]\
                .view(np.float16).astype('float32', subok=False).view(np.complex64)[s_lin:e_lin, s_pix:e_pix]
        else:
            self.data_memory[step][file_type] = self.data_disk[step][file_type][s_lin:e_lin, s_pix:e_pix]

        return True

    def clean_memory(self, step='', file_type=''):
        # Clean the image from all data loaded in memory. Useful step before parallel processing.
        if not step:
            for step in self.data_memory.keys():
                self.data_memory[step] = defaultdict()
                self.data_memory_limits[step] = defaultdict()
                self.data_memory_sizes[step] = defaultdict()
        elif step and not file_type:
            self.data_memory[step] = defaultdict()
            self.data_memory_limits[step] = defaultdict()
            self.data_memory_sizes[step] = defaultdict()
        elif step and file_type:
            self.data_memory[step].pop(file_type)
            self.data_memory_limits[step].pop(file_type)
            self.data_memory_sizes[step].pop(file_type)

    def clean_memmap_files(self, step='', file_type=''):
        # Clean the image from all data loaded in memory. Useful step before parallel processing.
        if not step:
            for step in self.data_files.keys():
                self.data_files[step] = defaultdict()
        elif step and not file_type:
            self.data_files[step] = defaultdict()
        elif step and file_type:
            self.data_files[step].pop(file_type)

    # Next function is used to read all data files as memmaps from the slice (minimal memory use)
    def read_data(self):
        # This function reads all data in after reading the .res file
        for step in self.processes.keys():
            # First find the different datafiles per step

            file_types = [string[:-12] for string in self.processes[step].keys() if string.endswith('_output_file')]

            for file_type in file_types:
                self.add_data_step(step, file_type)

    def read_data_memmap(self, step, file_type):

        if self.check_datafile(step, file_type, exist=True, warn=False):
            # Finally read in as a memmap file
            dat = open(self.data_files[step][file_type], 'r+')
            self.data_disk[step][file_type] = np.memmap(dat,
                                                   dtype=self.dtype_disk[self.data_types[step][file_type]],
                                                   shape=self.data_sizes[step][file_type])
            return True

        else:
            return False

    # Next processing_steps is used to read info from certain steps in the metadata.
    def image_add_processing_step(self, step, step_dict):
        # This function adds a new processing step to the image.
        # Before actual processing of an image this should be done first.

        # First add to resfile
        self.insert(step_dict, step)

        file_types = [string[:-12] for string in self.processes[step].keys() if string.endswith('_output_file')]

        for file_type in file_types:
            self.add_data_step(step, file_type)

    def add_data_step(self, step, file_type='Data'):
        # Read the path, data size and data format from metadata.

        if step == 'readfiles':
            data_string = file_type + 'file'
            type_string = file_type + 'format'
        else:
            data_string = file_type + '_output_file'
            type_string = file_type + '_output_format'

        # Check if this step even exists. And whether it has an outputfile
        if step not in self.processes.keys():
            print('Step does not exist')
            return False
        if any(x not in self.processes[step].keys() for x in [data_string, type_string]):
            print('No datafile defined or no data format specified')
            return False
        # Now check whether there is a useful data format
        if self.processes[step][type_string] not in self.dtype_numpy.keys():
            print('This data format does not exist')
            return False

        if step == 'readfiles':
            step = 'crop'

        if file_type + '_output_file' in self.processes[step]:
            keys = self.processes[step]

            if file_type + '_first_pixel' in self.processes[step].keys():
                lin_min = int(self.processes[step][file_type + '_first_line'])
                pix_min = int(self.processes[step][file_type + '_first_pixel'])
                lines = int(self.processes[step][file_type + '_lines'])
                pixels = int(self.processes[step][file_type + '_pixels'])
            elif file_type + '_size_in_longitude' in keys:
                pix_min = 1
                pixels = int(self.processes[step][file_type + '_size_in_longitude'])
                lin_min = 1
                lines = int(self.processes[step][file_type + '_size_in_latitude'])
            else:
                warnings.warn('No image size information found')
                return False

            if file_type + '_interval_range' in self.processes[step].keys():
                pix_int = int(self.processes[step][file_type + '_interval_range'])
                lin_int = int(self.processes[step][file_type + '_interval_azimuth'])
            elif file_type + '_multilook_range' in self.processes[step].keys():
                pix_int = int(self.processes[step][file_type + '_multilook_range'])
                lin_int = int(self.processes[step][file_type + '_multilook_azimuth'])
            else:
                pix_int = 1
                lin_int = 1
            if file_type + '_buffer_range' in self.processes[step].keys():
                pix_off = int(self.processes[step][file_type + '_buffer_range'])
                lin_off = int(self.processes[step][file_type + '_buffer_azimuth'])
            elif file_type + '_offset_range' in self.processes[step].keys():
                pix_off = int(self.processes[step][file_type + '_offset_range'])
                lin_off = int(self.processes[step][file_type + '_offset_azimuth'])
            else:
                pix_off = 0
                lin_off = 0
        else:
            warnings.warn('No image size information found')
            return False

        # If we passed the checks we can add the needed information
        if step not in self.data_types.keys():
            self.data_types[step] = defaultdict()
            self.data_sizes[step] = defaultdict()
            self.data_limits[step] = defaultdict()
            self.data_disk[step] = defaultdict()
            self.data_files[step] = defaultdict()
            self.data_intervals[step] = defaultdict()
            self.data_offset[step] = defaultdict()

            self.data_memory_sizes[step] = defaultdict()
            self.data_memory_limits[step] = defaultdict()
            self.data_memory[step] = defaultdict()
            self.data_memory_intervals[step] = defaultdict()
            self.data_memory_offset[step] = defaultdict()

        self.data_disk[step][file_type] = ''
        self.data_files[step][file_type] = os.path.join(self.folder, self.processes[step][file_type + '_output_file'])
        self.data_intervals[step][file_type] = (lin_int, pix_int)
        self.data_offset[step][file_type] = (lin_off, pix_off)
        self.data_types[step][file_type] = self.processes[step][type_string]
        self.data_sizes[step][file_type] = (lines, pixels)
        self.data_limits[step][file_type] = (lin_min, pix_min)

        return True

    # Following processing_steps are used to check whether a certain file exists, is loaded or covers a certain region
    def check_datafile(self, step, file_type='Data', exist=True, warn=True):
        # Function to check whether file exists or not. This assumes that all metadata is already read in.
        if not self.check_step_exist(step):
            return False

        data_string = file_type + '_output_file'

        if os.path.dirname(self.processes[step][data_string]) == '':
            self.data_files[step][file_type] = os.path.join(self.folder, self.processes[step][data_string])
        else:
            self.data_files[step][file_type] = self.processes[step][data_string]

        if not exist and exist != os.path.exists(self.data_files[step][file_type]):
            if warn:
                warnings.warn(self.data_files[step][file_type] + ' already exists')
            return False
        if exist and exist != os.path.exists(self.data_files[step][file_type]):
            if warn:
                warnings.warn(self.data_files[step][file_type] + ' does not exist')
            return False

        # All tests are passed.
        return True

    def check_step_exist(self, step):
        # Check if this step even exists.
        if step not in self.processes.keys():
            warnings.warn('Step does not exist')
            return False

        return True

    def check_loaded(self, step, loc='disk', file_type='Data', warn=True):
        # Check if this datafile is loaded.
        if not self.check_step_exist(step):
            return False

        if loc == 'disk':
            if file_type not in self.data_disk[step].keys():
                if warn:
                    warnings.warn('No datafile defined')
                return False
            elif len(self.data_disk[step][file_type]) == 0:
                if warn:
                    warnings.warn('Data file is empty')
                return False

        if loc == 'memory':
            if file_type not in self.data_memory[step].keys():
                if warn:
                    warnings.warn('No datafile defined')
                return False
            elif len(self.data_memory[step][file_type]) == 0:
                if warn:
                    warnings.warn('Data file is empty')
                return False

        return True

    def check_coverage(self, step, s_lin, s_pix, shape, loc='disk', file_type='Data', warn=True):
        # Check if the coverage of our file is large enough for the requested area
        # Or check if the replacing file has the same size as defined in the metadata

        e_lin = s_lin + shape[0]
        e_pix = s_pix + shape[1]

        if loc == 'disk':
            if file_type not in self.data_sizes[step].keys():
                if warn:
                    warnings.warn('No information on image size available!')
                return False
            if s_pix >= 0 and \
                    self.data_sizes[step][file_type][0] >= e_lin and \
                    s_lin >= 0 and \
                    self.data_sizes[step][file_type][1] >= e_pix:
                return True
            else:
                if warn:
                    warnings.warn('Area not covered by dataset')
                return False

        elif loc == 'memory':
            if file_type not in self.data_sizes[step].keys():
                if warn:
                    warnings.warn('No information on image size available!')
                return False

            mem_s_lin = self.data_memory_limits[step][file_type][0]
            mem_s_pix = self.data_memory_limits[step][file_type][1]

            if s_pix >= mem_s_pix and \
                            self.data_memory_sizes[step][file_type][0] + mem_s_lin >= e_lin and \
                            s_lin >= mem_s_lin and \
                            self.data_memory_sizes[step][file_type][1] + mem_s_pix >= e_pix:
                return True
            else:
                if warn:
                    warnings.warn('Area not covered by dataset')
                return False
