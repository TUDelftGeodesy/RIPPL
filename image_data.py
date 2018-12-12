import os
import numpy as np
import warnings
from collections import defaultdict
import sys
import inspect
import osr
import gdal
from gdalconst import *

from image_metadata import ImageMetadata
from coordinate_system import CoordinateSystem


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
        self.data_multilook = defaultdict()
        self.data_oversample = defaultdict()
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
    def image_read_disk(self, step, file_type=''):
        # This function initializes a dataset using a memmap or a simple numpy matrix. This can be used to store data
        # from a processing step that is performed in parallel blocks.

        if file_type == '':
            file_type = step

        if not self.check_datafile(step, file_type, exist=False):
            return

        self.add_data_step(step, file_type=file_type)
        self.data_disk[step][file_type] = np.memmap(self.data_files[step][file_type], mode='r+',
                                               dtype=self.dtype_disk[self.data_types[step][file_type]],
                                               shape=self.data_sizes[step][file_type])

    def images_create_disk(self, step, file_types='', coordinates='', coor_out=''):

        if len(file_types) == 0:
            meta_info = self.processes[step]
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            file_types = [filename[:-12] for filename in output_file_keys]

        if isinstance(file_types, str):
            file_types = [file_types]

        elif isinstance(coordinates, CoordinateSystem):
            file_types = [file_type + coordinates.sample for file_type in file_types]

            # Only used for the conversion grid
            if isinstance(coor_out, CoordinateSystem):
                file_types = [file_type + coor_out.sample for file_type in file_types]

        for file_type in file_types:
            self.image_create_disk(step, file_type)

    def image_create_disk(self, step, file_type='', overwrite=True):
        # With this function we add a certain data file to the image data structure. This is only used to make it
        # easily accessible, but the file is not saved to disk!
        # Metadata should be given to in this case before adding the file (to keep track of the location and size)

        if len(file_type) == 0:
            file_type = step

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

    def image_create_geotiff(self, step, file_type):
        # Save the file as an geotiff file. In case of a complex file a seperate amplitude and phase file are generated
        # as geotiff does not support complex data.

        # First check if file is available and read from disk if needed
        if not self.check_loaded(step, loc='disk', file_type=file_type, warn=False):
            if not self.read_data_memmap(step, file_type):
                return []

        # Get the coordinate system for this step and file type
        coordinates = self.read_res_coordinates(step, [file_type])[0]

        # Get geo transform and projection
        projection, geo_transform = coordinates.create_gdal_projection(self)

        # Create an empty geotiff with the right coordinate system
        gtiff_type = gdal.GDT_Float32
        driver = gdal.GetDriverByName('GTiff')

        # Save data to geotiff (if complex to two geotiff files)
        dat_type = self.data_types[step][file_type]
        file_name = os.path.join(os.path.dirname(self.res_path), self.processes[step][file_type + '_output_file'])

        if dat_type in ['complex_int', 'complex_short', 'complex_real4']:
            print('File converted to amplitude and phase image')

            if dat_type == 'complex_int':
                complex_data = self.data_disk[step][file_type].view(np.int16).astype('float32').view(np.complex64)
            elif dat_type == 'complex_short':
                complex_data = self.data_disk[step][file_type].view(np.float16).astype('float32').view(np.complex64)
            else:
                complex_data = self.data_disk[step][file_type]

            amp_data = driver.Create(file_name + '_amp.tiff', coordinates.shape[1], coordinates.shape[0], 1, gtiff_type,)
            amp_data.SetGeoTransform(tuple(geo_transform))
            amp_data.SetProjection(projection.ExportToWkt())
            amp_data.GetRasterBand(1).WriteArray(np.abs(complex_data).astype(np.float32))
            amp_data.FlushCache()  # Write to disk.
            del amp_data

            phase_data = driver.Create(file_name + '_phase.tiff', coordinates.shape[1], coordinates.shape[0], 1, gtiff_type)
            phase_data.SetGeoTransform((tuple(geo_transform)))
            phase_data.SetProjection(projection.ExportToWkt())
            phase_data.GetRasterBand(1).WriteArray(np.angle(complex_data).astype(np.float32))
            phase_data.FlushCache()  # Write to disk.

        else:   # All other cases.
            in_data = self.data_disk[step][file_type]

            phase_data = driver.Create(file_name + '.tiff', coordinates.shape[1], coordinates.shape[2], 1, gtiff_type)
            phase_data.SetGeoTransform((tuple(geo_transform)))
            phase_data.SetProjection(projection.ExportToWkt())
            phase_data.GetRasterBand(1).WriteArray(np.angle(in_data).astype(np.float32))
            phase_data.FlushCache()  # Write to disk.

        print('Saved ' + file_type + ' from ' + step + ' step of ' + os.path.dirname(self.res_path))


    def images_memory_to_disk(self, step, file_types='', coordinates='', coor_out=''):

        if len(file_types) == 0:
            meta_info = self.processes[step]
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            file_types = [filename[:-13] for filename in output_file_keys]

        if isinstance(file_types, str):
            file_types = [file_types]

        elif isinstance(coordinates, CoordinateSystem):
            file_types = [file_type + coordinates.sample for file_type in file_types]

            # Only used for the conversion grid
            if isinstance(coor_out, CoordinateSystem):
                file_types = [file_type + coor_out.sample for file_type in file_types]

        for file_type in file_types:
            self.image_memory_to_disk(step, file_type)

    def image_memory_to_disk(self, step, file_type=''):
        # This function is used to save a data crop to the image file. Normally this can be done directly but in case
        # the format is not a native numpy dtype we have to adjust.

        if file_type == '':
            file_type = step

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

    def image_load_data_memory(self, step, s_lin, s_pix, shape, file_type='', warn=True):
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

        if file_type == '':
            file_type = step

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
        return []

    def image_new_data_memory(self, in_data, step, s_lin, s_pix, file_type=''):
        # This function replaces the in memory data for the image. To do so we need the file together with the
        # first line and pixels of the data. If not defined we assume it starts at the first line and pixel.

        if file_type == '':
            file_type = step

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

    def image_subset_memory(self, step, s_lin, s_pix, shape, file_type=''):
        # This function subsets the data stored in memory.

        if file_type == '':
            file_type = step

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

    def image_disk_to_memory(self, step, s_lin, s_pix, shape, file_type=''):
        # This function loads a part of the information either from a memmap or standard numpy file. If needed the data
        # is also converted to a usable format (in case of complex int16 or complex float 16 datasets)

        if file_type == '':
            file_type = step

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

    def images_clean_memory(self, step, file_types='', coordinates='', coor_out=''):

        if not file_types:
            meta_info = self.processes[step]
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            file_types = [filename[:-13] for filename in output_file_keys]

        elif isinstance(coordinates, CoordinateSystem):
            file_types = [file_type + coordinates.sample for file_type in file_types]

            # Only used for the conversion grid
            if isinstance(coor_out, CoordinateSystem):
                file_types = [file_type + coor_out.sample for file_type in file_types]

        for file_type in file_types:
            self.clean_memory(step, file_type)

    def clean_memory(self, step='', file_type=''):
        # Clean the image from all data loaded in memory. Useful step before parallel processing.
        if not step:
            for step in self.data_memory.keys():
                self.data_memory[step] = defaultdict()
                self.data_memory_limits[step] = defaultdict()
                self.data_memory_sizes[step] = defaultdict()
        elif step not in list(self.data_memory.keys()):
            pass
        elif step and not file_type:
            self.data_memory[step] = defaultdict()
            self.data_memory_limits[step] = defaultdict()
            self.data_memory_sizes[step] = defaultdict()
        elif step and file_type:
            if file_type in list(self.data_memory[step].keys()):
                self.data_memory[step].pop(file_type)
                self.data_memory_limits[step].pop(file_type)
                self.data_memory_sizes[step].pop(file_type)

    def clean_memmap_files(self, step='', file_type=''):
        # Clean the image from all data loaded in memory. Useful step before parallel processing.
        if not step:
            for step in self.data_files.keys():
                for file_type in self.data_files[step].keys():
                    self.data_disk[step][file_type] = ''
        elif step and not file_type:
            for file_type in self.data_files[step].keys():
                self.data_disk[step][file_type] = ''
        elif step and file_type:
            self.data_disk[step][file_type] = ''

    # Next function is used to read all data files as memmaps from the slice (minimal memory use)
    def read_data(self):
        # This function reads all data in after reading the .res file
        for step in self.processes.keys():
            # First find the different datafiles per step

            file_types = [string[:-12] for string in self.processes[step].keys() if string.endswith('_output_file')]

            for file_type in file_types:
                self.add_data_step(step, file_type)

    def read_data_memmap(self, step='', file_type=''):

        if len(step) > 0:
            if isinstance(step, list):
                step = step
            else:
                step = [step]
        else:
            step = self.data_disk.keys()

        if isinstance(file_type, list):
            file_type = file_type
        elif len(file_type) == 0:
            file_type = []
        else:
            file_type = [file_type]

        steps = []
        file_types = []

        if len(step) == len(file_type):
            for s, f in zip(step, file_type):
                if s in self.data_disk.keys():
                    if f in self.data_disk[s].keys():
                        steps.append(s)
                        file_types.append(f)
        else:
            for s in step:
                if s in self.data_disk.keys():
                    for f in self.data_disk[s].keys():
                        steps.append(s)
                        file_types.append(f)

        succes = False
        for step, file_type in zip(steps, file_types):
            if self.check_datafile(step, file_type, exist=True, warn=False):
                # Finally read in as a memmap file
                self.data_disk[step][file_type] = np.memmap(self.data_files[step][file_type], mode='r+',
                                                       dtype=self.dtype_disk[self.data_types[step][file_type]],
                                                       shape=self.data_sizes[step][file_type])
                succes = True

            else:
                succes = False

        return succes

    # Next processing_steps is used to read info from certain steps in the metadata.
    def image_add_processing_step(self, step, step_dict):
        # This function adds a new processing step to the image.
        # Before actual processing of an image this should be done first.

        # First add to resfile
        self.insert(step_dict, step)

        file_types = [string[:-12] for string in self.processes[step].keys() if string.endswith('_output_file')]

        for file_type in file_types:
            self.add_data_step(step, file_type)

    def add_data_step(self, step, file_type=''):
        # Read the path, data size and data format from metadata.

        if file_type == '':
            file_type = step

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

            if file_type + '_pixels' in self.processes[step].keys():
                lin_min = int(self.processes[step][file_type + '_first_line'])
                pix_min = int(self.processes[step][file_type + '_first_pixel'])
                lines = int(self.processes[step][file_type + '_lines'])
                pixels = int(self.processes[step][file_type + '_pixels'])
            else:
                warnings.warn('No image size information found')
                return False

            if file_type + '_multilook_range' in self.processes[step].keys():
                pix_ml = int(self.processes[step][file_type + '_multilook_range'])
                lin_ml = int(self.processes[step][file_type + '_multilook_azimuth'])
            else:
                pix_ml = 1
                lin_ml = 1

            if file_type + '_offset_range' in self.processes[step].keys():
                pix_off = int(self.processes[step][file_type + '_offset_range'])
                lin_off = int(self.processes[step][file_type + '_offset_azimuth'])
            else:
                pix_off = 0
                lin_off = 0

            if file_type + '_offset_range' in self.processes[step].keys():
                pix_ovr = int(self.processes[step][file_type + '_oversample_range'])
                lin_ovr = int(self.processes[step][file_type + '_oversample_azimuth'])
            else:
                pix_ovr = 1
                lin_ovr = 1
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
            self.data_multilook[step] = defaultdict()
            self.data_oversample[step] = defaultdict()
            self.data_offset[step] = defaultdict()

            self.data_memory_sizes[step] = defaultdict()
            self.data_memory_limits[step] = defaultdict()
            self.data_memory[step] = defaultdict()
            self.data_memory_offset[step] = defaultdict()

        self.data_disk[step][file_type] = ''
        self.data_files[step][file_type] = os.path.join(self.folder, self.processes[step][file_type + '_output_file'])
        self.data_multilook[step][file_type] = (lin_ml, pix_ml)
        self.data_oversample[step][file_type] = (lin_ovr, pix_ovr)
        self.data_offset[step][file_type] = (lin_off, pix_off)
        self.data_types[step][file_type] = self.processes[step][type_string]
        self.data_sizes[step][file_type] = (lines, pixels)
        self.data_limits[step][file_type] = (lin_min, pix_min)

        return True

    def image_get_data_size(self, step, file_type, loc='disk'):
        # Returns the shape of the file on disk. If not found it returns a warning and [0, 0] size.

        if self.check_file_type_exist(step, file_type):
            if loc == 'disk':
                shape = np.array(self.data_sizes[step][file_type])
            elif loc == 'memory':
                shape = np.array(self.data_memory_sizes[step][file_type])
            else:
                print('variable loc should either be disk or memory')
                shape = [0, 0]
        else:
            shape = [0, 0]

        return shape

    # Following processing_steps are used to check whether a certain file exists, is loaded or covers a certain region
    def check_datafile(self, step, file_type='', exist=True, warn=True):
        # Function to check whether file exists or not. This assumes that all metadata is already read in.
        if not self.check_step_exist(step):
            return False

        if file_type == '':
            file_type = step

        data_string = file_type + '_output_file'
        if not data_string in self.processes[step].keys():
            return False

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
        if step not in list(self.processes.keys()):
            # warnings.warn('Step does not exist')
            return False

        return True

    def check_file_type_exist(self, step, file_type):
        # Check if this step even exists.
        if step not in self.processes.keys():
            warnings.warn('Step ' + step + ' does not exist')
            return False
        if file_type in self.processes[step].keys():
            warnings.warn('Data type' + file_type + ' in step ' + step + ' does not exist')

        return True

    def check_loaded(self, step, loc='disk', file_type='', warn=True):
        # Check if this datafile is loaded.
        if not self.check_step_exist(step):
            return False

        if file_type == '':
            file_type = step

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

    def check_coverage(self, step, s_lin, s_pix, shape, loc='disk', file_type='', warn=True):
        # Check if the coverage of our file is large enough for the requested area
        # Or check if the replacing file has the same size as defined in the metadata

        if file_type == '':
            file_type = step

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

    def get_size_res(self):
        size = ImageData.get_size(self)

        return size

    @staticmethod
    def get_size(obj, seen=None):
        """Recursively finds size of objects in bytes"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if hasattr(obj, '__dict__'):
            for cls in obj.__class__.__mro__:
                if '__dict__' in cls.__dict__:
                    d = cls.__dict__['__dict__']
                    if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                        size += ImageData.get_size(obj.__dict__, seen)
                    break
        if isinstance(obj, dict):
            size += sum((ImageData.get_size(v, seen) for v in obj.values()))
            size += sum((ImageData.get_size(k, seen) for k in obj.keys()))
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum((ImageData.get_size(i, seen) for i in obj))

        if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
            size += sum(ImageData.get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

        return size