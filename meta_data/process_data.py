"""
This function creates an interface between the processes and the data on disk. The function does:
- create new data files in memory and on disk
- write data from memory to disk
- load data in memory
- check the coverage of input/output datasets
- convert data types to data efficient datasets on disk (complex data)

"""
import numpy as np
from collections import OrderedDict
import os
import copy
import gdal

from rippl.meta_data.process_meta import ProcessMeta


class ProcessData():

    def __init__(self, process='', meta_file_name='', coordinates=[], process_name=[], settings=[], output_files=[],
                 input_files='', polarisation='', data_id='', json_data='', json_path=''):

        # If the the process metadata is already given we add that file as metadata
        if isinstance(process, ProcessMeta):
            self.meta = process
        else:
            self.meta = ProcessMeta(meta_file_name, coordinates, process_name, settings, output_files, input_files,
                                polarisation, data_id, json_data)

        self.dtype_disk, self.dtype_memory, self.dtype_size, self.dtype_gdal, self.dtype_gdal_numpy = self.load_dtypes()

        self.data_disk_meta = self.meta.output_files
        self.data_disk = OrderedDict()
        self.data_memory_meta = OrderedDict()
        self.data_memory = OrderedDict()

        self.coordinates = self.meta.coordinates
        self.process_name = self.meta.process_name
        self.process_id = self.meta.process_id

        # add settings to memory and disk data.
        self.sync_memory_disk_files()

    # Next functions are for writing/creating files and exchange between disk and memory

    '''
    Functions to communicate with data on disk
    - create_disk_data > create a new file on disk
    - load_disk_data > load disk data into a memmap file. To use data in a function data should be loaded to memory too
    - remove_disk_data_memmap > removes the memmap file of the file on disk (file on disk will stay there)
    - remove_disk_data > removes the data file from disk
    '''
    def create_disk_data(self, process_type):
        # Check if the file exists and whether there is a valid data file already.
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False
        if self.check_data_disk_valid(file_type)[1]:
            return False

        meta = self.data_disk_meta[file_type]
        self.data_disk[file_type] = np.memmap(meta['file_path'], mode='w+',ndtype=self.dtype_disk[meta['data_type']],
                                              shape=meta['shape'])
        return True

    def load_disk_data(self, process_type):
        # Check if file exists and whether it is valid
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False
        if not self.check_data_disk_valid(file_type)[1]:
            return False

        meta = self.data_disk_meta[file_type]
        self.data_disk[file_type] = np.memmap(meta['file_path'], mode='r+', ndtype=self.dtype_disk[meta['data_type']],
                                              shape=meta['shape'])
        return True

    def remove_disk_data_memmap(self, process_type):
        # Check if it exists
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False
        if isinstance(self.data_disk[file_type], np.memmap):
            self.data_disk[file_type] = []

        return True

    def remove_disk_data(self, process_type):
        # Check if it exists
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False
        # If file exists
        if self.check_data_disk_valid()[0]:
            os.remove(self.data_disk_meta[file_type]['file_path'])

        return True

    '''
    Functions needed to load or remove data in memory
        - new_memory_data > Create a new empty dataset in memory
        - add_memory_data > Add new memory data with a pre-existing numpy array

    '''
    def new_memory_data(self, process_type, shape, s_lin=0, s_pix=0):
        # Create a new memory dataset
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False

        self.data_memory[file_type] = np.zeros(shape).astype(self.dtype_memory[self.data_disk_meta[file_type]['data_type']])

        # Updata meta data
        self.data_memory_meta[file_type]['s_lin'] = s_lin
        self.data_memory_meta[file_type]['s_pix'] = s_pix
        self.data_memory_meta[file_type]['shape'] = shape

    def add_memory_data(self, process_type, data, s_lin=0, s_pix=0):
        # Find the correct data type
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False

        # Check if data formats are correct
        dtype = self.dtype_memory[self.data_disk_meta[file_type]['data_type']]
        if not data.dtype == dtype:
            print('Input data type is not correct. It should be ' + str(dtype))
        self.data_memory[file_type] = data

        # Updata meta data
        self.data_memory_meta[file_type]['s_lin'] = s_lin
        self.data_memory_meta[file_type]['s_pix'] = s_pix
        self.data_memory_meta[file_type]['shape'] = data.shape

    def load_memory_data(self, process_type, shape=[], s_lin=0, s_pix=0):
        """
        This function uses the following steps:
        1. Check if this file type exists > if not return error
        2. Checks if coverage of original file is fine > if not return error
        3. Checks if there is already data loaded in memory. If so:
            3.1. Checks if data is already available from memory in correct shape. -> nothing needed to do
            3.2. Checks if data can be subsetted from current data in memory. -> subset data
        4. Checks if data is loaded in memory and available on disk. > if not return error
            4.1. Load data as a memmap if not done already
            4.2. Load needed data to memory

        :return:
        """

        # Check if it exists
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False

        # Check if coverage is the same otherwise we have to load new data
        if shape == self.data_memory_meta[process_type]['shape'] and \
            s_lin == self.data_memory_meta[process_type]['s_lin'] and \
            s_pix == self.data_memory_meta[process_type]['s_pix']:
            return True
        # Try to subset
        if self.subset_memory_data(process_type, shape=shape, s_lin=s_lin, s_pix=s_pix):
            return True
        # Try to load from disk. As a last step.
        if self.load_memory_data_from_disk(process_type, shape=shape, s_lin=s_lin, s_pix=s_pix):
            return True
        else:
            return False

    def subset_memory_data(self, process_type, shape, s_lin=0, s_pix=0):
        # Check if subsetting is possible
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False

        # Check if the datasets are overlapping
        if not self.check_overlapping(file_type, data_type='memory', shape=shape, s_lin=s_lin, s_pix=s_pix):
            return False

        dat_s_lin = self.data_memory_meta[file_type]['s_lin'] - s_lin
        dat_s_pix = self.data_memory_meta[file_type]['s_pix'] - s_pix

        self.data_memory[file_type] = self.data_memory[file_type][dat_s_lin:dat_s_lin + shape[0],
                                                                  dat_s_pix:dat_s_pix + shape[1]]

        # Updata meta data
        self.data_memory_meta[file_type]['s_lin'] = s_lin
        self.data_memory_meta[file_type]['s_pix'] = s_pix
        self.data_memory_meta[file_type]['shape'] = shape

        return True

    def load_memory_data_from_disk(self, process_type, shape, s_lin=0, s_pix=0):
        # Load data from data on disk. This can be slice used for multiprocessing.
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False

        # Check if the datasets are overlapping
        if not self.check_overlapping(file_type, data_type='data', shape=shape, s_lin=s_lin, s_pix=s_pix):
            return False

        # Get the data from disk
        disk_data = self.disk2memory(self.data_disk[file_type], self.data_disk_meta[file_type]['data_type'])
        self.data_memory[file_type] = np.copy(disk_data[s_lin:s_lin + shape[0], s_pix:s_pix + shape[1]])

        # Updata meta data
        self.data_memory_meta[file_type]['s_lin'] = s_lin
        self.data_memory_meta[file_type]['s_pix'] = s_pix
        self.data_memory_meta[file_type]['shape'] = shape

    def save_memory_data_to_disk(self, process_type):
        # Save data to disk from memory
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False

        # Check if a memory file is loaded
        if not self.check_memory_file(file_type):
            return False

        # Find settings
        s_lin = self.data_memory_meta[file_type]['s_lin']
        s_pix = self.data_memory_meta[file_type]['s_pix']
        shape = self.data_memory_meta[file_type]['shape']

        # Check if the datasets are overlapping
        if not self.check_overlapping(file_type, data_type='data', shape=shape, s_lin=s_lin, s_pix=s_pix):
            return False

        # Check if memmap is loaded.
        if not self.check_disk_file(file_type):
            # If not loaded, a new data file is created. (In this way we can create new datasets, but better initialize
            # beforehand, otherwise it can create problems with parallel processing.)
            if not self.load_disk_data(process_type):
                return False

        # Save data to disk
        memory_data = self.memory2disk(self.data_memory[file_type], self.data_disk_meta[file_type]['data_type'])
        self.data_disk[file_type][s_lin:s_lin + shape[0], s_pix:s_pix + shape[1]] = memory_data

    def remove_memory_data(self, process_type):
        # Remove data if it is loaded
        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False

        if isinstance(self.data_memory[file_type], np.ndarray):
            self.data_memory[file_type] = []

    # Helper functions for reading writing functions
    def check_file_type_exists(self, process_type):

        if process_type in self.data_disk_meta.keys():
            file_type = process_type
        else:
            file_type = self.meta.get_filename(process_type)

        if not file_type in self.data_disk_meta.keys():
            print(file_type + ' does not exist for this processing step.')
            return False
        return file_type

    def check_memory_file(self, file_type):
        # Check if there is a memory file.
        if isinstance(self.data_memory[file_type], np.ndarray):
            return True
        else:
            return False

    def check_disk_file(self, file_type):
        # Check if there is a memory file.
        if isinstance(self.data_disk[file_type], np.memmap):
            return True
        else:
            return False

    def check_data_disk_valid(self, file_type):
        # Function checks whether the datafile exists and whether is has the correct size.
        # We assume 2D files here...
        # Check if file exists
        if not self.check_file_type_exists(file_type):
            return False, False

        exist = os.path.exists(self.data_disk_meta[file_type]['file_path'])
        self.data_disk_meta[file_type]['file_exist'] = exist

        if exist:
            shape = self.data_disk_meta[file_type]['shape']
            file_dat_size = self.dtype_size[self.data_disk_meta[file_type]['data_type']] * shape[0] * shape[1]
            if int(os.path.getsize(self.data_disk_meta[file_type]['file_path'])) == file_dat_size:
                valid_size = True
            else:
                valid_size = False
            self.data_disk_meta[file_type]['file_valid'] = valid_size

        return exist, valid_size

    def check_overlapping(self, file_type, data_type='memory', shape=[0, 0], s_lin=0, s_pix=0):
        # Check if the regions we want to select are within the boundary of the original file.
        if data_type == 'memory':
            orig_shape = self.data_memory_meta[file_type]['shape']
            orig_s_lin = self.data_memory_meta[file_type]['s_lin']
            orig_s_pix = self.data_memory_meta[file_type]['s_pix']
        else:       # If it is a data file on disk.
            orig_shape = self.data_disk_meta[file_type]['shape']
            orig_s_lin = 0
            orig_s_pix = 0

        # Check s_lin/s_pix
        if (s_lin < orig_s_lin or s_lin + shape[0] > orig_s_lin + orig_shape[0]) or \
                (s_pix < orig_s_pix or s_pix + shape[1] > orig_s_pix + orig_shape[1]):
            return False
        else:
            return True

    # Methods to add new filenames and or remove ones.
    def create_processing_filenames(self, process_types=[], data_types=[], shapes=[]):
        # Delegate to meta data method
        self.data_disk_meta = self.meta.create_processing_filenames(self.data_disk_meta, process_types, data_types, shapes)

    def sync_memory_disk_files(self):
        # This function sets up the structure for both the data and metadata structures from the original

        memory_meta_dummy = OrderedDict([('exist', None), ('shape', [0, 0]), ('s_lin', 0), ('s_pix', 0)])
        keys = self.data_disk_meta.keys()

        for key in keys:
            if key not in self.data_disk.keys():
                self.data_disk[key] = []
            if key not in self.data_memory.keys():
                self.data_memory[key] = []
            if key not in self.data_memory_meta.keys():
                self.data_memory_meta[key] = copy.deepcopy(memory_meta_dummy)

    # Conversion between data on disk and in memory.
    @staticmethod
    def disk2memory(data, dtype):

        if dtype == 'complex_int':
            data = ProcessData.complex_int2complex(data)
        if dtype == 'complex_short':
            data = ProcessData.complex_short2complex(data)

        return data

    @staticmethod
    def memory2disk(data, dtype):

        if dtype == 'complex_int':
            data = ProcessData.complex2complex_int(data)
        if dtype == 'complex_short':
            data = ProcessData.complex2complex_short(data)

        return data

    @staticmethod
    def complex2complex_int(data):
        data = data.view(np.int16).astype('float32').view(np.complex64)
        return data

    @staticmethod
    def complex2complex_short(data):
        data = data.view(np.float16).astype('float32').view(np.complex64)
        return data

    @staticmethod
    def complex_int2complex(data):
        data = data.view(np.float32).astype(np.int16).view(np.dtype([('re', np.int16), ('im', np.int16)]))
        return data

    @staticmethod
    def complex_short2complex(data):
        data = data.view(np.float32).astype(np.int16).view(np.dtype([('re', np.float16), ('im', np.float16)]))
        return data

    @staticmethod
    def load_dtypes(self):

        dtype_disk = {'complex_int': np.dtype([('re', np.int16), ('im', np.int16)]),
                           'complex_short': np.dtype([('re', np.float16), ('im', np.float16)]),
                           'complex_real4': np.complex64,
                           'real8': np.float64,
                           'real4': np.float32,
                           'real2': np.float16,
                           'bool' : np.bool,
                           'int8': np.int8,
                           'int16': np.int16,
                           'int32': np.int32,
                           'int64': np.int64,
                           'tiff': np.dtype([('re', np.int16), ('im', np.int16)])}  # source file format (sentinel)

        # Size of these files in bytes
        dtype_size = {'complex_int': 4, 'complex_short': 4, 'complex_real4': 8, 'real8': 8, 'real4': 4, 'real2': 2,
                      'bool' : 1, 'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8, 'tiff': 4}
        dtype_numpy = {'complex_int': np.complex64,
                            'complex_short': np.complex64,
                            'complex_real4': np.complex64,
                            'real8': np.float64,
                            'real4': np.float32,
                            'real2': np.float16,
                            'bool' : np.bool,
                            'int8': np.int8,
                            'int16': np.int16,
                            'int32': np.int32,
                            'int64': np.int64,
                            'tiff': np.complex64}
        dtype_gdal = {'complex_int': gdal.GDT_Float32,
                            'complex_short':gdal.GDT_Float32,
                            'complex_real4': gdal.GDT_Float32,
                            'real8': gdal.GDT_Float64,
                            'real4': gdal.GDT_Float32,
                            'real2': gdal.GDT_Float32,
                            'bool' : gdal.GDT_Byte,
                            'int8': gdal.GDT_Byte,
                            'int16': gdal.GDT_Int16,
                            'int32': gdal.GDT_Int32,
                            'int64': gdal.GDT_Int32,
                            'tiff': gdal.GDT_CInt16}
        dtype_gdal_numpy = {'complex_int': np.float32,
                            'complex_short': np.float32,
                            'complex_real4': np.float32,
                            'real8': np.float64,
                            'real4': np.float32,
                            'real2': np.float32,
                            'bool' : np.int8,
                            'int8': np.int8,
                            'int16': np.int16,
                            'int32': np.int32,
                            'int64': np.int64,
                            'tiff': np.dtype([('re', np.int16), ('im', np.int16)])}

        return dtype_disk, dtype_numpy, dtype_size, dtype_gdal, dtype_gdal_numpy

    def export_tiff(self, process_type):
        # Create geotiff images

        file_type = self.check_file_type_exists(process_type)
        if file_type == False:
            return False

        # Get geo transform and projection
        projection, geo_transform = self.coordinates.create_gdal_projection(self)

        # Save data to geotiff (if complex a file with two bands)
        file_name = self.data_disk_meta[file_type]['file_path'][:-4] + '.tiff'
        data_type = self.data_disk_meta[file_type]['data_type']
        data = self.disk2memory(self.data_disk[file_type], data_type)

        # Create an empty geotiff with the right coordinate system
        gtiff_type = self.dtype_gdal[data_type]
        np_type = self.dtype_gdal_numpy[data_type]
        driver = gdal.GetDriverByName('GTiff')

        # For complex numbers
        if data_type in ['complex_int', 'complex_short', 'complex_real4']:
            layers = 2
        else:
            layers = 1

        amp_data = driver.Create(file_name + '.tiff', self.coordinates.shape[1], self.coordinates.shape[0], layers,
                                 gtiff_type, )
        amp_data.SetGeoTransform(tuple(geo_transform))
        amp_data.SetProjection(projection.ExportToWkt())

        # Save data to tiff file
        if data_type in ['complex_int', 'complex_short', 'complex_real4']:
            print('File converted to amplitude and phase image')

            amp_data.GetRasterBand(1).WriteArray(np.abs(data).astype(np_type))
            amp_data.FlushCache()
            amp_data.GetRasterBand(2).WriteArray(np.angle(data).astype(np_type))
            amp_data.FlushCache()
        else:
            amp_data.GetRasterBand(1).WriteArray(np.abs(data).astype(np_type))
            amp_data.FlushCache()

        print('Saved ' + file_type + ' from ' + self.process_name + ' step of ' +
              os.path.dirname(self.data_disk_meta[file_type]['path_name']))
