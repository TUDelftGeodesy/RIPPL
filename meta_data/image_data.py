"""

Class to read and write image data from and to memory.

"""

import numpy as np
import gdal
from collections import OrderedDict
import os

from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ImageData():

    def __init__(self, json_dict='', file_type='', dtype='', shape='',
                 folder='', process_name='', coordinates='', polarisation='', data_id='', in_coordinates=''):
        
        """
        This class organizes the reading and writing of an image from and to disk.

        :param dict json_dict: Information about image based on information from .json data
        :param file_type:
        :param dtype:
        :param shape:
        :param folder:
        :param process_name:
        :param CoordinateSystem coordinates:
        :param polarisation:
        :param data_id:
        """

        # This information is not necessarily needed if the json dict
        self.folder = folder
        self.process_name = process_name
        self.coordinates = coordinates
        self.in_coordinates = in_coordinates
        self.polarisation = polarisation
        self.data_id = data_id
        self.dtype_disk, self.dtype_memory, self.dtype_size, self.dtype_gdal, self.dtype_gdal_numpy = self.load_dtypes()

        self.disk = OrderedDict()                   # type: OrderedDict(OrderedDict or np.memmap)
        self.disk['data'] = []                      # type: np.memmap
        self.disk['meta'] = OrderedDict()           # type: OrderedDict
        self.memory = OrderedDict()                 # type: OrderedDict(OrderedDict or np.ndarray)
        self.memory['data'] = []                    # type: np.ndarray
        self.memory['meta'] = OrderedDict([('shape', [0, 0]), ('s_lin', 0), ('s_pix', 0)])
        
        if isinstance(json_dict, OrderedDict):
            self.json_dict = json_dict
            self.disk['meta'] = self.json_dict
            self.file_type = self.json_dict['file_type']
            self.dtype = self.json_dict['dtype']
            self.shape = self.json_dict['shape']
            self.file_name = self.json_dict['file_name']
            self.slice_name = self.json_dict['slice_name']
            self.image_name = self.json_dict['image_name']
            self.file_path = os.path.join(self.folder, self.file_name)
            self.json_dict['exist'], self.json_dict['valid'] = self.check_data_disk_valid()
        else:
            self.file_type = file_type
            self.dtype = dtype
            if shape == '':
                self.shape = self.coordinates.shape
            else:
                self.shape = shape
            self.create_file_name()
            self.file_path = os.path.join(self.folder, self.file_name)
            self.create_meta_data()
        
        self.key = self.file_name[:-4]
        
        # Check if dtype exists
        if self.dtype not in list(self.dtype_disk.keys()):
            print(self.dtype + ' does not exist.')
        
        # Store the datasets in a dict to. To be able to link to these from other functions.
        self.file_path = os.path.join(self.folder, self.file_name)

    def create_meta_data(self):
        """

        :return:
        """

        self.json_dict = OrderedDict()
        self.json_dict['process_name'] = self.process_name
        self.json_dict['file_type'] = self.file_type
        self.json_dict['shape'] = [int(self.shape[0]), int(self.shape[1])]
        self.json_dict['file_name'] = self.file_name
        if self.coordinates.slice:
            self.json_dict['slice_name'] = os.path.basename(self.folder)
            self.json_dict['image_name'] = os.path.basename(os.path.dirname(self.folder))
        else:
            self.json_dict['slice_name'] = 'none'
            self.json_dict['image_name'] = os.path.basename(self.folder)
        self.json_dict['dtype'] = self.dtype
        self.json_dict['file_exist'], self.json_dict['file_valid'] = self.check_data_disk_valid()
        self.disk['meta'] = self.json_dict

    def create_file_name(self):
        """
        Get the filename for using id/coordinates/polarisation

        :return:
        """

        if self.coordinates.short_id_str == '':
            short_id = ''
        else:
            short_id = '_' + self.coordinates.short_id_str

        if not self.file_type:
            self.file_name = self.process_name + short_id
        else:
            self.file_name = self.file_type + short_id

        if self.data_id != 'none':
            self.file_name = self.file_name + '_' + self.data_id
        if self.polarisation != 'none':
            self.file_name = self.file_name + '_' + self.polarisation
            
        self.file_name += '.raw'

    # Next functions are for writing/creating files and exchange between disk and memory
    '''
    Functions to communicate with data on disk
    - create_disk_data > create a new file on disk
    - load_disk_data > load disk data into a memmap file. To use data in a function data should be loaded to memory too
    - remove_disk_data_memmap > removes the memmap file of the file on disk (file on disk will stay there)
    - remove_disk_data > removes the data file from disk
    '''

    def create_disk_data(self, overwrite=False):
        # Check if the file exists and whether there is a valid data file already.
        if not os.path.exists(os.path.dirname(self.file_path)):
            print('Folder does not exist')
            return False
        if self.check_data_disk_valid()[1] and not overwrite:
            return False

        meta = self.json_dict
        self.disk['data'] = np.memmap(self.file_path, mode='w+', dtype=self.dtype_disk[self.dtype], shape=tuple(self.shape))
        return True

    def load_disk_data(self):
        # Check if file exists and whether it is valid
        if not self.check_data_disk_valid()[1]:
            return False

        meta = self.json_dict
        self.disk['data'] = np.memmap(self.file_path, mode='r+', dtype=self.dtype_disk[self.dtype], shape=tuple(self.shape))
        return True

    def remove_disk_data_memmap(self):
        # Check if it exists
        if isinstance(self.disk['data'], np.memmap):
            self.disk['data'] = []

        return True

    def remove_disk_data(self):
        # Check if it exists

        # If file exists
        if self.check_data_disk_valid()[0]:
            os.remove(self.file_path)

        return True

    '''
    Functions needed to load or remove data in memory
        - new_memory_data > Create a new empty dataset in memory
        - add_memory_data > Add new memory data with a pre-existing numpy array

    '''

    def new_memory_data(self, shape, s_lin=0, s_pix=0):
        # Create a new memory dataset

        self.memory['data'] = np.zeros(shape).astype(
            self.dtype_memory[self.json_dict['dtype']])

        # Updata meta data
        self.memory['meta']['s_lin'] = s_lin
        self.memory['meta']['s_pix'] = s_pix
        self.memory['meta']['shape'] = shape

    def add_memory_data(self, data, s_lin=0, s_pix=0):
        # Check if data formats are correct
        dtype = self.dtype_memory[self.json_dict['dtype']]
        if not data.dtype == dtype:
            print('Input data type is not correct. It should be ' + str(dtype))
            return
        self.memory['data'] = data

        # Updata meta data
        self.memory['meta']['s_lin'] = s_lin
        self.memory['meta']['s_pix'] = s_pix
        self.memory['meta']['shape'] = data.shape

    def load_memory_data(self, shape=[], s_lin=0, s_pix=0):
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

        # Check if coverage is the same otherwise we have to load new data
        if shape == self.memory['meta']['shape'] and \
                s_lin == self.memory['meta']['s_lin'] and \
                s_pix == self.memory['meta']['s_pix']:
            return True
        # Try to subset
        if self.subset_memory_data(shape=shape, s_lin=s_lin, s_pix=s_pix):
            return True
        # Try to load from disk. As a last step.
        if self.load_memory_data_from_disk(shape=shape, s_lin=s_lin, s_pix=s_pix):
            return True
        else:
            return False

    def subset_memory_data(self, shape, s_lin=0, s_pix=0):
        # Check if subsetting is possible

        # Check if the datasets are overlapping
        if not self.check_overlapping(data_type='memory', shape=shape, s_lin=s_lin, s_pix=s_pix):
            return False

        dat_s_lin = self.memory['meta']['s_lin'] - s_lin
        dat_s_pix = self.memory['meta']['s_pix'] - s_pix

        self.memory['data'] = self.memory['data'][dat_s_lin:dat_s_lin + shape[0],
                                      dat_s_pix:dat_s_pix + shape[1]]

        # Updata meta data
        self.memory['meta']['s_lin'] = s_lin
        self.memory['meta']['s_pix'] = s_pix
        self.memory['meta']['shape'] = shape

        return True

    def load_memory_data_from_disk(self, shape, s_lin=0, s_pix=0):
        # Load data from data on disk. This can be slice used for multiprocessing.

        # Check if the datasets are overlapping
        if not self.check_overlapping(data_type='data', shape=shape, s_lin=s_lin, s_pix=s_pix):
            return False

        # Get the data from disk
        disk_data = self.disk2memory(self.disk['data'], self.json_dict['dtype'])
        self.memory['data'] = np.copy(disk_data[s_lin:s_lin + shape[0], s_pix:s_pix + shape[1]])

        # Updata meta data
        self.memory['meta']['s_lin'] = s_lin
        self.memory['meta']['s_pix'] = s_pix
        self.memory['meta']['shape'] = shape

        return True

    def save_memory_data_to_disk(self):
        # Save data to disk from memory

        # Check if a memory file is loaded
        if not self.check_memory_file():
            return False

        # Find settings
        s_lin = self.memory['meta']['s_lin']
        s_pix = self.memory['meta']['s_pix']
        shape = self.memory['meta']['shape']

        # Check if the datasets are overlapping
        if not self.check_overlapping(data_type='data', shape=shape, s_lin=s_lin, s_pix=s_pix):
            return False

        # Check if memmap is loaded.
        if not self.check_disk_file():
            # If not loaded, a new data file is created. (In this way we can create new datasets, but better initialize
            # beforehand, otherwise it can create problems with parallel processing.)
            if not self.load_disk_data():
                return False

        # Save data to disk
        memory_data = self.memory2disk(self.memory['data'], self.json_dict['dtype'])
        self.disk['data'][s_lin:s_lin + shape[0], s_pix:s_pix + shape[1]] = memory_data
        self.disk['data'].flush()

        return True

    def remove_memory_data(self):
        # Remove data if it is loaded

        if isinstance(self.memory['data'], np.ndarray):
            self.memory['data'] = []
            self.memory['meta']['shape'] = [0, 0]

    # Helper functions for reading writing functions
    def check_memory_file(self):
        # Check if there is a memory file.
        if isinstance(self.memory['data'], np.ndarray):
            return True
        else:
            return False

    def check_disk_file(self):
        # Check if there is a memory file.
        if isinstance(self.disk['data'], np.memmap):
            return True
        else:
            return False

    def check_data_disk_valid(self):
        # Function checks whether the datafile exists and whether is has the correct size.
        # We assume 2D files here...
        # Check if file exists

        exist = os.path.exists(self.file_path)
        self.json_dict['file_exist'] = exist

        if exist:
            shape = self.json_dict['shape']
            file_dat_size = self.dtype_size[self.json_dict['dtype']] * shape[0] * shape[1]
            if int(os.path.getsize(self.file_path)) == file_dat_size:
                valid_size = True
            else:
                valid_size = False
            self.json_dict['file_valid'] = valid_size
        else:
            return False, False

        return exist, valid_size

    def check_overlapping(self, data_type='memory', shape=[0, 0], s_lin=0, s_pix=0):
        # Check if the regions we want to select are within the boundary of the original file.
        if data_type == 'memory':
            orig_shape = self.memory['meta']['shape']
            orig_s_lin = self.memory['meta']['s_lin']
            orig_s_pix = self.memory['meta']['s_pix']
        else:  # If it is a data file on disk.
            orig_shape = self.json_dict['shape']
            orig_s_lin = 0
            orig_s_pix = 0

        # Check s_lin/s_pix
        if (s_lin < orig_s_lin or s_lin + shape[0] > orig_s_lin + orig_shape[0]) or \
                (s_pix < orig_s_pix or s_pix + shape[1] > orig_s_pix + orig_shape[1]):
            return False
        else:
            return True

    # Conversion between data on disk and in memory.
    @staticmethod
    def disk2memory(data, dtype):

        if dtype == 'complex_int':
            data = ImageData.complex_int2complex(data)
        if dtype == 'complex_short':
            data = ImageData.complex_short2complex(data)

        return data

    @staticmethod
    def memory2disk(data, dtype):

        if dtype == 'complex_int' or dtype == 'complex_short':
            if not data.dtype == np.complex64:
                raise TypeError('Input data should be a complex64 type')

            if dtype == 'complex_int':
                data = ImageData.complex2complex_int(data)
            if dtype == 'complex_short':
                data = ImageData.complex2complex_short(data)

        return data

    @staticmethod
    def complex_int2complex(data):
        data = data.view(np.int16).astype('float32').view(np.complex64)
        return data

    @staticmethod
    def complex_short2complex(data):
        data = data.view(np.float16).astype('float32').view(np.complex64)
        return data

    @staticmethod
    def complex2complex_int(data):
        data = data.view(np.float32).astype(np.int16).view(np.dtype([('re', np.int16), ('im', np.int16)]))
        return data

    @staticmethod
    def complex2complex_short(data):
        data = data.view(np.float32).astype(np.float16).view(np.dtype([('re', np.float16), ('im', np.float16)]))
        return data

    @staticmethod
    def load_dtypes():

        dtype_disk = {'complex_int': np.dtype([('re', np.int16), ('im', np.int16)]),
                      'complex_short': np.dtype([('re', np.float16), ('im', np.float16)]),
                      'complex_real4': np.complex64,
                      'real8': np.float64,
                      'real4': np.float32,
                      'real2': np.float16,
                      'bool': np.bool,
                      'int8': np.int8,
                      'int16': np.int16,
                      'int32': np.int32,
                      'int64': np.int64,
                      'tiff': np.dtype([('re', np.int16), ('im', np.int16)])}  # source file format (sentinel)

        # Size of these files in bytes
        dtype_size = {'complex_int': 4, 'complex_short': 4, 'complex_real4': 8, 'real8': 8, 'real4': 4, 'real2': 2,
                      'bool': 1, 'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8, 'tiff': 4}
        dtype_numpy = {'complex_int': np.complex64,
                       'complex_short': np.complex64,
                       'complex_real4': np.complex64,
                       'real8': np.float64,
                       'real4': np.float32,
                       'real2': np.float16,
                       'bool': np.bool,
                       'int8': np.int8,
                       'int16': np.int16,
                       'int32': np.int32,
                       'int64': np.int64,
                       'tiff': np.complex64}
        dtype_gdal = {'complex_int': gdal.GDT_Float32,
                      'complex_short': gdal.GDT_Float32,
                      'complex_real4': gdal.GDT_Float32,
                      'real8': gdal.GDT_Float64,
                      'real4': gdal.GDT_Float32,
                      'real2': gdal.GDT_Float32,
                      'bool': gdal.GDT_Byte,
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
                            'bool': np.int8,
                            'int8': np.int8,
                            'int16': np.int16,
                            'int32': np.int32,
                            'int64': np.int64,
                            'tiff': np.dtype([('re', np.int16), ('im', np.int16)])}

        return dtype_disk, dtype_numpy, dtype_size, dtype_gdal, dtype_gdal_numpy
