"""

Class to read and write image data from and to memory.

"""

import numpy as np
from osgeo import gdal
gdal.DontUseExceptions()
import shutil
from collections import OrderedDict
import os
import logging

from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ImageData():

    def __init__(self, json_dict='', file_type='', dtype='', shape='',
                 folder='', process_name='', coordinates='', polarisation='', data_id='', in_coordinates=''):
        
        """
        This class organizes the reading and writing of an image from and to disk.

        :param dict json_dict: Information about image based on information from .json data
        :param str file_type: File type of image as part of a process
        :param str dtype: Data type of image.
        :param tuple shape: Shape of the image. Can be left empty if the same as the coordinates system
        :param str folder: Path to the folder where data on disk of this image is stored
        :param str process_name: Name of the process this image is part of
        :param CoordinateSystem coordinates: Coordinate system for this image
        :param str polarisation: Polarisation of dataset
        :param str data_id: Data ID, in most cases not used, but usefull if comparable datasets are used.
        """

        # This information is not necessarily needed if the json dict
        self.folder = folder
        if self.folder:
            self.date = os.path.basename(self.folder)
        elif coordinates.date:
            self.date = coordinates.date[:4] + coordinates.date[5:7] + coordinates.date[8:10]
        elif not isinstance(json_dict, OrderedDict):
            raise LookupError('Not possible to get the date of the output image. Please provide source folder, '
                              'coordinates containing a date of the .json metadata')

        self.process_name = process_name
        self.coordinates = coordinates
        self.in_coordinates = in_coordinates
        self.polarisation = polarisation
        self.data_id = data_id
        self.dtype_disk, self.dtype_memory, self.dtype_size, self.dtype_gdal, self.dtype_gdal_numpy = self.load_dtypes()
        self.tmp_path = ''

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
        else:
            self.file_type = file_type
            self.dtype = dtype
            if not shape:
                self.shape = self.coordinates.shape
            else:
                self.shape = shape
            self.create_file_name()
            self.create_meta_data()
        
        self.key = self.file_name[:-4]
        
        # Check if dtype exists
        if self.dtype not in list(self.dtype_disk.keys()):
            logging.info(self.dtype + ' does not exist.')

    def create_meta_data(self):
        """
        This method creates the metadata information of this image. This will be saved to the .json metadata file.

        :return:
        """

        self.json_dict = OrderedDict()
        self.json_dict['process_name'] = self.process_name
        self.json_dict['file_type'] = self.file_type
        if len(self.shape) != 2:
            raise TypeError('Make sure that the shape size of the output coordinate system is defined before creating'
                            'output data for file ' + self.file_path)
        self.json_dict['shape'] = [int(self.shape[0]), int(self.shape[1])]
        self.json_dict['file_name'] = self.file_name
        if self.coordinates.slice:
            self.json_dict['slice_name'] = os.path.basename(self.folder)
            self.json_dict['image_name'] = os.path.basename(os.path.dirname(self.folder))
        else:
            self.json_dict['slice_name'] = 'none'
            self.json_dict['image_name'] = os.path.basename(self.folder)
        self.json_dict['dtype'] = self.dtype
        self.disk['meta'] = self.json_dict

    def create_file_name(self):
        """
        This method creates the id and filename of the data on disk. This file name is used when data is saved to disk.

        :return:
        """

        file_name = os.path.basename(self.folder) + '_' + self.file_type

        if self.polarisation and self.polarisation != 'none':
            file_name += '_' + self.polarisation
        if self.data_id and self.data_id != 'none':
            file_name += '_' + self.data_id

        file_name += '@'
        file_name += self.coordinates.short_id_str
        if isinstance(self.in_coordinates, CoordinateSystem):
            file_name += '_in_coor_' + self.in_coordinates.short_id_str

        self.file_name = file_name + '.raw'
        self.file_path = os.path.join(self.folder, self.file_name)

    '''
    Functions to communicate with data on disk
    - create_disk_data > create a new file on disk
    - load_disk_data > load disk data into a memmap file. To use data in a function data should be loaded to memory too
    - remove_disk_data_memmap > removes the memmap file of the file on disk (file on disk will stay there)
    - remove_disk_data > removes the data file from disk
    '''

    def create_disk_data(self, overwrite=False, tmp_directories=[]):
        """
        Create data on disk for this image data.

        :param bool overwrite: If file already exists do we overwrite.
        :return:
        """

        # Check if file already exist
        if isinstance(tmp_directories, str):
            tmp_directories = [tmp_directories]

        if tmp_directories:
            for tmp_directory in tmp_directories:

                if not os.path.exists(tmp_directory):
                    continue
                self.tmp_path = os.path.join(tmp_directory, os.path.basename(self.file_path))
                # If file already exists
                if os.path.exists(self.tmp_path) and not overwrite and self.check_data_disk_valid()[1]:
                    return True
                else:
                    self.disk['data'] = np.memmap(self.tmp_path, mode='w+', dtype=self.dtype_disk[self.dtype],
                                              shape=tuple(self.shape))
                    logging.info('File created at ' + self.file_path)
                    return True

        if not os.path.exists(os.path.dirname(self.file_path)):
            raise FileExistsError(os.path.dirname(self.file_path) + ' folder does not exist! Not able to create file'
                                                                    '' + self.file_path)
        else:
            if os.path.exists(self.file_path) and not overwrite and self.check_data_disk_valid()[1]:
                return True
            else:
                self.disk['data'] = np.memmap(self.file_path, mode='w+', dtype=self.dtype_disk[self.dtype], shape=tuple(self.shape))
                logging.info('File created at ' + self.file_path)
                return True

    def load_disk_data(self, tmp_directories=[]):
        """
        Load the data from disk as a memmap file.

        :return:
        """

        # Check if file exists and whether it is valid
        if isinstance(tmp_directories, str):
            tmp_directories = [tmp_directories]

        if tmp_directories:
            for tmp_directory in tmp_directories:
                # Files are copied to the same temporary folder using all the child folders within the path name as the
                # filename. This prevents any unnecessary creation of folders without issues due to the
                if not os.path.exists(tmp_directory):
                    continue
                tmp_path = os.path.join(tmp_directory, os.path.basename(self.file_path))

                if os.path.exists(tmp_path):
                    # In this case the tmp_tmp_path did exist, so the file is actually being copied.
                    self.disk['data'] = np.memmap(tmp_path, mode='r+', dtype=self.dtype_disk[self.dtype], shape=tuple(self.shape))
                    self.tmp_path = tmp_path
                    logging.info(tmp_path + ' loaded from temporary directory')
                    return True
                else:
                    continue

        if os.path.exists(self.file_path):
            self.disk['data'] = np.memmap(self.file_path, mode='r+', dtype=self.dtype_disk[self.dtype], shape=tuple(self.shape))
        else:
            return False

        return True

    def save_tmp_data(self):
        """
        Save created tmp file to disk

        Returns
        -------

        """

        if not self.tmp_path:
            raise FileNotFoundError('Temporary file does not exist')
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        # Copy using chunks of 100MB
        with open(self.tmp_path, 'rb') as src:
            with open(self.file_path, 'wb') as dst:
                shutil.copyfileobj(src, dst, 100000000)
        os.remove(self.tmp_path)

    def remove_disk_data_memmap(self):
        """
        Remove the memmap of the data on disk (But not the data on disk itself!)

        :return:
        """

        # Check if it exists
        if isinstance(self.disk['data'], np.memmap):
            self.disk['data'] = []

        return True

    def remove_disk_data(self):
        """
        Remove the memmap and the data on disk. (This will remove the data on disk which cannot be recovered afterward!)

        :return:
        """

        # If file exists
        if self.check_data_disk_valid()[0]:
            self.remove_disk_data_memmap()
            os.remove(self.file_path)

        return True

    '''
    Functions needed to load or remove data in memory
        - new_memory_data > Create a new empty dataset in memory
        - add_memory_data > Add new memory data with a pre-existing numpy array
    '''

    def new_memory_data(self, shape, s_lin=0, s_pix=0):
        # Create a new memory dataset

        self.memory['data'] = np.zeros(shape).astype(self.dtype_memory[self.json_dict['dtype']])

        # Updata meta data
        self.memory['meta']['s_lin'] = s_lin
        self.memory['meta']['s_pix'] = s_pix
        self.memory['meta']['shape'] = shape

    def add_memory_data(self, data, s_lin=0, s_pix=0):
        # Check if data formats are correct
        dtype = self.dtype_memory[self.json_dict['dtype']]
        if not data.dtype == dtype:
            logging.info('Input data type is not correct. It should be ' + str(dtype))
            return
        self.memory['data'] = data

        # Updata meta data
        self.memory['meta']['s_lin'] = s_lin
        self.memory['meta']['s_pix'] = s_pix
        self.memory['meta']['shape'] = data.shape

    def load_memory_data(self, shape=[], s_lin=0, s_pix=0, tmp_directories=[]):
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

        if s_lin < 0 or s_pix < 0:
            raise ImportError('It is not possible to load data with negative start lines or pixels')
        if shape[0] < 0 or shape[1] < 0:
            raise ImportError('It is not possible to load data with negative number of lines or pixels')

        # Check if coverage is the same otherwise we have to load new data
        if shape == self.memory['meta']['shape'] and \
                s_lin == self.memory['meta']['s_lin'] and \
                s_pix == self.memory['meta']['s_pix']:
            return True
        # Try to subset
        if self.subset_memory_data(shape=shape, s_lin=s_lin, s_pix=s_pix):
            return True
        # Try to load from disk. As a last step.
        if self.load_memory_data_from_disk(shape=shape, s_lin=s_lin, s_pix=s_pix, tmp_directories=tmp_directories):
            return True
        else:
            return False

    def subset_memory_data(self, shape, s_lin=0, s_pix=0):
        # Check if subsetting is possible

        # Check if the datasets are overlapping
        if not self.check_overlapping(data_type='memory', shape=shape, s_lin=s_lin, s_pix=s_pix):
            # logging.info('Requested dataset does not overlap with data in memory. Data cannot be loaded from memory.')
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

    def load_memory_data_from_disk(self, shape, s_lin=0, s_pix=0, tmp_directories=[]):
        # Load data from data on disk. This can be a chunk used for multiprocessing.

        # Check if the datasets are overlapping
        if not self.check_overlapping(data_type='data', shape=shape, s_lin=s_lin, s_pix=s_pix):
            logging.info('Requested dataset does not overlap with data on disk. Not able to load data in memory.')
            return False

        # Load data as memmap file
        if len(self.disk['data']) == 0:
            loaded = self.load_disk_data(tmp_directories)
            if not loaded:
                return False

        # Get the data from disk
        try:
            disk_data = self.disk2memory(self.disk['data'][s_lin:s_lin + shape[0], s_pix:s_pix + shape[1]], self.json_dict['dtype'])
            self.memory['data'] = np.copy(disk_data)
        except Exception as e:
            raise TypeError('Not possible to load ' + self.file_path + ' with shape ' + str(self.shape) + '. Tried to ' +
                            'load data starting at ' + str(s_pix) + ' pixels and ' + str(s_lin) + ' lines with shape' +
                            str(shape) + '. ' + str(e))

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
        if not self.disk['data'].dtype == memory_data.dtype:
            raise TypeError('Data from process calculation is not the same as defined at process input for file' +
                             self.file_path + '. Please review '
                            'your process output data and process calculations! (For example use numpy astype function)')

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

    def get_output_filename(self, file_path, file_folder, type_str='.tiff'):
        """
        Get the output tiff name

        """

        if not file_path:
            file_path = self.file_path

        basename = os.path.basename(file_path)[:-4] + type_str
        folder_name = os.path.basename(os.path.dirname(file_path))

        if 'slice' in folder_name:
            slice_name = folder_name
            file_name = slice_name + '_' + basename
            stack_folder = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
        else:
            slice_name = ''
            file_name = basename
            stack_folder = os.path.dirname(os.path.dirname(file_path))

        if not file_folder:
            file_folder = stack_folder + '_output_products'
            if not os.path.exists(file_folder):
                os.mkdir(file_folder)

        # Output folder. If no slice, slice name will be empty and therefore not added to path.
        output_folder = os.path.join(file_folder, self.process_name, self.coordinates.short_id_str, slice_name)

        # Create file path by adding filename
        out_file_path = os.path.join(output_folder, file_name)

        if not os.path.exists(file_folder):
            os.makedir(file_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        return out_file_path

    def save_tiff(self, file_path='', tiff_folder='', no_sub_folders=False, overwrite=False):
        """
        Save data as geotiff. Complex data will be saved as a two layer tiff with amplitude and phase values.

        :param str file_path: If a file name is defined data is saved in this file. Otherwise, the name is directly
                copied from the .raw name but replaced with a .tiff value.
        :return:
        """

        self.coordinates.create_short_coor_id()

        if not file_path:
            file_path = self.get_output_filename(file_path, tiff_folder, type_str='.tiff')

        # Check if file already exists.
        if os.path.exists(file_path) and not overwrite:
            logging.info('File ' + file_path + ' does already exist')
            return
        elif os.path.exists(file_path) and overwrite:
            os.remove(file_path)

        projection, geo_transform, flipped = self.coordinates.create_gdal_projection()
        driver = gdal.GetDriverByName('GTiff')

        self.load_memory_data(self.shape)

        if self.dtype.startswith('complex'):
            data = driver.Create(file_path, self.shape[1], self.shape[0], 2, self.dtype_gdal[self.dtype])
            data.SetGeoTransform(geo_transform)
            data.SetProjection(projection.ExportToWkt())
            if flipped:
                data.GetRasterBand(1).WriteArray(np.flipud(np.log(np.abs(self.memory['data']))))
                data.GetRasterBand(2).WriteArray(np.flipud(np.angle(self.memory['data'])))
            else:
                data.GetRasterBand(1).WriteArray(np.log(np.abs(self.memory['data'])))
                data.GetRasterBand(2).WriteArray(np.angle(self.memory['data']))
        else:
            data = driver.Create(file_path, self.shape[1], self.shape[0], 1, self.dtype_gdal[self.dtype])
            data.SetGeoTransform(geo_transform)
            data.SetProjection(projection.ExportToWkt())
            if flipped:
                data.GetRasterBand(1).WriteArray(np.flipud(self.memory['data']))
            else:
                data.GetRasterBand(1).WriteArray(self.memory['data'])

        data.FlushCache()
        data = None
        self.remove_memory_data()
        self.remove_disk_data_memmap()

    # Conversion between data on disk and in memory.
    @staticmethod
    def disk2memory(data, dtype):
        """
        For some format there is no proper numpy format. These should therefore be corrected when loaded to memory to
        be able to do calculations.

        :param np.ndarray data: Input data that should be converted from disk to memory
        :param str dtype: Data type of dataset
        :return:
        """

        if dtype == 'complex_int':
            data = ImageData.complex_int2complex(data)
        if dtype in ['complex_short', 'complex32']:
            data = ImageData.complex_short2complex(data)

        return data

    @staticmethod
    def memory2disk(data, dtype):
        """
        For some format there is no proper numpy format. These should therefore be corrected when save to disk to
        save space on disk.

        :param np.ndarray data: Input data that should be converted from disk to memory
        :param str dtype: Data type of dataset
        :return:
        """

        if dtype in ['complex_int', 'complex_short', 'complex32']:
            if not data.dtype == np.complex64:
                raise TypeError('Input data should be a complex64 type')

            if dtype == 'complex_int':
                data = ImageData.complex2complex_int(data)
            if dtype in ['complex_short', 'complex32']:
                data = ImageData.complex2complex_short(data)

        return data

    @staticmethod
    def complex_int2complex(data):
        """
        Convert complex integers to regular numpy complex64 data.

        :param np.ndarray data: Input data to be converted
        :return: data
        """
        data = np.copy(data).view(np.int16).astype('float32').view(np.complex64)
        return data

    @staticmethod
    def complex_short2complex(data):
        """
        Convert complex short values to regular numpy complex64 data.

        :param np.ndarray data: Input data to be converted
        :return: data
        """
        data = np.copy(data).view(np.float16).astype('float32').view(np.complex64)
        return data

    @staticmethod
    def complex2complex_int(data):
        """
        Convert from regular complex 64 data to complex integers.

        :param np.ndarray data: Input data to be converted
        :return: data
        """
        data = data.view(np.float32).astype(np.int16).view(np.dtype([('re', np.int16), ('im', np.int16)]))
        return data

    @staticmethod
    def complex2complex_short(data):
        """
        Convert from regular complex 64 data to complex shorts.

        :param np.ndarray data: Input data to be converted
        :return: data
        """
        data = data.view(np.float32).astype(np.float16).view(np.dtype([('re', np.float16), ('im', np.float16)]))
        return data

    @staticmethod
    def load_dtypes():

        dtype_disk = {'complex_int': np.dtype([('re', np.int16), ('im', np.int16)]),
                      'complex_short': np.dtype([('re', np.float16), ('im', np.float16)]),
                      'complex_real4': np.complex64,
                      'complex32': np.dtype([('re', np.float16), ('im', np.float16)]),
                      'complex64': np.complex64,
                      'complex128': np.complex128,
                      'real8': np.float64,
                      'real4': np.float32,
                      'real2': np.float16,
                      'float64': np.float64,
                      'float32': np.float32,
                      'float16': np.float16,
                      'bool': np.bool_,
                      'int8': np.int8,
                      'int16': np.int16,
                      'int32': np.int32,
                      'int64': np.int64,
                      'tiff': np.dtype([('re', np.int16), ('im', np.int16)])}  # source file format (sentinel)

        # Size of these files in bytes
        dtype_size = {'complex_int': 4, 'complex_short': 4, 'complex_real4': 8, 'real8': 8, 'real4': 4, 'real2': 2,
                      'bool': 1, 'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8, 'tiff': 4, 'complex32': 4,
                      'complex64': 8, 'complex128': 16, 'float64': 8, 'float32': 4, 'float16': 2}
        dtype_numpy = {'complex_int': np.complex64,
                       'complex_short': np.complex64,
                       'complex_real4': np.complex64,
                       'complex32': np.complex64,
                       'complex64': np.complex64,
                       'complex128': np.complex128,
                       'real8': np.float64,
                       'real4': np.float32,
                       'real2': np.float16,
                       'float64': np.float64,
                       'float32': np.float32,
                       'float16': np.float16,
                       'bool': np.bool_,
                       'int8': np.int8,
                       'int16': np.int16,
                       'int32': np.int32,
                       'int64': np.int64,
                       'tiff': np.complex64}
        dtype_gdal = {'complex_int': gdal.GDT_Float32,
                      'complex_short': gdal.GDT_Float32,
                      'complex_real4': gdal.GDT_Float32,
                      'complex32': gdal.GDT_Float32,
                      'complex64': gdal.GDT_Float32,
                      'complex128': gdal.GDT_Float64,
                      'real8': gdal.GDT_Float64,
                      'real4': gdal.GDT_Float32,
                      'real2': gdal.GDT_Float32,
                      'float64': gdal.GDT_Float64,
                      'float32': gdal.GDT_Float32,
                      'float16': gdal.GDT_Float32,
                      'bool': gdal.GDT_Byte,
                      'int8': gdal.GDT_Byte,
                      'int16': gdal.GDT_Int16,
                      'int32': gdal.GDT_Int32,
                      'int64': gdal.GDT_Int32,
                      'tiff': gdal.GDT_CInt16}
        dtype_gdal_numpy = {'complex_int': np.float32,
                            'complex_short': np.float32,
                            'complex_real4': np.float32,
                            'complex32': np.float32,
                            'complex64': np.float32,
                            'complex128': np.float64,
                            'real8': np.float64,
                            'real4': np.float32,
                            'real2': np.float32,
                            'float64': np.float64,
                            'float32': np.float32,
                            'float16': np.float32,
                            'bool': np.int8,
                            'int8': np.int8,
                            'int16': np.int16,
                            'int32': np.int32,
                            'int64': np.int64,
                            'tiff': np.dtype([('re', np.int16), ('im', np.int16)])}

        return dtype_disk, dtype_numpy, dtype_size, dtype_gdal, dtype_gdal_numpy
