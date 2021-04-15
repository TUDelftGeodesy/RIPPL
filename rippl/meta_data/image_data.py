"""

Class to read and write image data from and to memory.

"""

import numpy as np
from osgeo import gdal
import shutil
from collections import OrderedDict
import os
from rippl.user_settings import UserSettings
import time


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
            self.json_dict['exist'], self.json_dict['valid'] = self.check_data_disk_valid()
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
            print(self.dtype + ' does not exist.')

    def create_meta_data(self):
        """
        This method creates the meta data information of this image. This will be save to the .json meta data file.

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
        This method creates the id and filename of the data on disk. This file name is used when data is save to disk.

        :return:
        """

        file_name = self.file_type

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

    def create_disk_data(self, overwrite=False, tmp_directory=''):
        """
        Create data on disk for this image data.

        :param bool overwrite: If file already exists do we overwrite.
        :return:
        """

        # Check if file already exist
        if not os.path.exists(os.path.dirname(self.file_path)):
            print('Folder does not exist')
            return False
        if self.check_data_disk_valid()[1] and not overwrite:
            return False

        if tmp_directory:
            if os.name == 'nt':
                tmp_file = self.file_path.replace('\\', '_')
            else:
                tmp_file = self.file_path.replace('/', '_')
            if not os.path.exists(tmp_directory):
                raise FileExistsError('Temp directory ' + tmp_directory + ' does not exist. Aborting...')
            self.tmp_path = os.path.join(tmp_directory, tmp_file)

            self.disk['data'] = np.memmap(self.tmp_path, mode='w+', dtype=self.dtype_disk[self.dtype],
                                          shape=tuple(self.shape))
        else:
            self.disk['data'] = np.memmap(self.file_path, mode='w+', dtype=self.dtype_disk[self.dtype], shape=tuple(self.shape))

        return True

    def load_disk_data(self, tmp_directory=''):
        """
        Load the data from disk as a memmap file.

        :return:
        """

        # Check if file exists and whether it is valid
        if not self.check_data_disk_valid()[1]:
            return False

        if tmp_directory:
            # Files are copied to the same temporary folder using all the child folders within the path name as the
            # filename. This prevents any unnecessary creation of folders without issues due to the
            if os.name == 'nt':
                tmp_file = self.file_path.replace('\\', '_')
                tmp_file = tmp_file.replace('/', '_')
            else:
                tmp_file = self.file_path.replace('/', '_')
            if not os.path.exists(tmp_directory):
                raise FileExistsError('Temp directory ' + tmp_directory + ' does not exist. Aborting...')
            tmp_path = os.path.join(tmp_directory, tmp_file)
            tmp_tmp_path = os.path.join(tmp_directory, tmp_file + '.temporary')

            time.sleep(np.random.random(1)[0])      # A random sleep time before starting to copy (max 1 second)
            if not os.path.exists(tmp_path):
                time.sleep(np.random.random(1)[0])      # Another wait to get one of the processes ahead of the others
                if not os.path.exists(tmp_tmp_path):
                    # Copy using blocks of 100MB
                    with open(self.file_path, 'rb') as src:
                        # We add an extra check for the case that 2 processes enter this loop at the same time.
                        if not os.path.exists(tmp_tmp_path):
                            file_copied = True
                            with open(tmp_tmp_path, 'wb') as dst:
                                print('Copying ' + self.file_path + ' to temporary storage ' + tmp_path)
                                shutil.copyfileobj(src, dst, 100000000)
                                # Another sleep to make sure copying finished
                                time.sleep(0.1)
                        else:
                            print('Temporary file already created when reading original file, skipping.')
                            file_copied = False

                    # Now if the the temporary file is copied, we rename it. Because it sometimes takes some time
                    # before the file becomes available we add a try/except loop to make sure it works.
                    # In the very unlikely event two processes copied the file at the same time (think this is not
                    # possible...) the process just moves on and tries to load the renamed file without renaming.
                    name_changed = False
                    no_tries = 0
                    while not name_changed and no_tries < 10 and file_copied:
                        try:
                            if os.path.exists(tmp_tmp_path):
                                os.rename(tmp_tmp_path, tmp_path)
                                name_changed = True
                        except:
                            print('Cannot find the temporary file. Trying again in 10 seconds')
                            time.sleep(10)
                            no_tries += 1
                else:
                    print('Temporary file does not exist, but is being copied.')

            # If the temporary file exists, maybe the file is in the process of being copied. This can take several
            # minutes but we check for a full hour every 10 seconds whether the file is safely copied and load the file
            # if available.
            n = 0
            while n < 360:
                if os.path.exists(tmp_path):
                    # In this case the tmp_tmp_path did exist, so the file is actually being copied.
                    self.disk['data'] = np.memmap(tmp_path, mode='r+', dtype=self.dtype_disk[self.dtype], shape=tuple(self.shape))
                    self.tmp_path = tmp_path
                    break
                else:
                    print('File ' + tmp_path + ' is not readable. Likely because it is being copied. Trying again in '
                                               '10 seconds. Total waiting time is ' + str(n * 10) + ' seconds.')
                    n += 1
                    time.sleep(10)
            else:
                raise FileExistsError('File ' + tmp_path + ' is not readable!')
        else:
            self.disk['data'] = np.memmap(self.file_path, mode='r+', dtype=self.dtype_disk[self.dtype], shape=tuple(self.shape))
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

        # Copy using blocks of 100MB
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
        Remove the memmap and the data on disk. (This will remove the data on disk which cannot be recovered afterwards!)

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
            print('Input data type is not correct. It should be ' + str(dtype))
            return
        self.memory['data'] = data

        # Updata meta data
        self.memory['meta']['s_lin'] = s_lin
        self.memory['meta']['s_pix'] = s_pix
        self.memory['meta']['shape'] = data.shape

    def load_memory_data(self, shape=[], s_lin=0, s_pix=0, tmp_directory=''):
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
        if self.load_memory_data_from_disk(shape=shape, s_lin=s_lin, s_pix=s_pix, tmp_directory=tmp_directory):
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

    def load_memory_data_from_disk(self, shape, s_lin=0, s_pix=0, tmp_directory=''):
        # Load data from data on disk. This can be slice used for multiprocessing.

        # Check if the datasets are overlapping
        if not self.check_overlapping(data_type='data', shape=shape, s_lin=s_lin, s_pix=s_pix):
            return False

        # Load data as memmap file
        if len(self.disk['data']) == 0:
            self.load_disk_data(tmp_directory)

        # Get the data from disk
        disk_data = self.disk2memory(self.disk['data'][s_lin:s_lin + shape[0], s_pix:s_pix + shape[1]], self.json_dict['dtype'])
        self.memory['data'] = np.copy(disk_data)

        # Updata meta data
        self.memory['meta']['s_lin'] = s_lin
        self.memory['meta']['s_pix'] = s_pix
        self.memory['meta']['shape'] = shape

        return True

    def save_memory_data_to_disk(self, tmp_directory=''):
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

    def save_tiff(self, file_path='', tiff_folder='', no_sub_folders=False, overwrite=False):
        """
        Save data as geotiff. Complex data will be saved as a two layer tiff with amplitude and phase values.

        :param str file_path: If a file name is defined data is saved in this file. Otherwise the name is directly
                copied from the .raw name but replaced with a .tiff value.
        :return:
        """

        self.coordinates.create_short_coor_id()

        if not file_path:
            basename = os.path.basename(self.file_path)[:-4] + '.tiff'
            folder_name = os.path.basename(os.path.dirname(self.file_path))

            if folder_name.startswith('slice'):
                slice_name = folder_name
                date_name = os.path.basename(os.path.dirname(os.path.dirname(self.file_path)))
                file_name = date_name + '_' + slice_name + '_' + basename
                stack_folder = os.path.dirname(os.path.dirname(os.path.dirname(self.file_path)))
            else:
                slice_name = ''
                date_name = folder_name
                file_name = date_name + '_' + basename
                stack_folder = os.path.dirname(os.path.dirname(self.file_path))

            if not tiff_folder:
                settings = UserSettings()
                settings.load_settings()

                satellite_folder = os.path.join(settings.radar_data_products, os.path.basename(os.path.dirname(stack_folder)))
                tiff_folder = os.path.join(satellite_folder, os.path.basename(stack_folder))
                if not os.path.exists(tiff_folder):
                    os.mkdir(tiff_folder)

            # Output folder. If no slice, slice name will be empty and therefore not added to path.
            if no_sub_folders:
                output_folder = tiff_folder
            else:
                output_folder = os.path.join(tiff_folder, self.process_name, self.coordinates.short_id_str, slice_name)

            # Create file path by adding filename
            file_path = os.path.join(output_folder, file_name)

            if not os.path.exists(tiff_folder):
                raise FileExistsError('Folder to write tiff files ' + tiff_folder + ' does not exist.')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        else:
            if not os.path.exists(tiff_folder):
                raise FileExistsError('Folder to write tiff files ' +tiff_folder + ' does not exist.')

        # Check if file already exists.
        if os.path.exists(file_path) and not overwrite:
            print('File ' + file_path + ' does already exist')
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
        if dtype == 'complex_short':
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
