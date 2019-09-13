
import numpy as np
from collections import OrderedDict
import os
import gdal

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.process_meta import ProcessMeta
from rippl.meta_data.image_data import ImageData


class ProcessData():
    """
    This class creates an interface between the processes and the data on disk. The function does:
    - create new data files in memory and on disk
    - write data from memory to disk
    - load data in memory
    - check the coverage of input/output datasets
    - convert data types to data efficient datasets on disk (complex data)
    """

    def __init__(self, folder, process_name=[], coordinates=[], settings=[], polarisation='', data_id='',
                 json_data='', json_path='', process_meta=''):
        """
        Class that connects a process with the datafiles on disk. This enables easy access of the data files and loading
        in memory of these files in the processing code.

        :param str folder: Folder of dataset. This is not stored in the meta data as processing folders can be moved.
        :param str process_name: Name of the processing methot (for example, crop, resampling, deramping etc.)
        :param CoordinateSystem coordinates: Coordinate system of the output of this processing step.
        :param dict settings: Specific settings for this processing method
        :param str polarisation: Polarisation of the input data. Leave blank if not relevant
        :param str data_id: ID of the processing step, if used in different parts of the processing. Leave blank if not
                relevant
        :param OrderedDict json_data: If this step is read as part of a .json file this is the source data
        :param str json_path: If the process was saved as a .json it will be loaded using this path
        :param ProcessMeta process_meta: If the metadata of this step is already loaded it can be provided here
        """

        # If the the process metadata is already given we add that file as metadata
        self.folder = folder
        if not json_path:
            self.json_path = os.path.join(self.folder, 'info.json')
        else:
            self.json_path = json_path

        if isinstance(process_meta, ProcessMeta):
            self.meta = process_meta
        else:
            self.meta = ProcessMeta(folder, process_name, coordinates, settings, polarisation, data_id, json_data)

        self.dtype_disk, self.dtype_memory, self.dtype_size, self.dtype_gdal, self.dtype_gdal_numpy = ImageData.load_dtypes()

        self.input_files = self.meta.input_files
        self.output_files = self.meta.output_files

        self.images = OrderedDict()
        self.data_disk_meta = self.meta.output_files
        self.data_disk = OrderedDict()
        self.data_memory_meta = OrderedDict()
        self.data_memory = OrderedDict()

        self.coordinates = self.meta.coordinates
        self.process_name = self.meta.process_name
        self.process_id = self.meta.process_id
        self.data_id = self.meta.data_id
        self.polarisation = self.meta.polarisation

        # add settings to memory and disk data.
        self.add_process_images()

    def process_data_exists(self, file_type, data_disk=True):
        """
        Check if data already exists
        
        :param str file_type: Name of file type
        :param bool data_disk: Only returns True when the file exists on disk if True
        :return: 
        """

        file_type = self.check_file_type_exists(file_type)
        if not file_type:
            return False
        else:
            if file_type in self.images.keys():
                if data_disk:
                    return self.images[file_type].check_data_disk_valid()[0]
                else:
                    return True
            else:
                return False

    def process_data_iterator(self, file_types=[], data=True):
        """
        This function is used to retrieve individual images based on chosen file types.

        :param list[str] file_types: Defines which file types are selected. All are selected when left blank.
        :param bool data: Do we want to get descriptive data only, or also the image data itself too? Default is True.
        :return: List of file types, coordinate systems and links to datafiles
        :rtype: tuple[list[str], list[CoordinateSystem], list[ImageData]]
        """

        if len(file_types) == 0:
            file_types = list(self.data_disk_meta.keys())
        else:
            file_types = [self.check_file_type_exists(file_type) for file_type in file_types]
            file_types = [file_type for file_type in file_types if file_type != False]

        images = []
        coordinate_systems = []
        if data:
            for file_type in file_types:
                images.append(self.images[file_type])
                coordinate_systems.append(self.coordinates)

        return file_types, coordinate_systems, images

    def check_file_type_exists(self, file_type):
        """
        Check if a certain file type exists.
        
        :param str file_type: Name of file type 
        :return: 
        """

        file_types = self.data_disk_meta.keys()
        if file_type in file_types:
            return file_type
        else:
            return False

    def load_process_images(self, overwrite=False):
        """
        Load the images which are already part of this processing step. (Only used for initialization)

        :param bool overwrite: Do we overwrite? Only usefull in cases we want to reset the outputs of a processing step.
        :return:
        """

        for key in self.data_disk_meta.keys():
            if key not in list(self.images.keys()) or overwrite:
                self.images[key] = ImageData(self.data_disk_meta[key],
                                             folder=self.folder,
                                             process_name=self.process_name,
                                             coordinates=self.coordinates,
                                             polarisation=self.polarisation,
                                             data_id=self.data_id)
                self.data_disk[key] = self.images[key].data_disk
                self.data_disk_meta[key] = self.images[key].json_dict
                self.data_memory[key] = self.images[key].data_memory
                self.data_memory_meta[key] = self.images[key].data_memory_meta

    def add_process_images(self, file_types=[], data_types='', shapes=[], overwrite=False):
        """

        :param list[str] file_types:
        :param list[str] data_types:
        :param list[tuple] shapes:
        :param bool overwrite:
        :return:
        """

        if not file_types:
            file_types = list(self.data_disk_meta.keys())
            data_types = [self.data_disk_meta[key]['dtype'] for key in file_types]
            shapes = [self.data_disk_meta[key]['shape'] for key in file_types]
        elif not shapes:
            shapes = [self.coordinates.shape for i in range(len(file_types))]

        for file_type, data_type, shape in zip(file_types, data_types, shapes):
            self.add_process_image(file_type, data_type, shape, overwrite)

    # Methods to add new filenames and or remove ones.
    def add_process_image(self, file_type='', data_type='', shape=[], overwrite=False):
        """
        Add output data file that does not exist yet.

        :param str file_type: Name of output processing datafield (for example latitude for geocoding)
        :param str data_type: Datatype of values in dataset (check ImageData class for possibilities)
        :param tuple shape: Shape of output dataset if not the same as the coordinate system.
        :param bool overwrite: Do we overwrite the data if it already existst?
        :return:
        """

        if file_type in list(self.images.keys()) and not overwrite:
            return

        self.images[file_type] = ImageData(file_type=file_type,
                                              dtype=data_type,
                                              shape=shape,
                                              folder=self.folder,
                                              process_name = self.process_name,
                                              coordinates = self.coordinates,
                                              polarisation = self.polarisation,
                                              data_id = self.data_id)
        self.data_disk_meta[file_type] = self.images[file_type].disk['meta']
        self.data_memory_meta[file_type] = self.images[file_type].memory['meta']
        self.data_disk[file_type] = self.images[file_type].disk
        self.data_memory[file_type] = self.images[file_type].memory

    def export_tiff(self, file_type):
        """
        Method to export data files to geotiff format. This step is mainly used as a last part of the processing to
        generate output files which are readable using GIS programs.

        :param str file_type: Name of the file type that is converted to geotiff
        :return:
        """

        file_type = self.check_file_type_exists(file_type)
        if file_type == False:
            return False

        # Get geo transform and projection
        projection, geo_transform = self.coordinates.create_gdal_projection(self)

        # Save data to geotiff (if complex a file with two bands)
        file_name = self.data_disk_meta[file_type]['file_name'][:-4] + '.tiff'
        file_path = os.path.join(self.folder, file_name)
        data_type = self.data_disk_meta[file_type]['data_type']
        data = ImageData.disk2memory(self.data_disk[file_type], data_type)

        # Create an empty geotiff with the right coordinate system
        gtiff_type = self.dtype_gdal[data_type]
        np_type = self.dtype_gdal_numpy[data_type]
        driver = gdal.GetDriverByName('GTiff')

        # For complex numbers
        if data_type in ['complex_int', 'complex_short', 'complex_real4']:
            layers = 2
        else:
            layers = 1

        amp_data = driver.Create(file_path, self.coordinates.shape[1], self.coordinates.shape[0], layers,
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

    # Delegate to process ProcessMeta class describing the
    def update_json(self, save=True, json_path=''):
        self.meta.update_json(save, json_path)

    def load_json(self, json_data='', json_path=''):
        self.meta.load_json(json_data, json_path)

    def split_process_id(self, process_id):
        return self.meta.split_process_id(process_id)
