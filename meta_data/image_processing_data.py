from collections import OrderedDict
import numpy as np
import os

from rippl.meta_data.process_data import ProcessData
from rippl.meta_data.process_meta import ProcessMeta
from rippl.meta_data.orbit import Orbit
from rippl.meta_data.readfile import Readfile
from rippl.meta_data.image_processing_meta import ImageProcessingMeta
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ImageProcessingData(object):

    def __init__(self, folder, overwrite=False, json_path='', image_processing_meta=''):
        """
        This class gathers the data on disk of an image using ProcessData object. The main function is to connect all
        the processes and makes it possible to search the processes done for any image.

        :param str folder: Location of processes for this image.
        :param bool overwrite: Do we overwrite existing metadata files? Default is False
        :param str json_path: Path to json file to read meta data. If empty we assume it is folder + info.json
        :param ImageProcessingMeta image_processing_meta: If the meta file is already loaded it can also be added as an
                    ImageProcessingMeta object
        """

        self.folder = folder
        if not json_path:
            self.json_path = os.path.join(self.folder, 'info.json')
        else:
            self.json_path = json_path
        self.json_path = os.path.join(self.folder, 'info.json')

        if isinstance(image_processing_meta, ImageProcessingMeta):
            self.meta = image_processing_meta
        else:
            self.meta = ImageProcessingMeta(folder, overwrite)

        self.folder = self.meta.folder                      # type: str
        self.reference_paths = self.meta.reference_paths    # type: OrderedDict(str)
        self.orbits = self.meta.orbits                      # type: OrderedDict(Orbit)
        self.processes = self.meta.processes                # type: OrderedDict(OrderedDict(ProcessMeta))
        self.readfiles = self.meta.readfiles                # type: OrderedDict(Readfile)

        self.processes_data = OrderedDict()                 # type: OrderedDict(OrderedDict(ProcessData))

        self.data_disk_meta = OrderedDict()                 # type: OrderedDict(OrderedDict(OrderedDict()))
        self.data_disk = OrderedDict()                      # type: OrderedDict(OrderedDict(np.memmap))
        self.data_memory_meta = OrderedDict()               # type: OrderedDict(OrderedDict(OrderedDict()))
        self.data_memory = OrderedDict()                    # type: OrderedDict(OrderedDict(np.ndarray))

        self.load_processing_data()

    def load_processing_data(self):
        """
        Create processing data information from the individual process data objects.

        :return:
        """

        for process_key in self.meta.processes.keys():
            # Check if this type of process data already exists.
            if process_key not in self.processes_data.keys():
                self.processes_data[process_key] = OrderedDict()

            for id_key in self.meta.processes[process_key].keys():
                # Check if it is already load as a ProcessData object
                if id_key not in list(self.processes_data[process_key].keys()):
                    new_process = True
                elif not isinstance(self.processes_data[process_key][id_key], ProcessData):
                    new_process = True
                else:
                    new_process = False

                if new_process:
                    self.processes_data[process_key][id_key] = ProcessData(process_meta=self.meta.processes[process_key][id_key], folder=self.folder)
                self.load_process_data_info(process_key, id_key)

    def load_process_data_info(self, process_key, id_key):
        # Synchronize meta data of image.

        if process_key not in self.data_disk.keys():
            self.data_disk_meta[process_key] = OrderedDict()
            self.data_disk[process_key] = OrderedDict()
            self.data_memory_meta[process_key] = OrderedDict()
            self.data_memory[process_key] = OrderedDict()

        # If the images are not yet loaded, load them as ImageData objects.
        self.processes_data[process_key][id_key].load_process_images()

        # Extract information for the image data object
        self.data_disk_meta[process_key][id_key] = self.processes_data[process_key][id_key].data_disk_meta
        self.data_disk[process_key][id_key] = self.processes_data[process_key][id_key].data_disk
        self.data_memory_meta[process_key][id_key] = self.processes_data[process_key][id_key].data_memory_meta
        self.data_memory[process_key][id_key] = self.processes_data[process_key][id_key].data_memory

    def add_process(self, process):
        """
        Directly linked to the add process step in the image data object.

        :param ProcessData or ProcessMeta process: Input process to add to the image processing data
        :return:
        """

        # If it is not a ProcessData file
        if isinstance(process, ProcessMeta):
            self.meta.add_process(process)
        elif isinstance(process, ProcessData):
            self.meta.add_process(process.meta)
            if process.process_name not in self.processes_data.keys():
                self.processes_data[process.process_name] = OrderedDict()
            self.processes_data[process.process_name][process.process_id] = process

        # Synchronize the image oversight
        self.load_process_data_info(process.process_name, process.process_id)

    def processing_image_data_exists(self, process, coordinates, in_coordinates='', data_id='', polarisation='',
                                     file_type='', data=True, message=True, multiple=True):
        """
        Check if a specific image exists and load it. If more than one image is selected, give a warning.

        :param str process: Name process
        :param CoordinateSystem coordinates: Coordinate system
        :param CoordinateSystem in_coordinates: Coordinate system of input processing step
        :param str data_id: ID of dataset
        :param str polarisation: Polarisation of the image
        :param str file_type: Output file type within the processing step
        :return: ImageData object
        """

        images = self.processing_image_data_iterator([process], [coordinates], [in_coordinates], [data_id],
                                                     [polarisation], [file_type], data=data)[-1]
        if len(images) == 0:
            if message:
                print('No image data found')
            return False
        elif len(images) > 1 and not multiple:
            if message:
                print('Search criterea for image selection not specific enough. More than one image selected. Note that if '
                      'data_id or polarisation should be empty use "none" as keyword.')
            return False
        else:
            return images[0]

    def processing_image_data_iterator(self, processes=[], coordinates=[], in_coordinates=[], data_ids=[],
                                       polarisations=[], file_types=[], data=True):
        """
        This function find all the ImageData objects in this processing image that fullfill the set requirements in the
        inputs. If inputs are left blank, this parameter is not taken into account.

        :param list(str) processes: Name of processes to select
        :param list(CoordinateSystem) coordinates: Coordinate systems to select
        :param list(CoordinateSystem) in_coordinates: Input coordinate systems to select
        :param list(str) data_ids: Name of IDs for selection
        :param list(str) polarisations: Types of polarisation to select
        :param list(str) file_types: File types to select
        :param bool data:
        :return:
        """

        # get the coordinate ids, data_ids and polarisations
        if len(processes) == 0:
            processes = list(self.data_disk_meta.keys())

        coor_strs = []
        for coordinate in coordinates:
            coordinate.create_short_coor_id()
            coor_strs.append(coordinate.short_id_str)
        in_coor_strs = []
        for coordinate in in_coordinates:
            if isinstance(coordinate, CoordinateSystem):
                coordinate.create_short_coor_id()
                in_coor_strs.append(coordinate.short_id_str)

        processes_out = []
        process_ids_out = []
        coordinates_out = []
        in_coordinates_out = []
        file_types_out = []
        images_out = []

        for process_name in processes:
            if process_name not in self.data_disk_meta.keys():
                continue

            # If it does exist.
            process_ids = self.data_disk_meta[process_name].keys()
            for process_id in process_ids:
                proc, coor_str, in_coor_str, id_str, pol_str = ProcessMeta.split_process_id(process_id)

                if len(coor_strs) > 0 and coor_strs != ['']:
                    if coor_str not in coor_strs:
                        continue
                if len(in_coor_strs) > 0 and in_coor_strs != ['']:
                    if in_coor_str not in in_coor_strs:
                        continue
                if len(data_ids) > 0 and data_ids != ['']:
                    if id_str not in data_ids:
                        continue
                if len(polarisations) > 0 and polarisations != ['']:
                    if pol_str not in polarisations:
                        continue

                file_types_process, coordinate_systems_process, in_coordinate_systems_process, images_process = \
                    self.processes_data[process_name][process_id].process_data_iterator(file_types, data)

                # All the same for every file type
                processes_out += [process_name for i in range(len(file_types_process))]
                process_ids_out += [process_id for i in range(len(file_types_process))]

                file_types_out += file_types_process
                coordinates_out += coordinate_systems_process
                in_coordinates_out += in_coordinate_systems_process
                if data:
                    images_out += images_process

        return processes_out, process_ids_out, coordinates_out, in_coordinates_out, file_types_out, images_out

    def all_data_iterator(self):
        """
        # Get all ImageData objects available

        :return: All image data objects
        """

        processes_out, process_ids_out, coordinates_out, in_coordinates_out, file_types_out, images_out = \
            self.processing_image_data_iterator([], [], [], [], [])

        return processes_out, process_ids_out, coordinates_out, in_coordinates_out, file_types_out, images_out

    def check_process_exist(self, process_name, coordinates, in_coordinates='', data_id='', polarisation='',
                            process_id=''):
        """
        Check if a certain process exists.

        :param str process_name: Name of process
        :param CoordinateSystem coordinates: Coordinate system of process
        :param CoordinateSystem in_coordinates: Input coordinate system of process
        :param str data_id: ID of data set
        :param str polarisation: Polarisation of data
        :param str process_id: You can give also a process id directly. Generally left blank.
        :return:
        """

        if not process_id:
            process_id = ProcessMeta.create_process_id(process_name, coordinates, in_coordinates, data_id, polarisation)
        if process_name in self.processes_data.keys():
            if process_id in self.processes_data[process_name].keys():
                return True, process_id
        return False, process_id

    def select_process(self, process_name, coordinates, data_id='', polarisation='', process_id=''):
        """
        Select a specific process to read or write data.

        :param str process_name: Name of process
        :param CoordinateSystem coordinates: Process coordinate system
        :param str data_id: ID of process (often empty)
        :param str polarisation: polarisation of process, if any
        :param str process_id: Direct input of process id if known. Can be left blank if other parameters are given.
        :return:
        """

        exist, process_id = self.check_process_exist(process_name, coordinates, data_id, polarisation, process_id)

        if exist:
            return self.processes_data[process_name][process_id]
        else:
            return False

    def load_memmap_files(self):
        """
        Load all memmaps files of this image.

        :return:
        """
        processes, process_ids, coordinates, in_coordinates, file_types, images = self.all_data_iterator()
        for image in images:
            image.load_disk_data()

    def remove_memmap_files(self):
        """
        Remove all memmaps files of this image.

        :return:
        """
        processes, process_ids, coordinates, in_coordinates, file_types, images = self.all_data_iterator()
        for image in images:
            image.remove_disk_data_memmap()

    def remove_memory_files(self):
        """
        Remove all memory data of this image.

        :return:
        """
        processes, process_ids, coordinates, in_coordinates, file_types, images = self.all_data_iterator()
        for image in images:
            image.remove_memory_data()

    # Delegate functions to image processing meta object.
    def update_json(self, json_path=''):
        self.meta.update_json(json_path)

    def load_json(self, json_path=''):
        self.meta.load_json(json_path)

    def process_id_exist(self, process_id):
        # Checks if a process id exists
        return self.meta.process_id_exist(process_id)

    def find_best_orbit(self, orbit_type='original'):
        # Find best orbits
        return self.meta.find_best_orbit(orbit_type)
