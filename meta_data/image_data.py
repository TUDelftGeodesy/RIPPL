from collections import OrderedDict

from rippl.meta_data.process_data import ProcessData
from rippl.meta_data.process_meta import ProcessMeta
from rippl.meta_data.image_meta import ImageMeta

'''
This class gathers the data on disk of an image using ProcessData object. The main function is to connect all the 
processes and makes it possible to search the processes done for any image.
'''

class ImageData(object):

    def __init__(self, image_meta='', path='', overwrite=False):

        if isinstance(image_meta, ImageMeta):
            self.meta = image_meta
        else:
            self.meta = ImageMeta(path, overwrite)

        self.path = self.meta.path
        self.reference_paths = self.meta.reference_paths
        self.orbits = self.meta.orbits
        self.processes = self.meta.processes
        self.readfiles = self.meta.readfiles

        self.processes_data = OrderedDict()
        self.sync_memory_disk_files()

        self.data_disk_meta = OrderedDict()
        self.data_disk = OrderedDict()
        self.data_memory_meta = OrderedDict()
        self.data_memory = OrderedDict()

    def sync_memory_disk_files(self):
        # Create process data information from the process data objects.
        for process_key in self.meta.processes.keys():
            if process_key not in self.processes_data.keys():
                self.processes_data[process_key] = OrderedDict()

            for id_key in self.processes_data[process_key].keys():
                self.processes_data[process_key][id_key] = ProcessData(self.meta.processes[process_key][id_key])

            self.processes_data[process_key][id_key].sync_memory_disk_files()

            # Extract information for the image data object
            self.data_disk_meta[process_key][id_key] = self.processes_data[process_key][id_key].data_disk_meta
            self.data_disk[process_key][id_key] = self.processes_data[process_key][id_key].data_disk
            self.data_memory_meta[process_key][id_key] = self.processes_data[process_key][id_key].data_memory_meta
            self.data_memory[process_key][id_key] = self.processes_data[process_key][id_key].data_memory

    def add_process(self, process):
        # Directly linked to the add process step in the image data object.

        # If it is not a ProcessData file
        if isinstance(process, ProcessMeta):
            self.meta.add_process(process)
        elif isinstance(process, ProcessData):
            self.meta.add_process(process.meta)
        self.sync_memory_disk_files()

    def check_process_exist(self, process_name, coordinates, data_id='', polarisation='', process_id=''):

        if not process_id:
            process_id = process_name + '_#coor#_' + coordinates.short_id_str + '_#id#_' + data_id + \
                         '_#pol#_' + polarisation
        if process_name in self.processes_data.keys():
            if process_id in self.processes_data[process_name].keys():
                return True, process_id
        return False, process_id

    def select_process(self, process_name, coordinates, data_id='', polarisation='', process_id=''):
        # Select a specific process to read or write data.
        exist, process_id = self.check_process_exist(process_name, coordinates, data_id, polarisation, process_id)

        if exist:
            return self.processes_data[process_name][process_id]
        else:
            return False

    def data_file_iterator(self):

        process_keys = []
        id_keys = []
        file_types = []

        # Get the individual processes.
        for process_key in self.processes_data.keys():
            for id_key in self.processes_data[process_key].keys():

                # Get the process output files
                for file_type in self.processes_data[process_key][id_key].data_disk.keys():
                    process_keys.append(process_key)
                    id_keys.append(id_key)
                    file_types.append(file_type)

        return process_keys, id_keys, file_types

    def load_memmap_files(self):
        # Load all memmaps files of this image.
        process_keys, id_keys, file_types = self.data_file_iterator()
        for process_key, id_key, file_type in zip(process_keys, id_keys, file_types):
            self.processes_data[process_key][id_key].load_disk_data(file_type)

    def remove_memmap_files(self):
        # Load all memmaps files of this image.
        process_keys, id_keys, file_types = self.data_file_iterator()
        for process_key, id_key, file_type in zip(process_keys, id_keys, file_types):
            self.processes_data[process_key][id_key].remove_disk_data_memmap(file_type)
