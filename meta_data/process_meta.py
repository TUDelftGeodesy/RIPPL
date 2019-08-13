# This object defines a run process for a specific image.
import json
from collections import OrderedDict
import os

from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ProcessMeta():

    def __init__(self, file_path='', coordinates=[], process_name=[], settings=[], output_files=[], input_files='',
                 polarisation='', data_id='', json_data='', json_path=''):
        
        self.file_path = file_path
        self.process_name = ''

        self.output_files = []
        self.input_files = []

        self.coordinates = []
        self.data_id = ''
        self.polarisation = ''
        self.process_settings = []

        self.json_dict = OrderedDict()

        if json_data != '' or json_path != '':
            self.load_json(json_data, self.file_path)
            return

        if not isinstance(coordinates, CoordinateSystem):
            print('variable coordinates should be an CoordinateSystem object')
            return
        if input_files == '':
            input_files = OrderedDict()
        if not isinstance(output_files, dict) or not isinstance(input_files, dict):
            print('input and output files should be dictionaries')
            return

        self.process_name = process_name
        self.coordinates = coordinates
        if not polarisation:
            self.polarisation = 'not_defined'
        else:
            self.polarisation = polarisation
        if not data_id:
            self.data_id = 'none'
        else:
            self.data_id = data_id

        # Both input and output files are dicts that come with their coor_id and file_path
        self.input_files = input_files
        self.output_files = output_files

        # Specific settings for this step. You can label specific settings using a data_id. This will allow you to
        # run the same code with different settings in parallel.
        self.process_settings = settings

        # Data from readfiles and orbit. This information is usefull for a number of processing steps.
        # This data is only added in case they are used for the processing. (e.g. deramping/reramping, geocoding or geometrical coreg)
        self.json_dict['orbits'] = OrderedDict()
        self.orbits = self.json_dict['orbits']
        self.json_dict['readfiles'] = OrderedDict()
        self.readfiles = self.json_dict['readfiles']

        # Create the processing ID
        self.coor_id = self.coordinates.id_str
        self.process_id = self.process_name + '_#coor#_' + self.coordinates.short_id_str + '_#id#_' + self.data_id + '_#pol#_' + self.polarisation

    def update_json(self, save=True, json_path=''):
        # Update json data dict with current data.

        self.json_dict['output_files'] = self.output_files
        self.json_dict['input_files'] = self.input_files

        self.json_dict['process_name'] = self.process_name
        self.json_dict['process_settings'] = self.process_settings
        self.json_dict['data_id'] = self.data_id
        self.json_dict['polarisation'] = self.polarisation

        self.coordinates.update_json()
        self.json_dict['coordinates'] = self.coordinates.json_dict
        
        if save:
            if json_path:
                self.json_dict['file_path'] = json_path
                json.dump(json_path, self.json_dict)
            else:
                self.json_dict['file_path'] = self.file_path
                json.dump(self.file_path, self.json_dict)

    def load_json(self, json_data='', json_path=''):
        # Load json data
        if isinstance(json_data, OrderedDict):
            self.json_dict = json_data
        elif json_path:
            self.file_path = json_path
            self.json_dict = json.load(json_path, object_pairs_hook=OrderedDict)
            self.json_dict['file_path'] = self.file_path
        else:
            self.json_dict = json.load(self.file_path, object_pairs_hook=OrderedDict)

        self.output_files = self.json_dict['output_files']
        self.input_files = self.json_dict['input_files']

        self.process_name = self.json_dict['process_name']
        self.process_settings = self.json_dict['process_settings']
        self.polarisation = self.json_dict['polarisation']
        self.data_id = self.json_dict['data_id']

        self.coordinates = CoordinateSystem(json_data=self.json_dict['coordinates'])

    def create_processing_filenames(self, orig_file_list=[], process_types=[], data_types=[], shapes=[]):
        # In case of multiple process_types and filenames.

        if isinstance(orig_file_list, OrderedDict):
            file_list = orig_file_list

        for process_type, data_type in zip(process_types, data_types):
            file_meta, key = self.create_processing_filename(process_type, data_type)
            file_list[key] = file_meta

        return file_list

    def create_processing_filename(self, process_type='', data_type='', shape=[]):
        # This code generates a filename dict using coordinates process_type and processing step.
        # This can either be used for the input files or output files.
        # If not data type is provided we assume a float32 as default
        # The shape variable only has to be defined when it is not the same as the coordinate system.

        output_file = OrderedDict()
        file_name = self.get_filename(process_type)

        key_file = file_name
        output_file['file_path'] = os.path.join(os.path.basename(self.file_path, file_name + '.raw'))

        # Only used when accessing files
        output_file['file_exist'] = None
        output_file['file_valid'] = None

        if not data_type:
            output_file['data_type'] = None
        else:
            output_file['data_type'] = data_type
        if len(shape) != 2:
            output_file['shape'] = None
        else:
            output_file['shape'] = shape

        return  output_file, key_file

    def get_filename(self, process_type):
        # Get the filename for using id/coordinates/polarisation
        
        if not process_type:
            file_name = self.process_name + '_' + self.coordinates.short_id_str
        else:
            file_name = process_type + '_' + self.coordinates.short_id_str

        if self.data_id != 'none':
            file_name = file_name + '_' + self.data_id
        if self.polarisation != 'not_defined':
            file_name = file_name + '_' + self.polarisation
            
        return file_name