# This object defines a run process for a specific image.
import json
from collections import OrderedDict
import os

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.orbit import Orbit
from rippl.meta_data.readfile import Readfile


class ProcessMeta():

    def __init__(self, folder, process_name=[], coordinates=[], settings=[], polarisation='', data_id='',
                 json_data='', json_path=''):

        self.folder = folder
        if not json_path:
            self.json_path = os.path.join(self.folder, 'info.json')
        else:
            self.json_path = json_path
        self.process_name = ''

        self.coordinates = []
        self.data_id = ''
        self.polarisation = ''
        self.process_settings = []

        self.json_dict = OrderedDict()

        if json_data != '' or json_path != '':
            self.load_json(json_data, json_path)
        else:
            if not isinstance(coordinates, CoordinateSystem):
                print('variable coordinates should be an CoordinateSystem object')
                return

            self.process_name = process_name
            self.coordinates = coordinates
            if not polarisation:
                self.polarisation = 'none'
            else:
                self.polarisation = polarisation
            if not data_id:
                self.data_id = 'none'
            else:
                self.data_id = data_id

            # Both input and output files are dicts that come with their coor_id and file_path
            self.input_files = OrderedDict()
            self.output_files = OrderedDict()

            # Specific settings for this step. You can label specific settings using a data_id. This will allow you to
            # run the same code with different settings in parallel.
            self.process_settings = settings

            # Data from readfiles and orbit. This information is usefull for a number of processing steps.
            # This data is only added in case they are used for the processing. (e.g. deramping/reramping, geocoding or geometrical coreg)
            self.orbits = OrderedDict()
            self.readfiles = OrderedDict()

        # Create the processing ID
        self.coordinates.create_short_coor_id()
        self.coor_id = self.coordinates.id_str
        self.process_id = self.process_name + '_#coor#_' + self.coordinates.short_id_str + '_#id#_' + self.data_id + '_#pol#_' + self.polarisation

    def update_json(self, json_path='', save_orbits=False, save_readfiles=False):
        # Update json data dict with current data.

        # Combine everything in a json_dict
        self.json_dict['readfiles'] = OrderedDict()
        if save_readfiles:
            for readfile_key in self.readfiles.keys():
                self.json_dict['readfiles'][readfile_key] = self.readfiles[readfile_key].update_json()

        self.json_dict['orbits'] = OrderedDict()
        if save_orbits:
            for orbit_key in self.orbits.keys():
                self.json_dict['orbits'][orbit_key] = self.orbits[orbit_key].update_json()

        self.json_dict['output_files'] = self.output_files
        self.json_dict['input_files'] = self.input_files

        self.json_dict['process_name'] = self.process_name
        self.json_dict['process_settings'] = self.process_settings
        self.json_dict['data_id'] = self.data_id
        self.json_dict['polarisation'] = self.polarisation

        self.coordinates.update_json()
        self.json_dict['coordinates'] = self.coordinates.json_dict
        
        if json_path:
            file = open(json_path, 'w+')
            json.dump(self.json_dict, file, indent=3)
            file.close()

        return self.json_dict

    def load_json(self, json_data='', json_path=''):
        # Load json data

        if isinstance(json_data, OrderedDict):
            self.json_dict = json_data
        else:
            file = open(json_path)
            self.json_dict = json.load(file, object_pairs_hook=OrderedDict)
            file.close()

        self.output_files = self.json_dict['output_files']
        self.input_files = self.json_dict['input_files']

        self.process_name = self.json_dict['process_name']
        self.process_settings = self.json_dict['process_settings']
        self.polarisation = self.json_dict['polarisation']
        self.data_id = self.json_dict['data_id']

        self.readfiles = OrderedDict()
        for readfile_key in self.json_dict['readfiles'].keys():
            self.readfiles[readfile_key] = Readfile(json_data=self.json_dict['readfiles'][readfile_key])

        self.orbits = OrderedDict()
        for orbit_key in self.json_dict['orbits'].keys():
            self.readfiles[readfile_key] = Orbit(json_data=self.json_dict['orbits'][orbit_key])

        self.coordinates = CoordinateSystem(json_data=self.json_dict['coordinates'])

    @staticmethod
    def split_process_id(process_id):

        strs = process_id.split('#')

        process_name = strs[0][:-1]
        coordinates_str = strs[2][1:-1]
        data_id = strs[4][1:-1]
        polarisation = strs[6][1:]

        return process_name, coordinates_str, data_id, polarisation
