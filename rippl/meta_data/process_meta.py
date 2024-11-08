# This object defines a run process for a specific image.
import json
from collections import OrderedDict
import os
import logging

from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.orbit import Orbit
from rippl.meta_data.readfile import Readfile


class ProcessMeta(object):

    def __init__(self, folder, process_name=[], coordinates=[], in_coordinates=[], settings=[], polarisation='', data_id='',
                 json_data='', json_path=''):

        self.folder = folder
        if not json_path:
            self.json_path = os.path.join(self.folder, os.path.basename(self.folder) + '.json')
        else:
            self.json_path = json_path
        self.process_name = ''

        self.in_coordinates = []
        self.coordinates = []
        self.data_id = ''
        self.polarisation = ''
        self.settings = []

        self.json_dict = OrderedDict()

        if json_data != '' or json_path != '':
            self.load_json(json_data, json_path)
        else:
            if not isinstance(coordinates, CoordinateSystem):
                logging.info('variable coordinates should be an CoordinateSystem object')
                return

            self.process_name = process_name
            self.coordinates = coordinates
            self.in_coordinates = in_coordinates
            self.polarisation = polarisation
            self.data_id = data_id

            # Both input and output files are dicts that come with their coor_id and file_path
            self.input_files = OrderedDict()
            self.output_files = OrderedDict()

            # Specific settings for this step. You can label specific settings using a data_id. This will allow you to
            # run the same code with different settings in parallel.
            self.settings = settings

            # Data from readfiles and orbit. This information is usefull for a number of processing steps.
            # This data is only added in case they are used for the processing. (e.g. deramping/reramping, geocoding or geometrical coreg)
            self.orbits = OrderedDict()
            self.readfiles = OrderedDict()

        # Create the processing ID
        self.coordinates.create_short_coor_id()
        if isinstance(self.in_coordinates, CoordinateSystem):
            self.in_coordinates.create_short_coor_id()
        self.process_id = self.create_process_id(self.process_name, self.coordinates, self.in_coordinates, self.data_id,
                                                 self.polarisation)

    @staticmethod
    def create_process_id(process_name, coordinates, in_coordinates, data_id, polarisation):

        process_id = process_name
        if len(coordinates.short_id_str) > 0:
            process_id += ('_#coor#_' + coordinates.short_id_str)
        if isinstance(in_coordinates, CoordinateSystem):
            in_coordinates.create_short_coor_id()
            if len(in_coordinates.short_id_str) > 0:
                process_id += ('_#in_coor#_' + in_coordinates.short_id_str)
        if data_id:
            process_id += '_#id#_' + data_id
        if polarisation:
            process_id += '_#pol#_' + polarisation

        return process_id

    def update_json(self, save_orbits=False, save_readfiles=False):
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
        self.json_dict['process_settings'] = self.settings

        self.json_dict['process_name'] = self.process_name
        self.json_dict['data_id'] = self.data_id
        self.json_dict['polarisation'] = self.polarisation

        self.coordinates.update_json()
        self.json_dict['coordinates'] = self.coordinates.json_dict

        if isinstance(self.in_coordinates, CoordinateSystem):
            self.json_dict['in_coordinates'] = self.in_coordinates.json_dict
        else:
            self.json_dict['in_coordinates'] = []

        return self.json_dict

    def save_json(self, json_path, save_orbits=False, save_readfiles=False):
        # Save json file
        self.update_json(save_orbits, save_readfiles)

        if json_path:
            with open(json_path, 'w+') as file:
                try:
                    json.dumps(self.json_dict)
                except Exception as e:
                    raise ValueError('Part of the .json file is not json serializable. Make sure that the processing'
                                     'step settings only accept dictionaries with regular int, float or string values. ' + str(e))
                json.dump(self.json_dict, file, indent=3)

    def load_json(self, json_data='', json_path=''):
        # Load json data

        if isinstance(json_data, OrderedDict):
            self.json_dict = json_data
        else:
            with open(json_path) as file:
                self.json_dict = json.load(file, object_pairs_hook=OrderedDict)

        self.output_files = self.json_dict['output_files']
        self.input_files = self.json_dict['input_files']

        self.process_name = self.json_dict['process_name']
        self.polarisation = self.json_dict['polarisation']
        self.data_id = self.json_dict['data_id']
        self.settings = self.json_dict['process_settings']

        self.readfiles = OrderedDict()
        for readfile_key in self.json_dict['readfiles'].keys():
            self.readfiles[readfile_key] = Readfile(json_data=self.json_dict['readfiles'][readfile_key])

        self.orbits = OrderedDict()
        for orbit_key in self.json_dict['orbits'].keys():
            self.readfiles[readfile_key] = Orbit(json_data=self.json_dict['orbits'][orbit_key])

        self.coordinates = CoordinateSystem(json_data=self.json_dict['coordinates'])
        self.coordinates.create_short_coor_id()
        self.coordinates.create_coor_id()
        if len(self.json_dict['in_coordinates']) > 0:
            self.in_coordinates = CoordinateSystem(json_data=self.json_dict['in_coordinates'])
            self.in_coordinates.create_short_coor_id()
            self.in_coordinates.create_coor_id()

    @staticmethod
    def split_process_id(process_id):

        strs = (process_id + '_').split('#')

        process_name = strs[0][:-1]
        coordinates_str = strs[strs.index('coor') + 1][1:-1]
        if 'in_coor' in strs:
            in_coordinates_str = strs[strs.index('in_coor') + 1][1:-1]
        else:
            in_coordinates_str = 'none'
        if 'id' in strs:
            data_id = strs[strs.index('id') + 1][1:-1]
        else:
            data_id = 'none'
        if 'pol' in strs:
            polarisation = strs[strs.index('pol') + 1][1:-1]
        else:
            polarisation = 'none'

        return process_name, coordinates_str, in_coordinates_str, data_id, polarisation
