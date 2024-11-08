"""
This class combines information for different processing steps, used orbit and radar coordinate systems for one SAR
data file.

"""
import json
from collections import OrderedDict
import datetime
import os
import copy
import logging

from rippl.meta_data.process_meta import ProcessMeta
from rippl.meta_data.orbit import Orbit
from rippl.meta_data.readfile import Readfile
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ImageProcessingMeta:

    def __init__(self, folder, overwrite=False, json_path=''):

        self.folder = folder                     # type: bool
        if not json_path:
            new_json_path = os.path.join(self.folder, os.path.basename(self.folder) + '.json')
            old_json_path = os.path.join(self.folder, 'info.json')      # For backward compatability
            if os.path.exists(old_json_path):
                self.json_path = old_json_path
            else:
                self.json_path = new_json_path
        else:
            self.json_path = json_path
        self.reference_paths = OrderedDict()  # Path to reference, primary, secondary images.
        self.radar_coordinates = OrderedDict()

        if os.path.exists(self.json_path) and overwrite == False:
            self.load_json(self.json_path)
            return

        self.header = OrderedDict()
        self.process_control = OrderedDict()
        self.readfiles = OrderedDict()
        self.orbits = OrderedDict()
        self.processes = OrderedDict()

        # The process control give a summary of all applied steps.
        self.process_control = OrderedDict([('readfiles', []), ('orbits', []), ('processes', OrderedDict())])

        self.json_dict = OrderedDict()
        self.json_dict['header'] = self.header
        self.json_dict['process_control'] = self.process_control

    def load_radar_coordinates(self):
        """
        This function creates radar coordinates from the orbit and readfile data
        """

        for key in ['original', 'primary', 'secondary', 'reference']:

            if key in self.readfiles.keys():
                readfile = self.readfiles[key]
                orbit = self.find_best_orbit(orbit_type=key)

                coor = CoordinateSystem()
                coor.create_radar_coordinates()
                coor.load_orbit(orbit)
                coor.load_readfile(readfile, radar_grid_type=key + '_slc')
                coor.create_short_coor_id()

                assigned = False
                if key == 'original':
                    # Loop over all the processes to find a dataset where a radar grid is already generated, to get
                    # the right first_line, first_pixel and shape for this image. Generally this
                    for process in ['crop', 'deramp', 'resample', 'reramp', 'earth_topo_phase']:
                        if process in list(self.processes.keys()):
                            if assigned:
                                continue

                            for process_key in self.processes[process].keys():
                                if coor.short_id_str == self.processes[process][process_key].coordinates.short_id_str:
                                    radar_coor = self.processes[process][process_key].coordinates

                                    coor.shape = radar_coor.shape
                                    coor.first_line = radar_coor.first_line
                                    coor.first_pixel = radar_coor.first_pixel
                                    assigned = True
                                    break

                self.radar_coordinates[key] = coor

    def add_process_meta(self, process):

        if not isinstance(process, ProcessMeta):
            logging.info('Any added processing step should be a Process object')
            return

        if process.process_name not in self.processes.keys():
            self.processes[process.process_name] = OrderedDict()
            self.process_control['processes'][process.process_name] = []

        self.processes[process.process_name][process.process_id] = process
        self.process_control['processes'][process.process_name].append(process.process_id)

    def add_orbit(self, orbit, orbit_type='original'):

        if orbit_type not in ['original', 'primary', 'secondary', 'reference']:
            logging.info('Type of orbit should be either original, primary, secondary or reference. Default to image data now')
            orbit_type = orbit.date

        if not isinstance(orbit, Orbit):
            logging.info('Any added orbit should be an Orbit object')
            return

        orbit_id = orbit_type + '_#type#_' + orbit.orbit_type
        self.orbits[orbit_id] = orbit
        self.process_control['orbits'].append(orbit_id)

    def add_readfile(self, readfile, readfile_type='original'):

        if readfile_type not in ['original', 'primary', 'secondary', 'reference']:
            logging.info('Type of readfile should be either original, primary, secondary or reference')

        if not isinstance(readfile, Readfile):
            logging.info('Any added readfile should be an Readfile object')
            return

        self.readfiles[readfile_type] = readfile
        self.process_control['readfiles'].append(readfile.date)

    def dump_process(self, process, coordinates, in_coordinates='', data_id='', polarisation=''):
        # This removes one of the processes and their respective datafiles.

        if not isinstance(coordinates, CoordinateSystem):
            logging.info('coordinates should be an Coordinatesystem object!')
            return

        process_id = ProcessMeta.create_process_id(process, coordinates, in_coordinates, data_id, polarisation)

        if process in self.process_control.keys():
            if process_id in self.process_control[process]:
                self.process_control['processes'][process].remove(process_id)
                self.processes[process].pop(process_id)
            else:
                logging.info('Cannot remove process. The process exists but not with this coordinate system, id or polarisation')
        else:
            logging.info('Process does not exist')

    def dump_orbit(self, date, orbit_type):
        # This removes one of the orbits from the metadata file

        orbit_id = date + '_#type#_' + orbit_type

        if orbit_id in self.process_control['orbits'].keys():
            self.process_control['orbits'].remove(orbit_id)
            self.processes['orbits'].pop(orbit_id)
        else:
            logging.info('Cannot remove orbit. Orbit date or type does not exist.')

    def dump_readfile(self, date):
        # This removes one of the readfiles from the metadata file

        if date in self.process_control['readfiles'].keys():
            self.process_control['readfiles'].remove(date)
            self.processes['readfiles'].pop(date)
        else:
            logging.info('Cannot remove readfile. Readfile date does not exist.')

    def create_header(self):
        # Create the header for a new json file.

        header = dict()
        header['processor'] = 'RIPPL version 2.0 Beta'
        header['processor source'] = '[url]'
        header['created_by'] = 'Gert Mulder'
        header['creation_date'] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")
        header['last_date_changed'] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")

    def update_json(self):
        # Only the header file has to be changed. Other processes are changed within their respective classes.

        # Combine everything in a json_dict
        self.json_dict['readfiles'] = OrderedDict()
        for readfile_key in self.readfiles.keys():
            self.json_dict['readfiles'][readfile_key] = self.readfiles[readfile_key].update_json()

        self.json_dict['orbits'] = OrderedDict()
        for orbit_key in self.orbits.keys():
            self.json_dict['orbits'][orbit_key] = self.orbits[orbit_key].update_json()

        self.json_dict['processes'] = OrderedDict()
        for process_key in self.processes.keys():
            self.json_dict['processes'][process_key] = OrderedDict()
            for process_type_key in self.processes[process_key].keys():
                self.json_dict['processes'][process_key][process_type_key] = self.processes[process_key][process_type_key].update_json()

        self.json_dict['header']['last_date_changed'] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")

    def save_json(self, json_path='', update=True):
        # Update and save the json file.
        if update:
            self.update_json()

        if json_path:
            file_path = json_path
        else:
            file_path = os.path.join(self.folder, os.path.basename(self.folder) + '.json')
        with open(file_path, 'w+') as file:
            try:
                json.dumps(self.json_dict)
            except:
                raise ValueError('Part of the .json file is not json serializable. Make sure that the processing'
                                 'step settings only accept dictionaries with regular int, float or string values.')
            json.dump(self.json_dict, file, indent=3)

    def get_json(self, json_path=''):
        # Update and get json file.
        self.update_json()

        if json_path:
            return copy.deepcopy(self.json_dict), json_path
        else:
            return copy.deepcopy(self.json_dict), os.path.join(self.folder, os.path.basename(self.folder) + '.json')

    def load_json(self, json_path=''):
        # Load json data from file.

        if json_path:
            file_path = json_path
        else:
            file_path = os.path.join(self.folder, os.path.basename(self.folder) + '.json')

        try:
            with open(file_path, 'r') as json_file:
                self.json_dict = json.load(json_file, object_pairs_hook=OrderedDict)
        except Exception as e:
            raise ImportError('Not able to load ' + os.path.join(self.folder, os.path.basename(self.folder) + '.json') + ' as .json info file. ' + str(e))

        # Load processes, readfiles and orbits
        self.header = self.json_dict['header']

        self.readfiles = OrderedDict()
        for readfile_key in self.json_dict['readfiles'].keys():
            self.readfiles[readfile_key] = Readfile(json_data=self.json_dict['readfiles'][readfile_key])

        self.orbits = OrderedDict()
        for orbit_key in self.json_dict['orbits'].keys():
            self.orbits[orbit_key] = Orbit(json_data=self.json_dict['orbits'][orbit_key])

        self.processes = OrderedDict()
        for process_key in self.json_dict['processes'].keys():
            self.processes[process_key] = OrderedDict()
            for process_id in self.json_dict['processes'][process_key].keys():
                self.processes[process_key][process_id] = ProcessMeta(self.folder, json_data=self.json_dict['processes'][process_key][process_id])

        # Load process control
        self.process_control = self.json_dict['process_control']

    def process_id_exist(self, process_id):
        # Checks if a process id exists

        process_name = ProcessMeta.split_process_id(process_id)[0]
        if process_name in self.processes.keys():
            if process_id in self.processes[process_name].keys():
                return True
            else:
                return False
            return False

    def find_best_orbit(self, orbit_type='original'):
        # Find the best orbit file from the available ones

        keys = [key for key in self.orbits.keys() if orbit_type in key]

        precise = [key for key in keys if 'precise' in key]
        if len(precise) == 1:
            return self.orbits[precise[0]]
        elif len(precise) > 1:
            precise_reference = [key for key in precise if 'reference' in key]
            return self.orbits[precise_reference[0]]
        else:
            restituted = [key for key in keys if 'restituted' in key]
            if len(restituted) > 0:
                return self.orbits[restituted[0]]
            else:
                metadata = [key for key in keys if 'metadata' in key]
                if len(metadata) > 0:
                    return self.orbits[metadata[0]]
                else:
                    logging.info('No usable orbit found.')
                    return False
