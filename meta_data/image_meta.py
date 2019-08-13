"""
This class combines information for different processing steps, used orbit and radar coordinate systems for one SAR
data file.

"""
import json
from collections import OrderedDict
import datetime
import os

from rippl.meta_data.process_meta import ProcessMeta
from rippl.meta_data.orbit import Orbit
from rippl.meta_data.readfile import Readfile
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ImageMeta(object):

    def __init__(self, path='', overwrite=False):

        self.path = path
        if os.path.exists(self.path) and overwrite == False:
            self.load_json()
            return

        self.reference_paths = OrderedDict()    # Path to coreg, master, slave images.
        self.header = OrderedDict()
        self.process_control = OrderedDict()
        self.readfiles = OrderedDict()
        self.orbits = OrderedDict()
        self.processes = OrderedDict()

        # Information about all the data (or possible data on disk). Not part of
        self.data_disk = OrderedDict()

        # The process control give a summary of all applied steps.
        self.process_control = OrderedDict([('readfiles', []), ('orbits', []), ('processes', OrderedDict())])

        # Combine everything in a json_dict
        self.json_dict = OrderedDict()

        self.json_dict['header'] = self.header
        self.json_dict['process_control'] = self.process_control
        self.json_dict['readfiles'] = self.readfiles
        self.json_dict['orbits'] = self.orbits
        self.json_dict['processes'] = self.processes

    def add_process(self, process):

        if not isinstance(process, ProcessMeta):
            print('Any added processing step should be an Process object')
            return

        if not process.process_name not in self.processes.keys():
            self.processes[process.process_name] = OrderedDict()
            self.process_control['processes'][process.process_name] = OrderedDict()

        self.processes[process.process_name][process.process_id] = process
        self.process_control['processes'][process.process_name].append(process.process_id)

    def add_orbit(self, orbit, orbit_type='original'):

        if orbit_type not in ['original', 'master', 'slave', 'coreg']:
            print('Type of orbit should be either original, master, slave or coreg. Default to image data now')
            orbit_type = orbit.date

        if not isinstance(orbit, Orbit):
            print('Any added orbit should be an Orbit object')
            return

        orbit_id = orbit_type + '_#type#_' + orbit.orbit_type
        self.orbits[orbit_id] = orbit
        self.process_control['orbits'].append(orbit_id)

    def add_readfile(self, readfile, readfile_type='original'):

        if readfile_type not in ['original', 'master', 'slave', 'coreg']:
            print('Type of readfile should be either original, master, slave or coreg')

        if not isinstance(readfile, Readfile):
            print('Any added readfile should be an Readfile object')
            return

        self.readfiles[readfile_type] = readfile
        self.process_control['readfiles'].append(readfile.date)

    def dump_process(self, process, coordinates='', polarisation='', data_id=''):
        # This removes one of the processes and their respective datafiles.

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an Coordinatesystem object!')
            return

        # Create id
        if not polarisation:
            polarisation = 'not_defined'
        if not data_id:
            data_id = 'none'

        process_id = process + '_#coor#_' + coordinates.short_id_str + '_#id#_' + data_id + '_#pol#_' + polarisation

        if process in self.process_control.keys():
            if process_id in self.process_control[process]:
                self.process_control['processes'][process].remove(process_id)
                self.processes[process].pop(process_id)
            else:
                print('Cannot remove process. The process exists but not with this coordinate system, id or polarisation')
        else:
            print('Process does not exist')

    def dump_orbit(self, date, orbit_type):
        # This removes one of the orbits from the metadata file

        orbit_id = date + '_#type#_' + orbit_type

        if orbit_id in self.process_control['orbits'].keys():
            self.process_control['orbits'].remove(orbit_id)
            self.processes['orbits'].pop(orbit_id)
        else:
            print('Cannot remove orbit. Orbit date or type does not exist.')

    def dump_readfile(self, date):
        # This removes one of the readfiles from the metadata file

        if date in self.process_control['readfiles'].keys():
            self.process_control['readfiles'].remove(date)
            self.processes['readfiles'].pop(date)
        else:
            print('Cannot remove readfile. Readfile date does not exist.')

    def create_header(self):
        # Create the header for a new json file.

        header = dict()
        header['processor'] = 'RIPPL version 2.0 Beta'
        header['processor source'] = '[url]'
        header['created_by'] = 'Gert Mulder'
        header['creation_date'] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")
        header['last_date_changed'] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")

    def update_json(self, path=''):
        # Only the header file has to be changed. Other processes are changed within their respective classes.

        self.json_dict['header']['last_date_changed'] = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")
        if path:
            json.dump(self.json_dict, path)
        else:
            json.dump(self.json_dict, self.path)

    def load_json(self, file):
        # Load json data from file.

        self.json_dict = json.load(file)

        # Load processes, readfiles and orbits
        self.header = self.json_dict['header']
        self.readfiles = self.json_dict['readfiles']
        self.orbits = self.json_dict['orbits']
        self.processes = self.json_dict['processes']

        # Load process control
        self.process_control
