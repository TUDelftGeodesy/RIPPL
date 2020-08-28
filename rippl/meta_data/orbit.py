# class to define satellite orbits.
# Meta data from satellite files will be translated to this class, which will be used to calculate geometry of
# satellite orbits and points on the ground.

import numpy as np
import json
from collections import OrderedDict


class Orbit():

    """
    :type json_dict = OrderedDict
    """

    def __init__(self, json_data=''):

        self.satellite = ''
        self.date = ''
        self.ref_date = ''
        self.orbit_type = ''
        self.t = []

        self.x = []
        self.y = []
        self.z = []

        self.v_x = []
        self.v_y = []
        self.v_z = []

        self.a_x = []
        self.a_y = []
        self.a_z = []

        if json_data == '':
            self.json_dict = OrderedDict()
        else:
            self.load_json(json_data=json_data)

    def create_orbit(self, t, x, y, z, v_x='', v_y='', v_z='', a_x='', a_y='', a_z='', date='', satellite='', type=''):
        # Create orbit object. This is done using external orbit information.
        # Variable t is given in seconds from midnight of first time step.

        self.date = date
        self.satellite = satellite
        self.orbit_type = type
        self.t = t

        self.x = x
        self.y = y
        self.z = z

        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z

        self.a_x = a_x
        self.a_y = a_y
        self.a_z = a_z

    def update_json(self):

        self.json_dict['date'] = self.date
        self.json_dict['satellite'] = self.satellite
        self.json_dict['orbit_type'] = self.orbit_type

        self.json_dict['t'] = list(self.t)
        self.json_dict['x'] = list(self.x)
        self.json_dict['y'] = list(self.y)
        self.json_dict['z'] = list(self.z)

        self.json_dict['v_x'] = list(self.v_x)
        self.json_dict['v_y'] = list(self.v_y)
        self.json_dict['v_z'] = list(self.v_z)

        self.json_dict['a_x'] = list(self.a_x)
        self.json_dict['a_y'] = list(self.a_y)
        self.json_dict['a_z'] = list(self.a_z)

        return self.json_dict

    def save_json(self, json_path):
        # Save .json file
        self.update_json()

        file = open(json_path, 'w+')
        json.dump(self.json_dict, file, indent=3)
        file.close()

    def load_json(self, json_data='', json_path=''):

        if len(json_data) == 0:
            file = open(json_path)
            self.json_dict = json.load(file, object_pairs_hook=OrderedDict)
            file.close()
        else:
            self.json_dict = json_data

        self.date = self.json_dict['date']
        self.satellite = self.json_dict['satellite']
        self.orbit_type = self.json_dict['orbit_type']

        self.t = np.array(self.json_dict['t'])
        self.x = np.array(self.json_dict['x'])
        self.y = np.array(self.json_dict['y'])
        self.z = np.array(self.json_dict['z'])

        self.v_x = np.array(self.json_dict['v_x'])
        self.v_y = np.array(self.json_dict['v_y'])
        self.v_z = np.array(self.json_dict['v_z'])

        self.a_x = np.array(self.json_dict['a_x'])
        self.a_y = np.array(self.json_dict['a_y'])
        self.a_z = np.array(self.json_dict['a_z'])
