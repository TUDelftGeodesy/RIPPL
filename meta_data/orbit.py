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

    def __init__(self, json_dict=''):

        self.satellite = ''
        self.date = ''
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

        if json_dict == '':
            self.json_dict = OrderedDict()
        else:
            self.json_dict = json_dict

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

    def update_json(self, json_path=''):

        variables = vars(self)

        for var, key in zip(variables, variables.keys()):
            if key == 'json_data':
                continue

            if isinstance(var, np.array):
                self.json_dict[key] = list(var)
            else:
                self.json_dict[key] = var

        if json_path:
            json.dump(self.json_dict)

    def load_json(self, json_data='', json_path=''):

        if len(json_data) == 0:
            self.json_dict = json.load(json_path, object_pairs_hook=OrderedDict)
        else:
            self.json_dict = json_data

        variables = vars(self)
        for var, key in zip(variables, variables.keys()):
            if key == 'json_data':
                continue

            if isinstance(var, list):
                self.json_dict[key] = np.array(var)
            else:
                self.json_dict[key] = var
