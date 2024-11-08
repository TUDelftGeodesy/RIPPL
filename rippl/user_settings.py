# This class loads and saves user settings.
import os
import inspect
import base64
from six.moves import urllib
import ftplib
import subprocess
from http.cookiejar import CookieJar
import logging
import ssl
import json
import requests
ssl._create_default_https_context = ssl._create_unverified_context

import rippl
from rippl.NWP_model_delay.load_NWP_data.harmonie_nl.harmonie_api_request import OpenDataAPI


class UserSettings(object):

    def __init__(self):

        # And the path of the settings file itself
        self.settings_path = os.path.join(os.path.dirname(inspect.getfile(rippl)), 'user_settings.json')
        self.settings = {}

        if os.path.exists(self.settings_path):
            self.load_settings()
        else:
            # Accounts information
            self.settings['accounts'] = {}
            self.settings['accounts']['Copernicus'] = {'api_key': None}
            self.settings['accounts']['EarthData'] = {'username': None, 'password': None}
            self.settings['accounts']['DLR'] = {'username': None, 'password': None}
            self.settings['accounts']['CDS'] = {'api_key': None, 'enabled': False}
            self.settings['accounts']['KNMI'] = {'api_key': None, 'enabled': False}

            # Paths
            self.settings['paths'] = {}
            self.settings['paths']['rippl'] = ''
            self.settings['paths']['radar_database'] = ''
            self.settings['paths']['radar_data_stacks'] = ''
            self.settings['paths']['DEM_database'] = ''
            self.settings['paths']['orbit_database'] = ''
            self.settings['paths']['GIS_database'] = ''
            self.settings['paths']['NWP_model_database'] = ''
            self.settings['paths']['rippl'] = ''
            self.settings['paths']['snaphu'] = None

            # Sensor names
            self.settings['path_names'] = {}
            self.settings['path_names']['SAR'] = {}
            self.settings['path_names']['DEM'] = {}
            self.settings['path_names']['NWP'] = {}

            # Initialize
            self.define_sensor_names()
            self.save_settings()

    def add_copernicus_settings(self, username, password):
        """
        Checks whether username and password are registered and throws an error if not

        """

        data = {"client_id": "cdse-public",
                "username": username,
                "password": password,
                "grant_type": "password"}

        # connect to server. Hopefully this works at once
        try:
            r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                              data=data,
                              )
            r.raise_for_status()
            logging.info('Scihub password valid!')
        except Exception as e:
            logging.warning('Your ESA account is not valid to use the API hub. Please wait one week after account registration '
                  'before using it here. Details https://documentation.dataspace.copernicus.eu/APIs/Token.html'
                  'You can run this notebook again later to add the ESA scihub account. ' + str(e))
            return False

        self.settings['accounts']['Copernicus']['username'] = username
        self.settings['accounts']['Copernicus']['password'] = password
        return True

    def add_earthdata_settings(self, username, password):
        """
        Checks whether username and password are registered and throws an error if not

        """

        # Just take one dataset as an example to login.
        url = 'https://datapool.asf.alaska.edu/SLC/SA/S1A_IW_SLC__1SDV_20180511T161620_20180511T161639_021859_025BF8_9FF0.zip'
        request = urllib.request.Request(url)
        cj = CookieJar()
        base64string = base64.b64encode(bytes('%s:%s' % (username, password), "utf-8")).decode()
        request.add_header("Authorization", "Basic " + base64string)

        # connect to server. Hopefully this works at once
        try:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
            opener.open(request)
            logging.info('Earthdata password valid!')
        except Exception as e:
            logging.warning('Your Earthdata account is not valid. ' + str(e))
            return False

        self.settings['accounts']['EarthData']['username'] = username
        self.settings['accounts']['EarthData']['password'] = password
        return True

    def add_DLR_settings(self, username, password, dem_type='TDM30'):
        """
        Checks whether username and password are registered and throws an error if not

        """

        # Check DLR passwords
        if dem_type == 'TDM30':
            url = 'https://download.geoservice.dlr.de/TDM30_EDEM/files/'
        elif dem_type == 'TDM90':
            url = 'https://download.geoservice.dlr.de/TDM90/files/'
        else:
            raise TypeError('Only TDM30 and TDM90 are possible DEM types')

        dat = requests.get(url, auth=(username, password))
        if dat.ok:
            logging.info('DLR password valid!')
        else:
            logging.warning('Account to download TanDEM-X DEM data not valid.')
            return False

        self.settings['accounts']['DLR'][dem_type]['username'] = username
        self.settings['accounts']['DLR'][dem_type]['password'] = password
        return True

    def add_ecmwf_settings(self, api_key):
        """
        Load username and password for CDS download

        Account can be created at
        https://cds.climate.copernicus.eu/user/register
        """

        # Try to make a connection with the API using the given uid and api_key
        self.settings['accounts']['CDS']['api_key'] = api_key

        try:
            import cdsapi
        except:
            logging.warning('Pleas install the CDS api using the guide here https://cds.climate.copernicus.eu/api-how-to')
            return False
        # Set enabled
        self.settings['accounts']['CDS']['enabled'] = True

    def add_knmi_settings(self, api_key):
        """
        Load the used API key for downloading Harmoniet data.

        You can create an account via
        https://developer.dataplatform.knmi.nl/register/

        """

        # Try to connect to KNMI database
        try:
            knmi_downloader = OpenDataAPI(api_token=api_key)
            # Use the standard dataset name we use for the Harmonie data
            dataset_name = 'harmonie_arome_cy40_p1'
            dataset_version = '0.2'
            params = {"maxKeys": 100, "orderBy": "created", "sorting": "desc"}
            available_files = knmi_downloader.list_files(dataset_name, dataset_version, params)
        except:
            logging.warning('Download of KNMI data not succeeded, try again with another API key.')
            return False

        # Save on succes
        self.settings['accounts']['KNMI']['api_key'] = api_key
        # Set enabled
        self.settings['accounts']['KNMI']['enabled'] = True
        return True

    def define_sensor_names(self, dem_path_names='', sar_path_names='', nwp_model_path_names=''):
        """

        Parameters
        ----------
        dem_path_names
        sar_path_names
        nwp_model_path_names

        Returns
        -------

        """

        dem_names = ['SRTM', 'TanDEM-X']
        if not dem_path_names:
            dem_path_names = ['srtm', 'tdx']

        sar_names = ['Sentinel-1', 'TerraSAR-X']
        if not sar_path_names:
            sar_path_names = ['sentinel1', 'tsx']

        nwp_model_names = ['Harmonie-Arome', 'ECMWF']
        if not nwp_model_path_names:
            nwp_model_path_names = ['harmonie', 'ecmwf']

        for sensor, sensor_name in zip(dem_names, dem_path_names):
            self.settings['path_names']['DEM'][sensor] = sensor_name

        for sensor, sensor_name in zip(sar_names, sar_path_names):
            self.settings['path_names']['SAR'][sensor] = sensor_name

        for nwp_model, nwp_model_name in zip(nwp_model_names, nwp_model_path_names):
            self.settings['path_names']['NWP'][nwp_model] = nwp_model_name

    def save_data_database(self, main_folder='', radar_database='', radar_data_stacks='', DEM_database='',
                           orbit_database='', NWP_model_database='', GIS_database=''):

        if main_folder:
            if not os.path.isdir(os.path.dirname(main_folder)):
                logging.info('The folder to write the main folder does not exist.')
                return False
            else:
                if not os.path.isdir(main_folder):
                    os.mkdir(main_folder)

            if not radar_database:
                radar_database = os.path.join(main_folder, 'radar_database')
            if not radar_data_stacks:
                radar_data_stacks = os.path.join(main_folder, 'radar_data_stacks')
            if not DEM_database:
                DEM_database = os.path.join(main_folder, 'DEM_database')
            if not orbit_database:
                orbit_database = os.path.join(main_folder, 'orbit_database')
            if not NWP_model_database:
                NWP_model_database = os.path.join(main_folder, 'NWP_model_database')
            if not GIS_database:
                GIS_database = os.path.join(main_folder, 'GIS_database')

        if not os.path.isdir(os.path.dirname(radar_database)):
            logging.info('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(radar_database):
                os.mkdir(radar_database)
            for sensor_type in self.settings['path_names']['SAR'].values():
                if not os.path.isdir(os.path.join(radar_database, sensor_type)):
                    os.mkdir(os.path.join(radar_database, sensor_type))

        if not os.path.isdir(os.path.dirname(radar_data_stacks)):
            logging.info('The folder to write radar data_stacks folder does not exist.')
            return False
        else:
            if not os.path.isdir(radar_data_stacks):
                os.mkdir(radar_data_stacks)
            for sensor_type in self.settings['path_names']['SAR'].values():
                if not os.path.isdir(os.path.join(radar_data_stacks, sensor_type)):
                    os.mkdir(os.path.join(radar_data_stacks, sensor_type))

        if not os.path.isdir(os.path.dirname(DEM_database)):
            logging.info('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(DEM_database):
                os.mkdir(DEM_database)
            if not os.path.isdir(os.path.join(DEM_database, 'geoid')):
                os.mkdir(os.path.join(DEM_database, 'geoid'))
            for dat_type in self.settings['path_names']['DEM'].values():
                if not os.path.isdir(os.path.join(DEM_database, dat_type)):
                    os.mkdir(os.path.join(DEM_database, dat_type))

        if not os.path.isdir(os.path.dirname(orbit_database)):
            logging.info('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(orbit_database):
                os.mkdir(orbit_database)
            for sensor_type in self.settings['path_names']['SAR'].values():
                if not os.path.isdir(os.path.join(orbit_database, sensor_type)):
                    os.mkdir(os.path.join(orbit_database, sensor_type))

            # TODO move to orbit download scripts
            if not os.path.isdir(os.path.join(orbit_database, self.settings['path_names']['SAR']['Sentinel-1'], 'precise')):
                os.mkdir(os.path.join(orbit_database, self.settings['path_names']['SAR']['Sentinel-1'], 'precise'))
            if not os.path.isdir(os.path.join(orbit_database, self.settings['path_names']['SAR']['Sentinel-1'], 'restituted')):
                os.mkdir(os.path.join(orbit_database, self.settings['path_names']['SAR']['Sentinel-1'], 'restituted'))

        if not os.path.isdir(os.path.dirname(NWP_model_database)):
            logging.info('The folder to write NWP database folder does not exist.')
            return False
        else:
            if not os.path.isdir(NWP_model_database):
                os.mkdir(NWP_model_database)
            for model_type in self.settings['path_names']['NWP'].values():
                if not os.path.isdir(os.path.join(NWP_model_database, model_type)):
                    os.mkdir(os.path.join(NWP_model_database, model_type))

        if not os.path.isdir(os.path.dirname(GIS_database)):
            logging.info('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(GIS_database):
                os.mkdir(GIS_database)

        self.settings['paths']['radar_database'] = radar_database
        self.settings['paths']['radar_data_stacks'] = radar_data_stacks
        self.settings['paths']['DEM_database'] = DEM_database
        self.settings['paths']['orbit_database'] = orbit_database
        self.settings['paths']['GIS_database'] = GIS_database
        self.settings['paths']['NWP_model_database'] = NWP_model_database

        return True

    def add_snaphu_path(self, snaphu_path):
        """

        Parameters
        ----------
        snaphu_path

        Returns
        -------

        """

        try:
            subprocess.call(snaphu_path)
            logging.info('Snaphu path found and added to RIPPL installation')
            self.settings['paths']['snaphu'] = snaphu_path
            return True
        except Exception as e:
            logging.warning('Snaphu path incorrect. The full RIPPL code will still work except the unwrapping. ' + str(e))
            return False

    def load_settings(self):
        """
        Load user settings from file

        """

        # Open json file
        with open(self.settings_path, 'r') as f:
            self.settings = json.load(f)

    def save_settings(self):
        """
        Save user settings to file

        """

        with open(self.settings_path, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=4)

        logging.info('Settings saved to ' + self.settings_path)
