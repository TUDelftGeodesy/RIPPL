# This class loads and saves user settings.
import os
import inspect
import base64
from six.moves import urllib
import ftplib
import subprocess

import rippl
from rippl.download_login import DownloadLogin


class UserSettings(object):

    def __init__(self):
        
        # Passwords
        self.ESA_username = ''
        self.ESA_password = ''
        self.NASA_username = ''
        self.NASA_password = ''
        self.DLR_username = ''
        self.DLR_password = ''
    
        # Paths
        self.radar_database = ''
        self.radar_datastacks = ''
        self.DEM_database = ''
        self.orbit_database = ''
        self.GIS_database = ''
        self.NWP_model_database = ''
        self.rippl_folder = ''
        self.snaphu_path = ''

        # And the path of the settings file itself
        self.settings_path = os.path.join(os.path.dirname(inspect.getfile(rippl)), 'user_settings.txt')

    def add_ESA_settings(self, username, password):
        """
        Checks whether username and password are registered and throws an error if not

        """

        url = 'https://scihub.copernicus.eu/apihub/odata/v1'
        request = urllib.request.Request(url)
        base64string = base64.b64encode(bytes('%s:%s' % (username, password), "utf-8")).decode()
        request.add_header("Authorization", "Basic " + base64string)

        # connect to server. Hopefully this works at once
        try:
            urllib.request.urlopen(request)
            print('ESA password valid!')
        except:
            print('Your ESA account is not valid.')
            return False

        self.ESA_username = username
        self.ESA_password = password
        return True

    def add_NASA_settings(self, username, password):
        """
        Checks whether username and password are registered and throws an error if not

        """

        # check if wget is installed
        if os.name != 'nt':
            if os.system('wget') != 0:
                print('Please install the program wget on your system to enable downloading from the ASF server.')
                return False

        # Check whether we are allowed to download a file
        url = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N17E055.SRTMGL1.hgt.zip.xml'
        if os.name == 'nt':
            try:
                check_download = DownloadLogin('', username, password)
                path = os.path.join(os.path.dirname(self.settings_path), 'external_dems', 'test_file')
                check_download.download_file(url, path)
                os.remove(path)
            except:
                print('Your NASA account is not valid.')
                return False
        else:
            command = 'wget ' + url + ' -O /dev/null --password ' + password + ' --user ' + username
            output = os.system(command)

            if output != 0:
                print('Your NASA account is not valid.')
                return False

        self.NASA_username = username
        self.NASA_password = password
        print('NASA password valid!')
        return True

    def add_DLR_settings(self, username, password):
        """
        Checks whether username and password are registered and throws an error if not

        """

        # Check DLR passwords
        server = 'tandemx-90m.dlr.de'

        ftp = ftplib.FTP_TLS()
        ftp.connect(host=server)

        try:
            ftp.login(user=username, passwd=password)
            print('DLR password valid!')
        except:
            print('Account to download TanDEM-X DEM data not valid')
            return False

        self.DLR_username = username
        self.DLR_password = password
        return True

    def save_data_database(self, main_folder='', radar_database='', radar_datastacks='', DEM_database='',
                           orbit_database='', NWP_model_database='', GIS_database=''):
        
        if main_folder:
            if not os.path.isdir(os.path.dirname(main_folder)):
                print('The folder to write the main folder does not exist.')
                return False
            else:
                if not os.path.isdir(main_folder):
                    os.mkdir(main_folder)

            if not radar_database:
                radar_database = os.path.join(main_folder, 'radar_database')
            if not radar_datastacks:
                radar_datastacks = os.path.join(main_folder, 'radar_datastacks')
            if not DEM_database:
                DEM_database = os.path.join(main_folder, 'DEM_data')
            if not orbit_database:
                orbit_database = os.path.join(main_folder, 'orbit_folder')
            if not NWP_model_database:
                NWP_model_database = os.path.join(main_folder, 'NWP_model_database')
            if not GIS_database:
                GIS_database = os.path.join(main_folder, 'GIS_database')
            
        if not os.path.isdir(os.path.dirname(radar_database)):
            print('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(radar_database):
                os.mkdir(radar_database)
            for sensor_type in ['Sentinel-1', 'TanDEM-X', 'Radarsat']:
                if not os.path.isdir(os.path.join(radar_database, sensor_type)):
                    os.mkdir(os.path.join(radar_database, sensor_type))

        if not os.path.isdir(os.path.dirname(radar_datastacks)):
            print('The folder to write radar datastacks folder does not exist.')
            return False
        else:
            if not os.path.isdir(radar_datastacks):
                os.mkdir(radar_datastacks)
            for sensor_type in ['Sentinel-1', 'TanDEM-X', 'Radarsat']:
                if not os.path.isdir(os.path.join(radar_datastacks, sensor_type)):
                    os.mkdir(os.path.join(radar_datastacks, sensor_type))

        if not os.path.isdir(os.path.dirname(DEM_database)):
            print('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(DEM_database):
                os.mkdir(DEM_database)
            for dat_type in ['SRTM', 'TanDEM-X', 'geoid']:
                if not os.path.isdir(os.path.join(DEM_database, dat_type)):
                    os.mkdir(os.path.join(DEM_database, dat_type))
        
        if not os.path.isdir(os.path.dirname(orbit_database)):
            print('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(orbit_database):
                os.mkdir(orbit_database)
            for sensor_type in ['Sentinel-1']:
                if not os.path.isdir(os.path.join(orbit_database, sensor_type)):
                    os.mkdir(os.path.join(orbit_database, sensor_type))

            if not os.path.isdir(os.path.join(orbit_database, 'Sentinel-1', 'precise')):
                os.mkdir(os.path.join(orbit_database, 'Sentinel-1', 'precise'))
            if not os.path.isdir(os.path.join(orbit_database, 'Sentinel-1', 'restituted')):
                os.mkdir(os.path.join(orbit_database, 'Sentinel-1', 'restituted'))

        if not os.path.isdir(os.path.dirname(NWP_model_database)):
            print('The folder to write NWP database folder does not exist.')
            return False
        else:
            if not os.path.isdir(NWP_model_database):
                os.mkdir(NWP_model_database)

        if not os.path.isdir(os.path.dirname(GIS_database)):
            print('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(GIS_database):
                os.mkdir(GIS_database)

        self.radar_datastacks = radar_datastacks
        self.radar_database = radar_database
        self.orbit_database = orbit_database
        self.DEM_database = DEM_database
        self.NWP_model_database = NWP_model_database
        self.GIS_database = GIS_database

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
            print('Snaphu path found and added to RIPPL installation')
            self.snaphu_path = snaphu_path
            return True
        except:
            print('Snaphu path incorrect. The full RIPPL code will still work except the unwrapping')
            return False

    def load_settings(self):
        """
        Load user settings from file

        """

        # Open text file
        if not os.path.exists(self.settings_path):
            raise ValueError('Settings file not found. First setup user settings using user_setup.ipynb before processing!')

        user_settings = open(self.settings_path, 'r')

        # Get paths
        self.radar_datastacks = user_settings.readline().split()[1]
        self.radar_database = user_settings.readline().split()[1]
        self.orbit_database = user_settings.readline().split()[1]
        self.DEM_database = user_settings.readline().split()[1]
        self.NWP_model_database = user_settings.readline().split()[1]
        self.GIS_database = user_settings.readline().split()[1]
        self.snaphu_path = user_settings.readline().split()[1]

        # Get passwords
        self.ESA_username = user_settings.readline().split()[1]
        self.ESA_password = user_settings.readline().split()[1]
        self.NASA_username = user_settings.readline().split()[1]
        self.NASA_password = user_settings.readline().split()[1]
        self.DLR_username = user_settings.readline().split()[1]
        self.DLR_password = user_settings.readline().split()[1]

        user_settings.close()

    def save_settings(self):
        """
        Save user settings to file

        """

        # Create text file
        if os.path.exists(self.settings_path):
            os.remove(self.settings_path)
        user_settings = open(self.settings_path, 'w')

        # Write paths
        user_settings.write('radar_datastacks: ' + str(self.radar_datastacks) + '\n')
        user_settings.write('radar_database: ' + str(self.radar_database) + '\n')
        user_settings.write('orbit_database: ' + str(self.orbit_database) + '\n')
        user_settings.write('DEM_database: ' + str(self.DEM_database) + '\n')
        user_settings.write('NWP_model_database: ' + str(self.NWP_model_database) + '\n')
        user_settings.write('GIS_database: ' + str(self.GIS_database) + '\n')
        user_settings.write('Snaphu_path: ' + str(self.snaphu_path) + '\n')

        # Write passwords
        user_settings.write('ESA_username: ' + self.ESA_username + '\n')
        user_settings.write('ESA_password: ' + self.ESA_password + '\n')
        user_settings.write('NASA_username: ' + self.NASA_username + '\n')
        user_settings.write('NASA_password: ' + self.NASA_password + '\n')
        user_settings.write('DLR_username: ' + self.DLR_username + '\n')
        user_settings.write('DLR_password: ' + self.DLR_password + '\n')

        user_settings.close()
