# This class loads and saves user settings.
import os
import inspect
import base64
from six.moves import urllib
import ftplib

import rippl


class UserSettings():
    
    
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
            raise ConnectionError('Your ESA account is not valid.')

        self.ESA_username = username
        self.ESA_password = password

    def add_NASA_settings(self, username, password):
        """
        Checks whether username and password are registered and throws an error if not

        """

        # Check whether we are allowed to download a file
        url = 'http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N17E055.SRTMGL1.hgt.zip.xml'
        command = 'wget ' + url + ' -O /dev/null --password ' + password + ' --user ' + username
        output = os.system(command)

        if output == 0:
            print('NASA password valid!')
        else:
            raise ConnectionError('Your NASA account is not valid.')

        self.NASA_username = username
        self.NASA_password = password

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
            raise ConnectionError('Account to download TanDEM-X DEM data not valid')

        self.DLR_username = username
        self.DLR_password = password
    
    def save_data_database(self, main_folder='', radar_database='', radar_datastacks='', DEM_database='',
                           orbit_database='', NWP_model_database='', GIS_database=''):
        
        if main_folder:
            if not os.path.isdir(os.path.dirname(main_folder)):
                raise FileNotFoundError('The folder to write the main folder does not exist.')
            else:
                if not os.path.isdir(main_folder):
                    os.mkdir(main_folder)
            
            radar_database = os.path.join(main_folder, 'radar_database')
            radar_datastacks = os.path.join(main_folder, 'radar_datastacks')
            DEM_database = os.path.join(main_folder, 'DEM_data')
            orbit_database = os.path.join(main_folder, 'orbit_folder')
            
        if not os.path.isdir(os.path.dirname(radar_database)):
            raise FileNotFoundError('The folder to write radar database folder does not exist.')
        else:
            if not os.path.isdir(radar_database):
                os.mkdir(radar_database)
            for sensor_type in ['Sentinel-1', 'TanDEM-X', 'Radarsat']:
                if not os.path.isdir(os.path.join(radar_database, sensor_type)):
                    os.mkdir(os.path.join(radar_database, sensor_type))

        if not os.path.isdir(os.path.dirname(radar_datastacks)):
            raise FileNotFoundError('The folder to write radar datastacks folder does not exist.')
        else:
            if not os.path.isdir(radar_datastacks):
                os.mkdir(radar_datastacks)
            for sensor_type in ['Sentinel-1', 'TanDEM-X', 'Radarsat']:
                if not os.path.isdir(os.path.join(radar_datastacks, sensor_type)):
                    os.mkdir(os.path.join(radar_datastacks, sensor_type))

        if not os.path.isdir(os.path.dirname(DEM_database)):
            raise FileNotFoundError('The folder to write radar database folder does not exist.')
        else:
            if not os.path.isdir(DEM_database):
                os.mkdir(DEM_database)
            for dat_type in ['SRTM', 'TanDEM-X', 'geoid']:
                if not os.path.isdir(os.path.join(DEM_database, dat_type)):
                    os.mkdir(os.path.join(DEM_database, dat_type))
        
        if not os.path.isdir(os.path.dirname(orbit_database)):
            raise FileNotFoundError('The folder to write radar database folder does not exist.')
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
            raise FileNotFoundError('The folder to write NWP database folder does not exist.')
        else:
            if not os.path.isdir(NWP_model_database):
                os.mkdir(NWP_model_database)

        if not os.path.isdir(os.path.dirname(GIS_database)):
            raise FileNotFoundError('The folder to write radar database folder does not exist.')
        else:
            if not os.path.isdir(GIS_database):
                os.mkdir(GIS_database)

        self.radar_datastacks = radar_datastacks
        self.radar_database = radar_database
        self.orbit_database = orbit_database
        self.DEM_database = DEM_database
        self.NWP_model_database = NWP_model_database
        self.GIS_database = GIS_database
        
    def load_settings(self):
        """
        Load user settings from file

        """

        # Open text file
        if not os.path.exists(self.settings_path):
            raise FileNotFoundError('Settings file not found. First setup user settings using user_setup.ipynb before processing!')

        user_settings = open(self.settings_path, 'r')

        # Get paths
        self.radar_datastacks = user_settings.readline().split()[1]
        self.radar_database = user_settings.readline().split()[1]
        self.orbit_database = user_settings.readline().split()[1]
        self.DEM_database = user_settings.readline().split()[1]
        self.NWP_model_database = user_settings.readline().split()[1]
        self.GIS_database = user_settings.readline().split()[1]

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

        # Write passwords
        user_settings.write('ESA_username: ' + self.ESA_username + '\n')
        user_settings.write('ESA_password: ' + self.ESA_password + '\n')
        user_settings.write('NASA_username: ' + self.NASA_username + '\n')
        user_settings.write('NASA_password: ' + self.NASA_password + '\n')
        user_settings.write('DLR_username: ' + self.DLR_username + '\n')
        user_settings.write('DLR_password: ' + self.DLR_password + '\n')

        user_settings.close()
