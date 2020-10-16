# This class loads and saves user settings.
import os
import inspect
import base64
from six.moves import urllib
import ftplib
import subprocess
from http.cookiejar import CookieJar
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import rippl


class UserSettings(object):

    def __init__(self):
        
        # Passwords
        self.ESA_username = 'None'
        self.ESA_password = 'None'
        self.NASA_username = 'None'
        self.NASA_password = 'None'
        self.DLR_username = 'None'
        self.DLR_password = 'None'
    
        # Paths
        self.radar_database = ''
        self.radar_data_stacks = ''
        self.DEM_database = ''
        self.orbit_database = ''
        self.GIS_database = ''
        self.NWP_model_database = ''
        self.rippl_folder = ''
        self.snaphu_path = 'None'

        # sensor names
        self.sar_sensors = ''
        self.sar_sensor_names = ''
        self.sar_sensor_name = dict()
        self.dem_sensors = ''
        self.dem_sensor_names = ''
        self.dem_sensor_name = dict()

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
            print('Your ESA account is not valid to use the API hub. Please wait one week after account registration '
                  'before using it here. Details https://scihub.copernicus.eu/twiki/do/view/SciHubWebPortal/APIHubDescription'
                  'You can run this notebook again later to add the ESA scihub account.')
            return False

        if os.name != 'nt':
            try:
                subprocess.call('wget')
            except:
                print('You need to install the program wget to download Sentinel-1 data, or make it available from'
                      'the terminal.')

        self.ESA_username = username
        self.ESA_password = password
        return True

    def add_NASA_settings(self, username, password):
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
            print('NASA password valid!')
        except:
            print('Your NASA account is not valid.')
            return False

        if os.name != 'nt':
            try:
                subprocess.call('wget')
            except:
                print('You need to install the program wget to download Sentinel-1 data, or make it available from'
                      'the terminal.')

        self.NASA_username = username
        self.NASA_password = password
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

    def define_sensor_names(self, dem_sensors='', dem_sensor_names='', sar_sensors='', sar_sensor_names=''):
        """

        Parameters
        ----------
        dem_sensors
        dem_names
        sar_sensors
        sar_names

        Returns
        -------

        """

        if dem_sensors == '':
            self.dem_sensors = ['srtm', 'tdx']
        else:
            self.dem_sensors = dem_sensors
        if dem_sensor_names == '':
            self.dem_sensor_names = ['SRTM', 'TanDEM-X']
        else:
            self.dem_sensor_names = dem_sensor_names

        if sar_sensors == '':
            self.sar_sensors = ['sentinel1', 'tdx']
        else:
            self.sar_sensors = sar_sensors
        if sar_sensor_names == '':
            self.sar_sensor_names = ['Sentinel-1', 'TanDEM-X']
        else:
            self.sar_sensor_names = sar_sensor_names

        # Finally create the dictionaries
        self.create_sensor_dicts()

    def create_sensor_dicts(self):
        """

        Returns
        -------

        """

        self.dem_sensor_name = dict()
        for sensor, sensor_name in zip(self.dem_sensors, self.dem_sensor_names):
            self.dem_sensor_name[sensor] = sensor_name

        self.sar_sensor_name = dict()
        for sensor, sensor_name in zip(self.sar_sensors, self.sar_sensor_names):
            self.sar_sensor_name[sensor] = sensor_name

    def save_data_database(self, main_folder='', radar_database='', radar_data_stacks='', DEM_database='',
                           orbit_database='', NWP_model_database='', GIS_database='', radar_data_products=''):

        if not isinstance(self.sar_sensor_names, list) or not isinstance(self.dem_sensor_names, list):
            print('First run the define_sensor_names functions before creating database folders.')

        if main_folder:
            if not os.path.isdir(os.path.dirname(main_folder)):
                print('The folder to write the main folder does not exist.')
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
            if not radar_data_products:
                radar_data_products = os.path.join(main_folder, 'radar_data_products')

        if not os.path.isdir(os.path.dirname(radar_database)):
            print('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(radar_database):
                os.mkdir(radar_database)
            for sensor_type in self.sar_sensor_names:
                if not os.path.isdir(os.path.join(radar_database, sensor_type)):
                    os.mkdir(os.path.join(radar_database, sensor_type))

        if not os.path.isdir(os.path.dirname(radar_data_stacks)):
            print('The folder to write radar data_stacks folder does not exist.')
            return False
        else:
            if not os.path.isdir(radar_data_stacks):
                os.mkdir(radar_data_stacks)
            for sensor_type in self.sar_sensor_names:
                if not os.path.isdir(os.path.join(radar_data_stacks, sensor_type)):
                    os.mkdir(os.path.join(radar_data_stacks, sensor_type))

        if not os.path.isdir(os.path.dirname(radar_data_products)):
            print('The folder to write radar data products folder does not exist.')
            return False
        else:
            if not os.path.isdir(radar_data_products):
                os.mkdir(radar_data_products)
            for sensor_type in self.sar_sensor_names:
                if not os.path.isdir(os.path.join(radar_data_products, sensor_type)):
                    os.mkdir(os.path.join(radar_data_products, sensor_type))

        if not os.path.isdir(os.path.dirname(DEM_database)):
            print('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(DEM_database):
                os.mkdir(DEM_database)
            for dat_type in self.dem_sensor_names + ['geoid']:
                if not os.path.isdir(os.path.join(DEM_database, dat_type)):
                    os.mkdir(os.path.join(DEM_database, dat_type))

            # Create SRTM1 and SRTM3 paths.
            for srtm_type in ['srtm1', 'srtm3']:
                if not os.path.isdir(os.path.join(DEM_database, self.dem_sensor_name['srtm'], srtm_type)):
                    os.mkdir(os.path.join(DEM_database, self.dem_sensor_name['srtm'], srtm_type))

        if not os.path.isdir(os.path.dirname(orbit_database)):
            print('The folder to write radar database folder does not exist.')
            return False
        else:
            if not os.path.isdir(orbit_database):
                os.mkdir(orbit_database)
            for sensor_type in self.sar_sensor_names:
                if not os.path.isdir(os.path.join(orbit_database, sensor_type)):
                    os.mkdir(os.path.join(orbit_database, sensor_type))

            if not os.path.isdir(os.path.join(orbit_database, self.sar_sensor_name['sentinel1'], 'precise')):
                os.mkdir(os.path.join(orbit_database, self.sar_sensor_name['sentinel1'], 'precise'))
            if not os.path.isdir(os.path.join(orbit_database, self.sar_sensor_name['sentinel1'], 'restituted')):
                os.mkdir(os.path.join(orbit_database, self.sar_sensor_name['sentinel1'], 'restituted'))

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

        self.radar_data_stacks = radar_data_stacks
        self.radar_database = radar_database
        self.orbit_database = orbit_database
        self.DEM_database = DEM_database
        self.NWP_model_database = NWP_model_database
        self.GIS_database = GIS_database
        self.radar_data_products = radar_data_products

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

        settings_dict = dict()

        with open(self.settings_path, 'r') as user_settings:
            valid_line = True
            while valid_line:
                line = user_settings.readline()
                if line == '':
                    valid_line = False
                    continue
                settings_dict[line.split(':')[0]] = line

        # Get paths
        if 'radar_datastacks' in settings_dict.keys():
            self.radar_data_stacks = ' '.join(settings_dict.get('radar_datastacks', 'None None').split()[1:])
        else:
            self.radar_data_stacks = ' '.join(settings_dict.get('radar_data_stacks', 'None None').split()[1:])
        self.radar_database = ' '.join(settings_dict.get('radar_database', 'None None').split()[1:])
        self.orbit_database = ' '.join(settings_dict.get('orbit_database', 'None None').split()[1:])
        self.DEM_database = ' '.join(settings_dict.get('DEM_database', 'None None').split()[1:])
        self.NWP_model_database = ' '.join(settings_dict.get('NWP_model_database', 'None None').split()[1:])
        self.GIS_database = ' '.join(settings_dict.get('GIS_database', 'None None').split()[1:])
        self.snaphu_path = ' '.join(settings_dict.get('Snaphu_path', 'None None').split()[1:])
        self.radar_data_products = ' '.join(settings_dict.get('radar_data_products', 'None None').split()[1:])

        # Get passwords
        self.ESA_username = settings_dict.get('ESA_username', 'None None').split()[1]
        self.ESA_password = settings_dict.get('ESA_password', 'None None').split()[1]
        self.NASA_username = settings_dict.get('NASA_username', 'None None').split()[1]
        self.NASA_password = settings_dict.get('NASA_password', 'None None').split()[1]
        self.DLR_username = settings_dict.get('DLR_username', 'None None').split()[1]
        self.DLR_password = settings_dict.get('DLR_password', 'None None').split()[1]

        # Finally get the sensor names if these lines are not empty
        self.dem_sensors = settings_dict.get('DEM_sensors', 'None None').split()[1:]
        self.dem_sensor_names = settings_dict.get('DEM_sensor_names', 'None None').split()[1:]
        self.sar_sensors = settings_dict.get('SAR_sensors', 'None None').split()[1:]
        self.sar_sensor_names = settings_dict.get('SAR_sensor_names', 'None None').split()[1:]
        if self.dem_sensors != ['None']:
            self.create_sensor_dicts()
        else:
            self.define_sensor_names()

    def save_settings(self):
        """
        Save user settings to file

        """

        if len(self.radar_database) == 0 or len(self.radar_data_stacks) == 0 or len(self.orbit_database) == 0 or \
            len(self.NWP_model_database) == 0 or len(self.GIS_database) == 0 or len(self.DEM_database) == 0:
            print('Paths to RIPPL processing paths are empty. Please create these paths first using the '
                  'save_data_database function')

        # Create text file
        if os.path.exists(self.settings_path):
            os.remove(self.settings_path)
        with open(self.settings_path, 'w') as user_settings:

            # Write paths
            user_settings.write('radar_data_stacks: ' + str(self.radar_data_stacks) + '\n')
            user_settings.write('radar_database: ' + str(self.radar_database) + '\n')
            user_settings.write('radar_data_products: ' + str(self.radar_data_products) + '\n')
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

            # Write satellite sensor names
            user_settings.write('DEM_sensors: ' + ' '.join(self.dem_sensors) + '\n')
            user_settings.write('DEM_sensor_names: ' + ' '.join(self.dem_sensor_names) + '\n')
            user_settings.write('SAR_sensors: ' + ' '.join(self.sar_sensors) + '\n')
            user_settings.write('SAR_sensor_names: ' + ' '.join(self.sar_sensor_names) + '\n')

        print('Settings saved to ' + self.settings_path)
