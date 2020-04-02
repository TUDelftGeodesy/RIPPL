# This file contains a function to check which files for sentinel are available, which ones are downloaded and a quality
# check for the files which are downloaded.

import numpy as np
from six.moves import urllib
import ssl
import re
import os, sys
import datetime
import base64
import subprocess
from fiona import collection
from lxml import etree
from shapely.geometry import Polygon


class DownloadSentinel(object):
    
    def __init__(self, start_date, end_date, shape, track='', polarisation='', level='',
                 orbit_direction='', sensor_mode='IW', product='SLC'):
        # Following variables can be used to make a selection.
        # shape > defining shape file or .kml
        # start_date > first day for downloads (default one month before now) [yyyymmdd]
        # end_date > last day for downloads (default today)
        # track > the tracks we want to check (default all)
        # polarisation > which polarisation will be used. (default all)
    
        # string is the field we enter as url
        self.products = []
        self.links = []
        self.dates = []
        self.string = ''
        self.shape = shape

        self.valid_files = []
        self.invalid_files = []

        if sensor_mode:
            self.string = self.string + ' AND ' + 'sensoroperationalmode:' + sensor_mode
        if product:
            self.string = self.string + ' AND ' + 'producttype:' + product
        if level:
            self.string = self.string + ' AND ' + level
        if orbit_direction:
            self.string = self.string + ' AND ' + 'orbitdirection:' + orbit_direction
        if track:
            self.string = self.string + ' AND ' + 'relativeorbitnumber:' + str(track)
        if isinstance(start_date, str):
            if start_date:
                start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start = (datetime.datetime.now() - datetime.timedelta(days=350))
        else:
            start = start_date
        if isinstance(end_date, str):
            if end_date:
                end = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1)
            else:
                end = datetime.datetime.now()
        else:
            end = end_date
            
        if polarisation:
            self.string = self.string + ' AND ' + 'polarisationmode:' + polarisation
        if shape:
            self.load_shape_info()
            self.string = self.string + ' AND footprint:"Intersects(POLYGON(' + self.shape_string + '))"'
    
        date_string = 'beginposition:[' + start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z TO ' + \
                      end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z] AND endposition:[' + \
                      start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z TO ' + end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z]'
        self.string = self.string + ' AND ' + date_string
        self.string = self.string[5:]
    
    def sentinel_available(self, username, password):
        # All available sentinel 1 images are detected and printed on screen.

        self.products = []
        self.links = []
        self.dates = []
        page = 0

        all_loaded = False
        while all_loaded == False:
            # Finally we do the query to get the search result.
            url = 'https://scihub.copernicus.eu/dhus/search?start=' + str(page) + '&rows=100&q=' + urllib.parse.quote_plus(self.string)

            request = urllib.request.Request(url)
            base64string = base64.b64encode(bytes('%s:%s' % (username, password), "utf-8")).decode()
            request.add_header("Authorization", "Basic " + base64string)

            # connect to server. Hopefully this works at once
            # Check for paging https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/OpenSearchAPI#Paging_results
            try:
                dat = urllib.request.urlopen(request)
            except:
                print('not possible to connect this time')
                return [], [], []

            html_dat = ''
            for line in dat:
                html_dat = html_dat + line.decode('UTF-8')

            parser = etree.HTMLParser()
            tree = etree.fromstring(html_dat.encode('utf-8'), parser)
            # Check whether we got all results
            total_items = int(tree[0][0][5].text)
            start_index = int(tree[0][0][6].text)
            page_items = int(tree[0][0][7].text)
            if start_index + page_items >= total_items:
                all_loaded = True
            else:
                page += 1

            self.products.extend([data for data in tree.iter(tag='entry')])
            self.links.extend([data.find('link').attrib for data in tree.iter(tag='entry')])
            self.dates.extend([data.findall('date')[1].text for data in tree.iter(tag='entry')])

    def sentinel_download_ESA(self, database_folder, username, password):
        # Download the files which are found by the sentinel_available script.
    
        if not self.products:
            print('No files to project_functions')
            return

        wget_base = 'wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --continue --tries=20 ' \
                    '--no-check-certificate --user=' + username + ' --password=' + password + ' '

        for product in self.products:
            file_dir = DownloadSentinel.find_database_folder(database_folder, product)
            url = str('"' + product.findall('link')[0].attrib['href'][:-6] + urllib.parse.quote_plus('$value') + '"')

            # Download data files and create symbolic link
            if not os.path.exists(file_dir):
                wget_data = wget_base + url + ' -O ' + file_dir
                print(wget_data)
                os.system(wget_data)

    def sentinel_download_ASF(self, database_folder, username, password):
        # Download data from ASF (Generally easier and much faster to download from this platform.)

        if not self.products:
            print('No files to project_functions')
            return

        wget_base = 'wget -c --http-user=' + username + ' --http-password=' + password + ' '

        for product in self.products:
            file_dir = DownloadSentinel.find_database_folder(database_folder, product)
            uuid = product.find('id').text
            url = '"https://datapool.asf.alaska.edu/SLC/SA/' + os.path.basename(file_dir) + '"'

            # Download data files and create symbolic link
            if not os.path.exists(file_dir):
                wget_data = wget_base + url + ' -O ' + file_dir
                print(wget_data)
                os.system(wget_data)

    def sentinel_check_validity(self, database_folder, username, password):
        # Check whether the zip files can be unpacked or not. This is part of the project_functions procedure.

        for product in self.products:
            uuid = product.find('id').text
            checksum_url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('" + uuid + "')/Checksum/Value/" + urllib.parse.quote_plus('$value')
            request = urllib.request.Request(checksum_url)
            base64string = base64.b64encode(bytes('%s:%s' % (username, password), "utf-8")).decode()
            request.add_header("Authorization", "Basic " + base64string)

            # connect to server. Hopefully this works at once
            try:
                dat = urllib.request.urlopen(request)
            except:
                print('not possible to connect this time')
                return False

            html_dat = ''
            for line in dat:
                html_dat = html_dat + line.decode('UTF-8')

            # Check file on disk
            file_dir = DownloadSentinel.find_database_folder(database_folder, product)
            if sys.platform == 'darwin':
                md5 = subprocess.check_output('md5 ' + file_dir, shell=True)[-33:-1]
            elif sys.platform == 'linux2':
                md5 = subprocess.check_output('md5sum ' + file_dir, shell=True)[:32]
            else:
                'This function only works on mac or linux systems!'
                return False

            if md5 != html_dat.lower():
                os.remove(file_dir)

        return True

    @staticmethod
    def find_database_folder(database_folder, product):
        # Find the destination folder in the database.

        date = str(product.findall('date')[1].text)
        date = datetime.datetime.strptime(date[:19], '%Y-%m-%dT%H:%M:%S')

        name = str(product.find('title').text)

        track = str(product.find('int[@name="relativeorbitnumber"]').text)
        data_type = str(product.find(".//str[@name='filename']").text)[4:16]
        pol = str(product.find(".//str[@name='polarisationmode']").text).replace(' ', '')
        direction = str(product.find(".//str[@name='orbitdirection']").text)
        if direction == 'ASCENDING':
            direction = 'asc'
        elif direction == 'DESCENDING':
            direction = 'dsc'

        track_folder = os.path.join(database_folder, 's1_' + direction + '_t' + track)
        if not os.path.exists(track_folder):
            os.mkdir(track_folder)
        type_folder = os.path.join(track_folder, data_type + '_' + pol)
        if not os.path.exists(type_folder):
            os.mkdir(type_folder)
        date_folder = os.path.join(type_folder, date.strftime('%Y%m%d'))
        if not os.path.exists(date_folder):
            os.mkdir(date_folder)

        file_dir = os.path.join(date_folder, name + '.zip')

        return file_dir

    def load_shape_info(self):
        # This script converts .kml, .shp and .txt files to the right format. If multiple shapes are available the script
        # will select the first one.

        if isinstance(self.shape, str):
            if not os.path.exists(self.shape):
                raise FileExistsError('Defined shapefile does not exist')

            if self.shape.endswith('.shp'):
                with collection(self.shape, "r") as inputshape:
                    for shape in inputshape:
                        # only first shape
                        dat = shape['geometry']['coordinates']

                        if shape['geometry']['type'] != 'Polygon':
                           raise TypeError('Shape type should be Polygon (no Point, MultiPolygon etc..)')

                        self.shape_string = '('
                        shape_len = len(dat[0])
                        for p in dat[0]:
                            self.shape_string = self.shape_string + str(p[0]) + ' ' + str(p[1]) + ','
                        self.shape_string = self.shape_string[:-1] + ')'

                        break
            else:
                print('format not recognized! Pleas creat either a .kml or .shp file.')
                return []
            if shape_len > 100:
                print('The shapesize is larger than 100 points, this could cause problems with the download')
        elif isinstance(self.shape, Polygon):
            dat = self.shape.exterior.coords

            self.shape_string = '('
            shape_len = len(dat)
            for p in dat:
                self.shape_string = self.shape_string + str(p[0]) + ' ' + str(p[1]) + ','
            self.shape_string = self.shape_string[:-1] + ')'


class DownloadSentinelOrbit(object):

    def __init__(self, start_date='', end_date='', precise_folder='', restituted_folder='', download_source='ESA'):
        # This script downloads all orbits files from the precise orbits website, when pages is set to a very high number.
        # By default only the first page for the last two days (restituted) is checked.

        self.precise_folder = precise_folder
        self.restituted_folder = restituted_folder

        last_precise = '' # Last precise orbit file. Used to remove unused restituted orbit files.
        gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)

        # From now on the start date and end date should be given to find the right path.
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        t_step = datetime.timedelta(days=1)
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        self.precise_files = []
        self.precise_links = []
        self.precise_dates = []

        date = start_date

        if download_source == 'ESA':
            if self.precise_folder:
                while (end_date + t_step * 21)  > date:
                    # First extract the orbitfiles from the page.
                    url = 'https://aux.sentinel1.eo.esa.int/POEORB/' + str(date.year) + '/' + str(date.month).zfill(2) + '/' + str(date.day).zfill(2) +  '/'

                    try:
                        page = urllib.request.urlopen(url, context=gcontext)
                        html = page.read().decode("utf8").split('\n')

                        for line in html:
                            if re.search('<a .*href=.*>', line):
                                if re.search('EOF', line):
                                    dat = re.search('<a href=.*>(.*)</a>', line)
                                    self.precise_files.append(dat.group(1))
                                    print('Linked ' + dat.group(1))
                                    self.precise_links.append(url + dat.group(1))
                                    date_poeorb = datetime.datetime.strptime(dat.group(1)[42:50], '%Y%m%d') + t_step
                                    self.precise_dates.append(date_poeorb)
                    except:
                        print('No precise orbit found for ' + date.strftime('%Y-%m-%d'))

                    date += t_step

            self.restituted_files = []
            self.restituted_links = []
            self.restituted_dates = []

            date = np.max(self.precise_dates)

            if self.restituted_folder:
                while (end_date + 2 * t_step) > date:
                    # First extract the orbitfiles from the page.
                    url = 'https://aux.sentinel1.eo.esa.int/RESORB/' + str(date.year) + '/' + str(date.month).zfill(2) + '/' + str(date.day).zfill(2) + '/'

                    try:
                        page = urllib.request.urlopen(url, context=gcontext)
                        html = page.read().decode("utf8").split('\n')

                        for line in html:
                            if re.search('<a .*href=.*>', line):
                                if re.search('EOF', line):
                                    dat = re.search('<a href=.*>(.*)</a>', line)
                                    self.restituted_files.append(dat.group(1))
                                    print('Linked ' + dat.group(1))
                                    self.restituted_links.append(url + dat.group(1))
                    except:
                        print('No restituted orbit found for ' + date.strftime('%Y-%m-%d'))

                    date += t_step

    def download_orbits(self):
        # Do the actual download

        gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)

        if self.precise_folder:
            for orb, url in zip(self.precise_files, self.precise_links):
                # Download the orbit files
                filename = os.path.join(self.precise_folder, orb)
                if not os.path.exists(filename):
                    url_dat = urllib.request.urlopen(url, context=gcontext).read().decode("utf8")
                    f = open(filename, 'w')
                    f.write(url_dat)
                    f.close()
                    print(filename + ' downloaded')

        if self.restituted_folder:
            for orb, url in zip(self.restituted_files, self.restituted_links):
                # Download the orbit files
                filename = os.path.join(self.restituted_folder, orb)
                if not os.path.exists(filename):
                    url_dat = urllib.request.urlopen(url, context=gcontext).read().decode("utf8")
                    f = open(filename, 'w')
                    f.write(url_dat)
                    f.close()
                    print(filename + ' downloaded')
