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
from fastkml import kml
from lxml import etree
from shapely.geometry import Polygon


class DownloadSentinel(object):
    
    def __init__(self, start_date, end_date, user, password, shape, track='', polarisation='', level='',
                 orbit_direction='', sensor_mode='IW', product='SLC'):
        # Following variables can be used to make a selection.
        # shape > defining shape file or .kml
        # start_date > first day for downloads (default one month before now) [yyyymmdd]
        # end_date > last day for downloads (default today)
        # track > the tracks we want to check (default all)
        # polarisation > which polarisation will be used. (default all)
    
        # string is the field we enter as url
        self.user = user
        self.password = password
        
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
            self.string = self.string + ' AND ' + 'relativeorbitnumber:' + track
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
    
        date_string = 'beginPosition:[' + start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z TO ' + \
                      end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z] AND endPosition:[' + \
                      start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z TO ' + end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z]'
        self.string = self.string + ' AND ' + date_string
    
    def sentinel_available(self):
        # All available sentinel 1 images are detected and printed on screen.

        # Finally we do the query to get the search result.
        self.string = self.string[5:]
        url = 'https://scihub.copernicus.eu/dhus/search?q=' + urllib.quote_plus(self.string)
    
        request = urllib.Request(url)
        base64string = base64.b64encode('%s:%s' % (self.user, self.password))
        request.add_header("Authorization", "Basic %s" % base64string)
    
        # connect to server. Hopefully this works at once
        try:
            dat = urllib.urlopen(request)
        except:
            print('not possible to connect this time')
            return [], [], []
    
        html_dat = ''
        for line in dat:
            html_dat = html_dat + line
    
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_dat, parser)
        self.products = [data for data in tree.iter(tag='entry')]
        self.links = [data.find('link').attrib for data in tree.iter(tag='entry')]
        self.dates = [data.findall('date')[1].text for data in tree.iter(tag='entry')]    

    def sentinel_download(self, xml_only=False, destination_folder=''):
        # Download the files which are found by the sentinel_available script.
    
        if not self.products:
            print('No files to project_functions')
            return
    
        wget_base = 'wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --continue --tries=20 ' \
                    '--no-check-certificate --user=' + self.user + ' --password=' + self.password + ' '
    
        for product in self.products:
            date = str(product.findall('date')[1].text)
            date = datetime.datetime.strptime(date[:19], '%Y-%m-%dT%H:%M:%S')
    
            url = str('"'+product.findall('link')[0].attrib['href'][:-6]+ urllib.quote_plus('$value') +'"')
            name = str(product.find('title').text)
    
            track = str(product.find('int[@name="relativeorbitnumber"]').text)
            data_type = str(product.find(".//str[@name='filename']").text)[4:16]
            pol = str(product.find(".//str[@name='polarisationmode']").text).replace(' ', '')
            direction = str(product.find(".//str[@name='orbitdirection']").text)
            if direction == 'ASCENDING':
                direction = 'asc'
            elif direction == 'DESCENDING':
                direction = 'dsc'
    
            track_folder = os.path.join(destination_folder, 's1_' + direction + '_t' + track)
            if not os.path.exists(track_folder):
                os.mkdir(track_folder)
            type_folder = os.path.join(track_folder, data_type + '_' + pol)
            if not os.path.exists(type_folder):
                os.mkdir(type_folder)
            date_folder = os.path.join(type_folder, date.strftime('%Y%m%d'))
            if not os.path.exists(date_folder):
                os.mkdir(date_folder)
    
            xml_dir = os.path.join(date_folder, name + '.xml')
            file_dir = os.path.join(date_folder, name + '.SAFE.zip')
            preview_dir = os.path.join(date_folder, name + '.jpg')
    
            # Save .xml files
            prod = etree.ElementTree(product)
    
            if not os.path.exists(xml_dir):
                prod.write(xml_dir, pretty_print = True)
    
            prev = "'preview'"
            png = "'quick-look.png'"
            dat = "'" + name + ".SAFE'"
    
            preview_url = url[:-10] + '/Nodes(' + dat + ')/Nodes(' + prev + ')/Nodes(' + png + ')/' + urllib.quote_plus('$value') + '"'
    
            # Download data files and create symbolic link
            if not xml_only: # So we also project_functions the file
                if not os.path.exists(file_dir):
                    wget_data = wget_base + url + ' -O ' + file_dir
                    os.system(wget_data)
    
                    # Finally check whether the file is downloaded correctly. Otherwise delete file and wait for next round of
                    # downloads.
                    uuid = product.find('id').text
                    valid = self.sentinel_quality_check(file_dir, uuid, self.user, self.password)
                else: # If the file already exists we assume it is valid.
                    valid = True
    
                if valid == True:
                    # First project_functions additional files
                    if not os.path.exists(preview_dir):
                        wget_preview = wget_base + preview_url + ' -O ' + preview_dir
                        os.system(wget_preview)
                else:
                    if os.path.exists(file_dir):
                        os.system('rm ' + file_dir)
                    if os.path.exists(xml_dir):
                        os.system('rm ' + xml_dir)
                    if os.path.exists(preview_dir):
                        os.system('rm ' + preview_dir)

    def sentinel_check_validity(self, destination_folder='', remove=True):
        # Check if the downloaded files are valid and remove if not

        if not self.products:
            print('Nothing to check')
            return
    
        for product in self.products:
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
    
            track_folder = os.path.join(destination_folder, 's1_' + direction + '_t' + track.zfill(3))
            type_folder = os.path.join(track_folder, data_type + '_' + pol)
            date_folder = os.path.join(type_folder, date.strftime('%Y%m%d'))
    
            xml_dir = os.path.join(date_folder, name + '.xml')
            file_dir = os.path.join(date_folder, name + '.SAFE.zip')
            kml_dir = os.path.join(date_folder, name + '.kml')
            preview_dir = os.path.join(date_folder, name + '.jpg')
    
            # First check the file
            if os.path.exists(file_dir):
                uuid = product.find('id').text
                valid_dat = self.sentinel_quality_check(file_dir, uuid)
            else:
                valid_dat = False
    
            if not valid_dat:
                if os.path.exists(file_dir) and remove == True:
                    os.system('rm ' + file_dir)
                if os.path.exists(xml_dir) and remove == True:
                    os.system('rm ' + xml_dir)
                if os.path.exists(kml_dir) and remove == True:
                    os.system('rm ' + kml_dir)
                if os.path.exists(preview_dir) and remove == True:
                    os.system('rm ' + preview_dir)
    
                self.invalid_files.append(file_dir)
            else:
                self.valid_files.append(file_dir)

    def load_shape_info(self):
        # This script converts .kml, .shp and .txt files to the right format. If multiple shapes are available the script
        # will select the first one.
        
        if isinstance(self.shape, str):
            if not os.path.exists(self.shape):
                print('Defined shapefile does not exist')
                return
    
            if self.shape.endswith('.shp'):
                with collection(self.shape, "r") as inputshape:
                    for shape in inputshape:
                        # only first shape
                        dat = shape['geometry']['coordinates']
        
                        self.shape_string='('
                        shape_len = len(dat[0])
                        for p in dat[0]:
                            self.shape_string = self.shape_string + str(p[0]) + ' ' + str(p[1]) + ','
                        self.shape_string = self.shape_string[:-1] + ')'
        
                        break
            elif self.shape.endswith('.kml'):
                doc = open(self.shape).read()
                k = kml.KML()
                k.from_string(doc)
                shape = list(list(k.features())[0].features())[0].geometry[0].exterior.coords[:]
                self.shape_string='('
                shape_len = len(shape)
                for p in shape:
                    self.shape_string = self.shape_string + str(p[0]) + ' ' + str(p[1]) + ','
                self.shape_string = self.shape_string[:-1] + ')'
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

    @staticmethod
    def sentinel_quality_check(filename, uuid, user, password):
        # Check whether the zip files can be unpacked or not. This is part of the project_functions procedure.
    
        checksum_url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('" + uuid + "')/Checksum/Value/" + urllib.quote_plus('$value')
        request = urllib.Request(checksum_url)
        base64string = base64.b64encode('%s:%s' % (user, password))
        request.add_header("Authorization", "Basic %s" % base64string)
    
        # connect to server. Hopefully this works at once
        try:
            dat = urllib.urlopen(request)
        except:
            print('not possible to connect this time')
            return False
    
        html_dat = ''
        for line in dat:
            html_dat = html_dat + line
    
        # Check file on disk
        if sys.platform == 'darwin':
            md5 = subprocess.check_output('md5 ' + filename, shell=True)[-33:-1]
        elif sys.platform == 'linux2':
            md5 = subprocess.check_output('md5sum ' + filename, shell=True)[:32]
        else:
            'This function only works on mac or linux systems!'
            return False
    
        if md5 == html_dat.lower():
            return True
        else:
            return False


class DownloadSentinelOrbit(object):

    def __init__(self, start_date='', end_date='', precise_folder='', restituted_folder=''):
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

        if self.precise_folder:
            while (end_date + t_step * 21)  > date:
                # First extract the orbitfiles from the page.
                url = 'https://aux.sentinel1.eo.esa.int/POEORB/' + str(date.year) + '/' + str(date.month).zfill(2) + '/' + str(date.day).zfill(2) +  '/'

                try:
                    page = urllib.urlopen(url, context=gcontext)
                    html = page.read().split('\n')

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
                    page = urllib.urlopen(url, context=gcontext)
                    html = page.read().split('\n')

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
                    urllib.urlretrieve(url, filename, context=gcontext)
                    print(filename + ' downloaded')

        if self.restituted_folder:
            for orb, url in zip(self.restituted_files, self.restituted_links):
                # Download the orbit files
                filename = os.path.join(self.restituted_folder, orb)
                if not os.path.exists(filename):
                    urllib.urlretrieve(url, filename, context=gcontext)
                    print(filename + ' downloaded')