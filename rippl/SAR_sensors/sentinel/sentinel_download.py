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
from lxml import etree
import json
from shapely.geometry import Polygon
import shapely
from shapely.ops import cascaded_union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mpl_polygon

from rippl.user_settings import UserSettings
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes

"""
Test 

import datetime

start_date = datetime.datetime(year=2018, month=4, day=7)
end_date = datetime.datetime(year=2018, month=5, day=21)

coordinates = [(0, 32.28), (2, 32.28), (2, 33.24), (0, 33.24), (0, 32.28)]
study_area = ReadWriteShapes()
study_area(coordinates)

shape = study_area.shape

mode = 'IW'
product = 'SLC'
orbit_direction = 'DSC'

self = DownloadSentinel(start_date, end_date, '', '', shape, '', '', orbit_direction, 'IW', product)
self.sentinel_search_ASF()
self.sentinel_search_ESA()
self.summarize_search_results()

self.sentinel_download_ESA()
self.sentinel_download_ASF()

"""


class DownloadSentinel(object):

    def __init__(self, start_date='', end_date='', start_dates='', end_dates='', date='', dates='', time_window='',
                 shape='', track='', polarisation='',
                 orbit_direction='', sensor_mode='IW', product='SLC', instrument_name='Sentinel-1'):
        # Following variables can be used to make a selection.
        # shape > defining shape file or .kml
        # start_date > first day for downloads (default one month before now) [yyyymmdd]
        # end_date > last day for downloads (default today)
        # track > the tracks we want to check (default all)
        # polarisation > which polarisation will be used. (default all)

        # string is the field we enter as url
        self.settings = UserSettings()
        self.settings.load_settings()
        self.products = []
        self.polarisations = []
        self.footprints = []
        self.tracks = []
        self.orbit_directions = []
        self.ids = []
        self.dates = []

        if not isinstance(shape, Polygon):
            shape_data = ReadWriteShapes()
            shape_data(shape)
            self.shape = shape_data.shape  # type:Polygon
        else:
            self.shape = shape
        self.shape_string = ''
        self.create_shape_str()

        self.valid_files = []
        self.invalid_files = []

        self.instrument_name = instrument_name
        self.sensor_mode = sensor_mode
        self.product = product
        if orbit_direction in ['a', 'A', 'ASC', 'Asc', 'ASCENDING', 'Ascending', 'ascending', 'asc']:
            self.orbit_direction_ESA = 'ascending'
            self.orbit_direction_ASF = 'ASC'
        elif orbit_direction in ['D', 'd', 'DESC', 'desc', 'Desc', 'DESCENDING', 'Descending', 'descending']:
            self.orbit_direction_ESA = 'descending'
            self.orbit_direction_ASF = 'DESC'
        else:
            self.orbit_direction_ESA = ''
            self.orbit_direction_ASF = ''
        self.track = track
        self.shape = shape
        self.polarisation = polarisation

        self.start_dates = []
        self.end_dates = []

        if isinstance(date, datetime.datetime):
            dates = [date]
        elif isinstance(dates, datetime.datetime):
            dates = [dates]

        # Create a list of search windows with start and end dates
        if isinstance(dates, list):
            for date in dates:
                if not isinstance(date, datetime.datetime):
                    raise TypeError('Input dates should be datetime objects.')

            if isinstance(time_window, datetime.timedelta):
                self.start_dates = [date - time_window for date in dates]
                self.end_dates = [date + time_window for date in dates]
            else:
                self.start_dates = [date.replace(hour=0, minute=0, second=0, microsecond=0) for date in dates]
                self.end_dates = [date + datetime.timedelta(days=1) for date in self.start_dates]

        elif isinstance(start_date, datetime.datetime) and isinstance(end_date, datetime.datetime):
            self.start_dates = [start_date]
            self.end_dates = [end_date]
        elif isinstance(start_dates, list) and isinstance(end_dates, list):
            self.start_dates = start_dates
            self.end_dates = end_dates
            valid_dates = [isinstance(start_date, datetime.datetime) * isinstance(end_date, datetime.datetime) for
                           start_date, end_date in zip(start_dates, end_dates)]
            if np.sum(valid_dates) < len(valid_dates):
                raise TypeError('Input dates should be datetime objects.')
        else:
            raise TypeError('You should define a start or end date or a list of dates to search for Sentinel-1 data! '
                            'Dates should be datetime objects')

        self.start_dates = np.array(self.start_dates)
        self.end_dates = np.array(self.end_dates)

    def search_string_ESA(self, start, end):

        ESA_string = ''
        if not isinstance(start, datetime.datetime) or not isinstance(end, datetime.datetime):
            print('Start and end date should be datetime objects!')

        if self.instrument_name:
            ESA_string += ' AND ' + 'platformname:' + self.instrument_name
        if self.sensor_mode:
            ESA_string += ' AND ' + 'sensoroperationalmode:' + self.sensor_mode
        if self.product:
            ESA_string += ' AND ' + 'producttype:' + self.product
        if self.orbit_direction_ESA:
            ESA_string += ' AND ' + 'orbitdirection:' + self.orbit_direction_ESA
        if self.track:
            ESA_string += ' AND ' + 'relativeorbitnumber:' + str(self.track)
        if self.polarisation:
            ESA_string += ' AND ' + 'polarisationmode:' + self.polarisation
        if self.shape:
            ESA_string += ' AND footprint:"Intersects(POLYGON(' + self.shape_string + '))"'

        date_string = 'beginposition:[' + start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z TO ' + \
                      end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z] AND endposition:[' + \
                      start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z TO ' + end.strftime('%Y-%m-%dT%H:%M:%S.%f')[
                                                                              :-3] + 'Z]'
        ESA_string += ' AND ' + date_string
        ESA_string = ESA_string[5:]

        return ESA_string

    def search_string_ASF(self, start, end):

        ASF_string = ''
        if not isinstance(start, datetime.datetime) or not isinstance(end, datetime.datetime):
            print('Start and end date should be datetime objects!')

        if self.instrument_name:
            ASF_string += '&platform=' + self.instrument_name
        if self.sensor_mode:
            ASF_string += '&beamMode=' + self.sensor_mode
        if self.product:
            ASF_string += '&processingLevel=' + self.product
        if self.orbit_direction_ASF:
            ASF_string += '&flightDirection=' + self.orbit_direction_ASF
        if self.track:
            ASF_string += '&relativeOrbit=' + str(self.track)
        if self.polarisation:
            if self.polarisation == 'HH':
                pol_str = 'HH,HH+HV'
            elif self.polarisation == 'HV':
                pol_str = 'HH+HV'
            elif self.polarisation == 'VV':
                pol_str = 'VV,VV+VH'
            elif self.polarisation == 'VH':
                pol_str = 'VV+VH'
            elif self.polarisation == ['HH', 'HV']:
                pol_str = 'HH+HV'
            elif self.polarisation == ['VV', 'VH']:
                pol_str = 'VV+VH'
            else:
                pol_str = self.polarisation
            ASF_string += '&polarization=' + pol_str
        if self.shape:
            ASF_string += '&intersectsWith=polygon(' + self.shape_string + ')'
        ASF_string += '&output=JSON'

        ASF_string += '&start=' + start.strftime('%Y-%m-%dT%H:%M:%S') + 'UTC' + \
                      '&end=' + end.strftime('%Y-%m-%dT%H:%M:%S') + 'UTC'
        ASF_string = ASF_string[1:]

        return ASF_string

    def sentinel_search_ASF(self, username='', password=''):
        # All available sentinel 1 images are detected and printed on screen.

        if not username:
            username = self.settings.NASA_username
        if not password:
            password = self.settings.NASA_password
        self.products = []
        self.footprints = []
        self.tracks = []
        self.orbit_directions = []
        self.ids = []
        self.dates = []
        self.polarisations = []

        for start, end in zip(self.start_dates, self.end_dates):

            ASF_string = self.search_string_ASF(start, end)

            # Finally we do the query to get the search result.
            url = 'https://api.daac.asf.alaska.edu/services/search/param?' + urllib.parse.quote_plus(ASF_string, safe='=&')

            request = urllib.request.Request(url)
            base64string = base64.b64encode(bytes('%s:%s' % (username, password), "utf-8")).decode()
            request.add_header("Authorization", "Basic " + base64string)

            # connect to server. Hopefully this works at once
            # Check for paging https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/OpenSearchAPI#Paging_results
            try:
                dat = urllib.request.urlopen(request)
            except:
                raise ConnectionError('Not possible to connect to ASF server')

            html_dat = ''
            for line in dat:
                html_dat = html_dat + line.decode('UTF-8')

            products = json.loads(html_dat)[-1]
            self.products.extend(products)

            for product in products:
                self.polarisations.append(product['polarization'].replace('+', ''))
                self.footprints.append(shapely.wkt.loads(product['stringFootprint']))
                self.tracks.append(int(product['track']))
                self.orbit_directions.append(product['flightDirection'])
                self.ids.append(product['sceneId'])
                self.dates.append(datetime.datetime.strptime(product['sceneDate'], '%Y-%m-%dT%H:%M:%S.%f'))

        if len(self.dates) == 0:
            print('No images found!')

    def sentinel_search_ESA(self, username='', password=''):
        # All available sentinel 1 images are detected and printed on screen.

        if not username:
            username = self.settings.ESA_username
        if not password:
            password = self.settings.ESA_password
        self.products = []
        self.footprints = []
        self.tracks = []
        self.orbit_directions = []
        self.ids = []
        self.dates = []
        self.polarisations = []
        page = 0

        all_loaded = False
        for start_date, end_date in zip(self.start_dates, self.end_dates):

            ESA_string = self.search_string_ESA(start_date, end_date)

            while all_loaded == False:
                # Finally we do the query to get the search result.
                url = 'https://scihub.copernicus.eu/dhus/search?start=' + str(
                    page) + '&rows=100&q=' + urllib.parse.quote_plus(ESA_string)

                request = urllib.request.Request(url)
                base64string = base64.b64encode(bytes('%s:%s' % (username, password), "utf-8")).decode()
                request.add_header("Authorization", "Basic " + base64string)

                # connect to server. Hopefully this works at once
                # Check for paging https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/OpenSearchAPI#Paging_results
                try:
                    dat = urllib.request.urlopen(request)
                except:
                    raise ConnectionError('Not possible to connect to ESA server')

                html_dat = ''
                for line in dat:
                    html_dat = html_dat + line.decode('UTF-8')

                parser = etree.HTMLParser()
                tree = etree.fromstring(html_dat.encode('utf-8'), parser)
                # Check whether we got all result.
                if not tree[0][0][5].text:
                    raise LookupError('No images found')

                total_items = int(tree[0][0][5].text)
                start_index = int(tree[0][0][6].text)
                page_items = int(tree[0][0][7].text)
                if start_index + page_items >= total_items:
                    all_loaded = True
                else:
                    page += 1

                self.products.extend([data for data in tree.iter(tag='entry')])
                self.polarisations.extend([str(data.find("str[@name='polarisationmode']").text).replace(' ', '') for data in tree.iter(tag='entry')])
                self.ids.extend([data.find("str[@name='identifier']").text for data in tree.iter(tag='entry')])
                self.tracks.extend([int(data.find("int[@name='relativeorbitnumber']").text) for data in tree.iter(tag='entry')])
                self.footprints.extend([shapely.wkt.loads(data.find("str[@name='footprint']").text) for data in tree.iter(tag='entry')])
                self.orbit_directions.extend([data.find("str[@name='orbitdirection']").text for data in tree.iter(tag='entry')])
                self.dates.extend([datetime.datetime.strptime(data.find("date[@name='beginposition']").text[:19], '%Y-%m-%dT%H:%M:%S') for data in tree.iter(tag='entry')])

        if len(self.dates) == 0:
            print('No images found!')

    def sentinel_download_ESA(self, database_folder='', username='', password=''):
        # Download the files which are found by the sentinel_available script.

        if not username:
            username = self.settings.ESA_username
        if not password:
            password = self.settings.ESA_password
        if not database_folder:
            database_folder = os.path.join(self.settings.radar_database, 'Sentinel-1')

        if not self.products:
            raise FileNotFoundError('No images found!')

        wget_base = 'wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 --continue --tries=20 ' \
                    '--no-check-certificate --user=' + username + ' --password=' + password + ' '

        for product_name, date, track, polarisation, direction in zip(self.ids, self.dates, self.tracks, self.polarisations, self.orbit_directions):
            file_dir = DownloadSentinel.find_database_folder(database_folder, product_name, date, track, polarisation, direction)
            url = "https://scihub.copernicus.eu/dhus/odata/v1/Products?$filter=Name%20eq%20'" + product_name + "'"
            request = urllib.request.Request(url)

            base64string = base64.b64encode(bytes('%s:%s' % (username, password), "utf-8")).decode()
            request.add_header("Authorization", "Basic " + base64string)
            try:
                dat = urllib.request.urlopen(request)
                html_dat = ''
                for line in dat:
                    html_dat = html_dat + line.decode('UTF-8')

                parser = etree.HTMLParser()
                tree = etree.fromstring(html_dat.encode('utf-8'), parser)
                uuid = [data.find('id').text.split("'")[-2] for data in tree.iter(tag='entry')][0]
            except:
                raise ConnectionError('Not possible to connect to ESA server')
            # Now extract the uuid from here.

            url = str('"' + "https://scihub.copernicus.eu/dhus/odata/v1/Products('" + uuid + "')/" + urllib.parse.quote_plus('$value') + '"')

            # Download data files and create symbolic link
            if not os.path.exists(file_dir):
                wget_data = wget_base + url + ' -O ' + file_dir
                print(wget_data)
                os.system(wget_data)

    def sentinel_download_ASF(self, database_folder='', username='', password=''):
        # Download data from ASF (Generally easier and much faster to download from this platform.)

        if not username:
            username = self.settings.NASA_username
        if not password:
            password = self.settings.NASA_password
        if not database_folder:
            database_folder = os.path.join(self.settings.radar_database, 'Sentinel-1')

        if not self.products:
            raise FileNotFoundError('No images found!')

        wget_base = 'wget -c --http-user=' + username + ' --http-password=' + password + ' --no-check-certificate '

        for product_name, date, track, polarisation, direction in zip(self.ids, self.dates, self.tracks, self.polarisations, self.orbit_directions):
            file_dir = DownloadSentinel.find_database_folder(database_folder, product_name, date, track, polarisation, direction)
            url = '"https://datapool.asf.alaska.edu/SLC/SA/' + os.path.basename(file_dir) + '"'

            # Download data files and create symbolic link
            if not os.path.exists(file_dir):
                wget_data = wget_base + url + ' -O ' + file_dir
                print(wget_data)
                os.system(wget_data)

    def summarize_search_results(self, plot_cartopy=True, buffer=5):

        if len(self.dates) == 0:
            print('No images found to visualize')
            return

        # Sort by date and track
        date_track = np.array([d.date().isoformat() + '_' + str(t) for d, t in zip(self.dates, self.tracks)])
        unique, unique_ids = np.unique(date_track, return_index=True)

        dates = np.array(self.dates)[unique_ids]
        tracks = np.array(self.tracks)[unique_ids]
        ids_list = []
        direction = np.array(self.orbit_directions)[unique_ids]
        polygons = []
        coverage = []

        AOI_area = self.shape.area

        # Create polygon per date
        for u_val in unique:
            ids = np.ravel(np.argwhere(date_track == u_val))
            ids_list.append(ids)
            polygons.append(cascaded_union(np.array(self.footprints)[ids]))
            coverage.append(polygons[-1].intersection(self.shape).area / AOI_area)

        # Create polygon per track
        polygons = np.array(polygons)
        coverage = np.array(coverage)
        unique_tracks, unique_ids = np.unique(tracks, return_index=True)
        direction = direction[unique_ids]
        track_polygon = []
        average_coverages = []
        image_nos = []

        for track in unique_tracks:
            ids = np.ravel(np.argwhere(tracks == track))
            track_polygon.append(cascaded_union(polygons[ids]))
            image_nos.append(len(ids))
            average_coverages.append(np.mean(coverage[ids]))

        bb_buffer = np.array(cascaded_union(track_polygon + [self.shape]).buffer(buffer).bounds)[np.array([0,2,1,3])]

        # Plot polygon for different tracks
        for n, polygon in enumerate(track_polygon):
            title = str(int(average_coverages[n] * 100)) + '% coverage for ' + direction[n].lower() + ' track ' + str(unique_tracks[n])
            if isinstance(polygon, Polygon):
                polygons = [polygon]
            else:
                polygons = [pol for pol in polygon]

            if plot_cartopy:
                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_title(title)
                ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='g'))
                ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
                ax.add_geometries(polygons, ccrs.PlateCarree(), facecolor='b', alpha=0.5)
                ax.add_geometries([self.shape], ccrs.PlateCarree(), facecolor='r', alpha=0.8)
                ax.set_extent(list(bb_buffer), ccrs.PlateCarree())
                plt.show()
            else:
                ax = plt.axes()
                shape_x, shape_y = self.shape.exterior.xy
                for polygon in polygons:
                    dat_x, dat_y = polygon.exterior.xy
                    ax.fill(dat_x, dat_y, alpha=0.5, fc='blue', ec='none')
                ax.fill(shape_x, shape_y, alpha=0.5, fc='r', ec='none')
                ax.set_title(title)
                ax.set_xlim([bb_buffer[0], bb_buffer[1]])
                ax.set_ylim([bb_buffer[2], bb_buffer[3]])
                plt.show()

        # Finally print list available images with their coverage.
        print('Summary statistics for Sentinel-1 search:')
        for track, average_coverage, image_no, direc in zip(unique_tracks, average_coverages, image_nos, direction):
            print('Stack for ' + direc.lower() + ' track ' + str(track) + ' contains ' + str(image_no) + ' images with ' + str(int(average_coverage * 100)) + '% coverage of area of interest')

        for track in unique_tracks:
            print('')
            print('Full list of images for track ' + str(track) + ':')

            # Sort all images for this track
            id_dates = np.ravel(np.argwhere(tracks == track))
            sorted_ids = id_dates[np.ravel(np.argsort(dates[id_dates]))]
            for id in sorted_ids:

                print(dates[id].isoformat() + ' with a coverage of ' + str(int(coverage[id] * 100)) + '% consists of SAR products:')
                for id_image in ids_list[id]:
                    print('          ' + self.ids[id_image])

    def sentinel_check_validity(self, database_folder, username, password):
        # Check whether the zip files can be unpacked or not. This is part of the project_functions procedure.

        if not username:
            username = self.settings.ESA_username
        if not password:
            password = self.settings.ESA_password
        if not database_folder:
            database_folder = os.path.join(self.settings.radar_database, 'Sentinel-1')

        for product in self.products:
            uuid = product.find('id').text
            checksum_url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('" + uuid + "')/Checksum/Value/" + urllib.parse.quote_plus(
                '$value')
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
                raise FileNotFoundError('md5 check sum failed for ' + file_dir)

        return True

    @staticmethod
    def find_database_folder(database_folder, product_name, date, track, polarisation, direction):
        # Find the destination folder in the database.

        data_type = product_name[4:16]
        if direction == 'ASCENDING':
            direction = 'asc'
        elif direction == 'DESCENDING':
            direction = 'dsc'

        track_folder = os.path.join(database_folder, 's1_' + direction + '_t' + str(track))
        if not os.path.exists(track_folder):
            os.mkdir(track_folder)
        type_folder = os.path.join(track_folder, data_type + '_' + polarisation)
        if not os.path.exists(type_folder):
            os.mkdir(type_folder)
        date_folder = os.path.join(type_folder, date.strftime('%Y%m%d'))
        if not os.path.exists(date_folder):
            os.mkdir(date_folder)

        file_dir = os.path.join(date_folder, product_name + '.zip')

        return file_dir

    def create_shape_str(self):
        # This script converts .shp files to the right format. If multiple shapes are available the script
        # will select the first one.

        dat = self.shape.exterior.coords

        if not isinstance(self.shape, Polygon):
            raise TypeError('Shape type should be Polygon (no Point, MultiPolygon etc..)')

        self.shape_string = '('
        for p in dat:
            self.shape_string = self.shape_string + str(p[0]) + ' ' + str(p[1]) + ','
        self.shape_string = self.shape_string[:-1] + ')'


class DownloadSentinelOrbit(object):

    def __init__(self, start_date='', end_date='', precise_folder='', restituted_folder='', download_source='ESA'):
        # This script downloads all orbits files from the precise orbits website, when pages is set to a very high number.
        # By default only the first page for the last two days (restituted) is checked.

        settings = UserSettings()
        settings.load_settings()

        if not restituted_folder:
            restituted_folder = os.path.join(settings.orbit_database, 'Sentinel-1', 'restituted')
        if not precise_folder:
            precise_folder = os.path.join(settings.orbit_database, 'Sentinel-1', 'precise')
        self.precise_folder = precise_folder
        self.restituted_folder = restituted_folder

        last_precise = ''  # Last precise orbit file. Used to remove unused restituted orbit files.
        gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)

        # From now on the start date and end date should be given to find the right path.
        if not isinstance(start_date, datetime.datetime):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        t_step = datetime.timedelta(days=1)
        if not isinstance(end_date, datetime.datetime):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        self.precise_files = []
        self.precise_links = []
        self.precise_dates = []

        date = start_date
        print('Loading available orbit data files')

        if download_source == 'ESA':
            if self.precise_folder:
                while (end_date + t_step * 21) > date:
                    # First extract the orbitfiles from the page.
                    url = 'https://aux.sentinel1.eo.esa.int/POEORB/' + str(date.year) + '/' + str(date.month).zfill(
                        2) + '/' + str(date.day).zfill(2) + '/'

                    try:
                        page = urllib.request.urlopen(url, context=gcontext)
                        html = page.read().decode("utf8").split('\n')

                        for line in html:
                            if re.search('<a .*href=.*>', line):
                                if re.search('EOF', line):
                                    dat = re.search('<a href=.*>(.*)</a>', line)
                                    self.precise_files.append(dat.group(1))
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

            # Download only restituted files for the last 3 weeks.
            if self.restituted_folder and (datetime.datetime.today() - end_date).days < 21:
                while (end_date + 2 * t_step) > date:
                    # First extract the orbitfiles from the page.
                    url = 'https://aux.sentinel1.eo.esa.int/RESORB/' + str(date.year) + '/' + str(date.month).zfill(
                        2) + '/' + str(date.day).zfill(2) + '/'

                    try:
                        page = urllib.request.urlopen(url, context=gcontext)
                        html = page.read().decode("utf8").split('\n')

                        for line in html:
                            if re.search('<a .*href=.*>', line):
                                if re.search('EOF', line):
                                    dat = re.search('<a href=.*>(.*)</a>', line)
                                    self.restituted_files.append(dat.group(1))
                                    self.restituted_links.append(url + dat.group(1))
                    except:
                        print('No restituted orbit found for ' + date.strftime('%Y-%m-%d'))

                    date += t_step

        print('Finished loading date files')

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
