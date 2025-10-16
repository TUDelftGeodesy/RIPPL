# This file contains a function to check which files for sentinel are available, which ones are downloaded and a quality
# check for the files which are downloaded.
from typing import Optional

import requests
from tqdm.auto import tqdm
import numpy as np
import logging
import os
import datetime
import base64
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
from multiprocessing import get_context

from rippl.SAR_sensors.sentinel.sentinel_burst_id import SentinelBurstId
from rippl.user_settings import UserSettings
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes

"""
# Test 
import rippl
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.SAR_sensors.sentinel.sentinel_image_download import DownloadSentinel
import datetime

start = datetime.datetime(year=2017, month=7, day=16)
end = datetime.datetime(year=2017, month=7, day=28)

Benelux_shape = [[7.218017578125001, 53.27178347923819],
                 [5.218505859375, 53.50111704294316],
                 [4.713134765624999, 53.20603255157844],
                 [3.3508300781249996, 51.60437164681676],
                 [3.8452148437499996, 50.127621728300475],
                 [4.493408203125, 49.809631563563094],
                 [6.35009765625, 49.36806633482156],
                 [6.83349609375, 52.5897007687178],
                 [7.218017578125001, 53.27178347923819]]
study_area = ReadWriteShapes()
study_area(Benelux_shape)
shape = study_area.shape.buffer(0.2)

mode = 'IW'
product = 'SLC'
orbit_direction = 'DSC'
n_processes = 4
track = 37

self = DownloadSentinel(start_date=start, end_date=end, track=track,
                        shape=shape, orbit_direction=orbit_direction, 
                        sensor_mode='IW', product=product, n_processes=n_processes)

# First test ASF normal search and download
self.sentinel_search_ASF()
self.summarize_search_results()
self.sentinel_download_ASF()

# Then test ESA search and download
self.sentinel_search_ESA()
self.summarize_search_results()
self.sentinel_download_ESA()

# Finally test ASF download burstwise (not working jet, most likely because not implemented at ASF. Possibly because
# only the database with extracted bursts is checked. However, over most areas they are not extracted yet.)
self = DownloadSentinel(start_date=start, end_date=end, use_burst_id=True,
                        shape=shape, orbit_direction=orbit_direction, 
                        sensor_mode='IW', product=product, n_processes=n_processes)
self.sentinel_search_ASF()
self.summarize_search_results()
self.sentinel_download_ASF()

"""


class ASFDownloader():

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __call__(self, download_data):
        # Do the download

        [product_name, date, track, polarisation, direction, size, file_path, pos] = download_data

        # Download
        url = 'https://datapool.asf.alaska.edu/SLC/SA/' + os.path.basename(file_path)
        edl_host = 'urs.earthdata.nasa.gov'
        edl_client_id = 'BO_n7nTIlMljdvU6kRRB3g'
        asf_auth_host = 'auth.asf.alaska.edu'
        login_url = f'https://{edl_host}/oauth/authorize?client_id={edl_client_id}&response_type=code&redirect_uri=https://{asf_auth_host}/login'

        req = requests.Session()
        req.auth = (self.username, self.password)
        response = req.get(login_url)

        # Download data files and create symbolic link
        response = req.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as file, \
                tqdm(desc=os.path.basename(file_path), total=total, unit='iB', unit_scale=True,
                     unit_divisor=1024, position=1) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = file.write(chunk)
                    bar.update(size)

class ESADownloader():

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __call__(self, download_data):

        [uuid, product_name, date, track, polarisation, direction, size, file_path, pos] = download_data

        # Get token
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                          data={"client_id": "cdse-public", "username": self.username, "password": self.password,
                                "grant_type": "password"})
        token = r.json()[("access_token")]

        # Now download the file
        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products(" + uuid + ")/$value"
        headers = {"Authorization": f"Bearer {token}"}

        response = requests.get(url, headers=headers, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as file, \
                tqdm(desc=os.path.basename(file_path), total=total, unit='iB', unit_scale=True,
                     unit_divisor=1024, position=1) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = file.write(chunk)
                    bar.update(size)


class DownloadSentinel(object):

    def __init__(self, start_date='', end_date='', start_dates='', end_dates='', date='', dates='', time_window='',
                 shape='', track='', polarisation='', n_processes=1, use_burst_id=False,
                 orbit_direction='', sensor_mode='IW', product='SLC', instrument_name='', max_num_products=1000):
        # Following variables can be used to make a selection.
        # shape > defining shape file or .kml
        # start_date > first day for downloads (default one month before now) [yyyymmdd]
        # end_date > last day for downloads (default today)
        # track > the tracks we want to check (default all)
        # polarisation > which polarisation will be used. (default all)

        # string is the field we enter as url
        self.n_processes = n_processes
        self.settings = UserSettings()
        self.settings.load_settings()
        self.sizes = []
        self.products = []
        self.polarisations = []
        self.footprints = []
        self.tracks = []
        self.orbit_directions = []
        self.ids = []
        self.dates = []
        self.use_burst_id = use_burst_id

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

        if not instrument_name in ['1A', '1B', '']:
            logging.error('Instrument name should either be 1A, 1B or left empty')
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
        self.track = str(track)
        self.orbit = ''
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
        self.data_collection = 'SENTINEL-1'
        self.max_num_products = str(max_num_products)

    def search_string_ESA(self,
                          start_date: Optional[datetime.datetime] = None,
                          end_date: Optional[datetime.datetime] = None) -> str:
        """"Get querying URL"""

        url_base = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
        if self.instrument_name:
            satellite_platform_letter = self.instrument_name[-1]
        else:
            satellite_platform_letter = ''

        # Initialize
        q = {}
        q_keys = ['name', 'area', 'date_start', 'date_end', 'orbit_direction',
                  'product_type', 'operational_mode', 'satellite_platform_letter',
                  'relative_orbit_number', 'orbit_number', 'polarisation_mode',
                  'include_string', 'exclude_string', 'limit']

        date_start_str = start_date.strftime('%Y-%m-%d')
        date_end_str = end_date.strftime('%Y-%m-%d')

        # Set
        q['name'] = f"Collection/Name eq '{self.data_collection}'" if self.data_collection else ""
        q['area'] = f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON({self.shape_string})')" if self.shape_string else ""
        q['date_start'] = f"ContentDate/Start gt {date_start_str}T00:00:00.000Z" if date_start_str else ""
        q['date_end'] = f"ContentDate/Start lt {date_end_str}T23:59:59.000Z" if date_end_str else ""
        q['orbit_direction'] = f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection' and att/OData.CSC.StringAttribute/Value eq '{self.orbit_direction_ESA}')" if self.orbit_direction_ESA else ""
        q['product_type'] = f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{self.product}')" if self.product else ""
        q['operational_mode'] = f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'operationalMode' and att/OData.CSC.StringAttribute/Value eq '{self.sensor_mode}')" if self.sensor_mode else ""
        q['satellite_platform_letter'] = f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'platformSerialIdentifier' and att/OData.CSC.StringAttribute/Value eq '{satellite_platform_letter}')" if satellite_platform_letter else ""
        q['relative_orbit_number'] = f"Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq 'relativeOrbitNumber' and att/OData.CSC.IntegerAttribute/Value eq {self.track})" if self.track else ""
        q['orbit_number'] = f"Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq 'orbitNumber' and att/OData.CSC.IntegerAttribute/Value eq {self.orbit})" if self.orbit else ""
        q['polarisation_mode'] = f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'polarisationChannels' and att/OData.CSC.StringAttribute/Value eq '{self.polarisation.replace(' ', '%26')}')" if self.polarisation else ""

        # limit is added separately
        q_limit = f"&$top={self.max_num_products}" if self.max_num_products else ""

        # Get all query items
        query_list = []
        for key, item in q.items():
            if item:
                query_list.append(item)

        # Join all query items
        query = " and ".join(query_list)

        # Prepend url
        url_req = f'{url_base}{query}{q_limit}&$expand=Attributes'

        return url_req

    def sentinel_search_ESA(self):
        # All available sentinel-1 images are detected and printed on screen.

        self.products = []
        self.footprints = []
        self.tracks = []
        self.orbit_directions = []
        self.ids = []
        self.dates = []
        self.polarisations = []
        self.product_names = []

        for start_date, end_date in zip(self.start_dates, self.end_dates):

            Copernicus_string = self.search_string_ESA(start_date, end_date)
            # Finally we do the query to get the search result.
            # connect to server. Hopefully this works at once
            try:
                r = requests.get(Copernicus_string)

                # Check status code
                if r.status_code != 200:
                    logging.error(f'Response code was {r.status_code}. Output: {r.text}. '
                                  f'Please verify.', stacklevel=2)
                else:
                    products = r.json()['value']
            except Exception as e:
                logging.error("Could not query CDSE.", exc_info=e, stacklevel=2)

            self.products.extend(products)

            for product in products:
                self.sizes.append(None)
                self.polarisations.append(product['Attributes'][17]['Value'].replace('&', ''))
                self.footprints.append(shapely.Polygon(product['GeoFootprint']['coordinates'][0]))
                self.tracks.append(int(product['Attributes'][16]['Value']))
                self.orbit_directions.append(product['Attributes'][8]['Value'])
                self.ids.append(product['Id'])
                self.product_names.append(product['Name'])
                if product['ContentDate']['Start'].endswith('Z'):
                    self.dates.append(
                        datetime.datetime.strptime(product['ContentDate']['Start'], '%Y-%m-%dT%H:%M:%S.%fZ'))
                else:
                    self.dates.append(
                        datetime.datetime.strptime(product['ContentDate']['Start'], '%Y-%m-%dT%H:%M:%S.%f'))

        if len(self.dates) == 0:
            logging.info('No images found!')

    def sentinel_download_ESA(self, database_folder='', username='', password=''):
        # Download the files which are found by the sentinel_available script.

        if not username:
            username = self.settings.settings['accounts']['Copernicus']['username']
        if not password:
            password = self.settings.settings['accounts']['Copernicus']['password']
        if not database_folder:
            radar_database = self.settings.settings['paths']['radar_database']
            folder = self.settings.settings['path_names']['SAR']['Sentinel-1']
            database_folder = os.path.join(radar_database, folder)

        if not self.products:
            raise FileNotFoundError('No images found!')

        downloader = ESADownloader(username, password)

        # Loop over all images
        download_dat = []
        pos = 0
        for uuid, product_name, date, track, polarisation, direction, size in zip(self.ids, self.product_names,
                                                                                  self.dates,
                                                                                  self.tracks, self.polarisations,
                                                                                  self.orbit_directions, self.sizes):
            file_path = DownloadSentinel.find_database_folder(database_folder, product_name, date, track,
                                                              polarisation, direction)
            if not os.path.exists(file_path):
                download_dat.append([uuid, product_name, date, track, polarisation, direction, size, file_path, pos])
                pos += 1

        if len(download_dat) == 0:
            logging.info('All images are already downloaded. Skipping download')
            return

        if self.n_processes > 1:
            with get_context("spawn").Pool(processes=self.n_processes, maxtasksperchild=5) as pool:
                # Process in blocks of 25
                block_size = 25
                for i in range(int(np.ceil(len(download_dat) / block_size))):
                    last_dat = np.minimum((i + 1) * block_size, len(download_dat))
                    pool.map(downloader, list(download_dat[i * block_size:last_dat]))
        else:
            for download_info in download_dat:
                downloader(download_info)

    def search_string_ASF(self, start, end):

        ASF_string = ''
        if not isinstance(start, datetime.datetime) or not isinstance(end, datetime.datetime):
            logging.info('Start and end date should be datetime objects!')

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
        # Get the burst IDs
        if self.use_burst_id:
            logging.info('Looking for burst IDs to search for')
            burst_ids_func = SentinelBurstId()
            self.burst_ids = burst_ids_func(aoi=self.shape, track=self.track)
            self.burst_id_str = ''
            for track in self.burst_ids.keys():
                for id in self.burst_ids[track]:
                    self.burst_id_str = self.burst_id_str + id.replace(' ', '_') + ','
            self.burst_id_str = self.burst_id_str[:-1]
            ASF_string += '&fullBurstID=' + self.burst_id_str
        elif self.shape:
            ASF_string += '&intersectsWith=polygon(' + self.shape_string + ')'
        ASF_string += '&output=JSON'

        ASF_string += '&start=' + start.strftime('%Y-%m-%dT%H:%M:%S') + 'UTC' + \
                      '&end=' + end.strftime('%Y-%m-%dT%H:%M:%S') + 'UTC'
        ASF_string = ASF_string[1:]

        return ASF_string

    def sentinel_search_ASF(self, username='', password=''):
        # All available sentinel-1 images are detected and printed on screen.

        if not username:
            username = self.settings.settings['accounts']['EarthData']['username']
        if not password:
            password = self.settings.settings['accounts']['EarthData']['password']
        self.products = []
        self.footprints = []
        self.tracks = []
        self.orbit_directions = []
        self.ids = []
        self.dates = []
        self.polarisations = []
        self.product_names = []

        for start, end in zip(self.start_dates, self.end_dates):

            ASF_string = self.search_string_ASF(start, end)

            # Finally we do the query to get the search result.
            url = f'https://api.daac.asf.alaska.edu/services/search/param?' + ASF_string

            base64string = base64.b64encode(bytes('%s:%s' % (username, password), "utf-8")).decode()
            headers = {"Authorization": "Basic " + base64string}

            # connect to server. Hopefully this works at once
            try:
                dat = requests.get(url, headers=headers)
            except Exception as e:
                raise ConnectionError('Not possible to connect to ASF server. ' + str(e))

            if len(dat.json()) == 0:
                raise ConnectionError('No images found!')

            products = dat.json()[-1]
            self.products.extend(products)

            for product in products:
                self.sizes.append(int(float(product['sizeMB'])))
                self.polarisations.append(product['polarization'].replace('+', ''))
                self.footprints.append(shapely.wkt.loads(product['stringFootprint']))
                self.tracks.append(int(product['track']))
                self.orbit_directions.append(product['flightDirection'])
                self.ids.append(None)
                self.product_names.append(product['productName'])
                if product['sceneDate'].endswith('Z'):
                    self.dates.append(datetime.datetime.strptime(product['sceneDate'][:-1], '%Y-%m-%dT%H:%M:%S.%f'))
                else:
                    self.dates.append(datetime.datetime.strptime(product['sceneDate'], '%Y-%m-%dT%H:%M:%S.%f'))

        if len(self.dates) == 0:
            logging.info('No images found!')

    def sentinel_download_ASF(self, database_folder='', username='', password=''):
        # Download data from ASF (Generally easier and much faster to download from this platform.)

        if not username:
            username = self.settings.settings['accounts']['EarthData']['username']
        if not password:
            password = self.settings.settings['accounts']['EarthData']['password']
        if not database_folder:
            radar_database = self.settings.settings['paths']['radar_database']
            folder = self.settings.settings['path_names']['SAR']['Sentinel-1']
            database_folder = os.path.join(radar_database, folder)

        if not self.products:
            raise FileNotFoundError('No images found!')

        downloader = ASFDownloader(username, password)

        # Loop over all images
        download_dat = []
        pos = 0
        for product_name, date, track, polarisation, direction, size in zip(self.product_names, self.dates, self.tracks,
                                                                            self.polarisations, self.orbit_directions,
                                                                            self.sizes):
            file_path = DownloadSentinel.find_database_folder(database_folder, product_name, date, track,
                                                              polarisation, direction)
            if not os.path.exists(file_path):
                download_dat.append([product_name, date, track, polarisation, direction, size, file_path, pos])
                pos += 1

        if len(download_dat) == 0:
            logging.info('All images are already downloaded. Skipping download')
            return

        download_dat = np.array(download_dat)

        if self.n_processes > 1:
            with get_context("spawn").Pool(processes=self.n_processes, maxtasksperchild=5) as pool:
                # Process in blocks of 25
                block_size = 25
                for i in range(int(np.ceil(len(download_dat) / block_size))):
                    last_dat = np.minimum((i + 1) * block_size, len(download_dat))
                    pool.map(downloader, list(download_dat[i * block_size:last_dat]))
        else:
            for download_info in download_dat:
                downloader(download_info)

    def summarize_search_results(self, plot=True, plot_cartopy=True, buffer=3):
        """
        Plot the overlap between the different tracks and the area of interest and summarize the available images for
        this search.

        :param plot:
        :param plot_cartopy:
        :param buffer:
        :return:
        """

        if len(self.dates) == 0:
            logging.info('No images found to visualize')
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
            polygons.append(unary_union(np.array(self.footprints)[ids]))
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
            track_polygon.append(unary_union(polygons[ids]))
            image_nos.append(len(ids))
            average_coverages.append(np.mean(coverage[ids]))

        bb_buffer = np.array(unary_union(track_polygon + [self.shape]).buffer(buffer).bounds)[np.array([0, 2, 1, 3])]

        # Plot polygon for different tracks
        for n, polygon in enumerate(track_polygon):
            title = str(int(average_coverages[n] * 100)) + '% coverage for ' + direction[n].lower() + ' track ' + str(
                unique_tracks[n])
            if isinstance(polygon, Polygon):
                polygons = [polygon]
            else:
                polygons = [pol for pol in polygon]

            if plot_cartopy:
                ax = plt.axes(projection=ccrs.PlateCarree())

                ax.set_title(title)
                ax.add_feature(cfeature.LAND, zorder=0, edgecolor='k')
                ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
                ax.add_geometries(polygons, ccrs.PlateCarree(), facecolor='b', alpha=0.5)
                ax.add_geometries([self.shape], ccrs.PlateCarree(), facecolor='r', alpha=0.5)

                # Add the coordinates on the sides
                ax.set_xticks(np.arange(-180, 181, 2), crs=ccrs.PlateCarree())
                lon_formatter = cticker.LongitudeFormatter()
                ax.xaxis.set_major_formatter(lon_formatter)

                ax.set_yticks(np.arange(-90, 91, 2), crs=ccrs.PlateCarree())
                lat_formatter = cticker.LatitudeFormatter()
                ax.yaxis.set_major_formatter(lat_formatter)

                ax.set_extent(list(bb_buffer), ccrs.PlateCarree())

                plt.show()
            elif plot:
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
        logging.info('Summary statistics for Sentinel-1 search:')
        for track, average_coverage, image_no, direc in zip(unique_tracks, average_coverages, image_nos, direction):
            logging.info('Stack for ' + direc.lower() + ' track ' + str(track).zfill(3) + ' contains ' + str(
                image_no) + ' images with ' + str(int(average_coverage * 100)) + '% coverage of area of interest')

        for track in unique_tracks:
            logging.info('')
            logging.info('Full list of images for track ' + str(track).zfill(3) + ':')

            # Sort all images for this track
            id_dates = np.ravel(np.argwhere(tracks == track))
            sorted_ids = id_dates[np.ravel(np.argsort(dates[id_dates]))]
            for id in sorted_ids:

                logging.info(dates[id].isoformat() + ' with a coverage of ' + str(
                    int(coverage[id] * 100)) + '% consists of SAR products:')
                for id_image in ids_list[id]:
                    logging.info(str('          ') + self.product_names[id_image])

    @staticmethod
    def find_database_folder(database_folder, product_name, date, track, polarisation, direction):
        # Find the destination folder in the database.

        data_type = product_name[4:16]
        if direction == 'ASCENDING':
            direction = 'asc'
        elif direction == 'DESCENDING':
            direction = 'dsc'

        track_folder = os.path.join(database_folder, 's1_' + direction + '_t' + str(track).zfill(3))
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
