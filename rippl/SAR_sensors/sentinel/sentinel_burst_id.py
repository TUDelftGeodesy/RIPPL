import copy
import os
import warnings
import zipfile
import urllib.request
import logging
from typing import Optional, List, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy

import fiona
import geopandas as gpd
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
import numpy as np
from shapely.geometry import Polygon

from rippl.user_settings import UserSettings

"""
# Test sentinel burst download and use 
from rippl.SAR_sensors.sentinel.sentinel_burst_id import SentinelBurstId
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes

# Initialize burst id and download burst id data
burst_ids = SentinelBurstId(mode='IW')

# Define the area of interest
Benelux_shape = [[7.218017578125001, 53.27178347923819],
                 [6.1962890625, 53.57293832648609],
                 [5.218505859375, 53.50111704294316],
                 [4.713134765624999, 53.20603255157844],
                 [3.3508300781249996, 51.60437164681676],
                 [3.284912109375, 51.41291212935532],
                 [2.39501953125, 51.103521942404186],
                 [3.8452148437499996, 50.127621728300475],
                 [4.493408203125, 49.809631563563094],
                 [5.361328125, 49.475263243037986],
                 [6.35009765625, 49.36806633482156],
                 [6.2841796875, 51.754240074033525],
                 [6.844482421875, 52.482780222078226],
                 [6.83349609375, 52.5897007687178],
                 [7.0751953125, 52.6030475337285],
                 [7.218017578125001, 53.27178347923819]]
study_area = ReadWriteShapes()
study_area(Benelux_shape)
aoi = study_area.shape.buffer(0)

# Find the relevant bursts and track
bursts = burst_ids(aoi, min_overlap_area_track=50, min_overlap_area_burst=0)

# Plot the bursts of different tracks
burst_ids.plot_coverage()

"""


class SentinelBurstId:

    def __init__(self, mode='IW', burst_id_folder: Optional[str] = None):

        self.tracks = []
        self.track_shapes = []
        self.mode = mode
        self.burst_ids = []
        self.bursts = dict()
        self.burst_shapes = dict()
        self.aoi = None

        # Store the loaded shapes from database
        self.track_database = None
        self.burst_database = dict()

        # Load burst id folder
        if not burst_id_folder:
            settings = UserSettings()
            settings.load_settings()
            settings = settings.settings
            sar_database = settings['paths']['orbit_database']
            sentinel_name = settings['path_names']['SAR']['Sentinel-1']
            self.burst_id_folder = os.path.join(sar_database, sentinel_name, 'burst_ids')
        else:
            self.burst_id_folder = burst_id_folder

        # Download the data if necessary
        self.download_burst_ids()

    def __call__(self, aoi: Polygon, track: Optional[Union[List[int], int]] = None,
                 min_overlap_area_track=0, min_overlap_area_burst=0):
        """
        Based on the polygon shape the burst id is selected. If coverage is not more than 80% a warning is given.

        """

        if not isinstance(aoi, Polygon):
            raise ValueError('The area of interest should be given as a shapely Polygon')
        else:
            self.aoi = aoi

        # Select relevant tracks
        if (isinstance(track, int) or isinstance(track, str)) and len(track) > 0:
            self.tracks = [track]
        elif isinstance(track, list) and len(track) > 0:
            self.tracks = track
        else:
            self.tracks, self.track_shapes = self.select_tracks(self.aoi, min_overlap_area_track)

        # Then loop over the different tracks
        self.bursts = dict()
        for track in self.tracks:
            self.bursts[track], self.burst_shapes[track] = self.select_bursts(track, self.aoi, min_overlap_area_burst)

        return self.bursts

    def select_bursts(self, track: Union[int, str], aoi: Polygon, min_overlap_area=0):
        """
        Select the burst ids for a specific track.

        :return:
        """

        track_str = str(track).zfill(3)

        if not track_str in self.burst_database.keys():
            track_file_folder = os.path.join(self.burst_id_folder, self.mode)
            track_files = os.listdir(track_file_folder)
            for track_file in track_files:
                if track_file.startswith(track_str) and track_file.endswith('kmz'):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.burst_database[track_str] = gpd.read_file(os.path.join(track_file_folder, track_file))
                    break

        # Select the overlapping bursts
        burst_shapes = self.burst_database[track_str][self.burst_database[track_str].intersects(aoi)]

        # Select the overlapping burst with minimal overlap area
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aoi_area = aoi.area
            burst_areas = burst_shapes.overlay(gpd.GeoDataFrame(geometry=[aoi]).set_crs(epsg=4326), how='intersection').area

        used_bursts = (np.array(burst_areas) / aoi_area) > (min_overlap_area / 100)
        bursts = list(np.array(burst_shapes['Name'])[used_bursts])
        burst_shapes = list(np.array(burst_shapes['geometry'])[used_bursts])

        return bursts, burst_shapes

    def select_tracks(self, aoi, min_overlap_area=0):
        """

        :param burst_id_folder:
        :return:
        """

        # Read tracks
        track_file = os.path.join(self.burst_id_folder, self.mode.lower() + '_tracks.gpkg')
        if not self.track_database:
            self.track_database = gpd.read_file(track_file)
        track_shapes = copy.deepcopy(self.track_database)

        # Find all overlapping tracks
        track_shapes = track_shapes[track_shapes.overlaps(aoi)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aoi_area = aoi.area
            track_areas = track_shapes.overlay(gpd.GeoDataFrame(geometry=[aoi]).set_crs(epsg=4326), how='intersection').area

        used_tracks = np.array(track_areas) / aoi_area > (min_overlap_area / 100)
        tracks = list(np.array(track_shapes['track'])[used_tracks])
        track_shapes = list(np.array(track_shapes['geometry'])[used_tracks])

        return tracks, track_shapes

    def download_burst_ids(self):
        """
        Download the Burst IDs

        """

        if not os.path.exists(self.burst_id_folder):
            os.makedirs(self.burst_id_folder)

        iw_folder = os.path.join(self.burst_id_folder, 'IW')
        ew_folder = os.path.join(self.burst_id_folder, 'EW')

        if not os.path.exists(iw_folder) or not os.path.exists(ew_folder):
            url = 'https://sar-mpc.eu/files/S1_burstid_20220530.zip'

            zip_path = os.path.join(self.burst_id_folder, 'burst_ids.zip')
            urllib.request.urlretrieve(url, zip_path)

            with zipfile.ZipFile(zip_path) as zip_file:
                for folder, file_type in zip([iw_folder, ew_folder], ['IW', 'EW']):
                    logging.info('Save burst id files to disk')

                    for zip_info in zip_file.infolist():
                        if zip_info.is_dir():
                            continue
                        zip_info.filename = os.path.basename(zip_info.filename)
                        if zip_info.filename.endswith('.kmz') and file_type in zip_info.filename:
                            zip_file.extract(zip_info, folder)

        # Create the shapes for the individual tracks. This way we can also check which tracks cover
        iw_tracks = os.path.join(self.burst_id_folder, 'iw_tracks.gpkg')
        ew_tracks = os.path.join(self.burst_id_folder, 'ew_tracks.gpkg')

        for track_file, track_folder in zip([iw_tracks, ew_tracks], [iw_folder, ew_folder]):
            if not os.path.exists(track_file):
                print('Combining burst to tracks for later search. This may take a while')
                burst_files = os.listdir(track_folder)
                combined_tracks = []

                for n, burst_file in enumerate(burst_files):
                    print('Convert to track shape ' + burst_file + '. ' + str(n + 1) + ' out of ' + str(len(burst_files)))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        input_shapes = gpd.read_file(os.path.join(track_folder, burst_file))
                    track = input_shapes.dissolve()

                    if len(combined_tracks) == 0:
                        combined_tracks = copy.deepcopy(track)
                    else:
                        combined_tracks = combined_tracks._append(track)

                # Simplify shape to save space on disk and speed up processing.
                combined_tracks['track'] = combined_tracks['Name'].apply(lambda x: x[:3])
                combined_tracks = combined_tracks.drop(columns=['description', 'timestamp', 'begin', 'end', 'altitudeMode', 'extrude',
                                              'visibility', 'drawOrder', 'icon', 'tessellate', 'Name'])
                combined_tracks['geometry'] = combined_tracks.simplify(0.05)
                combined_tracks.to_file(track_file)

    def plot_coverage(self):
        """
        Plot the coverage of the bursts over the area of interest.

        :return:
        """

        # Make a seperate plot for every track
        for track, track_shape in zip(self.tracks, self.track_shapes):
            bursts_shapes = self.burst_shapes[track]

            bb = np.array(gpd.GeoDataFrame(geometry=bursts_shapes).total_bounds) + np.array([-1, -1, 1, 1])

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([bb[0], bb[2], bb[1], bb[3]], ccrs.PlateCarree())

            plt.title('Coverage track and bursts from ' + self.mode + ' track ' + track)
            # Add coast and land
            ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='k')
            ax.coastlines(lw=1, resolution='10m')

            # Add AOI
            aoi_l = ax.add_geometries([self.aoi], ccrs.PlateCarree(), zorder=0, color='g', alpha=0.3)

            # Add track
            track_l = ax.add_geometries([track_shape], ccrs.PlateCarree(), color='b', alpha=0.2)

            # Add bursts
            bursts_l = ax.add_geometries(bursts_shapes, ccrs.PlateCarree(), facecolor='r', alpha=0.4)

            aoi_l = mpatches.Rectangle((0, 0), 1, 1, facecolor="g")
            track_l = mpatches.Rectangle((0, 0), 1, 1, facecolor="b")
            bursts_l = mpatches.Rectangle((0, 0), 1, 1, facecolor="r")
            within_2_deg = mpatches.Rectangle((0, 0), 1, 1, facecolor="#FF7E00")
            plt.legend([aoi_l, track_l, bursts_l], ['aoi', 'track ' + track, 'bursts'],
                       loc='lower left', bbox_to_anchor=(0.025, 0.025), fancybox=True)
            plt.show()
