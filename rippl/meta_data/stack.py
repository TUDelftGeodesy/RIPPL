"""
This class contains a full doris data_stack. This links also to several functions which can be solved based on a
full stack approach. This is also the basis from which we create network of interferograms, which can be used for
different purposes.

Examples of functions are:
- network solution for ESD
- network solution for Ionosphere (or combined with ESD)
- sbas or squisar solutions for the whole network
- network solution by including harmonie_nl data or ERA5 data with the interferograms

"""

import os
import copy
import datetime
import numpy as np
import logging
from shapely.ops import unary_union
from shapely import speedups
speedups.disable()
from multiprocessing import get_context

from rippl.run_parallel import run_parallel
from rippl.meta_data.slc import SLC
from rippl.meta_data.interferogram import Interferogram
from rippl.external_dems.srtm.srtm_download import SrtmDownload
from rippl.external_dems.tandem_x.tandem_x_download import TandemXDownload
from rippl.meta_data.interferogram_network import InterferogramNetwork
from rippl.meta_data.image_processing_concatenate import ImageConcatData, ImageProcessingData
from rippl.user_settings import UserSettings
from rippl.processing_steps.import_dem import ImportDem
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates
from rippl.resampling.coor_new_extend import CoorNewExtend


class Stack(object):

    def __init__(self, data_stack_folder='', SAR_type='Sentinel-1', data_stack_name=''):

        self.settings = UserSettings()
        self.settings.load_settings()

        if not data_stack_folder:
            if not data_stack_name:
                raise NotADirectoryError('The data_stack name is empty, so directory cannot be created!')
            if SAR_type not in list(self.settings.settings['path_names']['SAR'].keys()):
                raise ValueError('SAR_type should be ' + ' '.join(list(self.settings.settings['path_names']['SAR'].keys())))
            else:
                SAR_folder = self.settings.settings['path_names']['SAR'][SAR_type]
            self.data_stack_folder = os.path.join(self.settings.settings['paths']['radar_data_stacks'], SAR_folder, data_stack_name)
        else:
            self.data_stack_folder = data_stack_folder

        # List of images and interferograms
        self.slcs = dict()
        self.slc_dates = []
        self.ifgs = dict()
        self.ifg_dates = []

        self.dates = []
        self.coordinates = dict()

        # Oversight of the different interferograms and images
        self.ifg_matrix = []
        self.baseline_matrix = []

        # reference
        self.reference_date = ''

        # Specific information reference slices
        self.reference_slice_id = []
        self.reference_slice_date = []
        self.reference_slice_datetime = []
        self.reference_slice_azimuth_seconds = []
        self.reference_slice_range_seconds = []

    def get_processing_data(self, data_type, slice=False, include_reference_slc=False):
        """
        This function loads the data for the different data types (ifg, reference_slc, secondary_slc, slc)
        or their default radar coordinate systems.

        If slice is True all slices are loaded, otherwise only the full images.
        - slc > All slcs in the stack are loaded. If you want to work on coregistration or similar it is better to use
                the secondary_slc, as it gives the reference_slc images alongside these images.
        - ifg > All the ifg images of the stack. It will also add primary_slc, secondary_slc and reference_slc
        - reference_slc > The reference_slc of the stack, or the slices that are part of it
        - secondary_slc > Get all the secondary_slc images and the corresponding reference_slc images.

        :param data_type:
        :param slice:
        :return:
        """

        if data_type not in ['slc', 'ifg', 'secondary_slc', 'reference_slc']:
            raise TypeError('Only processing data for slcs, ifgs, secondary_slcs and reference_slc can be requested.'
                            'Change the data_type variable to one of these values.')

        reference_slc_name = self.reference_date
        slice_names = np.sort(self.slcs[reference_slc_name].slice_names)

        images = {'ifg': dict(), 'primary_slc': dict(), 'secondary_slc': dict(), 'reference_slc': dict()}
        coordinates = {'ifg': dict(), 'primary_slc': dict(), 'secondary_slc': dict(), 'reference_slc': dict()}

        if data_type == 'slc':
            slc_names = list(self.slcs.keys())
            for slc_name in slc_names:
                if slice:
                    for slice_name in slice_names:
                        if slice_name in self.slcs[slc_name].slice_data.keys():
                            images['secondary_slc'][slc_name + '_' + slice_name] = self.slcs[slc_name].slice_data[slice_name]
                            coordinates['secondary_slc'][slc_name + '_' + slice_name] = self.slcs[slc_name].slice_data[slice_name].radar_coordinates['original']
                else:
                    images['secondary_slc'][slc_name + '_full'] = self.slcs[slc_name]
                    coordinates['secondary_slc'][slc_name + '_full'] = self.slcs[slc_name].radar_coordinates['original']

        elif data_type == 'ifg':
            ifg_names = list(self.ifgs.keys())
            primary_slc_names = [ifg_key[:8] for ifg_key in ifg_names]
            secondary_slc_names = [ifg_key[9:] for ifg_key in ifg_names]

            for ifg_name, primary_slc_name, secondary_slc_name in zip(ifg_names, primary_slc_names, secondary_slc_names):
                if slice:
                    for slice_name in slice_names:
                        if slice_name in self.slcs[primary_slc_name].slice_data.keys() and \
                            slice_name in self.slcs[secondary_slc_name].slice_data.keys():

                            images['ifg'][ifg_name + '_' + slice_name] = self.ifgs[ifg_name].slice_data[slice_name]
                            images['primary_slc'][ifg_name + '_' + slice_name] = self.slcs[primary_slc_name].slice_data[slice_name]
                            images['secondary_slc'][ifg_name + '_' + slice_name] = self.slcs[secondary_slc_name].slice_data[slice_name]
                            images['reference_slc'][ifg_name + '_' + slice_name] = self.slcs[reference_slc_name].slice_data[slice_name]
                            coordinates['ifg'][ifg_name + '_' + slice_name] = self.slcs[reference_slc_name].slice_data[slice_name].radar_coordinates['original']
                            coordinates['primary_slc'][ifg_name + '_' + slice_name] = self.slcs[primary_slc_name].slice_data[slice_name].radar_coordinates['original']
                            coordinates['secondary_slc'][ifg_name + '_' + slice_name] = self.slcs[secondary_slc_name].slice_data[slice_name].radar_coordinates['original']
                            coordinates['reference_slc'][ifg_name + '_' + slice_name] = self.slcs[reference_slc_name].slice_data[slice_name].radar_coordinates['original']
                else:
                        images['ifg'][ifg_name + '_full'] = self.ifgs[ifg_name]
                        images['primary_slc'][ifg_name + '_full'] = self.slcs[primary_slc_name]
                        images['secondary_slc'][ifg_name + '_full'] = self.slcs[secondary_slc_name]
                        images['reference_slc'][ifg_name + '_full'] = self.slcs[reference_slc_name]
                        coordinates['ifg'][ifg_name + '_full'] = self.slcs[reference_slc_name].radar_coordinates['original']
                        coordinates['primary_slc'][ifg_name + '_full'] = self.slcs[primary_slc_name].radar_coordinates['original']
                        coordinates['secondary_slc'][ifg_name + '_full'] = self.slcs[secondary_slc_name].radar_coordinates['original']
                        coordinates['reference_slc'][ifg_name + '_full'] = self.slcs[reference_slc_name].radar_coordinates['original']

            if len(images['ifg']) == 0:
                logging.info('You are trying to get interferogram data, but there are no interferograms available. '
                             'Maybe you have to create the interferograms first?')

        elif data_type == 'reference_slc':

            if slice:
                for slice_name in slice_names:
                    images['reference_slc'][reference_slc_name + '_' + slice_name] = self.slcs[reference_slc_name].slice_data[slice_name]
                    coordinates['reference_slc'][reference_slc_name + '_' + slice_name] = self.slcs[reference_slc_name].slice_data[slice_name].radar_coordinates['original']
            else:
                images['reference_slc'][reference_slc_name + '_full'] = self.slcs[reference_slc_name]
                coordinates['reference_slc'][reference_slc_name + '_full'] = self.slcs[reference_slc_name].radar_coordinates['original']

        elif data_type == 'secondary_slc':

            if include_reference_slc:
                secondary_slc_names = list(self.slcs.keys())
            else:
                secondary_slc_names = [secondary_slc_name for secondary_slc_name in list(self.slcs.keys()) if secondary_slc_name != reference_slc_name]
            for secondary_slc_name in secondary_slc_names:
                if slice:
                    for slice_name in slice_names:
                        if slice_name in self.slcs[secondary_slc_name].slice_data.keys():
                            images['secondary_slc'][secondary_slc_name + '_' + slice_name] = self.slcs[secondary_slc_name].slice_data[slice_name]
                            images['reference_slc'][secondary_slc_name + '_' + slice_name] = self.slcs[reference_slc_name].slice_data[slice_name]
                            coordinates['secondary_slc'][secondary_slc_name + '_' + slice_name] = self.slcs[secondary_slc_name].slice_data[slice_name].radar_coordinates['original']
                            coordinates['reference_slc'][secondary_slc_name + '_' + slice_name] = self.slcs[reference_slc_name].slice_data[slice_name].radar_coordinates['original']
                else:
                    images['secondary_slc'][secondary_slc_name + '_full'] = self.slcs[secondary_slc_name]
                    images['reference_slc'][secondary_slc_name + '_full'] = self.slcs[reference_slc_name]
                    coordinates['secondary_slc'][secondary_slc_name + '_full'] = self.slcs[secondary_slc_name].radar_coordinates['original']
                    coordinates['reference_slc'][secondary_slc_name + '_full'] = self.slcs[reference_slc_name].radar_coordinates['original']

        return images, coordinates

    def load_radar_coordinates(self):
        """
        Create the radar coordinate system.

        :return:
        """

        # Radar grid without shapes defined
        self.coordinates['reference_slc'] = {'meta': dict()}
        self.coordinates['reference_slc']['meta']['buffer'] = 0
        self.coordinates['reference_slc']['meta']['rounding'] = 0
        self.coordinates['reference_slc']['meta']['min_height'] = 0
        self.coordinates['reference_slc']['meta']['max_height'] = 0

        self.coordinates['reference_slc']['full'] = self.get_processing_data('reference_slc', slice=False)[1]['reference_slc'][self.reference_date + '_full']
        self.coordinates['reference_slc']['full'].radar_grid_type = 'reference_slc' # type: CoordinateSystem
        slice_coors = self.get_processing_data('reference_slc', slice=True)[1]['reference_slc']
        for full_slice_name in slice_coors.keys():
            slice_name = 'slice' + full_slice_name.split('slice')[-1]
            slice_coor = slice_coors[full_slice_name]
            slice_coor.radar_grid_type = 'reference_slc'
            self.coordinates['reference_slc'][slice_name] = slice_coor

    def create_dem_coordinates(self, dem_type, lon_resolution=None, buffer=0, rounding=0, min_height=0, max_height=500):
        """
        Create the coordinate system for the dem

        :param dem_type:
        :param lon_resolution: Resolution in longitude in arc-seconds. This is used for TanDEM-X DEM, which is always
            3 arc-seconds in latitude, but varies in longitude.
        :return:
        """

        if dem_type not in ['SRTM1', 'SRTM3', 'TDM30', 'TDM90']:
            raise TypeError('dem_type should be SRTM1, SRTM3, TDM30 or TDM90'
                            'If you using a different external DEM, use the create_ml_coordinates function instead.')

        self.coordinates[dem_type] = {'meta': dict()}
        self.coordinates[dem_type]['meta']['buffer'] = buffer
        self.coordinates[dem_type]['meta']['rounding'] = rounding
        self.coordinates[dem_type]['meta']['min_height'] = min_height
        self.coordinates[dem_type]['meta']['max_height'] = max_height

        if not lon_resolution:
            if dem_type == 'TDM30':
                lon_resolution = 1
            elif dem_type == 'TDM90':
                lon_resolution = 3
        self.coordinates[dem_type]['full'] = ImportDem.create_dem_coor(dem_type, self.coordinates['reference_slc']['full'],
             buffer=buffer, rounding=rounding, min_height=min_height, max_height=max_height, lon_resolution=lon_resolution)

        # Do the same thing for the slices
        slice_names = [slice_name for slice_name in self.coordinates['reference_slc'].keys() if 'slice' in slice_name]
        for slice_name in slice_names:
            radar_coor = self.coordinates['reference_slc'][slice_name]
            new_coor = ImportDem.create_dem_coor(dem_type, radar_coor, buffer=buffer, rounding=rounding,
                                  min_height=min_height, max_height=max_height, lon_resolution=lon_resolution)
            self.coordinates[dem_type][slice_name] = new_coor

    def create_ml_coordinates(self, name='ml_coor', coor_type='geographic', multilook=[1,1], oversample=[1,1], shape='',
                              dlat=0.001, dlon=0.001, lat0=-90, lon0=-180, buffer=0, rounding=0,
                              min_height=0, max_height=0, overwrite=False,
                              dx=1, dy=1, x0=0, y0=0, projection_string='', UTM=False, oblique_mercator=False):
        """
        Create the coordinate system for multilooking. This can either be in radar coordinates, geographic or projected.

        :return:
        """

        if name in self.coordinates.keys() and not overwrite:
            logging.info('Coordinate system ' + name + ' already exists. You can overwrite by setting overwrite to True, '
                  'but if other datasets with this coordinate system already exists this can lead to processing errors '
                  'later on.')
            return

        # Initialize this coordinate system
        self.coordinates[name] = {'meta': dict()}
        self.coordinates[name]['meta']['buffer'] = buffer
        self.coordinates[name]['meta']['rounding'] = rounding
        self.coordinates[name]['meta']['min_height'] = min_height
        self.coordinates[name]['meta']['max_height'] = max_height

        if UTM:
            orbit_in = OrbitCoordinates(self.coordinates['reference_slc']['full'])
            proj_string = orbit_in.create_mercator_projection(UTM=True)
        elif oblique_mercator:
            orbit_in = OrbitCoordinates(self.coordinates['reference_slc']['full'])
            proj_string = orbit_in.create_mercator_projection(UTM=False)

        # First define the coordinate system. (for oblique mercator take te full image as a reference.)
        ml_coor = CoordinateSystem()
        if coor_type == 'radar_grid':
            ml_coor.create_radar_coordinates(multilook=multilook, oversample=oversample, shape=shape)
            ml_coor.orbit = copy.deepcopy(self.coordinates['reference_slc']['full'].orbit)
            ml_coor.readfile = copy.deepcopy(self.coordinates['reference_slc']['full'].readfile)
        elif coor_type == 'geographic':
            ml_coor.create_geographic(dlat, dlon, shape=shape, lon0=lon0, lat0=lat0)
        elif coor_type == 'projection':
            ml_coor.create_projection(dx, dy, shape=shape, projection_type=name, proj4_str=proj_string, x0=x0, y0=y0)

        # Then define the sizes for the full image and slices.
        self.coordinates[name]['full'] = CoorNewExtend(self.coordinates['reference_slc']['full'], ml_coor, buffer=buffer, rounding=rounding,
                                     dx=dx, dy=dy, dlat=dlat, dlon=dlon).out_coor
        slice_names = [slice_name for slice_name in self.coordinates['reference_slc'].keys() if 'slice' in slice_name]
        for slice_name in slice_names:
            reference_radar_coor = self.coordinates['reference_slc'][slice_name]
            new_coor = CoorNewExtend(reference_radar_coor, ml_coor, buffer=buffer, rounding=rounding,
                                     dx=dx, dy=dy, dlat=dlat, dlon=dlon).out_coor
            self.coordinates[name][slice_name] = new_coor

    def write_reference_slice_list(self):
        # az_time, slice id

        l = open(os.path.join(self.data_stack_folder, 'reference_slice_list'), 'w+')
        for az_time, date, id, range_seconds, azimuth_seconds in (
                zip(self.reference_slice_datetime, self.reference_slice_date, self.reference_slice_id,
                    self.reference_slice_range_seconds, self.reference_slice_azimuth_seconds)):
            l.write('id ' + id + ', datetime ' + az_time + ', reference_date '+ date +
                    ', range_seconds ' + str(range_seconds) + ', azimuth_seconds ' + str(azimuth_seconds) + ' \n')

        l.close()

    def read_reference_slice_list(self, force=False):
        # az_time, yyyy-mm-ddThh:mm:ss.ssssss, swath x, slice i, x xxxx, y yyyy, z zzzz, lat ll.ll, lon ll.ll, pol pp

        list_file = os.path.join(self.data_stack_folder, 'reference_slice_list')

        if not os.path.exists(list_file):
            logging.info('No existing reference slices list found')
            return False
        if len(self.reference_slice_datetime) != 0 and not force:
            logging.info('primary slice list already loaded!')
            return False

        self.reference_slice_datetime = []
        self.reference_slice_date = []
        self.reference_slice_id = []
        self.reference_slice_azimuth_seconds = []
        self.reference_slice_range_seconds = []
        self.reference_slices = []

        with open(list_file, 'r') as l:
            for line in l:
                sl = line.split(', ')
                id = sl[0].split(' ')[1]
                datetime_str = sl[1].split(' ')[1]
                date = sl[2].split(' ')[1]
                range_seconds = sl[3].split(' ')[1]
                azimuth_seconds = sl[4].split(' ')[1]
                if not id in self.reference_slice_id:
                    self.reference_slice_datetime.append(datetime_str)
                    self.reference_slice_id.append(id)
                    self.reference_slice_date.append(date)
                    self.reference_slice_azimuth_seconds.append(float(azimuth_seconds))
                    self.reference_slice_range_seconds.append(float(range_seconds))
                    self.reference_slices.append([])

        return True

    def read_stack(self, start_date='', end_date='', start_dates='', end_dates='', date='', dates='', time_window=''):
        # This function reads the whole stack in memory. A stack consists of:
        # - images > with individual slices (yyyymmdd)
        # - interferograms > with individual slices if needed. (yyyymmdd_yyyymmdd)
        # First date and last give the maximum and minimum date to load (in case we want to load only a part of the stack.)
        # Note: The reference date is always loaded!

        # Create a list of search windows with start and end dates
        if isinstance(date, datetime.datetime):
            dates = [date]
        elif isinstance(dates, datetime.datetime):
            dates = [dates]

        if not self.reference_date:
            self.read_reference_slice_list(force=True)

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
            raise TypeError('You should define a start or end date or a list of dates to read a stack '
                            'Dates should be datetime objects')

        self.start_dates = np.array(self.start_dates)
        self.end_dates = np.array(self.end_dates)

        dirs = next(os.walk(self.data_stack_folder))[1]
        all_image_dirs = sorted([os.path.join(self.data_stack_folder, x) for x in dirs if len(x) == 8 and x != 'coverage'])
        image_dirs = []
        all_ifg_dirs = sorted([os.path.join(self.data_stack_folder, x) for x in dirs if len(x) == 17])
        ifg_dirs = []

        for image_dir in all_image_dirs:
            date = datetime.datetime.strptime(os.path.basename(image_dir), '%Y%m%d')
            if np.sum((self.start_dates < date) * (self.end_dates > date)) > 0 or (
                    os.path.basename(image_dir) == self.reference_date):
                image_dirs.append(image_dir)

        for ifg_dir in all_ifg_dirs:
            date_1 = datetime.datetime.strptime(os.path.basename(ifg_dir)[:8], '%Y%m%d')
            date_2 = datetime.datetime.strptime(os.path.basename(ifg_dir)[9:], '%Y%m%d')
            if np.sum((self.start_dates < date_1) * (self.end_dates > date_1)) > 0 and \
                    np.sum((self.start_dates < date_2) * (self.end_dates > date_2)):
                ifg_dirs.append(ifg_dir)

        # Load individual images.
        for image_dir in image_dirs:
            if image_dir not in self.slc_dates:
                image_key = os.path.basename(image_dir)
                self.slcs[image_key] = SLC(image_dir, slice_list=self.reference_slice_id)
                self.slcs[image_key].load_meta()
                self.slcs[image_key].load_slice_meta()
                self.slc_dates.append(os.path.basename(image_dir))

        # Load reference date information.
        self.reference_date = self.reference_slice_date[0].replace('-', '')
        self.read_reference_slice_list(force=True)
        reference_slc = self.slcs[self.reference_date]

        # Load ifgs
        for ifg_dir in ifg_dirs:
            if ifg_dir not in self.ifg_dates:

                if os.path.basename(ifg_dir)[:8] in self.slcs.keys():
                    primary_slc = self.slcs[os.path.basename(ifg_dir)[:8]]
                else:
                    primary_slc = ''
                if os.path.basename(ifg_dir)[9:] in self.slcs.keys():
                    secondary_slc = self.slcs[os.path.basename(ifg_dir)[9:]]
                else:
                    secondary_slc = ''

                self.ifgs[ifg_dir[-17:]] = Interferogram(ifg_dir, primary_slc=primary_slc, secondary_slc=secondary_slc,
                                                         reference_slc=reference_slc, slice_list=self.reference_slice_id)
                self.ifg_dates.append(os.path.basename(ifg_dir))

        # combine the ifg and image dates
        self.dates = sorted(set(self.ifg_dates) - set(self.slc_dates))

        # Load the radar coordinates of the reference image
        self.load_radar_coordinates()

    def reload_stack(self):
        """
        Reload the metadata for the full stack

        :return:
        """

        for slc_key in self.slcs.keys():
            slc = self.slcs[slc_key]            # type: SLC
            slc.load_meta()
            slc.load_slice_meta()

        for ifg_key in self.ifgs.keys():
            ifg = self.ifgs[ifg_key]            # type: Interferogram
            ifg.load_meta()
            ifg.load_slice_meta()

    def perpendicular_baselines(self, mid_image_lat=0, mid_image_lon=0, mid_image_height=0):
        """
        Calculate the perpendicular baselines
        TODO Calculate the perpendicular baselines for a fixed mid lat/lon. (If not defined derived from reference image)
        """

        # Load all orbits

        # Get xyz of lat/lon location

        # Calc where orbit is perpendicular to orbit

        # Get perpendicular baselines

        # Get total baselines

        # Derive parallel baselines


    def create_interferogram_network(self, single_reference_date='',
                                     max_temporal_baseline=0, daisy_chain_width=0, max_perpendicular_baseline=0):
        # This method will create an interferogram network.
        # Run after reading in the data_stack.

        if max_perpendicular_baseline:
            perpendicular_baselines = self.perpendicular_baselines()
        else:
            perpendicular_baselines = []

        network = InterferogramNetwork(self.slcs.keys(),
                                       perpendicular_baselines=perpendicular_baselines,
                                       max_temporal_baseline=max_temporal_baseline,
                                       max_perpendicular_baseline=max_perpendicular_baseline,
                                       single_reference_date=single_reference_date,
                                       daisy_chain_width=daisy_chain_width)
        ifg_pairs = network.ifg_pairs

        # Finally create the requested ifg if they do not exist already
        ifg_ids = self.ifgs.keys()
        reference_key = self.reference_date
        date_int = np.sort([int(key) for key in self.slcs.keys()])

        for ifg_pair in ifg_pairs:

            primary_key = str(date_int[ifg_pair[0]])
            secondary_key = str(date_int[ifg_pair[1]])

            ifg_key_1 = primary_key + '_' + secondary_key
            ifg_key_2 = secondary_key + '_' + primary_key

            if not ifg_key_1 in ifg_ids and not ifg_key_2 in ifg_ids:
                folder = os.path.join(self.data_stack_folder, ifg_key_1)
                ifg = Interferogram(folder, secondary_slc=self.slcs[secondary_key], primary_slc=self.slcs[primary_key], reference_slc=self.slcs[reference_key])
                self.ifgs[ifg_key_1] = ifg

    def stack_data_iterator(self, processes=[], coordinates=[], in_coordinates=[], data_ids=[], polarisations=[], process_types=[],
                            slc_date=False, ifg_date=False, slc=True, ifg=False, full_image=True, slices=False, data=True, load_memmap=True):

        processes_out = []
        process_ids_out = []
        file_types_out = []
        coordinates_out = []
        in_coordinates_out = []
        slice_names_out = []
        images_out = []
        image_dates_out = []
        image_types_out = []

        if slc_date:
            if not slc_date in self.slcs.keys():
                logging.info('The selected SLC data should be part of the stack')
                return
        if ifg_date:
            if not ifg_date in self.ifgs.keys():
                logging.info('The selected interferogram date should be part of the stack')
                return

        for used, data, str_data, date in zip([slc, ifg], [self.slcs, self.ifgs], ['slc', 'ifg'], [slc_date, ifg_date]):
            if used:
                if date:
                    dates = [date]
                else:
                    dates = data.keys()

                for date in dates:
                    for iterator, data_used in zip([data[date].processing_image_data_iterator,
                                                    data[date].processing_slice_image_data_iterator], [full_image, slices]):
                        if data_used:
                            out_processes, out_process_ids, out_coordinates, out_in_coordinates, out_file_types, out_images \
                                = iterator(processes, coordinates, in_coordinates, data_ids, polarisations,
                                                                             process_types, data)
                            process_ids_out += out_process_ids
                            file_types_out += out_file_types
                            #slice_names_out += slice_names_slc
                            images_out += out_images
                            coordinates_out += out_coordinates
                            in_coordinates_out += out_in_coordinates
                            image_dates_out += [date for i in range(len(out_processes))]
                            image_types_out += [str_data for i in range(len(out_processes))]

        return image_types_out, image_dates_out, slice_names_out, processes_out, process_ids_out, coordinates_out, \
               in_coordinates_out, file_types_out, images_out

    def download_SRTM_dem(self, srtm_folder=None, username=None, password=None, srtm_type='SRTM3', parallel=True, n_processes=4):
        """
        Downloads the needed srtm data for this data_stack. srtm_folder is the folder the downloaded srtm tiles are
        stored.
        Username and password can be obtained at https://lpdaac.usgs.gov
        Documentation: https://lpdaac.usgs.gov/sites/default/files/public/measures/docs/NASA_SRTM_V3.pdf

        Description srtm data: https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/SRTMGL1_v003
        Description srtm q data: https://lpdaac.usgs.gov/node/505

        :param srtm_folder:
        :param username:
        :param password:
        :param srtm_type:
        :param parallel:
        :return:
        """

        if not srtm_folder:
            srtm_folder = os.path.join(self.settings.settings['paths']['DEM_database'], self.settings.settings['path_names']['DEM']['SRTM'])
        if not username:
            username = self.settings.settings['accounts']['EarthData']['username']
        if not password:
            password = self.settings.settings['accounts']['EarthData']['password']

        download = SrtmDownload(srtm_folder, username, password, srtm_type, n_processes=n_processes)
        download(coordinates=self.coordinates[srtm_type]['full'])

    def download_Tandem_X_dem(self, tandem_x_folder=None, username=None, password=None,
                              parallel=True, n_processes=1, tandem_x_type='TDM30'):
        """
        Downloads the needed TanDEM-X data for this data_stack. srtm_folder is the folder the downloaded srtm tiles are
        stored. Details on the product and login can be found under: https://geoservice.dlr.de/web/dataguide/tdm90/

        :param TanDEM-X_folder:
        :param username:
        :param password:
        :param parallel:
        :param n_processes:
        :return:
        """

        if not tandem_x_folder:
            tandem_x_folder = os.path.join(self.settings.settings['paths']['DEM_database'], self.settings.settings['path_names']['DEM']['TanDEM-X'])
        if not username:
            username = self.settings.settings['accounts']['DLR'][tandem_x_type]['username']
        if not password:
            password = self.settings.settings['accounts']['DLR'][tandem_x_type]['password']

        coordinates = self.coordinates[tandem_x_type]['full']
        download = TandemXDownload(tandem_x_folder, username, password, tandem_x_type=tandem_x_type,
                                   lon_resolution=int(coordinates.dlon * 3600), n_processes=n_processes)
        download(coordinates=coordinates)

    def create_coverage_shp_kml_geojson(self):
        """
        Create shapefiles for the full coverage and the individual swaths and bursts.

        Returns
        -------

        """

        # First load the bursts of the reference date.
        reference = self.slcs[self.reference_date]            # type: ImageConcatData
        reference.load_slice_meta()

        slice_shapes = []
        slice_names = []
        for key in reference.slice_data:
            slice = reference.slice_data[key]              # type: ImageProcessingData
            slice_shapes.append(slice.readfiles['original'].polygon.buffer(0.0001))
            slice_names.append(key)

        if not os.path.exists(os.path.join(self.data_stack_folder, 'coverage')):
            os.mkdir(os.path.join(self.data_stack_folder, 'coverage'))

        # Create the burst image file
        shapes = ReadWriteShapes()
        shapes.shapes = slice_shapes
        shapes.shape_names = slice_names
        shapes.write_kml(os.path.join(self.data_stack_folder, 'coverage', 'bursts.kml'))
        shapes.write_geo_json(os.path.join(self.data_stack_folder, 'coverage', 'bursts.geojson'))
        shapes.write_shapefile(os.path.join(self.data_stack_folder, 'coverage', 'bursts.shp'))

        # Now concatenate the individual swaths and create the swath shp/kml/geojson files
        swath_names = [slice_name[-7:] for slice_name in slice_names]
        swath_unique_names = np.unique(np.array(swath_names))
        swath_shapes = []

        for swath in swath_unique_names:
            swath_ids = np.ravel(np.argwhere(np.array(swath_names) == swath))
            swath_shapes.append(unary_union(np.array(slice_shapes)[swath_ids]))
        swaths = ReadWriteShapes()
        swaths.shapes = swath_shapes
        swaths.shape_names = list(swath_unique_names)
        swaths.write_kml(os.path.join(self.data_stack_folder, 'coverage', 'swaths.kml'))
        swaths.write_geo_json(os.path.join(self.data_stack_folder, 'coverage', 'swaths.geojson'))
        swaths.write_shapefile(os.path.join(self.data_stack_folder, 'coverage', 'swaths.shp'))

        # Finally do the same thing for the full image
        full_image_shape = unary_union(slice_shapes)
        full_image = ReadWriteShapes()
        full_image.shapes = [full_image_shape]
        full_image.shape_names = ['full_image']
        full_image.write_kml(os.path.join(self.data_stack_folder, 'coverage', 'full_image.kml'))
        full_image.write_geo_json(os.path.join(self.data_stack_folder, 'coverage', 'full_image.geojson'))
        full_image.write_shapefile(os.path.join(self.data_stack_folder, 'coverage', 'full_image.shp'))

    def get_overpasses(self):
        """
        Get a list of overpass dates for the first slice.

        :return:
        """

        overpasses = []

        overpass_time = datetime.datetime.strptime(self.reference_slice_datetime[0], '%Y-%m-%dT%H:%M:%S.%f').time()
        for str_date in self.slcs.keys():
            overpass_date = datetime.datetime.strptime(str_date, '%Y%m%d')
            overpass = datetime.datetime.combine(overpass_date, overpass_time)
            overpass = overpass.replace(second=0, microsecond=0)
            overpasses.append(overpass)

        return overpasses

    def create_concatenate_images(self, image_type, process, file_type, coor, data_id='', polarisation='', overwrite=False,
                                 output_type='disk', transition_type='full_weight', replace=False, cut_off=10,
                                 remove_input=False, tmp_directory='', no_processes=1, parallel=True):
        """
        This method is used to concatenate slices. Be sure that before this step is run the metadata is first created
        using the create_concatenate_meta_data function.

        :param str process: The process of which the result should be concatenated
        :param str file_type: Actual file type of the process that will be concatenated
        :param CoordinateSystem coor: Coordinatesystem of image. If size is already defined these will be used,
                    otherwise it will be calculated.
        :param str data_id: Data ID of process/file_type. Normally left empty
        :param str polarisation: Polarisation of data set
        :param bool overwrite: If data already exist, should we overwrite?
        :param str output_type: This is either memory or disk (Generally disk is preferred unless this dataset is not
                    saved to disk and is part of a processing pipeline.)
        :param str transition_type: Type of transition between burst. There are 3 types possible: 1) full weight, this
                    simply adds all values on top of each other. 2) linear, creates a linear transition zone between
                    the bursts. 3) cut_off, this creates a hard break between the different bursts and swaths without
                    overlap. (Note that option 2 and 3 are only possible when working in radar coordinates!)
        :param int cut_off: Number of pixels of the outer part of the image that will not be used because it could still
                    contain zeros.
        :param bool remove_input: Remove the disk data of the input images.
        :return:
        """

        if image_type == 'slc':
            images = [self.slcs[key] for key in self.slcs.keys()]
        elif image_type == 'ifg':
            images = [self.ifgs[key] for key in self.ifgs.keys()]
        else:
            raise TypeError('Only types slc or ifg are possible for concatenation!')

        # Now create the parallel chunks
        datasets = []
        for image in images:
            dat = dict()
            dat['concat_data'] = image
            dat['process'] = process
            dat['file_type'] = file_type
            dat['coor'] = coor
            dat['transition_type'] = transition_type
            dat['remove_input'] = remove_input
            dat['polarisation'] = polarisation
            dat['data_id'] = data_id
            dat['output_type'] = output_type
            dat['tmp_directory'] = tmp_directory
            dat['replace'] = replace
            dat['overwrite'] = overwrite
            dat['cut_off'] = cut_off

            datasets.append(dat)
        datasets = np.array(datasets)

        if parallel and no_processes > 1:
            with get_context("spawn").Pool(processes=no_processes, maxtasksperchild=5) as pool:
                # Process in chunks of 16
                chunk_size = 16
                for i in range(int(np.ceil(len(datasets) / chunk_size))):
                    last_dat = np.minimum((i + 1) * chunk_size, len(datasets))
                    success = pool.map(run_parallel, list(datasets[i*chunk_size:last_dat]))
        else:
            for dat in datasets:
                success = run_parallel(dat)
